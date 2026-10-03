"""Optional TorchTitan 0.3.0 data-parallel backend for DFlash-family drafts.

TorchTitan owns the per-decoder-block FSDP2 policy. SpecForge continues to own
the loss, gradient normalization, FP32-master optimizer, and checkpoint format.
Only the draft is sharded: the composite model's frozen teacher tables remain
replicated and keep their original state-dict names.
"""

from __future__ import annotations

import contextlib
from importlib import metadata
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from specforge.training.backend import (
    FSDPTrainingBackend,
    ParallelConfig,
    _foreach_scale_,
)


def _load_torchtitan_components():
    """Keep the optional dependency out of the default FSDP import path."""
    try:
        version = metadata.version("torchtitan")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "training.backend='torchtitan' requires torchtitan==0.3.0 and its "
            "compatible PyTorch runtime; install the optional TorchTitan backend"
        ) from exc
    if version.split("+", 1)[0] != "0.3.0":
        raise ImportError(
            "training.backend='torchtitan' supports torchtitan==0.3.0; "
            f"found {version!r}"
        )
    try:
        from torch.distributed.fsdp import FSDPModule
        from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
    except ImportError as exc:
        raise ImportError(
            "Unable to load TorchTitan 0.3.0 FSDP2 components. Use the PyTorch "
            "version required by that TorchTitan release."
        ) from exc
    return apply_fsdp_to_decoder, FSDPModule


@contextlib.contextmanager
def _decoder_layout(draft: nn.Module):
    """Expose Titan's decoder accessors without renaming or moving parameters.

    Titan iterates ``layers.items()`` whereas the HF-derived drafts iterate a
    ModuleList. A temporary accessor is sufficient; even the container object
    stays unchanged. The absent embedding/head belong to the outer SpecForge
    model, so Titan leaves the draft's norm, feature projection, and any custom
    head in its root FSDP unit. All temporary attributes are restored on errors.
    """
    layers = getattr(draft, "layers", None)
    if not isinstance(layers, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError("TorchTitan draft layers must be a ModuleList or ModuleDict")
    if not layers:
        raise ValueError("TorchTitan backend requires at least one draft decoder layer")
    if any(getattr(layer, "moe_enabled", False) for layer in layers.children()):
        raise ValueError(
            "TorchTitan backend currently supports dense draft layers only"
        )
    for name in ("tok_embeddings", "lm_head"):
        if getattr(draft, name, None) is not None:
            raise ValueError(
                f"TorchTitan draft must keep teacher {name} outside the trainable draft"
            )

    missing = object()
    attributes = [(draft, "enable_weight_tying", False)]
    for name in ("tok_embeddings", "lm_head", "norm"):
        if not hasattr(draft, name):
            attributes.append((draft, name, None))
    if isinstance(layers, nn.ModuleList):
        attributes.append((layers, "items", lambda: layers._modules.items()))
    restored = []
    try:
        for obj, name, value in attributes:
            previous = obj.__dict__.get(name, missing)
            restored.append((obj, name, previous))
            # These are non-parameter accessors, not registered module aliases.
            object.__setattr__(obj, name, value)
        yield draft
    finally:
        for obj, name, previous in reversed(restored):
            if previous is missing:
                object.__delattr__(obj, name)
            else:
                object.__setattr__(obj, name, previous)


class TorchTitanTrainingBackend(FSDPTrainingBackend):
    """Use TorchTitan's FSDP2 decoder policy with SpecForge training semantics.

    The initial backend supports BF16, dense DFlash/DFlash2/DSpark drafts and
    data parallelism only. It deliberately does not install Titan's trainer,
    loss normalization, SPMD tensor annotations, or optimizer.
    """

    name = "torchtitan"

    def __init__(
        self,
        parallel_config: ParallelConfig,
        *,
        optimizer_factory=None,
    ) -> None:
        super().__init__(parallel_config, optimizer_factory=optimizer_factory)
        self._draft: Optional[nn.Module] = None
        self._fsdp_modules: tuple[nn.Module, ...] = ()
        self._dp_mesh = None
        pc = parallel_config
        if pc.tp_size != 1 or pc.sp_size != 1:
            raise ValueError("TorchTitan backend currently supports DP only (TP/SP=1)")
        if pc.sharding_strategy not in {"FULL_SHARD", "SHARD_GRAD_OP"}:
            raise ValueError(
                "TorchTitan backend requires FULL_SHARD or SHARD_GRAD_OP; "
                f"got {pc.sharding_strategy!r}"
            )
        if pc.param_dtype != torch.bfloat16:
            raise ValueError("TorchTitan backend currently requires BF16 parameters")

    def prepare_model(
        self,
        model: nn.Module,
        *,
        wrap: bool = True,
        optimizer_target: Optional[nn.Module] = None,
    ) -> nn.Module:
        if self.module is not None:
            raise RuntimeError("TorchTitanTrainingBackend.prepare_model called twice")
        if not wrap:
            self._draft = optimizer_target if optimizer_target is not None else model
            return super().prepare_model(
                model, wrap=False, optimizer_target=optimizer_target
            )
        if not dist.is_initialized():
            raise RuntimeError(
                "TorchTitan backend requires an initialized process group"
            )
        draft = optimizer_target if optimizer_target is not None else model
        draft_ids = {id(parameter) for parameter in draft.parameters()}
        if not any(module is draft for module in model.modules()):
            raise ValueError("optimizer_target must belong to the training model")
        if any(
            parameter.requires_grad and id(parameter) not in draft_ids
            for parameter in model.parameters()
        ):
            raise ValueError(
                "TorchTitan backend requires all trainable parameters in draft"
            )
        parameters = tuple(draft.parameters())
        if not parameters:
            raise ValueError("TorchTitan backend requires a nonempty draft model")
        device = parameters[0].device
        if device.type != "cuda":
            raise ValueError(
                "TorchTitan backend currently requires CUDA draft parameters"
            )
        if any(parameter.device != device for parameter in parameters):
            raise ValueError(
                "TorchTitan draft parameters must reside on one CUDA device"
            )
        apply_fsdp, fsdp_module_cls = _load_torchtitan_components()
        from torch.distributed.device_mesh import DeviceMesh

        pc = self.parallel_config
        group = pc.fsdp_process_group
        if group is None:
            group = dist.group.WORLD
        group_size = dist.get_world_size(group)
        if group_size != pc.world_size:
            raise ValueError("TorchTitan DP group must span the complete trainer world")
        # Reuse the process group established by SpecForge; do not create a
        # second distributed lifecycle or reorder process-group collectives.
        self._dp_mesh = DeviceMesh.from_group(
            group, device.type, mesh_dim_names=("fsdp",)
        )
        policy = "always" if pc.sharding_strategy == "FULL_SHARD" else "never"
        with _decoder_layout(draft):
            apply_fsdp(
                draft,
                self._dp_mesh,
                param_dtype=pc.param_dtype,
                reduce_dtype=pc.param_dtype,
                pp_enabled=False,
                reshard_after_forward_policy=policy,
            )
        self._fsdp_modules = tuple(
            module for module in draft.modules() if isinstance(module, fsdp_module_cls)
        )
        if not isinstance(draft, fsdp_module_cls):
            raise RuntimeError("TorchTitan did not apply FSDP2 to the draft root")
        for module in self._fsdp_modules:
            # Titan disables division to use its own global-token loss. The
            # SpecForge controller instead assumes DP-averaged gradients.
            module.set_gradient_divide_factor(float(group_size))
        # DFlash2/DSpark consume custom root head parameters after draft.forward
        # returns. They must stay materialized through the outer loss forward.
        draft.set_reshard_after_forward(False, recurse=False)
        self._draft = draft
        self.module = model
        self._wrapped = True
        self._wrapper_kind = "fsdp2"
        self.auto_wrap_block_classes = {
            type(layer) for layer in draft.layers.children()
        }
        self.ignored_frozen_modules = self._frozen_target_modules(model)
        if self._optimizer_factory is not None:
            self.optimizer = self._optimizer_factory(draft)
            self._configure_optimizer_grad_norm()
        return model

    @contextlib.contextmanager
    def forward_context(self, *, is_boundary: bool = True):
        """Configure FSDP2 accumulation before the complete forward/backward."""
        if self._wrapped:
            self._draft.set_requires_gradient_sync(is_boundary)
            self._draft.set_reshard_after_backward(is_boundary)
        try:
            yield
        finally:
            if self._wrapped:
                self._draft.set_requires_gradient_sync(True)
                self._draft.set_reshard_after_backward(True)

    def backward(self, loss: torch.Tensor, *, is_boundary: bool = True) -> None:
        # Also honor direct callers that do not use TrainerCore's context.
        if self._wrapped:
            self._draft.set_requires_gradient_sync(is_boundary)
            self._draft.set_reshard_after_backward(is_boundary)
        loss.backward()

    def scale_gradients(self, factor: torch.Tensor) -> None:
        if self.module is None:
            raise RuntimeError("scale_gradients called before prepare_model")
        from torch.distributed.tensor import DTensor

        # DTensor dispatch would perform distributed operations on a local
        # scale factor; the controller already globally reduced that factor.
        with torch.no_grad():
            gradients = []
            for parameter in self._draft.parameters():
                if parameter.grad is not None:
                    grad = parameter.grad
                    gradients.append(
                        grad.to_local() if isinstance(grad, DTensor) else grad
                    )
            _foreach_scale_(gradients, factor)

    def _module_state_dict(self) -> dict:
        if not self._wrapped:
            return super()._module_state_dict()
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        # With full_state_dict + cpu_offload, only rank zero receives full CPU
        # tensors. The original composite/draft prefixes remain unchanged.
        return get_model_state_dict(
            self.module,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

    def _load_module_state_dict(self, model_state: dict) -> None:
        if not self._wrapped:
            super()._load_module_state_dict(model_state)
            return
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        # Match SpecForge's existing contract: every rank reads the shared
        # full model state before restoring its own optimizer/RNG shard.
        set_model_state_dict(
            self.module,
            model_state,
            options=StateDictOptions(full_state_dict=True, strict=True),
        )


__all__ = ["TorchTitanTrainingBackend"]
