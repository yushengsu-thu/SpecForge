"""Forward-boundary state and shared objective integration for draft MoE."""

from dataclasses import asdict, replace

from .layer import MoELayer


def moe_resume_contract(model):
    """Persist topology and math; dispatch can change without changing weights."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, MoELayer):
            fields = asdict(module.moe_config)
            fields.pop("dispatch")
            layers.append((name, fields))
    return {"specforge_moe": {"version": 1, "layers": tuple(layers)}} if layers else {}


def apply_pending_balance_updates(model) -> None:
    """Begin one model forward, outside any activation-checkpoint region.

    Official Qwen routing has no correction-bias update. This boundary owns
    forward-local bookkeeping now; a future bias-balanced family must apply its
    detached pending update here, before any router can be recomputed.
    """
    for module in model.modules():
        if isinstance(module, MoELayer):
            module.begin_forward()


def finalize_step_output(model, output):
    """Add each layer's differentiable auxiliary term once, for every strategy.

    With additive loss_terms, the auxiliary mean is weighted by the main
    objective's detached denominator. TrainerCore's existing global denominator
    normalization therefore scales both losses identically under accumulation
    and data parallelism. Without loss_terms, both use the microbatch mean.
    """
    auxiliary = None
    ratios = dict(output.ratio_metrics)
    denominator = output.loss.detach().new_ones(())
    if output.loss_terms is not None:
        denominator = output.loss_terms[1].detach()
    layer_index = 0
    for module in model.modules():
        if not isinstance(module, MoELayer):
            continue
        records = module.take_auxiliary_records()
        if records:
            count = len(records)
            raw = sum(record[0] for record in records) / count
            weighted = raw * module.moe_config.aux_loss_coeff
            auxiliary = weighted if auxiliary is None else auxiliary + weighted
            prefix = f"moe/layer_{layer_index}"
            for name, value in (
                ("aux_loss", raw.detach()),
                ("router_entropy", sum(record[1] for record in records) / count),
                (
                    "active_expert_fraction",
                    sum(record[2] for record in records) / count,
                ),
            ):
                ratios[f"{prefix}/{name}"] = (value * denominator, denominator)
        layer_index += 1
    if auxiliary is None:
        return output
    loss_terms = output.loss_terms
    if loss_terms is not None:
        loss_terms = (loss_terms[0] + auxiliary * denominator, loss_terms[1])
    ratios["moe/aux_loss"] = (auxiliary.detach() * denominator, denominator)
    return replace(
        output,
        loss=output.loss + auxiliary,
        loss_terms=loss_terms,
        ratio_metrics=ratios,
    )
