# Corrected native CUDA graph continuation

Public `specforge train` CLI, core commit `5a35ff35b0ba939242f294907892a2d8955a52b9`, Torch 2.14.0+cu130 / TorchTitan 0.3.0 on H200. All 168 Python source hashes are retained per layout and checked against the committed tree.

| Layout | Visible GPUs | Uninterrupted 3 vs 1 + resume to 3 | HF export | CUDA capture |
| --- | --- | --- | --- | --- |
| DP2 | 2,3 | 152 tensors bitwise equal | 36 tensors bitwise equal | logged |
| TP2 | 4,5 | 152 tensors bitwise equal | 36 tensors bitwise equal | logged |
| CP2 | 6,7 | 152 tensors bitwise equal | 36 tensors bitwise equal | logged |

All nine child training commands exit 0. These are tiny synthetic DFlash2 correctness fixtures with accumulation 2, objective chunking and selector ramp. The three layouts ran concurrently on disjoint pairs after all performance trials ended; their elapsed times are not performance evidence.

Each directory retains the exact orchestration script, launcher output, resolved YAML, per-phase logs, source hashes and tensor-comparison report. Binary checkpoints/exports remain on the devbox and are omitted here; the report records the comparison performed before omission. The script and YAML provide full child commands and environment settings. Launcher command for each layout was:

```sh
/scratch/specforge-torchtitan-20261003/venv/bin/python /scratch/specforge-torchtitan-20261003/verify_native_loss_fix_LAYOUT.py > /scratch/specforge-torchtitan-20261003/native-cuda-loss-fix-LAYOUT.log 2>&1
```

These checks cover native Trainer compilation and CUDA capture/replay on the final numerical policy. They do not imply GraphTrainer TP/CP support, multi-node correctness, or real-data convergence.
