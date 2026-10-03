# Candidate3 focused regression

Source snapshot: `/scratch/specforge-torchtitan-20261003/graph-candidate3-focused`, copied from the current local candidate3 core. `graph-candidate3-focused-manifest.json` records SHA256 hashes for 176 production/test Python files and the full selected pytest command. This is a separate immutable-for-this-run test directory; it did not modify root's working remote checkout.

The combined H200 GPU0 run completed successfully: **82 passed, 8 skipped, 306 subtests passed**. The 8 skipped cases were the parameter-cache tests, whose fixture requires a clean single-process Gloo group; the CLI regression had already initialized a process group in that combined pytest process.

The complete parameter-cache file was therefore run in its own fresh process and all **8 passed**. Together these runs cover all **90 selected tests**, with no unresolved skips or failures. The raw combined and isolated logs are retained. Expected warnings concern optional flash-attn, existing FSDP API deprecations, and Dynamo's handling of the lazy lru_cache wrapper.

Isolated command, from the same snapshot:

```sh
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /scratch/specforge-torchtitan-20261003/venv/bin/python -m pytest tests/test_training/test_torchtitan_parameter_cache.py -q > /scratch/specforge-torchtitan-20261003/graph-candidate3-cache-isolated.log 2>&1
```

Torch2.13 legacy convolution/regression checks had already passed before this run and were not repeated; no relevant legacy production code changed after those checks. This focused suite is not a multi-GPU throughput, convergence, or serving acceptance test.

## Follow-up: resume compilation-mode identity

After the candidate3 run, the frontend resume contract also began recording the
exact Graph Inductor mode (`full` or `regional`). The entire frontend test file
passed **11 tests and 10 subtests** in a fresh CPU-only invocation, including
rejection of missing mode metadata. This follow-up changes checkpoint metadata
validation, not candidate3 arithmetic.

`frontend-mode-identity-verification.json` records the exact command, exit status,
tool-captured pytest summary, and SHA256 hashes of the frontend and test source.
Those hashes match both the tested mutable remote checkout and local commit
`5a35ff35b0ba939242f294907892a2d8955a52b9`. **No raw log was saved for this
follow-up**; the JSON explicitly records that limitation. No test rerun was needed
to produce this archive.
