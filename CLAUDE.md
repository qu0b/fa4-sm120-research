# CLAUDE.md

## Project Overview

Autonomous kernel optimization research for FlashAttention-4 on SM120 (RTX PRO 6000 Blackwell Workstation) GPUs. Modeled after karpathy/autoresearch — an AI agent iterates on kernel configurations, benchmarks, keeps improvements, discards regressions, and repeats.

## How to run experiments

```bash
uv sync                                          # install deps (first time)
uv run python experiment.py > run.log 2>&1       # run experiment
grep "^fwd_tflops_geomean:\|^bwd_tflops_geomean:" run.log
```

## File roles

- `prepare.py` — READ ONLY. Benchmark harness, problem sizes, GPU detection, metrics.
- `experiment.py` — AGENT EDITS THIS. Kernel configs, patches, new kernels.
- `program.md` — Full agent instructions. Read this first for any new experiment run.
- `patches/` — Modified FA4 source files for deeper changes.
- `results.tsv` — Experiment log (untracked by git).

## Key paths

FA4 source files (in your site-packages):
```
flash_attn/cute/interface.py     — dispatch logic
flash_attn/cute/flash_fwd.py     — forward kernels
flash_attn/cute/flash_bwd.py     — backward kernels
flash_attn/cute/copy_utils.py    — TMA and async copy
flash_attn/cute/mask.py          — attention masks
```

## Metrics

- **Primary**: `fwd_tflops_geomean` — geometric mean of forward TFLOPS across all problem sizes
- **Secondary**: `bwd_tflops_geomean`, `fwd_tflops_peak`, `peak_vram_mb`
- Higher TFLOPS = better. Lower VRAM = better (soft constraint).

## SM120 hardware facts

- sm_120a, 188 SMs, 95 GB VRAM, ~300 TFLOPS bf16 theoretical
- Supports: mma.sync, cp.async, TMA (cp.async.bulk.tensor)
- Does NOT support: WGMMA, TMEM/tcgen05
- Shared memory: 99KB optin (101376 bytes)
- Current baseline: 275.5 TFLOPS peak (92% utilization) via SM80 path
