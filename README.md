# fa4-sm120-research

Autonomous kernel optimization research for FlashAttention-4 on SM120 (RTX PRO 6000 Blackwell Workstation) GPUs.

Inspired by [autoresearch](https://github.com/karpathy/autoresearch) — the same "agent programs agent" pattern applied to GPU kernel optimization instead of model pretraining. An AI agent iterates on kernel configurations, benchmarks each change for a fixed time budget, keeps improvements, discards regressions, and repeats. You wake up to a log of experiments and faster kernels.

## How it works

The repo has three files that matter:

- **`prepare.py`** — fixed infrastructure: GPU detection, benchmark harness, metric extraction, reference implementations. Not modified by the agent.
- **`experiment.py`** — the single file the agent edits. Contains kernel configurations, dispatch logic, custom kernel classes, and any optimization code. Everything is fair game: tile sizes, pipeline stages, thread counts, memory layouts, copy strategies, new kernel variants.
- **`program.md`** — agent instructions. The human writes this; the agent follows it.

By design, each benchmark runs for a **fixed set of problem sizes** with warmup and repeated timing. The metrics are **TFLOPS** (higher is better) and **peak VRAM** (lower is better for equal TFLOPS). Results are directly comparable across experiments.

## Target hardware

- **GPU**: 2x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- **Arch**: SM 12.0 (sm_120a), 188 SMs, 95 GB VRAM
- **Supports**: mma.sync (SM80-style), cp.async, TMA (cp.async.bulk.tensor)
- **Does NOT support**: WGMMA (wgmma.mma_async), TMEM/tcgen05
- **Shared memory**: 48KB default, 99KB optin (101376 bytes)
- **Theoretical peak**: ~300 TFLOPS bf16

## Current state

As of March 2026, upstream FA4 has **official SM120 support** via dedicated kernel classes (`FlashAttentionForwardSm120`, `FlashAttentionBackwardSm120`) that subclass the SM80 kernels with a 99KB SMEM capacity check. Upstream uses 128 threads (4 warps) with conservative tile sizes.

Our earlier tuning found that **256 threads (8 warps) with larger tiles significantly outperforms** the upstream defaults. The first experiment should verify this on the latest codebase, then feed the improved config back upstream.

### Our tuned baseline (256 threads)

| Config | Fwd ms | Fwd TFLOPS |
|--------|--------|------------|
| b=1 s=8192 h=16 d=128 | 1.995 | 275.5 |
| b=2 s=4096 h=16 d=128 | 1.062 | 258.7 |
| b=2 s=4096 h=16 d=96 | 0.792 | 260.2 |
| b=4 s=2048 h=32 d=64 | 0.542 | 253.5 |

Peak: **275.5 TFLOPS** (92% utilization). The goal is to push beyond this.

## Quick start

**Requirements:** An SM120 GPU (RTX PRO 6000), Python 3.10+, `flash-attn-4` installed (`pip install flash-attn-4`).

```bash
# 1. Verify GPU
python -c "import torch; print(torch.cuda.get_device_name())"

# 2. Run baseline benchmark
python prepare.py --baseline

# 3. Run a single experiment
python experiment.py > run.log 2>&1
grep "^fwd_tflops_peak:\|^bwd_tflops_peak:\|^fwd_tflops_geomean:" run.log
```

## Research directions

Ordered roughly by expected impact:

1. **Validate 256-thread config on latest upstream** — Our tuning found 256 threads (8 warps) much faster than upstream's 128. First priority: reproduce this on the latest code, then PR.

2. **TMA (Tensor Memory Accelerator)** — SM120 supports cp.async.bulk.tensor. Previously blocked by CUTLASS DSL 4.4.1's `atom_tma_partition`. CUTLASS DSL is now at 4.4.2 — re-test. Could give 10-20% uplift.

3. **cp.async pipelining** — The SM80 path uses synchronous global loads. Multi-stage pipelining with cp.async could hide memory latency.

4. **Config search for SM120** — Upstream added `sm90_config_search.py`. Adapt it for SM120's 99KB SMEM / 128-thread constraints to systematically explore the config space.

5. **Backward kernel tuning** — Upstream uses conservative backward configs. Our earlier tuning found larger n_block (128 for d<=64) helps. Systematic sweep needed.

6. **Warp specialization** — Dedicated producer/consumer warp roles (like SM90 kernel) could help even without WGMMA.

7. **Feature parity** — Upstream SM120 lacks: block sparsity, paged KV, SplitKV, score_mod/mask_mod backward, deterministic backward. Each is a contribution opportunity.

## Project structure

```
prepare.py       — benchmark harness + GPU detection (do not modify)
experiment.py    — kernel configs + optimizations (agent modifies this)
program.md       — agent instructions
results.tsv      — experiment log (untracked)
```

## License

MIT
