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

## Current baseline

Using the SM80 (Ampere) code path with tuned block sizes:

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

1. **TMA (Tensor Memory Accelerator)** — SM120 supports cp.async.bulk.tensor. Currently blocked by CUTLASS DSL's `atom_tma_partition` MLIR op producing dynamic basis strides in standalone JIT functions. Unblocking this could give 10-20% uplift by eliminating thread-level address computation.

2. **cp.async pipelining** — The SM80 path uses synchronous global loads. SM120 supports cp.async (asynchronous copy to shared memory). Multi-stage pipelining could hide memory latency.

3. **Warp specialization** — SM80 path uses homogeneous warps. Dedicated producer/consumer warp roles (like SM90 kernel) could improve throughput even without WGMMA.

4. **Register-to-shared optimizations** — Explore swapAB MMA variants, V_in_regs, Q_in_regs for different head dimensions.

5. **Backward kernel tuning** — Current backward config is less optimized than forward. More systematic sweep of block sizes, pipeline stages, register allocation.

6. **Occupancy tuning** — Shared memory partitioning, register pressure management, persistent kernel scheduling.

7. **Block sparsity** — SM80 path lacks block sparse support. Adding it could unlock sparse attention patterns.

8. **GQA/MQA** — pack_gqa has a pre-existing compile error on SM80 path. Fixing it enables grouped-query attention.

9. **Variable-length sequences** — SM80 forward assumes 4D batched tensors. Varlen support needs substantial rewrite.

## Project structure

```
prepare.py       — benchmark harness + GPU detection (do not modify)
experiment.py    — kernel configs + optimizations (agent modifies this)
program.md       — agent instructions
results.tsv      — experiment log (untracked)
```

## License

MIT
