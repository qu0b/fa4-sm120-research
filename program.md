# fa4-sm120-research

Autonomous kernel optimization for FlashAttention-4 on SM120 (RTX PRO 6000 Blackwell Workstation).

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar19`). The branch `qu0b/sm120-opt/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b qu0b/sm120-opt/<tag>` from current main.
3. **Read context files**: The repo is small. Read these for full context:
   - `README.md` — hardware specs, current baseline, research directions.
   - `prepare.py` — benchmark harness, problem sizes, metric extraction. **Do not modify.**
   - `experiment.py` — the file you modify. Kernel configs, dispatch patches, new kernels.
4. **Read the FA4 codebase**: Find the installed FA4 package via `python -c "import flash_attn.cute; print(flash_attn.cute.__path__)"`. Read these key files:
   - `interface.py` — dispatch logic, `_arch_dispatch_family()`, compile/invoke flow.
   - `flash_fwd.py` — forward kernels. `FlashAttentionForwardSm80` is the current SM120 path.
   - `flash_bwd.py` — backward kernels. `FlashAttentionBackwardSm80` is the current SM120 path.
   - `copy_utils.py` — TMA and async copy utilities.
   - `mask.py` — attention mask application.
5. **Verify baseline**: Run `python prepare.py --baseline` to confirm the GPU works and get reference numbers.
6. **Initialize results.tsv**: Create `results.tsv` with the header row. The baseline will be the first entry.
7. **Confirm and go**: Confirm setup is ready.

Once confirmed, kick off the experimentation.

## Experimentation

Each experiment modifies `experiment.py`, runs the benchmark suite, and records results. The benchmark suite tests a fixed set of problem sizes (batch, seqlen, heads, headdim, causal) and reports TFLOPS.

**What you CAN do:**
- Modify `experiment.py` — this is the only file you commit changes to. Everything is fair game:
  - Tune kernel configurations (tile sizes, thread counts, pipeline stages)
  - Monkey-patch FA4 modules at runtime to change dispatch, kernel selection, or behavior
  - Write entirely new kernel variants (e.g., SM120-specific TMA kernels)
  - Modify the attention function to use different code paths
  - Copy and modify FA4 kernel files into `patches/` for deeper changes
  - Deploy patches to your FA4 site-packages installation for testing

**What you CANNOT do:**
- Modify `prepare.py`. The benchmark harness and problem sizes are fixed.
- Change the metric definition. TFLOPS as computed by the harness is ground truth.
- Sacrifice correctness for speed. The correctness check must pass.

**The goal is simple: maximize fwd_tflops_geomean and bwd_tflops_geomean across the benchmark suite.** Peak TFLOPS matters, but geometric mean across all problem sizes is the primary metric — it penalizes regressions on any config.

**VRAM** is a soft constraint. Some increase is acceptable for TFLOPS gains, but OOM on any benchmark config is a crash.

**Correctness** is a hard constraint. The attention function must produce results matching PyTorch SDPA within the tolerance defined in `prepare.py`. A fast but wrong kernel is worse than a slow correct one.

**Simplicity criterion**: All else being equal, simpler is better. A 0.5 TFLOPS geomean improvement from 100 lines of hacky code? Probably not worth it. A 0.5 TFLOPS improvement from changing one config value? Definitely keep. A clean architectural change that opens the door to bigger gains? Keep even if the immediate gain is small.

**The first run**: Always establish the baseline first. Run `experiment.py` unmodified to record reference numbers.

## Hardware context

SM120 is workstation Blackwell. It shares SM80's compute primitives (mma.sync) but has newer memory subsystem features:

- **TMA (cp.async.bulk.tensor)**: Hardware tensor copies. Near-zero thread overhead. SM90/SM100 kernels use this for all global→shared loads. Currently BLOCKED — CUTLASS DSL's `atom_tma_partition` produces dynamic basis strides for standalone JIT functions. This is the highest-value research direction.
- **cp.async**: Asynchronous copy to shared memory. The SM80 path uses synchronous `cute.copy()`. Switching to cp.async with proper pipelining could hide latency.
- **99KB shared memory**: Larger than SM80's default. Allows bigger tiles or more pipeline stages.
- **188 SMs**: More SMs than typical SM80 GPUs. Persistent kernels or work-stealing could help occupancy.

What SM120 does NOT have:
- **WGMMA** (SM90+): No warpgroup-level matrix multiply. Must use warp-level mma.sync.
- **TMEM/tcgen05** (SM100+): No tensor memory. Can't use SM100 kernel path.

## Research strategies

From highest to lowest expected impact:

### 1. TMA memory loads (HIGH)
The SM90 forward kernel uses TMA for Q/K/V loads via `cpasync.make_tiled_tma_atom()`. SM120 hardware supports TMA. The blocker is that CUTLASS DSL's `atom_tma_partition` C++ MLIR verifier rejects the tensor types when called from standalone `@cute.jit` functions (produces `?{i64}@0` dynamic basis strides instead of `1@0` static).

Investigation paths:
- Study why the SM90 kernel's class-based `@cute.jit` produces static strides while standalone functions don't
- Try creating tensors with fully static strides (no `mark_layout_dynamic`)
- Try matching the SM90 kernel's exact tensor processing flow (transpose, assume_aligned, etc.)
- Bypass `tma_partition` entirely and write raw PTX `cp.async.bulk.tensor` via inline asm
- Check if newer CUTLASS DSL versions fix this

### 2. cp.async pipelining (MEDIUM)
The SM80 forward path loads K/V synchronously. SM120 supports cp.async for async shared memory loads. Pipeline: load next K tile while computing current.

- Study `flash_fwd.py` SM80 load paths
- Replace synchronous copies with cp.async + commit groups
- Add multi-stage K/V buffering (2-stage should fit in 99KB smem for most configs)

### 3. Warp specialization (MEDIUM)
The SM90 kernel dedicates warp 0 to data loading (producer) and warps 1-4 to MMA (consumer). The SM80 kernel uses homogeneous warps. Even without WGMMA, separating load and compute could help.

### 4. Block size / thread count sweep (LOW-MEDIUM)
Current config: m=128, n=128, threads=256, stages=1. Systematic sweep of:
- m_block: 64, 128, 192
- n_block: 64, 128, 192, 256
- threads: 128, 192, 256, 384
- stages: 1, 2

### 5. Register optimizations (LOW)
- Q_in_regs: keep Q in registers instead of shared memory
- V_in_regs: keep V in registers (frees shared memory)
- swapAB MMA variants for different data layouts

### 6. Backward kernel (LOW-MEDIUM)
Less optimized than forward. Same techniques apply: cp.async, pipeline stages, block sizes.

## Dual-GPU parallel experiments

The system has **2 GPUs**. `experiment.py` defines two experiments — `EXPERIMENT_A` and `EXPERIMENT_B` — that run **simultaneously**, one per GPU. This doubles throughput: ~24 experiments/hour instead of ~12.

**How to use this**:

Each iteration, you test TWO ideas at once. Set `EXPERIMENT_A` to one variant and `EXPERIMENT_B` to another. Run them in parallel. The script prints a side-by-side comparison showing which is better.

Strategies for picking A vs B:
- **A/B split**: A = current best, B = new idea. If B wins, it becomes the new A next round.
- **Two ideas**: A = idea 1, B = idea 2. Keep whichever is better (or both if both improve).
- **Bisection**: A = aggressive change, B = conservative change. Narrow in on the sweet spot.
- **Baseline check**: A = unchanged baseline, B = experimental. Ensures no measurement noise.

Running:
```bash
# Parallel (default): runs EXPERIMENT_A on cuda:0 and EXPERIMENT_B on cuda:1
python experiment.py > run.log 2>&1

# Single GPU only (for debugging or when only 1 GPU is free)
python experiment.py --single > run.log 2>&1
```

## Output format

The experiment script prints a summary for each experiment:

```
--- baseline_A ---
fwd_tflops_peak:     275.5
fwd_tflops_geomean:  260.0
bwd_tflops_peak:     180.0
bwd_tflops_geomean:  165.0
peak_vram_mb:        4500.0
total_seconds:       120.5
configs_tested:      16
configs_crashed:     0

--- n_block_192_B ---
fwd_tflops_peak:     278.3
fwd_tflops_geomean:  263.1
...

  COMPARISON: baseline_A vs n_block_192_B
  fwd_tflops_geomean        A=  260.0  B=  263.1  (+1.2%)  -> B wins
```

Extract key metrics:
```
grep "^fwd_tflops_geomean:\|^bwd_tflops_geomean:" run.log
grep "COMPARISON" -A 5 run.log
```

## Logging results

When an experiment is done, log BOTH results to `results.tsv` (tab-separated).

Header row and 6 columns:

```
commit	fwd_geomean	bwd_geomean	peak_vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. fwd_tflops_geomean (e.g. 260.0) — use 0.0 for crashes
3. bwd_tflops_geomean (e.g. 165.0) — use 0.0 for crashes
4. peak VRAM in GB (divide peak_vram_mb by 1024, round to .1f) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short description — prefix with `[A]` or `[B]` to track which experiment

Example:

```
commit	fwd_geomean	bwd_geomean	peak_vram_gb	status	description
a1b2c3d	260.0	165.0	4.4	keep	[A] baseline
a1b2c3d	260.0	165.0	4.4	keep	[B] baseline
b2c3d4e	260.0	165.0	4.4	discard	[A] baseline (control)
b2c3d4e	265.3	165.0	4.4	keep	[B] increase n_block to 192
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `qu0b/sm120-opt/mar19`).

LOOP FOREVER:

1. Look at the git state: current branch/commit, recent results in results.tsv.
2. Decide what to try next. Pick TWO ideas — one for each GPU. Prioritize:
   - Fixing crashes from the previous experiment
   - High-impact changes (TMA, cp.async) over low-impact (config tweaks)
   - Simple changes before complex ones
   - Combining previous near-misses
   - Use one GPU as control (baseline) when testing risky changes
3. Set `EXPERIMENT_A` and `EXPERIMENT_B` in `experiment.py`.
4. git commit.
5. Run: `python experiment.py > run.log 2>&1`
6. Read results: `grep "COMPARISON" -A 5 run.log`
7. If grep output is empty, the run crashed. `tail -50 run.log` for stack trace.
8. Record BOTH results in results.tsv (do NOT commit results.tsv).
9. If either experiment improved fwd_tflops_geomean: keep the commit, update the winning config as the new baseline for next round.
10. If both equal or worse: `git reset --soft HEAD~1` (discard).

**Timeout**: Each benchmark run should take 2-5 minutes. If it exceeds 15 minutes, kill it and treat as failure.

**Crashes**: If it's a typo or import error, fix and re-run. If it's fundamental (e.g., SM120 doesn't support an instruction), log crash, revert, and move on.

**Deploying FA4 patches**: If you need to modify FA4 source files (not just experiment.py):
1. Find the FA4 install path: `python -c "import flash_attn.cute; print(flash_attn.cute.__path__)"`
2. Copy the file: `cp <fa4_path>/FILE.py ./patches/FILE.py`
3. Make modifications to the local copy
4. Deploy: `cp ./patches/FILE.py <fa4_path>/FILE.py`
5. Reference the patch in your experiment.py commit message
6. Consider making experiment.py apply the patch programmatically (monkey-patching) so the experiment is self-contained

**NEVER STOP**: Once the experiment loop begins, do NOT pause to ask. The human might be away. If you run out of ideas, think harder: re-read the FA4 source code for new angles, study the SM90 kernel for techniques to port, try combining approaches, try more radical changes. The loop runs until manually interrupted.

## Key FA4 code paths to understand

### Forward dispatch (interface.py)
```
flash_attn_func() → _flash_attn_fwd()
  → _arch_dispatch_family(arch)  # SM120 → family 8
  → creates FlashAttentionForwardSm80(...)
  → kernel.compile(...)  # JIT to CUBIN
  → kernel(...)          # launch
```

### SM80 forward kernel (flash_fwd.py)
```
FlashAttentionForwardSm80.__call__():
  → to_cute_tensor(q,k,v,o)     # torch → cute tensors
  → _setup_attributes()          # MMA atoms, tile layouts
  → cute.compile(self.kernel, ...) → launch

FlashAttentionForwardSm80.kernel():
  → loop over n_blocks:
    → cute.copy(gmem_K → smem_K)   # SYNCHRONOUS global→shared
    → cute.copy(gmem_V → smem_V)   # SYNCHRONOUS global→shared
    → mma(Q, K^T)                   # warp-level mma.sync
    → softmax
    → mma(P, V)                     # warp-level mma.sync
  → store O to global
```

### SM90 forward kernel (flash_fwd.py) — TMA reference
```
FlashAttentionForwardSm90.__call__():
  → to_cute_tensor(q,k,v,o)
  → assume_tensor_aligned()
  → layout_utils.select(mK, [1,3,2,0])  # transpose to (seq,hdim,head,batch)
  → sm90_utils.make_smem_layout(...)      # TMA-compatible smem layout with swizzle
  → cpasync.make_tiled_tma_atom(op, mK, smem_layout, tile_shape, mcast)  # TMA descriptor
  → kernel gets tma_tensor_K (not raw mK!)

FlashAttentionForwardSm90.kernel():
  → warp 0 (producer): TMA loads via cpasync.tma_partition → cute.copy
  → warps 1-4 (consumer): WGMMA compute via warpgroup MMA
  → pipeline synchronization via mbarriers
```

The SM90 kernel's TMA+WGMMA pattern achieves >90% MMA utilization on Hopper. On SM120, we can use TMA (if unblocked) but must use mma.sync instead of WGMMA. Even so, TMA alone should give 10-20% uplift by freeing threads from address computation.
