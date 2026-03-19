"""
FA4 SM120 kernel optimization experiment.
This is the file the agent modifies. Everything is fair game.

Usage: python experiment.py > run.log 2>&1

The script:
1. Applies any kernel configuration overrides
2. Patches FA4's dispatch/interface as needed
3. Runs the benchmark suite from prepare.py
4. Reports metrics in the standard output format

Key FA4 files that can be patched at runtime (in your site-packages flash_attn/cute/):
    interface.py    — dispatch logic, kernel selection, config (tile sizes, threads, stages)
    flash_fwd.py    — forward kernels (SM80 base, SM90 Hopper, SM120 dead code)
    flash_bwd.py    — backward kernels (SM80 base)
    mask.py         — attention mask application
    copy_utils.py   — TMA and async copy utilities
    softmax.py      — online softmax implementation
"""

import os
import sys
import time
import copy
import importlib

import torch

from prepare import (
    detect_gpu,
    run_benchmark_suite,
    check_correctness,
    compare_results,
    PROBLEM_SIZES_FULL,
    PROBLEM_SIZES_QUICK,
    ProblemSize,
    get_default_attn_fn,
)


# ---------------------------------------------------------------------------
# Configuration — the agent tunes these
# ---------------------------------------------------------------------------

# Forward kernel config overrides (applied via monkey-patching interface.py)
FWD_CONFIG = {
    # Tile sizes
    "m_block_size": 128,          # M tile (query block)
    "n_block_size": 128,          # N tile (key block)
    # Threading
    "num_threads": 256,           # total threads (warps * 32)
    # Pipeline
    "num_stages": 1,              # shared memory pipeline stages
}

# Backward kernel config overrides
BWD_CONFIG = {
    "m_block_size": 64,
    "n_block_size": 64,
    "num_threads": 256,
    "num_stages_Q": 2,            # Q double-buffering for d>64
    "num_stages_dO": 1,
}

# Which benchmark suite to run
PROBLEM_SIZES = PROBLEM_SIZES_FULL
DO_BACKWARD = True
DO_CORRECTNESS = True


# ---------------------------------------------------------------------------
# Kernel patching — override FA4 behavior at runtime
# ---------------------------------------------------------------------------

def apply_config_patches():
    """Apply configuration overrides to the installed FA4 package.

    This function monkey-patches the FA4 interface module to use our
    optimized SM120 configurations instead of the defaults.

    Modify this function to experiment with different kernel configs,
    dispatch strategies, or entirely new kernel implementations.
    """
    try:
        import flash_attn.cute.interface as interface
    except ImportError:
        print("ERROR: flash_attn.cute not importable. Is FA4 installed?")
        return False

    # The current baseline: SM120 dispatches to family 8 (SM80 path)
    # with the tuned config above. No patches needed for baseline.
    #
    # To experiment, you can:
    # 1. Override _arch_dispatch_family() to route SM120 differently
    # 2. Patch _flash_attn_fwd() to inject custom config
    # 3. Add entirely new kernel classes
    # 4. Patch copy_utils for different memory access patterns

    return True


def get_experiment_attn_fn():
    """Get the attention function with experimental patches applied.

    Returns the FA4 flash_attn_func with any patches from apply_config_patches().
    Override this to return a completely different implementation if needed.
    """
    apply_config_patches()

    from flash_attn.cute import flash_attn_func

    def attn_fn(q, k, v, causal=False):
        return flash_attn_func(q, k, v, causal=causal)

    return attn_fn


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("  FA4 SM120 Kernel Optimization Experiment")
    print("=" * 80)
    print()

    arch, gpu_name = detect_gpu()
    print()

    # Apply patches and get attention function
    attn_fn = get_experiment_attn_fn()

    # Correctness check
    if DO_CORRECTNESS:
        ok = check_correctness(attn_fn)
        if not ok:
            print("\nERROR: Correctness check failed! Aborting benchmark.")
            print("status: crash")
            sys.exit(1)

    # Run benchmark
    t0 = time.time()
    results = run_benchmark_suite(
        attn_fn,
        problems=PROBLEM_SIZES,
        do_backward=DO_BACKWARD,
        label="experiment",
    )
    total_time = time.time() - t0

    # Extract summary metrics
    fwd_ok = [r for r in results if r.status == "ok" and r.fwd_tflops > 0]
    bwd_ok = [r for r in results if r.status == "ok" and r.bwd_tflops > 0]

    from statistics import geometric_mean

    print("---")
    if fwd_ok:
        fwd_peak = max(r.fwd_tflops for r in fwd_ok)
        fwd_geomean = geometric_mean([r.fwd_tflops for r in fwd_ok])
        print(f"fwd_tflops_peak:     {fwd_peak:.1f}")
        print(f"fwd_tflops_geomean:  {fwd_geomean:.1f}")
    if bwd_ok:
        bwd_peak = max(r.bwd_tflops for r in bwd_ok)
        bwd_geomean = geometric_mean([r.bwd_tflops for r in bwd_ok])
        print(f"bwd_tflops_peak:     {bwd_peak:.1f}")
        print(f"bwd_tflops_geomean:  {bwd_geomean:.1f}")

    peak_vram = max(r.peak_vram_mb for r in results if r.status == "ok") if fwd_ok else 0
    n_crash = sum(1 for r in results if r.status == "crash")
    print(f"peak_vram_mb:        {peak_vram:.1f}")
    print(f"total_seconds:       {total_time:.1f}")
    print(f"configs_tested:      {len(results)}")
    print(f"configs_crashed:     {n_crash}")


if __name__ == "__main__":
    main()
