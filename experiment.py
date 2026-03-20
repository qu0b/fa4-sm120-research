"""
FA4 SM120 kernel optimization experiment.
This is the file the agent modifies. Everything is fair game.

Usage:
    python experiment.py                    # run two experiments in parallel (one per GPU)
    python experiment.py --single           # run one experiment on cuda:0 only
    CUDA_VISIBLE_DEVICES=1 python experiment.py --single  # run on specific GPU

The script:
1. Defines two experiment variants (A on GPU 0, B on GPU 1)
2. Applies kernel config patches for each
3. Runs benchmarks in parallel
4. Reports metrics for both, indicating which is better

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
import json
import time

import torch

from prepare import (
    detect_gpu,
    detect_all_gpus,
    run_benchmark_suite,
    check_correctness,
    compare_results,
    PROBLEM_SIZES_FULL,
    PROBLEM_SIZES_QUICK,
    ProblemSize,
    get_default_attn_fn,
)


# ---------------------------------------------------------------------------
# Experiment definitions — the agent modifies these
# ---------------------------------------------------------------------------
# Define two experiments to run in parallel. Each gets its own GPU.
# For baseline, both are identical. The agent changes one or both.

EXPERIMENT_A = {
    "name": "baseline_A",
    "description": "baseline (no changes)",
    # Forward config overrides
    "fwd_config": {
        "m_block_size": 128,
        "n_block_size": 128,
        "num_threads": 256,
        "num_stages": 1,
    },
    # Backward config overrides
    "bwd_config": {
        "m_block_size": 64,
        "n_block_size": 64,
        "num_threads": 256,
    },
}

EXPERIMENT_B = {
    "name": "baseline_B",
    "description": "baseline (no changes)",
    # Forward config overrides
    "fwd_config": {
        "m_block_size": 128,
        "n_block_size": 128,
        "num_threads": 256,
        "num_stages": 1,
    },
    # Backward config overrides
    "bwd_config": {
        "m_block_size": 64,
        "n_block_size": 64,
        "num_threads": 256,
    },
}

# Benchmark settings
PROBLEM_SIZES = PROBLEM_SIZES_FULL
DO_BACKWARD = True
DO_CORRECTNESS = True


# ---------------------------------------------------------------------------
# Kernel patching — override FA4 behavior at runtime
# ---------------------------------------------------------------------------

def apply_config_patches(experiment: dict):
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


def get_experiment_attn_fn(experiment: dict):
    """Get the attention function with experimental patches applied.

    Returns the FA4 flash_attn_func with any patches from apply_config_patches().
    Override this to return a completely different implementation if needed.
    """
    apply_config_patches(experiment)

    from flash_attn.cute import flash_attn_func

    def attn_fn(q, k, v, causal=False):
        return flash_attn_func(q, k, v, causal=causal)

    return attn_fn


# ---------------------------------------------------------------------------
# Single-GPU experiment runner
# ---------------------------------------------------------------------------

def run_single(experiment: dict, device: str = "cuda:0"):
    """Run one experiment on one GPU. Returns summary dict."""
    name = experiment["name"]
    print(f"\n{'#'*80}")
    print(f"  Experiment: {name}")
    print(f"  Description: {experiment['description']}")
    print(f"  Device: {device}")
    print(f"{'#'*80}")

    detect_gpu(int(device.split(":")[1]))
    print()

    attn_fn = get_experiment_attn_fn(experiment)

    if DO_CORRECTNESS:
        ok = check_correctness(attn_fn, device=device)
        if not ok:
            print(f"\nERROR: Correctness check failed for {name}!")
            return {"name": name, "status": "crash", "error": "correctness"}

    t0 = time.time()
    results = run_benchmark_suite(
        attn_fn,
        problems=PROBLEM_SIZES,
        do_backward=DO_BACKWARD,
        label=name,
        device=device,
    )
    elapsed = time.time() - t0

    from statistics import geometric_mean

    fwd_ok = [r for r in results if r.status == "ok" and r.fwd_tflops > 0]
    bwd_ok = [r for r in results if r.status == "ok" and r.bwd_tflops > 0]

    summary = {
        "name": name,
        "description": experiment["description"],
        "status": "ok",
        "fwd_tflops_peak": max(r.fwd_tflops for r in fwd_ok) if fwd_ok else 0,
        "fwd_tflops_geomean": geometric_mean([r.fwd_tflops for r in fwd_ok]) if fwd_ok else 0,
        "bwd_tflops_peak": max(r.bwd_tflops for r in bwd_ok) if bwd_ok else 0,
        "bwd_tflops_geomean": geometric_mean([r.bwd_tflops for r in bwd_ok]) if bwd_ok else 0,
        "peak_vram_mb": max(r.peak_vram_mb for r in results if r.status == "ok") if fwd_ok else 0,
        "total_seconds": elapsed,
        "configs_tested": len(results),
        "configs_crashed": sum(1 for r in results if r.status == "crash"),
    }
    return summary


# ---------------------------------------------------------------------------
# Parallel dual-GPU experiment runner
# ---------------------------------------------------------------------------

def run_parallel(exp_a: dict, exp_b: dict):
    """Run two experiments in parallel, one per GPU. Returns both summaries."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    gpus = detect_all_gpus()
    print(f"\nParallel mode: {len(gpus)} GPUs available")
    for gid, arch, name in gpus:
        print(f"  cuda:{gid} — {name} (sm_{arch}0)")

    if len(gpus) < 2:
        print("WARNING: Only 1 GPU available. Running sequentially.")
        sa = run_single(exp_a, "cuda:0")
        sb = run_single(exp_b, "cuda:0")
        return sa, sb

    # Run both experiments in parallel on separate GPUs using threads.
    # Each thread sets its own CUDA device. FA4 kernels are JIT-compiled
    # per device, so there's no interference.
    def _run_on_gpu(exp, device):
        torch.cuda.set_device(device)
        return run_single(exp, device)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(_run_on_gpu, exp_a, "cuda:0")
        fut_b = pool.submit(_run_on_gpu, exp_b, "cuda:1")
        sa = fut_a.result()
        sb = fut_b.result()

    return sa, sb


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_summary(summary: dict):
    """Print a single experiment summary in the standard grep-able format."""
    print(f"\n--- {summary['name']} ---")
    print(f"description:         {summary.get('description', '')}")
    print(f"fwd_tflops_peak:     {summary['fwd_tflops_peak']:.1f}")
    print(f"fwd_tflops_geomean:  {summary['fwd_tflops_geomean']:.1f}")
    print(f"bwd_tflops_peak:     {summary['bwd_tflops_peak']:.1f}")
    print(f"bwd_tflops_geomean:  {summary['bwd_tflops_geomean']:.1f}")
    print(f"peak_vram_mb:        {summary['peak_vram_mb']:.1f}")
    print(f"total_seconds:       {summary['total_seconds']:.1f}")
    print(f"configs_tested:      {summary['configs_tested']}")
    print(f"configs_crashed:     {summary['configs_crashed']}")


def print_comparison(sa: dict, sb: dict):
    """Print side-by-side comparison of two experiments."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON: {sa['name']} vs {sb['name']}")
    print(f"{'='*60}")

    for metric in ["fwd_tflops_geomean", "fwd_tflops_peak", "bwd_tflops_geomean", "bwd_tflops_peak"]:
        va, vb = sa.get(metric, 0), sb.get(metric, 0)
        if va == 0 and vb == 0:
            continue
        delta = ((vb - va) / va * 100) if va > 0 else 0
        sign = "+" if delta >= 0 else ""
        winner = "<- A wins" if va > vb else ("-> B wins" if vb > va else "   tie")
        print(f"  {metric:<25} A={va:>7.1f}  B={vb:>7.1f}  ({sign}{delta:.1f}%)  {winner}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true", help="Run experiment A only on cuda:0")
    args = parser.parse_args()

    print("=" * 80)
    print("  FA4 SM120 Kernel Optimization Experiment")
    print("=" * 80)

    if args.single:
        summary = run_single(EXPERIMENT_A, "cuda:0")
        print_summary(summary)
    else:
        sa, sb = run_parallel(EXPERIMENT_A, EXPERIMENT_B)
        print_summary(sa)
        print_summary(sb)
        print_comparison(sa, sb)

    # Write results to FA4_RESULT_FILE if set (used by parallel runner)
    result_file = os.environ.get("FA4_RESULT_FILE")
    if result_file:
        with open(result_file, "w") as f:
            json.dump(summary if args.single else {"A": sa, "B": sb}, f)


if __name__ == "__main__":
    main()
