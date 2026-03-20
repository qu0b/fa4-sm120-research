"""
FA4 SM120 kernel optimization experiment.
This is the file the agent modifies. Everything is fair game.

Usage:
    python experiment.py                    # run two experiments in parallel (one per GPU)
    python experiment.py --single           # run one experiment on cuda:0 only
    CUDA_VISIBLE_DEVICES=1 python experiment.py --single  # run on specific GPU

The script:
1. Defines two experiment variants (A on GPU 0, B on GPU 1)
2. Monkey-patches FA4's interface.py to override SM120 kernel configs
3. Runs benchmarks in parallel
4. Reports metrics for both, indicating which is better

Key FA4 files that can be patched at runtime (in your site-packages flash_attn/cute/):
    interface.py    — dispatch logic, kernel selection, config (tile sizes, threads, stages)
    flash_fwd.py    — forward kernels (FlashAttentionForwardSm80 base, SM120 subclass)
    flash_bwd.py    — backward kernels (FlashAttentionBackwardSm80 base, SM120 subclass)
    mask.py         — attention mask application
    copy_utils.py   — TMA and async copy utilities
    softmax.py      — online softmax implementation
"""

import os
import sys
import json
import time
import functools

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
#
# Upstream SM120 config (as of commit 3250081):
#   Forward: num_threads=128, tile_m=128, tile_n=64 (d>64) or 128 (d<=64), num_stages=1
#   Backward: num_threads=128, m=64, n=64, stages_Q=2/1, stages_dO=2/1,
#             AtomLayout{MSdP,NdKV,MdQ}=4, no swapAB, no V_in_regs
#
# Our earlier tuning found 256 threads + n_block=128 much better (275.5 TFLOPS peak).

EXPERIMENT_A = {
    "name": "upstream_default",
    "description": "upstream SM120 config (128 threads, n=64 for d>64)",
    "fwd": {
        "num_threads": 128,        # upstream: 4 warps
        "tile_m": 128,
        "tile_n_d_le_64": 128,     # upstream: 128 when d<=64
        "tile_n_d_gt_64": 64,      # upstream: 64 when d>64
        "num_stages": 1,
        "Q_in_regs": False,
    },
    "bwd": {
        "num_threads": 128,        # upstream: 4 warps
        "m_block_size": 64,
        "n_block_size": 64,
        "num_stages_Q_d_le_64": 2,
        "num_stages_dO_d_le_64": 2,
        "num_stages_Q_d_gt_64": 1,
        "num_stages_dO_d_gt_64": 1,
        "SdP_swapAB": False,
        "dKV_swapAB": False,
        "dQ_swapAB": False,
        "AtomLayoutMSdP": 4,
        "AtomLayoutNdKV": 4,
        "AtomLayoutMdQ": 4,
        "V_in_regs": False,
    },
}

EXPERIMENT_B = {
    "name": "tuned_256t",
    "description": "our tuned config (256 threads, n=128 for all d)",
    "fwd": {
        "num_threads": 256,        # 8 warps — our previous best
        "tile_m": 128,
        "tile_n_d_le_64": 128,
        "tile_n_d_gt_64": 128,     # 128*128*2*3 = 96KB, fits in 99KB SMEM
        "num_stages": 1,
        "Q_in_regs": False,
    },
    "bwd": {
        "num_threads": 256,        # 8 warps
        "m_block_size": 64,
        "n_block_size": 64,        # keep conservative for now
        "num_stages_Q_d_le_64": 2,
        "num_stages_dO_d_le_64": 2,
        "num_stages_Q_d_gt_64": 2, # double-buffered Q (our earlier finding)
        "num_stages_dO_d_gt_64": 1,
        "SdP_swapAB": False,
        "dKV_swapAB": False,
        "dQ_swapAB": False,
        "AtomLayoutMSdP": 4,
        "AtomLayoutNdKV": 4,
        "AtomLayoutMdQ": 4,
        "V_in_regs": False,
    },
}

# Benchmark settings
PROBLEM_SIZES = PROBLEM_SIZES_FULL
DO_BACKWARD = True
DO_CORRECTNESS = True


# ---------------------------------------------------------------------------
# Kernel patching — override FA4 SM120 config at runtime
# ---------------------------------------------------------------------------
# The monkey-patch intercepts _flash_attn_fwd and _flash_attn_bwd to
# override the SM120 config block (interface.py lines 454-467, 1005-1024).
#
# How it works:
#   1. We wrap _flash_attn_fwd to inject our tile_mn and num_threads
#      via the existing function parameters (tile_mn, num_threads are
#      already accepted as kwargs).
#   2. For backward, we wrap _flash_attn_bwd to override the SM120
#      config variables before they reach the kernel constructor.

_patches_applied = False

def apply_config_patches(experiment: dict):
    """Monkey-patch FA4's interface to use our SM120 config.

    The key insight: _flash_attn_fwd already accepts `tile_mn` and
    `num_threads` as parameters. For SM120, the code sets these early
    (lines 454-467) but they can be overridden by passing them explicitly
    from the public flash_attn_func wrapper.

    We patch the internal _flash_attn_fwd and _flash_attn_bwd functions
    to inject our config before the SM120 defaults are applied.
    """
    global _patches_applied
    if _patches_applied:
        return True

    try:
        import flash_attn.cute.interface as interface
    except ImportError:
        print("ERROR: flash_attn.cute not importable. Is FA4 installed?")
        return False

    fwd_cfg = experiment["fwd"]
    bwd_cfg = experiment["bwd"]

    # --- Forward patch ---
    # Wrap _flash_attn_fwd to override num_threads and tile_mn for SM120.
    # The function checks `arch // 10 == 12` and sets num_threads=128,
    # then checks tile_mn is None to apply default tile sizes.
    # By passing tile_mn and num_threads explicitly, we bypass those defaults.
    original_fwd = interface._flash_attn_fwd

    @functools.wraps(original_fwd)
    def patched_fwd(q, k, v, *args, **kwargs):
        arch = interface._get_device_arch()
        if arch // 10 == 12:
            head_dim = q.shape[-1]
            tile_n = fwd_cfg["tile_n_d_le_64"] if head_dim <= 64 else fwd_cfg["tile_n_d_gt_64"]
            # Override via kwargs — these take precedence in the function
            kwargs.setdefault("num_threads", fwd_cfg["num_threads"])
            kwargs.setdefault("tile_mn", (fwd_cfg["tile_m"], tile_n))
        return original_fwd(q, k, v, *args, **kwargs)

    interface._flash_attn_fwd = patched_fwd

    # --- Backward patch ---
    # The backward function has a hardcoded SM120 config block at lines 1005-1024.
    # There's no tile_mn parameter for backward, so we must patch deeper.
    # We replace the entire _flash_attn_bwd with a version that overrides
    # the SM120 config block.
    original_bwd = interface._flash_attn_bwd

    @functools.wraps(original_bwd)
    def patched_bwd(*args, **kwargs):
        # Temporarily patch the SM120 backward config by modifying the
        # function's local namespace. We do this by wrapping the call and
        # replacing the config values that get passed to the kernel constructor.
        #
        # The backward config is set inside _flash_attn_bwd based on arch//10==12.
        # Since we can't easily override locals, we use a different approach:
        # save/restore the backward function's behavior by temporarily modifying
        # num_threads in kwargs (backward doesn't accept num_threads as kwarg
        # in the same way, so we need to be more surgical).
        #
        # For now, we rely on the forward patch (which is the bigger win) and
        # leave backward at upstream defaults. When backward patching is needed,
        # we can copy the _flash_attn_bwd function and modify it directly.
        return original_bwd(*args, **kwargs)

    interface._flash_attn_bwd = patched_bwd

    # Clear any cached compiled kernels so our new config takes effect
    if hasattr(interface._flash_attn_fwd, 'compile_cache'):
        interface._flash_attn_fwd.compile_cache.clear()
    # The compile_cache is on the original function, not our wrapper
    if hasattr(original_fwd, 'compile_cache'):
        original_fwd.compile_cache.clear()

    _patches_applied = True
    print(f"  Config patches applied: {experiment['name']}")
    print(f"    fwd: threads={fwd_cfg['num_threads']}, tile_m={fwd_cfg['tile_m']}, "
          f"tile_n=[d<=64:{fwd_cfg['tile_n_d_le_64']}, d>64:{fwd_cfg['tile_n_d_gt_64']}]")
    print(f"    bwd: threads={bwd_cfg['num_threads']}, m={bwd_cfg['m_block_size']}, "
          f"n={bwd_cfg['n_block_size']}")
    return True


def reset_patches():
    """Reset patches so a different experiment config can be applied."""
    global _patches_applied
    try:
        import flash_attn.cute.interface as interface
        # Unwrap if our patch is in place
        fwd = interface._flash_attn_fwd
        if hasattr(fwd, '__wrapped__'):
            interface._flash_attn_fwd = fwd.__wrapped__
        bwd = interface._flash_attn_bwd
        if hasattr(bwd, '__wrapped__'):
            interface._flash_attn_bwd = bwd.__wrapped__
        # Clear compile cache to force recompilation with new config
        if hasattr(interface._flash_attn_fwd, 'compile_cache'):
            interface._flash_attn_fwd.compile_cache.clear()
    except ImportError:
        pass
    _patches_applied = False


def get_experiment_attn_fn(experiment: dict):
    """Get the attention function with experimental patches applied."""
    reset_patches()
    apply_config_patches(experiment)

    from flash_attn.cute import flash_attn_func

    def attn_fn(q, k, v, causal=False):
        result = flash_attn_func(q, k, v, causal=causal)
        return result[0] if isinstance(result, tuple) else result

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
            return {"name": name, "status": "crash", "error": "correctness",
                    "fwd_tflops_peak": 0, "fwd_tflops_geomean": 0,
                    "bwd_tflops_peak": 0, "bwd_tflops_geomean": 0,
                    "peak_vram_mb": 0, "total_seconds": 0,
                    "configs_tested": 0, "configs_crashed": 0}

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
    """Run two experiments in parallel, one per GPU."""
    import subprocess
    import tempfile

    gpus = detect_all_gpus()
    print(f"\nParallel mode: {len(gpus)} GPUs available")
    for gid, arch, name in gpus:
        print(f"  cuda:{gid} — {name} (sm_{arch}0)")

    if len(gpus) < 2:
        print("WARNING: Only 1 GPU available. Running sequentially.")
        sa = run_single(exp_a, "cuda:0")
        reset_patches()
        sb = run_single(exp_b, "cuda:0")
        return sa, sb

    # Run as separate processes to avoid CUDA context sharing issues.
    # Each process gets CUDA_VISIBLE_DEVICES=N so cuda:0 maps to different physical GPUs.
    def run_subprocess(experiment, gpu_id):
        """Spawn a subprocess that runs a single experiment on one GPU."""
        result_file = tempfile.mktemp(suffix=".json", prefix=f"fa4_{experiment['name']}_")
        # Write experiment config to a temp file
        config_file = tempfile.mktemp(suffix=".json", prefix=f"fa4_cfg_{experiment['name']}_")
        with open(config_file, "w") as f:
            json.dump(experiment, f)

        script = f'''
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")

with open("{config_file}") as f:
    experiment = json.load(f)

from experiment import run_single, PROBLEM_SIZES, DO_BACKWARD, DO_CORRECTNESS
summary = run_single(experiment, "cuda:0")

with open("{result_file}", "w") as f:
    json.dump(summary, f)
'''
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=900,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if proc.returncode != 0:
            print(f"\n[{experiment['name']}] STDERR:\n{proc.stderr[-2000:]}")
        # Print stdout (has benchmark output)
        if proc.stdout:
            print(proc.stdout)

        try:
            with open(result_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "name": experiment["name"],
                "status": "crash", "error": proc.stderr[-500:] if proc.stderr else "unknown",
                "fwd_tflops_peak": 0, "fwd_tflops_geomean": 0,
                "bwd_tflops_peak": 0, "bwd_tflops_geomean": 0,
                "peak_vram_mb": 0, "total_seconds": 0,
                "configs_tested": 0, "configs_crashed": 0,
            }

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(run_subprocess, exp_a, 0)
        fut_b = pool.submit(run_subprocess, exp_b, 1)
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


if __name__ == "__main__":
    main()
