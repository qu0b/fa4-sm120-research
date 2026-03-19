"""
Fixed benchmark infrastructure for FA4 SM120 kernel optimization.
GPU detection, benchmark harness, metric extraction, reference results.

Usage:
    python prepare.py                # run baseline benchmark suite
    python prepare.py --baseline     # same as above
    python prepare.py --quick        # quick smoke test (subset of configs)
    python prepare.py --profile      # run with torch profiler

This file is READ-ONLY. The agent does not modify it.
"""

import os
import sys
import time
import math
import argparse
import subprocess
from dataclasses import dataclass, field
from typing import Optional
from statistics import geometric_mean

import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

# SM120 hardware specs
SM120_ARCH = 120
SM120_SMEM_DEFAULT = 48 * 1024      # 48 KB
SM120_SMEM_OPTIN = 101376           # 99 KB (actual)
SM120_NUM_SMS = 188
SM120_PEAK_TFLOPS_BF16 = 300.0      # theoretical peak

# Benchmark parameters
WARMUP_ITERS = 10        # warmup iterations (includes compilation)
BENCH_ITERS = 50         # timed iterations
BENCH_REPEATS = 3        # repeat full benchmark N times, take best

# Dtypes to benchmark
BENCH_DTYPES = [torch.bfloat16]

# ---------------------------------------------------------------------------
# Problem sizes — the canonical benchmark suite
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProblemSize:
    """A single attention problem configuration."""
    batch: int
    seqlen_q: int
    seqlen_k: int
    nheads_q: int
    nheads_k: int
    headdim: int
    causal: bool = False
    dtype: torch.dtype = torch.bfloat16

    @property
    def label(self) -> str:
        c = "causal" if self.causal else "full"
        gqa = f"gqa{self.nheads_q // self.nheads_k}" if self.nheads_q != self.nheads_k else "mha"
        return f"b{self.batch}_s{self.seqlen_q}_h{self.nheads_q}_d{self.headdim}_{c}_{gqa}"

    @property
    def flops_fwd(self) -> float:
        """FLOPs for forward pass (2 * batch * nheads * seqlen_q * seqlen_k * headdim * 2)."""
        # Standard attention: Q@K^T (2*s*s*d) + softmax@V (2*s*s*d) per head per batch
        f = 4 * self.batch * self.nheads_q * self.seqlen_q * self.seqlen_k * self.headdim
        if self.causal:
            f //= 2  # approximately half the work
        return f

    @property
    def flops_bwd(self) -> float:
        """FLOPs for backward pass (~2.5x forward)."""
        return int(self.flops_fwd * 2.5)


# The canonical problem sizes. These define what "performance" means.
PROBLEM_SIZES_FULL = [
    # Large sequence lengths (memory-bound → bandwidth-sensitive)
    ProblemSize(1, 8192, 8192, 16, 16, 128, causal=False),
    ProblemSize(1, 8192, 8192, 16, 16, 128, causal=True),
    ProblemSize(1, 16384, 16384, 16, 16, 128, causal=False),
    ProblemSize(1, 16384, 16384, 16, 16, 128, causal=True),
    # Medium sequence lengths (balanced)
    ProblemSize(2, 4096, 4096, 16, 16, 128, causal=False),
    ProblemSize(2, 4096, 4096, 16, 16, 128, causal=True),
    ProblemSize(4, 2048, 2048, 32, 32, 128, causal=False),
    ProblemSize(4, 2048, 2048, 32, 32, 128, causal=True),
    # Different head dimensions
    ProblemSize(2, 4096, 4096, 16, 16, 96, causal=False),
    ProblemSize(2, 4096, 4096, 16, 16, 96, causal=True),
    ProblemSize(4, 2048, 2048, 32, 32, 64, causal=False),
    ProblemSize(4, 2048, 2048, 32, 32, 64, causal=True),
    # Short sequences (compute-bound → MMA throughput)
    ProblemSize(8, 1024, 1024, 16, 16, 128, causal=False),
    ProblemSize(16, 512, 512, 32, 32, 128, causal=False),
    # GQA configurations
    ProblemSize(2, 4096, 4096, 32, 8, 128, causal=False),   # GQA 4:1
    ProblemSize(2, 4096, 4096, 32, 8, 128, causal=True),    # GQA 4:1 causal
]

PROBLEM_SIZES_QUICK = [
    ProblemSize(1, 8192, 8192, 16, 16, 128, causal=False),
    ProblemSize(2, 4096, 4096, 16, 16, 128, causal=True),
    ProblemSize(4, 2048, 2048, 32, 32, 64, causal=False),
]


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpu():
    """Detect GPU and verify it's SM120."""
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available")
        sys.exit(1)

    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    cap = torch.cuda.get_device_capability(device)
    arch = cap[0] * 10 + cap[1]
    mem_gb = torch.cuda.get_device_properties(device).total_mem / (1024**3)

    print(f"gpu_name:      {name}")
    print(f"gpu_arch:      sm_{arch}0")
    print(f"gpu_vram_gb:   {mem_gb:.1f}")
    print(f"num_sms:       {torch.cuda.get_device_properties(device).multi_processor_count}")

    if arch != 12:
        print(f"WARNING: Expected SM120, got SM{arch}0. Results may not be meaningful.")

    return arch, name


# ---------------------------------------------------------------------------
# Tensor creation
# ---------------------------------------------------------------------------

def make_tensors(problem: ProblemSize, device="cuda:0", requires_grad=False):
    """Create Q, K, V tensors for a given problem size."""
    q = torch.randn(
        problem.batch, problem.seqlen_q, problem.nheads_q, problem.headdim,
        dtype=problem.dtype, device=device, requires_grad=requires_grad,
    )
    k = torch.randn(
        problem.batch, problem.seqlen_k, problem.nheads_k, problem.headdim,
        dtype=problem.dtype, device=device, requires_grad=requires_grad,
    )
    v = torch.randn(
        problem.batch, problem.seqlen_k, problem.nheads_k, problem.headdim,
        dtype=problem.dtype, device=device, requires_grad=requires_grad,
    )
    return q, k, v


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    """Result of benchmarking a single problem size."""
    problem: ProblemSize
    fwd_ms: float = 0.0
    bwd_ms: float = 0.0
    fwd_tflops: float = 0.0
    bwd_tflops: float = 0.0
    peak_vram_mb: float = 0.0
    status: str = "ok"      # ok, crash, skip
    error: str = ""


def benchmark_one(
    problem: ProblemSize,
    attn_fn,
    warmup_iters: int = WARMUP_ITERS,
    bench_iters: int = BENCH_ITERS,
    bench_repeats: int = BENCH_REPEATS,
    do_backward: bool = True,
) -> BenchResult:
    """Benchmark a single problem size with the given attention function.

    attn_fn signature: attn_fn(q, k, v, causal=False) -> output
    """
    result = BenchResult(problem=problem)

    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Create tensors
        q, k, v = make_tensors(problem, requires_grad=do_backward)

        # Warmup (includes JIT compilation)
        for _ in range(warmup_iters):
            out = attn_fn(q, k, v, causal=problem.causal)
            if do_backward:
                dout = torch.randn_like(out)
                out.backward(dout)
                q.grad = k.grad = v.grad = None
        torch.cuda.synchronize()

        # Forward benchmark
        best_fwd_ms = float("inf")
        for _ in range(bench_repeats):
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]

            for i in range(bench_iters):
                start_events[i].record()
                out = attn_fn(q, k, v, causal=problem.causal)
                end_events[i].record()

            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            # Drop highest and lowest, take mean of the rest
            times.sort()
            trimmed = times[2:-2] if len(times) > 4 else times
            mean_ms = sum(trimmed) / len(trimmed)
            best_fwd_ms = min(best_fwd_ms, mean_ms)

        result.fwd_ms = best_fwd_ms
        result.fwd_tflops = (problem.flops_fwd / (best_fwd_ms / 1000)) / 1e12

        # Backward benchmark
        if do_backward:
            best_bwd_ms = float("inf")
            for _ in range(bench_repeats):
                start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
                end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]

                for i in range(bench_iters):
                    q.grad = k.grad = v.grad = None
                    out = attn_fn(q, k, v, causal=problem.causal)
                    dout = torch.randn_like(out)
                    start_events[i].record()
                    out.backward(dout)
                    end_events[i].record()

                torch.cuda.synchronize()
                times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
                times.sort()
                trimmed = times[2:-2] if len(times) > 4 else times
                mean_ms = sum(trimmed) / len(trimmed)
                best_bwd_ms = min(best_bwd_ms, mean_ms)

            result.bwd_ms = best_bwd_ms
            result.bwd_tflops = (problem.flops_bwd / (best_bwd_ms / 1000)) / 1e12

        result.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        result.status = "ok"

    except Exception as e:
        result.status = "crash"
        result.error = str(e)

    # Cleanup
    torch.cuda.empty_cache()
    return result


def run_benchmark_suite(
    attn_fn,
    problems: list[ProblemSize] = None,
    do_backward: bool = True,
    label: str = "experiment",
) -> list[BenchResult]:
    """Run the full benchmark suite and print results."""
    if problems is None:
        problems = PROBLEM_SIZES_FULL

    print(f"\n{'='*80}")
    print(f"  Benchmark: {label}")
    print(f"{'='*80}\n")

    results = []
    fwd_tflops_list = []
    bwd_tflops_list = []

    for i, problem in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] {problem.label} ...", end=" ", flush=True)
        r = benchmark_one(problem, attn_fn, do_backward=do_backward)
        results.append(r)

        if r.status == "ok":
            fwd_str = f"fwd={r.fwd_ms:.3f}ms ({r.fwd_tflops:.1f} TF)"
            bwd_str = f"bwd={r.bwd_ms:.3f}ms ({r.bwd_tflops:.1f} TF)" if do_backward else ""
            print(f"{fwd_str}  {bwd_str}")
            fwd_tflops_list.append(r.fwd_tflops)
            if do_backward and r.bwd_tflops > 0:
                bwd_tflops_list.append(r.bwd_tflops)
        elif r.status == "skip":
            print(f"SKIP: {r.error}")
        else:
            print(f"CRASH: {r.error}")

    # Summary
    print(f"\n{'='*80}")
    print(f"  Summary: {label}")
    print(f"{'='*80}")

    if fwd_tflops_list:
        fwd_peak = max(fwd_tflops_list)
        fwd_geomean = geometric_mean(fwd_tflops_list)
        print(f"fwd_tflops_peak:     {fwd_peak:.1f}")
        print(f"fwd_tflops_geomean:  {fwd_geomean:.1f}")
        print(f"fwd_utilization:     {fwd_peak / SM120_PEAK_TFLOPS_BF16 * 100:.1f}%")

    if bwd_tflops_list:
        bwd_peak = max(bwd_tflops_list)
        bwd_geomean = geometric_mean(bwd_tflops_list)
        print(f"bwd_tflops_peak:     {bwd_peak:.1f}")
        print(f"bwd_tflops_geomean:  {bwd_geomean:.1f}")

    peak_vram = max(r.peak_vram_mb for r in results if r.status == "ok") if any(r.status == "ok" for r in results) else 0
    n_ok = sum(1 for r in results if r.status == "ok")
    n_crash = sum(1 for r in results if r.status == "crash")
    n_skip = sum(1 for r in results if r.status == "skip")

    print(f"peak_vram_mb:        {peak_vram:.1f}")
    print(f"configs_ok:          {n_ok}/{len(results)}")
    if n_crash > 0:
        print(f"configs_crash:       {n_crash}")
    if n_skip > 0:
        print(f"configs_skip:        {n_skip}")
    print()

    return results


# ---------------------------------------------------------------------------
# Reference: default FA4 attention function (current SM80 path)
# ---------------------------------------------------------------------------

def get_default_attn_fn():
    """Get the default FA4 attention function (whatever the installed package provides)."""
    try:
        from flash_attn.cute import flash_attn_func
        def attn_fn(q, k, v, causal=False):
            return flash_attn_func(q, k, v, causal=causal)
        return attn_fn
    except ImportError:
        print("ERROR: flash-attn-4 not installed. Install with: pip install flash-attn-4")
        sys.exit(1)


def get_torch_sdpa_fn():
    """Get PyTorch's scaled_dot_product_attention as a reference."""
    def attn_fn(q, k, v, causal=False):
        # SDPA expects (batch, nheads, seqlen, headdim)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=causal,
        )
        return out.transpose(1, 2)
    return attn_fn


# ---------------------------------------------------------------------------
# Results comparison
# ---------------------------------------------------------------------------

def compare_results(baseline: list[BenchResult], experiment: list[BenchResult]):
    """Print a comparison table between baseline and experiment results."""
    print(f"\n{'='*100}")
    print(f"  Comparison: baseline vs experiment")
    print(f"{'='*100}")
    print(f"{'Config':<45} {'Base FWD TF':>12} {'Exp FWD TF':>12} {'Delta':>8} {'Base BWD TF':>12} {'Exp BWD TF':>12} {'Delta':>8}")
    print("-" * 100)

    for b, e in zip(baseline, experiment):
        if b.status != "ok" or e.status != "ok":
            continue
        fwd_delta = ((e.fwd_tflops - b.fwd_tflops) / b.fwd_tflops * 100) if b.fwd_tflops > 0 else 0
        bwd_delta = ((e.bwd_tflops - b.bwd_tflops) / b.bwd_tflops * 100) if b.bwd_tflops > 0 else 0
        fwd_sign = "+" if fwd_delta >= 0 else ""
        bwd_sign = "+" if bwd_delta >= 0 else ""
        print(f"{b.problem.label:<45} {b.fwd_tflops:>10.1f}TF {e.fwd_tflops:>10.1f}TF {fwd_sign}{fwd_delta:>6.1f}% {b.bwd_tflops:>10.1f}TF {e.bwd_tflops:>10.1f}TF {bwd_sign}{bwd_delta:>6.1f}%")

    print()


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness(attn_fn, rtol=1e-2, atol=1e-2):
    """Verify the attention function produces correct results against PyTorch SDPA."""
    print("\nCorrectness check...")
    ref_fn = get_torch_sdpa_fn()

    test_cases = [
        ProblemSize(1, 256, 256, 4, 4, 64, causal=False),
        ProblemSize(1, 256, 256, 4, 4, 64, causal=True),
        ProblemSize(2, 128, 128, 8, 8, 128, causal=False),
    ]

    all_pass = True
    for problem in test_cases:
        q, k, v = make_tensors(problem)
        out = attn_fn(q, k, v, causal=problem.causal)
        ref = ref_fn(q, k, v, causal=problem.causal)

        max_diff = (out - ref).abs().max().item()
        mean_diff = (out - ref).abs().mean().item()
        ok = torch.allclose(out, ref, rtol=rtol, atol=atol)
        status = "PASS" if ok else "FAIL"
        print(f"  {problem.label}: {status} (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
        if not ok:
            all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FA4 SM120 benchmark infrastructure")
    parser.add_argument("--baseline", action="store_true", help="Run baseline benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--fwd-only", action="store_true", help="Forward only (skip backward)")
    parser.add_argument("--check", action="store_true", help="Run correctness check only")
    parser.add_argument("--sdpa", action="store_true", help="Benchmark PyTorch SDPA as reference")
    args = parser.parse_args()

    arch, gpu_name = detect_gpu()

    if args.check:
        attn_fn = get_default_attn_fn()
        ok = check_correctness(attn_fn)
        sys.exit(0 if ok else 1)

    problems = PROBLEM_SIZES_QUICK if args.quick else PROBLEM_SIZES_FULL

    if args.sdpa:
        attn_fn = get_torch_sdpa_fn()
        label = "PyTorch SDPA reference"
    else:
        attn_fn = get_default_attn_fn()
        label = "FA4 baseline (current installed)"

    results = run_benchmark_suite(
        attn_fn,
        problems=problems,
        do_backward=not args.fwd_only,
        label=label,
    )
