"""Hardware benchmark script: CPU vs NVIDIA GeForce RTX 5060 Laptop GPU.

Measures policy inference latency, batched backpropagation throughput,
and memory utilization across CPU and CUDA devices.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROCKETMIND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROCKETMIND_DIR / "src"))

import torch
import torch.nn as nn
from rocketmind.custom_ppo import ActorCritic


def benchmark_device(device_name: str, batch_sizes: list[int] = [64, 256, 1024, 4096]) -> dict:
    device = torch.device(device_name)
    results = {}

    print(f"\n[BENCHMARK] Testing Device: {device_name.upper()}")
    if device_name == "cuda":
        print(f"            Hardware: {torch.cuda.get_device_name(0)}")
        print(f"            Initial VRAM: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    model = ActorCritic(state_dim=8, action_dim=4, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for bs in batch_sizes:
        x = torch.randn(bs, 8, device=device)
        actions = torch.randint(0, 4, (bs,), device=device)

        # Warm-up
        for _ in range(50):
            _, log_prob, entropy, val = model.get_action_and_value(x, actions)
            loss = -(log_prob.mean()) + 0.5 * (val.mean() ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if device_name == "cuda":
            torch.cuda.synchronize()

        # Timed benchmark: 500 forward + backward iterations
        n_iters = 500
        start = time.perf_counter()
        for _ in range(n_iters):
            _, log_prob, entropy, val = model.get_action_and_value(x, actions)
            loss = -(log_prob.mean()) + 0.5 * (val.mean() ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if device_name == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        throughput = (bs * n_iters) / elapsed
        results[bs] = {
            "elapsed_sec": round(elapsed, 3),
            "throughput_fps": round(throughput, 1),
            "latency_ms_per_batch": round((elapsed / n_iters) * 1000, 3),
        }
        print(f"  Batch size {bs:4d} | Throughput: {throughput:8.1f} samples/sec | Latency: {(elapsed/n_iters)*1000:6.3f} ms/update")

    if device_name == "cuda":
        results["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(0) / 1024**2, 2)
        print(f"  Peak VRAM: {results['peak_vram_mb']} MB")

    return results


def main() -> None:
    print("=" * 70)
    print("🚀 RocketMind Hardware Acceleration Benchmark (CPU vs RTX 5060)")
    print("=" * 70)

    cpu_res = benchmark_device("cpu")

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_res = benchmark_device("cuda")
        print("\n" + "=" * 70)
        print("[SUMMARY] Speedup Factor (CUDA / CPU Throughput):")
        print("=" * 70)
        for bs in [64, 256, 1024, 4096]:
            speedup = cuda_res[bs]["throughput_fps"] / cpu_res[bs]["throughput_fps"]
            print(f"  Batch Size {bs:4d} -> Speedup: {speedup:5.2f}x")
    else:
        print("\n[WARN] CUDA is not available on this machine.")


if __name__ == "__main__":
    main()

