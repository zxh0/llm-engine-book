"""第十四章 · FlashAttention，这次量得出来

第十二章说过，FlashAttention在CPU上演示不出加速。这里只拿注意力那一段来比：
同样的q、k、v，形状是一层SmolLM2的(9, T, 64)，一个走我们手写的`attend_manual`，
一个走`attend_sdpa`，也就是PyTorch自带的F.scaled_dot_product_attention。
在CUDA上，后者会自动挑FlashAttention的实现。

量两样东西：

    峰值显存   手写的那一套要把(9, T, T)的分数表和权重表都存下来，随T平方增长
    耗时       同步之后再读表

CPU上量不了显存，只量耗时。

运行：
    cd code
    uv run python book/ch14/flash.py                        # 有什么设备用什么
    uv run python book/ch14/flash.py --device cuda --max-t 16384
"""

import argparse
import time

import torch

from engine import DTYPES, Engine, pick_device
from speed import sync
from weights import MODEL_DIR, MODELS


def peak_reset(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_bytes(device):
    """这一次计算期间，显存最多用了多少。只有CUDA记这个数。"""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    return None


def run(fn, q, k, v, device):
    peak_reset(device)
    base = torch.cuda.memory_allocated() if device.type == "cuda" else 0
    with torch.no_grad():
        fn(q, k, v, 0)                                 # 热身
        sync(device)
        peak_reset(device)
        start = time.perf_counter()
        fn(q, k, v, 0)
        sync(device)
    seconds = time.perf_counter() - start
    peak = peak_bytes(device)
    return seconds, None if peak is None else peak - base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR.name)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    ap.add_argument("--max-t", type=int, default=4096)
    args = ap.parse_args()
    device = pick_device(args.device)
    dtype = DTYPES[args.dtype]

    model = Engine(MODELS / args.model, device, dtype, "manual").model
    cfg = model.config
    device = model.device
    h, d = cfg.num_attention_heads, cfg.head_dim

    print(f"一层注意力，{h}个头，每头{d}维，{device}，{args.dtype}\n")
    print(f"  {'T':>6}{'分数表':>10}{'手写 耗时':>12}{'峰值':>10}{'sdpa 耗时':>12}{'峰值':>10}")
    T = 512
    while T <= args.max_t:
        g = torch.Generator().manual_seed(T)
        q, k, v = (torch.randn(h, T, d, generator=g).to(device, dtype) for _ in range(3))
        table = h * T * T * torch.finfo(dtype).bits // 8
        row = f"  {T:>6}{table / 1e6:>8.0f}MB"
        for fn in (model.attend_manual, model.attend_sdpa):
            try:
                s, peak = run(fn, q, k, v, device)
                row += f"{s * 1000:>10.1f}ms" + (f"{peak / 1e6:>8.0f}MB" if peak else f"{'-':>10}")
            except RuntimeError as e:                  # 显存不够
                row += f"{'放不下':>12}{'':>10}"
        print(row)
        T *= 2


if __name__ == "__main__":
    main()
