"""第十四章 · 批处理：读一遍权重，给好几条序列各算一个词元

第十一章说过，decode是内存带宽密集型的：每算一个词元，都要把全部权重读一遍。
那就让B条序列一起decode，读一遍权重，B条各算一个，权重的那一趟搬运就被B条分摊了。

这一章的`Model`允许张量前面多一维批次：ids从(T,)变成(B, T)，主干从(T, 576)
变成(B, T, 576)，KV Cache每层是(B, 3, 长度, 64)。为了不碰补齐（padding）和
对应的掩码，这里B条序列用的是同一个开头，只是各自按不同的随机种子采样，
所以永远一样长。

    一  正确性   B条完全一样的贪婪序列，每一条都要和一条一条单独生成的一字不差
    二  吞吐量   B从1加到64，每秒一共生成多少个词元

运行：
    cd code
    uv run python book/ch14/batch.py                        # 有什么设备用什么
    uv run python book/ch14/batch.py --device cuda --dtype bfloat16
"""

import argparse
import time

import torch

from engine import DTYPES, Engine, pick_device
from kv_cache import KVCache
from speed import sync
from weights import MODEL_DIR, MODELS

PROMPT = "Once upon a time"
N = 32  # 每条序列生成多少个


def generate_batch(model, ids, B, n, temperature=0.0, seed=0):
    """B条序列一起生成n个，返回(B, T + n)。temperature为0就是贪婪。"""
    T = len(ids)
    cache = KVCache(model.config, T + n, dtype=model.dtype, device=model.device, batch=(B,))
    batch = torch.tensor([ids] * B)                    # (B, T)，在CPU上
    gen = torch.Generator().manual_seed(seed)
    feed = batch
    for _ in range(n):
        logits = model.forward(feed, cache).float().cpu()  # (B, 词表大小)
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)          # (B, 1)
        else:
            probs = (logits / temperature).softmax(-1)
            nxt = torch.multinomial(probs, 1, generator=gen)  # 每一行各抽一个
        batch = torch.cat([batch, nxt], dim=1)
        feed = nxt                                     # 之后每条只喂刚生成的那1个
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR.name)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    ap.add_argument("--max-batch", type=int, default=64)
    args = ap.parse_args()
    device = pick_device(args.device)

    engine = Engine(MODELS / args.model, device, DTYPES[args.dtype], "sdpa")
    model = engine.model
    ids = engine.tok.encode(PROMPT, add_special_tokens=False).ids

    # ---- 一：正确性 -----------------------------------------------------------
    with torch.no_grad():
        one = engine.generate(ids, max_new_tokens=N)   # 一条一条生成，第十三章的走法
        many = generate_batch(model, ids, 4, N)
    ok = all(row.tolist() == one for row in many)
    print(f"模型 {args.model}，{device}，{args.dtype}\n")
    print(f"一  4条一起贪婪生成{N}个，每一条都和单独生成的{'一字不差' if ok else '不同'}")

    # ---- 二：吞吐量 -----------------------------------------------------------
    print(f"\n二  一起生成{N}个新词元，温度0.8，各用各的随机数")
    print(f"    {'B':>4}{'用时':>10}{'每秒词元':>12}{'是B=1的几倍':>14}")
    base = None
    B = 1
    while B <= args.max_batch:
        with torch.no_grad():
            generate_batch(model, ids, B, 2, 0.8)      # 热身
            sync(model.device)
            start = time.perf_counter()
            generate_batch(model, ids, B, N, 0.8)
            sync(model.device)
        seconds = time.perf_counter() - start
        rate = B * N / seconds
        base = base or rate
        print(f"    {B:>4}{seconds:>9.2f}秒{rate:>12.1f}{rate / base:>14.1f}")
        B *= 2


if __name__ == "__main__":
    main()
