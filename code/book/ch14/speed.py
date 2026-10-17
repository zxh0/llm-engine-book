"""第十四章 · GPU上快了多少

第十一章`speed.py`的第三项，换到GPU上再量一遍：同样128个词元，prefill一次喂完，
decode每次喂1个。每种设备、精度、注意力实现各量一次。

在GPU上量时间，有两件事和CPU不一样：

  - GPU是异步的。Python那一行返回了，GPU可能还在算。所以读表之前，
    先`synchronize()`，等GPU真算完。
  - 第一次调用要做一堆准备：加载计算核心、分配显存。所以先热身几次，再开始量。

运行：
    cd code
    uv run python book/ch14/speed.py                        # 有什么设备用什么
    uv run python book/ch14/speed.py --model Qwen3-0.6B-Base --device cuda
"""

import argparse
import time

import torch

from engine import DTYPES, Engine, pick_device
from kv_cache import KVCache
from weights import MODEL_DIR, MODELS

T = 128        # prefill多长
ROUNDS = 20    # decode量几次取平均


def sync(device):
    """等设备上排着队的计算都做完。CPU上什么都不用做。"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(model, ids):
    """返回(prefill一次的秒数, decode一次的秒数)。"""
    dev = model.device
    cache = KVCache(model.config, T + 1, dtype=model.dtype, device=dev)
    with torch.no_grad():
        for _ in range(3):                             # 热身
            cache.length = 0
            model.forward(ids, cache)
        sync(dev)

        cache.length = 0
        start = time.perf_counter()
        logits = model.forward(ids, cache)             # prefill：T个一次喂完
        sync(dev)
        prefill = time.perf_counter() - start

        nxt = [int(logits.argmax())]
        start = time.perf_counter()
        for _ in range(ROUNDS):                        # decode：每次1个
            cache.length = T                           # 退回去，每次都在同一个位置上量
            model.forward(nxt, cache)
        sync(dev)
        decode = (time.perf_counter() - start) / ROUNDS
    return prefill, decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR.name)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)

    cpu = Engine(MODELS / args.model)
    ids = cpu.generate(cpu.tok.encode("Once upon a time", add_special_tokens=False).ids,
                       max_new_tokens=T - 4)           # 凑一段T个词元的真文字
    combos = [("cpu", "float32", "manual"), (device, "float32", "manual"),
              (device, "float32", "sdpa"), (device, "bfloat16", "sdpa"),
              (device, "float16", "sdpa")]

    print(f"模型 {args.model}，在{T}个词元上\n")
    print(f"  {'设备':<6}{'精度':<10}{'注意力':<8}{'prefill':>12}{'每个':>10}"
          f"{'decode每个':>14}{'每秒词元':>10}")
    for dev, dt, attn in dict.fromkeys(combos):        # 去掉重复的（device就是cpu时）
        try:
            model = Engine(MODELS / args.model, dev, DTYPES[dt], attn).model
            prefill, decode = measure(model, ids)
        except Exception as e:
            print(f"  {dev:<6}{dt:<10}{attn:<8}   跳过：{type(e).__name__}: {str(e)[:40]}")
            continue
        print(f"  {dev:<6}{dt:<10}{attn:<8}{prefill * 1000:>10.1f}ms{prefill / T * 1000:>8.2f}ms"
              f"{decode * 1000:>12.1f}ms{1 / decode:>10.1f}")
        del model


if __name__ == "__main__":
    main()
