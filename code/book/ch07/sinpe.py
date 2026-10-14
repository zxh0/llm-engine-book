"""第七章 · 正弦位置编码（SinPE），只为看清思路

SmolLM2不用SinPE，引擎里也没有它。这个脚本只做两件事：

1. 打印一张很小的位置编码表（d=8，4个位置），和正文那张图对照
2. 验证SinPE那个性质：PE(pos+k)可以由PE(pos)转一个角度得到，
   而转多少只和k有关，和pos无关

公式出自《Attention Is All You Need》3.5节，维度编号沿用论文，从0开始：

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

运行：
    cd code
    uv run python book/ch07/sinpe.py
"""

import math

import torch

BASE = 10000.0


def sinpe(n, d):
    """n个位置、d维的位置编码表，(n, d)。偶数列用sin，奇数列用cos。"""
    pos = torch.arange(n, dtype=torch.float64)[:, None]         # (n, 1)
    freq = BASE ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)  # (d/2,)
    pe = torch.zeros(n, d, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


def shift(k, d):
    """只和k有关的那个矩阵M，(d, d)：每两维一块，块里是转Δ角的矩阵。"""
    m = torch.zeros(d, d, dtype=torch.float64)
    for i in range(d // 2):
        delta = k * BASE ** (-2 * i / d)
        c, s = math.cos(delta), math.sin(delta)
        m[2 * i:2 * i + 2, 2 * i:2 * i + 2] = torch.tensor([[c, s], [-s, c]], dtype=torch.float64)
    return m


def main():
    torch.set_printoptions(precision=3, sci_mode=False)
    print("1. 位置编码表，d=8，4个位置（行是位置，列是维度）：")
    for pos, row in enumerate(sinpe(4, 8)):
        print(f"   pos={pos}  " + "  ".join(f"{float(v):6.3f}" for v in row))

    d = 64
    pe = sinpe(200, d)
    print(f"\n2. PE(pos+k)和M·PE(pos)最大差多少（d={d}）：")
    for k in (1, 5, 50):
        m = shift(k, d)
        worst = max((pe[p + k] - m @ pe[p]).abs().max().item() for p in (0, 7, 100))
        print(f"   k={k:<3} pos取0、7、100，同一个M   最大差 {worst:.1e}")


if __name__ == "__main__":
    main()
