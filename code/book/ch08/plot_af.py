#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib>=3.8"]
# ///
"""第八章 · 画激活函数：Sigmoid、ReLU、GELU、SiLU

按出现的先后排。ReLU、GELU、SiLU画成实线，标出GELU和SiLU的最低点。
Sigmoid画成虚线：它的输出封顶在1，和另外三个不是一类；SiLU就是x乘上它，
放在一起能看出两条线为什么在x=1相交，SiLU负数那一侧为什么往下凹。

先打印正文引用的数字（取值表、两个最低点），再画图。

画图要用matplotlib，它不在项目依赖里，所以脚本开头用PEP 723单独声明。
运行时不要写`python`，让uv按那几行注释现装一个环境：
    cd code
    uv run book/ch08/plot_af.py

图写到本书的`images/ch08/af.png`；没有那个目录的话，写到当前目录。
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BOOK = Path(__file__).resolve().parents[3]  # code/book/ch08 往上三层
OUT = BOOK / "images" / "ch08" / "af.png" if (BOOK / "images").is_dir() else Path("af.png")


def relu(x):
    return max(0.0, x)


def gelu(x):  # 精确版：x 乘标准正态分布的累积分布函数
    return x * 0.5 * (1 + math.erf(x / math.sqrt(2)))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def silu(x):
    return x * sigmoid(x)


FUNCS = [("ReLU", relu), ("GELU", gelu), ("SiLU", silu)]
COLORS = {"ReLU": "#999999", "GELU": "#4C8BF5", "SiLU": "#E8710A", "Sigmoid": "#2E9E5B"}


def argmin(f, lo=-4.0, hi=0.0, n=400001):
    """在 [lo, hi] 上逐点找最低点，步长 1e-5，够正文保留三位小数用。"""
    best = min((lo + (hi - lo) * i / (n - 1) for i in range(n)), key=f)
    return best, f(best)


def report():
    xs = [-3, -1, 0, 1, 3]
    print("取值表")
    print(f"{'x':>8}" + "".join(f"{x:>10}" for x in xs))
    for name, f in [("Sigmoid", sigmoid)] + FUNCS:
        print(f"{name:>8}" + "".join(f"{f(x):>10.4f}" for x in xs))
    print("最低点")
    for name, f in FUNCS[1:]:
        x, y = argmin(f)
        print(f"{name:>8}  x = {x:.3f}  y = {y:.4f}")


def plot():
    n = 801
    xs = [-4 + 8 * i / (n - 1) for i in range(n)]

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.plot(xs, [sigmoid(x) for x in xs], label="Sigmoid", color=COLORS["Sigmoid"],
            linewidth=1.5, linestyle="--")
    for name, f in FUNCS:
        ax.plot(xs, [f(x) for x in xs], label=name, color=COLORS[name], linewidth=2)

    # 标出 GELU 和 SiLU 的最低点：负数那一侧往下凹的地方
    # 两个点离得近，标注一个放右下、一个放左下，免得叠在一起
    label_at = {"GELU": (0.3, -0.85), "SiLU": (-3.4, -0.85)}
    for name, f in FUNCS[1:]:
        x, y = argmin(f)
        ax.plot([x], [y], "o", color=COLORS[name], markersize=5)
        ax.annotate(
            f"({x:.2f}, {y:.3f})".replace("-", "−"),  # 和坐标轴一样用减号
            xy=(x, y),
            xytext=label_at[name],
            fontsize=9,
            color=COLORS[name],
            arrowprops=dict(arrowstyle="-", color=COLORS[name], linewidth=0.8),
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.2, 4)
    ax.set_xticks(range(-4, 5))
    ax.set_yticks(range(-1, 5))
    ax.grid(color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    ax.set_aspect("equal")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print(f"已写入 {OUT}")


if __name__ == "__main__":
    report()
    plot()
