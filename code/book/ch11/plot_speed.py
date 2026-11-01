#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib>=3.8"]
# ///
"""第十一章 · 画测速图：每生成一个词元花了多少毫秒

读`speed.py`写下的`speed.json`，不重新测。不带缓存的那条一路往上爬，
带缓存的那条从第2个词元起就是平的。每个词元的原始耗时画成淡色的点，
每20个的平均画成实线，和`speed.py`打印的那张表一一对应。

先打印正文引用的数字（每段的平均、合计），再画图。

画图要用matplotlib，它不在项目依赖里，所以脚本开头用PEP 723单独声明。
运行时不要写`python`，让uv按那几行注释现装一个环境：
    cd code
    uv run python book/ch11/speed.py     # 先测，写出speed.json
    uv run book/ch11/plot_speed.py       # 再画

图写到本书的`images/ch11/speed.png`；没有那个目录的话，写到当前目录。
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
BOOK = HERE.parents[2]  # code/book/ch11 往上三层
OUT = BOOK / "images" / "ch11" / "speed.png" if (BOOK / "images").is_dir() else Path("speed.png")

# 和第八章、第十章的图同一套颜色
SERIES = [("no_cache_ms", "不带缓存", "#E8710A"), ("cache_ms", "带缓存", "#4C8BF5")]


def windows(n, every):
    """第1个单独一段，之后每every个一段，和speed.py的分段一样。"""
    return [(0, 1)] + [(k, min(k + every, n)) for k in range(1, n, every)]


def report(data):
    every = data["every"]
    for key, name, _ in SERIES:
        ms = data[key]
        avgs = [sum(ms[a:b]) / (b - a) for a, b in windows(len(ms), every)]
        print(f"{name:　<4}" + " ".join(f"{v:6.1f}" for v in avgs)
              + f"   合计{sum(ms) / 1000:.1f}秒")


def plot(data):
    every = data["every"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    for key, name, color in SERIES:
        ms = data[key]
        xs = range(1, len(ms) + 1)
        ax.scatter(xs, ms, s=4, color=color, alpha=0.25, linewidths=0)
        # 每段的平均画在这一段的中点上
        segs = windows(len(ms), every)
        mid = [(a + b + 1) / 2 for a, b in segs]
        avg = [sum(ms[a:b]) / (b - a) for a, b in segs]
        ax.plot(mid, avg, color=color, linewidth=2, label=name)
        ax.annotate(name, xy=(mid[-1], avg[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9)

    ax.set_xlim(0, len(data["cache_ms"]) + 22)
    ax.set_ylim(0, None)
    ax.set_xlabel("第几个新词元")
    ax.set_ylabel("毫秒")
    ax.grid(color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print(f"已写入 {OUT}")


if __name__ == "__main__":
    data = json.loads((HERE / "speed.json").read_text(encoding="utf-8"))
    report(data)
    plot(data)
