#!/usr/bin/env python3
"""第十章 · 画两张图：过滤器怎么一步步改分布，温度怎么改分布

都用“Once upon a time”之后真实的49152个分数，横轴固定是原始概率最高的
那几个词元，最后多一根“其余”，把剩下几万个词元的概率加在一起。

- filters.png：原始分布 → 温度0.7 → top-k 4 → top-p 0.9，每一步之后
  过一次Softmax看概率。温度和top-p和正文“代码效果”那一节的演示一样；
  那里没开top-k，这里加一个k=4，和正文draw.io画的filter.png对齐，
  好让每一步都看得出变化。
- temperature.png：T取0.3、0.7、1.0、1.5、3.0，和正文那张表同样的5个词元。

画图要用matplotlib，它不在项目依赖里；又要加载模型，得用项目的环境。
所以运行时临时带上它（numpy要跟着项目锁在2以下）：
    cd code
    uv run --with matplotlib --with "numpy<2" python book/ch10/plot_sampling.py

图写到本书的`images/ch10/`；没有那个目录的话，写到当前目录。
"""

from pathlib import Path

import matplotlib
import torch
from tokenizers import Tokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import Model
from sampler import apply_temperature, apply_top_k, apply_top_p
from weights import MODEL_DIR

BOOK = Path(__file__).resolve().parents[3]  # code/book/ch10 往上三层
OUT = BOOK / "images" / "ch10" if (BOOK / "images").is_dir() else Path(".")
TEXT = "Once upon a time"

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

BAR = "#4C8BF5"
REST = "#BBBBBB"


def bars(ax, probs, ids, labels, title):
    """画一格：ids那几个词元各一根，再加一根“其余”。"""
    shown = [probs[i].item() for i in ids]
    rest = max(1.0 - sum(shown), 0.0)
    xs = range(len(ids) + 1)
    ax.bar(xs, shown + [rest], color=[BAR] * len(ids) + [REST], width=0.7)
    for x, v in zip(xs, shown + [rest]):
        if v >= 0.0005:
            ax.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels + ["其余"], rotation=60, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)


def save(fig, name):
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor="white")
    print(f"已写入 {OUT / name}")


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = Model()
    ids = tok.encode(TEXT, add_special_tokens=False).ids
    with torch.no_grad():
        logits = model.forward(ids)  # (49152,)

    top = logits.softmax(-1).topk(8).indices.tolist()
    labels = [repr(tok.decode([i])) for i in top]

    # ---- 一：过滤器一步一步改分布 ------------------------------------------------
    stages = [("原始分布", logits)]
    stages.append(("温度0.7之后", apply_temperature(stages[-1][1], 0.7)))
    stages.append(("top-k 4之后", apply_top_k(stages[-1][1], 4)))
    stages.append(("top-p 0.9之后", apply_top_p(stages[-1][1], 0.9)))

    fig, axes = plt.subplots(1, len(stages), figsize=(12.8, 3.4), dpi=200, sharey=True)
    for ax, (title, z) in zip(axes, stages):
        kept = int(torch.isfinite(z).sum())
        bars(ax, z.softmax(-1), top, labels, f"{title}（留{kept}个）")
        print(f"{title:<12} 留{kept:>6}个  " +
              "  ".join(f"{z.softmax(-1)[i]:.3f}" for i in top))
    axes[0].set_ylabel("概率")
    save(fig, "filters.png")

    # ---- 二：温度 --------------------------------------------------------------
    temps = (0.3, 0.7, 1.0, 1.5, 3.0)
    fig, axes = plt.subplots(1, len(temps), figsize=(12.8, 3.2), dpi=200, sharey=True)
    for ax, t in zip(axes, temps):
        bars(ax, apply_temperature(logits, t).softmax(-1), top[:5], labels[:5], f"T={t}")
    axes[0].set_ylabel("概率")
    save(fig, "temperature.png")


if __name__ == "__main__":
    main()
