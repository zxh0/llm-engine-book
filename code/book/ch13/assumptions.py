"""第十三章 · 前十二章悄悄做了哪些假设

拿Qwen3-0.6B-Base，把`Config`里的字段一个一个改回“SmolLM2的样子”，
看看会发生什么。有的当场报错，有的不报错，只是话说得越来越不对。

    head_dim = hidden_size / 头数   1024 / 16 = 64，不是文件里写的128
    不做QK-Norm                     q和k直接转RoPE
    rope_theta = 100000             SmolLM2的底数
    rms_norm_eps = 1e-5             SmolLM2的eps

每一种都贪婪生成20个词元，和transformers存下的那句话比。

运行：
    cd code
    uv run python book/ch13/baseline.py          # 还没有Qwen3的基准文件的话，先跑这个
    uv run python book/ch13/assumptions.py
"""

import torch

from engine import Engine
from model import Model
from weights import MODELS

NAME = "Qwen3-0.6B-Base"


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def main():
    ref = torch.load(MODELS.parent / "reference" / f"{NAME}.pt")
    ids = ref["input_ids"][0].tolist()
    theirs = ref["greedy"][0].tolist()
    engine = Engine(MODELS / NAME)
    model = engine.model
    tok = engine.tok
    cfg = model.config
    n = len(ids)

    def run(label):
        model.cos, model.sin = model.rope_table()      # 改了配置，表要重算
        model.qk_norm = model.rms_norm if cfg.qk_norm else model.no_norm
        try:
            out = engine.generate(ids)
        except Exception as e:
            print(f"  {label:<26}报错   {type(e).__name__}: {str(e)[:60]}")
            return
        d = first_diff(out, theirs)
        verdict = "一字不差" if d is None else f"第{d - n + 1}个新词元起不同，不报错"
        print(f"  {label:<26}{verdict}")
        if d is not None:
            print(f"  {'':<26}{tok.decode(out[n:])!r}")

    print(f"{NAME}，'{ref['prompt']}'，贪婪生成20个\n")
    run("照文件读（正确）")

    saved = cfg.head_dim
    cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    run(f"head_dim = {cfg.head_dim}")
    cfg.head_dim = saved

    cfg.qk_norm = False
    run("不做QK-Norm")
    cfg.qk_norm = True

    saved = cfg.rope_theta
    cfg.rope_theta = 100000
    run("rope_theta = 100000")
    cfg.rope_theta = saved

    saved = cfg.rms_norm_eps
    cfg.rms_norm_eps = 1e-5
    run("rms_norm_eps = 1e-5")
    cfg.rms_norm_eps = saved


if __name__ == "__main__":
    main()
