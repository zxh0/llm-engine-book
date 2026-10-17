"""第十四章 · 搬到GPU上以后，对拍还过不过

基准是CPU上fp32的那一份：第十三章它已经和transformers一字不差地对过了。
同一个模型，换个设备、换个精度、换个注意力的实现，分三项比：

    一  逐层比     每一层输出和CPU那一份差多少，相对误差max|diff| / max|ref|
    二  argmax     每个位置挑的那个词元一样不一样
    三  贪婪生成   N个新词元，从第几个开始和CPU那一份分叉

fp32在GPU上不会逐位相同：GPU的矩阵乘法用的是另一套代码，加法的顺序不一样。
bf16、fp16差得更多，贪婪生成迟早会分叉，问题只是什么时候。

运行：
    cd code
    uv run python book/ch14/check_device.py                       # 有什么设备用什么
    uv run python book/ch14/check_device.py --model Qwen3-0.6B-Base --device cuda
"""

import argparse

import torch

from engine import DTYPES, Engine, pick_device
from weights import MODEL_DIR, MODELS

PROMPT = "Once upon a time"


def hidden_states(model, ids):
    """逐层留下最后一个位置的主干，各(hidden_size,)，搬回CPU、升回fp32。"""
    ids = model.to_ids(ids)
    x = model.to_embeddings(ids)
    states = [x]
    for i, layer in enumerate(model.weights.layers):
        x = model.block(x, layer, i)
        states.append(x)
    states[-1] = model.rms_norm(x, model.weights.norm)
    return [s.float().cpu() for s in states]


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR.name)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=100, help="贪婪生成多少个")
    args = ap.parse_args()
    device = pick_device(args.device)

    ref = Engine(MODELS / args.model)                  # CPU、fp32、手写注意力
    ids = ref.tok.encode(PROMPT, add_special_tokens=False).ids
    n = len(ids)
    with torch.no_grad():
        ref_states = hidden_states(ref.model, ids)
        ref_out = ref.generate(ids, max_new_tokens=args.n)
    ref_argmax = ref.model.to_logits(ref_states[-1]).argmax(-1)

    # 最后一行是CPU上的bf16：精度带来的差别，和设备无关。
    combos = [(device, "float32", "manual"), (device, "float32", "sdpa"),
              (device, "bfloat16", "sdpa"), (device, "float16", "sdpa"),
              ("cpu", "bfloat16", "manual")]
    print(f"模型 {args.model}，基准是CPU、float32、manual，贪婪生成{args.n}个\n")
    print(f"  {'设备':<6}{'精度':<10}{'注意力':<8}{'逐层最大相对误差':>16}{'argmax':>8}"
          f"{'贪婪生成':>14}")
    for dev, dt, attn in combos:
        try:
            eng = Engine(MODELS / args.model, dev, DTYPES[dt], attn)
            with torch.no_grad():
                states = hidden_states(eng.model, ids)
                out = eng.generate(ids, max_new_tokens=args.n)
        except Exception as e:  # 比如MPS不支持bf16
            print(f"  {dev:<6}{dt:<10}{attn:<8}   跳过：{type(e).__name__}: {str(e)[:40]}")
            continue
        rel = max(((s - r).abs().max() / r.abs().max()).item()
                  for s, r in zip(states, ref_states))
        same = torch.equal(eng.model.to_logits(states[-1].to(eng.model.device, eng.model.dtype))
                           .argmax(-1).cpu(), ref_argmax)
        d = first_diff(out, ref_out)
        gen = "全部相同" if d is None else f"第{d - n + 1}个起分叉"
        print(f"  {dev:<6}{dt:<10}{attn:<8}{rel:>16.1e}{'相同' if same else '不同':>8}{gen:>14}")
        del eng


if __name__ == "__main__":
    main()
