"""第九章 · logit lens：在残差流的半路上，提前问一句“下一个词是什么”

残差流从词嵌入一路通到出口，形状始终是(T, 576)，每一层只是往上加一点东西。
既然形状一直没变，那么在任何一层停下来，把主干直接送去出口：

    主干 → 出口norm → lm_head → logits

也能得到一组分数。这个办法叫logit lens。它不改模型，只是多看几眼。

对每一层，打印最后一个位置排第一的词元和它的概率，以及最终答案的排名和概率。

运行：
    cd code
    uv run python book/ch09/logit_lens.py
    uv run python book/ch09/logit_lens.py "Once upon a time"
"""

import sys
import unicodedata

import torch
from tokenizers import Tokenizer

from model import Model
from weights import MODEL_DIR


def width(s):
    """显示宽度：中文占两列，其余占一列。str的对齐按字符数算，有中文就对不齐。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def ljust(s, n):
    return s + " " * (n - width(s))


def rjust(s, n):
    return " " * (n - width(s)) + s


def lens(model, h):
    """把主干上最后一个位置直接送去出口，得到概率：(576,) → (49152,)。"""
    logits = model.to_logits(model.rms_norm(h, model.weights.norm))
    return torch.softmax(logits, dim=-1)


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = Model()
    text = (sys.argv[1] if len(sys.argv) > 1
            else "The Eiffel Tower is located in the city of")
    ids = tok.encode(text, add_special_tokens=False).ids

    # 先完整走一遍，拿到最终答案；同时把每一层之后的主干留下来。
    x = model.to_embeddings(ids)
    stream = [x[-1]] # 第0份是词嵌入本身
    for layer in model.weights.layers:
        x = model.block(x, layer)
        stream.append(x[-1])
    final = int(lens(model, stream[-1]).argmax())

    print(f"输入   {text!r}   {ids}")
    print(f"最终答案   {tok.id_to_token(final)!r}\n")
    print("   " + ljust("位置", 10) + ljust("排第一的词元", 16) + rjust("概率", 8)
          + rjust("最终答案排第几", 16) + rjust("它的概率", 10))
    for i, h in enumerate(stream):
        p = lens(model, h)
        top = int(p.argmax())
        rank = int((p > p[final]).sum()) + 1
        where = "词嵌入" if i == 0 else f"第{i - 1}层后"
        print(f"   {ljust(where, 10)}{tok.id_to_token(top)!r:<16}{float(p[top]):8.3f}"
              f"{rank:>16}{float(p[final]):10.3f}")


if __name__ == "__main__":
    main()
