"""第九章 · 残差流：每一层往主干上加了多少

第九章没有新代码，`model.py`和第八章一字不差。这个脚本只是把`block`里那两个
加号拆开看：

    h = x + attn(rms_norm(x))     ← 加号右边是注意力这一层的“修改量”
    y = h + ffn(rms_norm(h))      ← 加号右边是FFN这一层的“修改量”

打印三样东西：

1. 最后一个位置，每一层的主干模长，以及注意力、FFN各加了多少
2. 每一层前后，主干的方向转了多少（前后两个向量的余弦）
3. 去掉某一层再跑20圈，生成的话和完整模型差多少；以及把残差整个去掉会怎样

运行：
    cd code
    uv run python book/ch09/residual_stream.py
    uv run python book/ch09/residual_stream.py "你想试的任意一段话"
"""

import sys

import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Model
from weights import MODEL_DIR

ROUNDS = 20


def forward(model, ids, skip=None, residual=True):
    """和`Model.forward`一样，只是可以跳过第skip层，或者把残差去掉。"""
    x = model.to_embeddings(ids) # (T, 576)
    for i, layer in enumerate(model.weights.layers):
        if i == skip:
            continue # 这一层整个不走
        a = model.attn(model.rms_norm(x, layer["input_norm"]), layer)
        x = x + a if residual else a
        f = model.ffn(model.rms_norm(x, layer["post_attn_norm"]), layer)
        x = x + f if residual else f
    x = model.rms_norm(x, model.weights.norm)
    return model.to_logits(x[-1]) # (49152,)


def generate(model, ids, **kw):
    ids = list(ids)
    for _ in range(ROUNDS):
        ids.append(int(forward(model, ids, **kw).argmax()))
    return ids


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = Model()
    text = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    ids = tok.encode(text, add_special_tokens=False).ids
    print(f"输入   {text!r}   {ids}")

    # ---- 1、2：拆开每一层的两个加号 -----------------------------------------
    print("\n1. 最后一个位置：主干模长，以及每一层两个子层各加了多少（模长）")
    print(f"   {'层':>3}{'进来时':>10}{'注意力加':>10}{'FFN加':>10}"
          f"{'出去时':>10}{'前后余弦':>10}")
    x = model.to_embeddings(ids)
    for i, layer in enumerate(model.weights.layers):
        before = x[-1]
        a = model.attn(model.rms_norm(x, layer["input_norm"]), layer)
        x = x + a
        f = model.ffn(model.rms_norm(x, layer["post_attn_norm"]), layer)
        x = x + f
        cos = float(F.cosine_similarity(before, x[-1], dim=0))
        norms = (before.norm(), a[-1].norm(), f[-1].norm(), x[-1].norm())
        print(f"   {i:>3}" + "".join(f"{float(v):10.2f}" for v in norms)
              + f"{cos:10.3f}")

    # ---- 3：去掉一层 --------------------------------------------------------
    full = generate(model, ids)
    n = len(ids)
    print(f"\n3. 完整模型   {tok.decode(full)!r}")
    print("   跳过某一层之后，20个新词元里和完整模型相同的个数：")
    for i in range(model.weights.n_layers):
        out = generate(model, ids, skip=i)
        same = sum(a == b for a, b in zip(out[n:], full[n:]))
        print(f"   跳过第{i:2d}层   {same:2d}/20   {tok.decode(out[n:])!r}")

    out = generate(model, ids, residual=False)
    print(f"\n   去掉残差     {tok.decode(out)!r}")


if __name__ == "__main__":
    main()
