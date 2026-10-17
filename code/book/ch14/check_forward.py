"""第十四章 · 整机对拍：四级验收，两个模型都要过

和第八、十一章那个是同一份，只是不再写死SmolLM2：模型从命令行来，基准文件
跟着模型走。SmolLM2用第一章存的`reference/baseline.pt`，Qwen3用这一章
`baseline.py`存的`reference/<模型名>.pt`。

    一  参数量            和transformers一个不多一个不少
    二  逐层比            层数+1份，相对误差max|diff| / max|ref|
    三  每个位置的argmax   一个都不能错
    四  贪婪解码20个词元    和transformers那句话一字不差

这个脚本不导入transformers，只读存下来的文件。

运行：
    cd code
    uv run python book/ch14/baseline.py                  # 先给Qwen3存一份基准
    uv run python book/ch14/check_forward.py Qwen3-0.6B-Base
    uv run python book/ch14/check_forward.py             # SmolLM2，和第十一章一样
"""

import sys

import torch

from engine import Engine
from weights import MODEL_DIR, MODELS


def baseline_path(name):
    """SmolLM2的基准是第一章存的，其余的按模型名存。"""
    ref = MODELS.parent / "reference"
    return ref / "baseline.pt" if name == MODEL_DIR.name else ref / f"{name}.pt"


def hidden_states(model, ids):
    """走一遍所有层，按transformers的约定留下层数+1份中间结果，各(T, hidden_size)。"""
    x = model.to_embeddings(ids)                       # (T, hidden_size)
    states = [x]
    for i, layer in enumerate(model.weights.layers):
        x = model.block(x, layer, i)
        states.append(x)
    states[-1] = model.rms_norm(x, model.weights.norm)  # 最后一份要过出口norm
    return states


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else MODEL_DIR.name
    ref = torch.load(baseline_path(name))
    engine = Engine(MODELS / name)
    model = engine.model
    ids = ref["input_ids"][0].tolist()
    n = model.weights.n_layers + 1                     # 逐层比一共几份

    print(f"模型   {name}")
    print(f"输入   {ref['prompt']!r}   {ids}")

    # ---- 一：参数量 -----------------------------------------------------------
    ours = model.weights.n_params
    print(f"\n一  参数量    我们{ours:,}   transformers{ref['n_params']:,}   "
          f"{'相同' if ours == ref['n_params'] else '不同'}")

    # ---- 二：31层逐层比 -------------------------------------------------------
    with torch.no_grad():
        states = hidden_states(model, ids)
    print(f"\n二  逐层比   {'下标':>4}{'绝对误差':>12}{'相对误差':>12}")
    worst = 0.0
    for i, (h, r) in enumerate(zip(states, ref["hidden_states"][:, 0])):
        abs_err = (h - r).abs().max().item()
        rel = abs_err / r.abs().max().item()
        worst = max(worst, rel)
        if i % 5 == 0 or i >= n - 2:
            print(f"            {i:>4}{abs_err:>14.1e}{rel:>14.1e}")
    print(f"            {n}份里最大的相对误差   {worst:.1e}")

    # ---- 三：argmax ----------------------------------------------------------
    logits = model.to_logits(states[-1])               # (T, 词表大小)
    ours = logits.argmax(-1).tolist()
    theirs = ref["logits"][0].argmax(-1).tolist()
    print(f"\n三  argmax   我们{ours}   transformers{theirs}   "
          f"{'相同' if ours == theirs else '不同'}")

    # ---- 四：贪婪解码 ---------------------------------------------------------
    out = engine.generate(ids)
    theirs = ref["greedy"][0].tolist()
    text = engine.tok.decode(out)
    print(f"\n四  贪婪解码")
    print(f"    我们     {text!r}")
    print(f"    transformers {ref['greedy_text']!r}")
    print(f"    ID{'相同' if out == theirs else '不同'}，"
          f"文字{'相同' if text == ref['greedy_text'] else '不同'}")


if __name__ == "__main__":
    main()
