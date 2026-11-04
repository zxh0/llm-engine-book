"""第十三章 · 两个分词器，同一句话

第三章拆过SmolLM2的`tokenizer.json`。Qwen3的分词器也是byte-level BPE，
加载方式一模一样，`Tokenizer.from_file`一行。这里把第三章用过的几句话，
分别交给两个分词器，看看切出来有什么不同；再看看两个文件的结构差在哪。

运行：
    cd code
    uv run python book/ch13/tokenizers_diff.py
"""

import json

from tokenizers import Tokenizer

from layout import width, wrap_items
from weights import MODELS

NAMES = ["SmolLM2-135M", "Qwen3-0.6B-Base"]
CHAT = "Qwen3-0.6B"  # 对话版：合并规则一样，追加词元多了4个
THINK = "<think>\n\n</think>"
TEXTS = [
    "To be, or not to be, that is the question:",
    "There are 49152 tokens in the vocabulary.",
    "The toy engine is unbelievable.",
    "你好吗？",
    "第十章：采样",
]


def structure(n, tok):
    """词表、追加词元（其中special为true的几个）、合并规则、预处理。"""
    with open(MODELS / n / "tokenizer.json", encoding="utf-8") as f:
        j = json.load(f)
    pre = j["pre_tokenizer"]
    steps = [p["type"] for p in pre.get("pretokenizers", [pre])]
    added = j["added_tokens"]
    special = sum(a["special"] for a in added)
    print(f"  {n:<18}词表{tok.get_vocab_size():>8}   追加词元{len(added):>4}个（special {special:>2}个）")
    print(f"  {'':<18}合并规则{len(j['model']['merges']):>7}条   预处理{steps}")


def main():
    toks = {n: Tokenizer.from_file(str(MODELS / n / "tokenizer.json")) for n in NAMES + [CHAT]}

    print("一  文件结构")
    for n in NAMES:
        structure(n, toks[n])

    print("\n二  同一句话")
    for text in TEXTS:
        print(f"  {text!r}")
        for n in NAMES:
            enc = toks[n].encode(text, add_special_tokens=False)
            print(wrap_items(enc.tokens, indent=f"    {n:<18}{len(enc.ids):>3}个   "))

    # 对话模板里那对<think>标签：基座版的分词器不认识它们，只能切碎。
    print("\n三  基座版和对话版")
    for n in NAMES[1:] + [CHAT]:
        structure(n, toks[n])
    print(f"  {THINK!r}")
    for n in NAMES[1:] + [CHAT]:
        enc = toks[n].encode(THINK, add_special_tokens=False)
        head = f"    {n:<18}{len(enc.ids):>3}个   "
        print(wrap_items(enc.tokens, indent=head))
        print(wrap_items(enc.ids, indent=" " * width(head)))


if __name__ == "__main__":
    main()
