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

from weights import MODELS

NAMES = ["SmolLM2-135M", "Qwen3-0.6B-Base"]
TEXTS = [
    "To be, or not to be, that is the question:",
    "There are 49152 tokens in the vocabulary.",
    "The toy engine is unbelievable.",
    "你好吗？",
    "第十章：采样",
]


def main():
    toks = {n: Tokenizer.from_file(str(MODELS / n / "tokenizer.json")) for n in NAMES}

    print("一  文件结构")
    for n in NAMES:
        with open(MODELS / n / "tokenizer.json", encoding="utf-8") as f:
            j = json.load(f)
        pre = j["pre_tokenizer"]
        steps = [p["type"] for p in pre.get("pretokenizers", [pre])]
        print(f"  {n:<18}词表{toks[n].get_vocab_size():>8}   特殊词元{len(j['added_tokens']):>4}个"
              f"   合并规则{len(j['model']['merges']):>7}条   预处理{steps}")

    print("\n二  同一句话")
    for text in TEXTS:
        print(f"  {text!r}")
        for n in NAMES:
            enc = toks[n].encode(text, add_special_tokens=False)
            print(f"    {n:<18}{len(enc.ids):>3}个   {enc.tokens}")


if __name__ == "__main__":
    main()
