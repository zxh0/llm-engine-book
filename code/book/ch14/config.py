"""第十三章 · 模型的规格说明，换一个模型也能读

`Config`是第五章写的，之后每章补一两个字段。前十二章它只读过SmolLM2的
`config.json`，有几处悄悄依赖了SmolLM2的写法。这一章要读Qwen3-0.6B，
这几处都得改成真正从文件里读：

    head_dim            SmolLM2没写，576 / 9 = 64；Qwen3写明了128，而1024 / 16 = 64
    rope_interleaved    SmolLM2写了false；Qwen3压根没这个字段，没写就是false
    eos_token_ids       SmolLM2只有一个0；Qwen3的对话版有两个，写在generation_config.json里
    qk_norm             SmolLM2没有；Qwen3的注意力里，q和k先各过一次RMSNorm

前三个是“文件里有就照文件，没有就按约定推出来”。最后一个文件里没有字段，
只能按`model_type`认：这是架构的差别，不是超参数的差别。

运行：
    cd code
    uv run python book/ch13/config.py
    uv run python book/ch13/config.py Qwen3-0.6B-Base
"""

import json
import sys

from weights import MODEL_DIR, MODELS


class Config:
    """一个模型的规格说明，读自`config.json`和`generation_config.json`。"""

    def __init__(self, model_dir=MODEL_DIR):
        with open(model_dir / "config.json", encoding="utf-8") as f:
            self.raw = json.load(f)
        gen_path = model_dir / "generation_config.json"
        with open(gen_path, encoding="utf-8") as f:
            self.gen = json.load(f)

        self.model_type = self.raw["model_type"]                      # llama / qwen3

        # 第二到四章：维度、层数、词表、权重共享。
        self.hidden_size = self.raw["hidden_size"]
        self.num_hidden_layers = self.raw["num_hidden_layers"]
        self.vocab_size = self.raw["vocab_size"]
        self.tie_word_embeddings = self.raw["tie_word_embeddings"]

        # 第五章：RMSNorm的eps。SmolLM2是1e-5，Qwen3是1e-6。
        self.rms_norm_eps = self.raw["rms_norm_eps"]

        # 第六章：头数。
        self.num_attention_heads = self.raw["num_attention_heads"]
        self.num_key_value_heads = self.raw["num_key_value_heads"]
        # 每个头多宽。文件里写了就用文件的，没写才按hidden_size / 头数推。
        # 前十二章一直是推出来的，SmolLM2恰好推得对；Qwen3写的是128，推出来是64。
        self.head_dim = self.raw.get(
            "head_dim", self.hidden_size // self.num_attention_heads)

        # 第七章：RoPE。没写配对方式，就是transformers默认的隔半个头配对。
        self.rope_theta = self.raw["rope_theta"]
        self.rope_interleaved = self.raw.get("rope_interleaved", False)
        self.max_position_embeddings = self.raw["max_position_embeddings"]

        # 第八章：FFN的中间层。
        self.intermediate_size = self.raw["intermediate_size"]

        # 第十章：EOS。generation_config.json里的才是生成时真正用的那一份，
        # 可能是一个数，也可能是一串，统一成一个集合。
        eos = self.gen.get("eos_token_id", self.raw["eos_token_id"])
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos])

        # 这一章新的一个：q和k在RoPE之前要不要先过RMSNorm。文件里没有字段，
        # 按架构认。真要是认错了，Weights那边对不上张量，当场就会报错。
        self.qk_norm = self.model_type == "qwen3"

    def __repr__(self):
        keys = ("model_type", "hidden_size", "num_hidden_layers", "vocab_size",
                "num_attention_heads", "num_key_value_heads", "head_dim",
                "rope_theta", "intermediate_size", "eos_token_ids", "qk_norm")
        return "Config(" + ", ".join(f"{k}={getattr(self, k)}" for k in keys) + ")"


def main():
    names = sys.argv[1:] or ["SmolLM2-135M", "Qwen3-0.6B-Base", "Qwen3-0.6B"]
    cfgs = [Config(MODELS / n) for n in names]
    fields = ("model_type", "hidden_size", "num_hidden_layers", "vocab_size",
              "tie_word_embeddings", "rms_norm_eps", "num_attention_heads",
              "num_key_value_heads", "head_dim", "rope_theta", "rope_interleaved",
              "max_position_embeddings", "intermediate_size", "eos_token_ids",
              "qk_norm")
    print(f"{'':<26}" + "".join(f"{n:>18}" for n in names))
    for k in fields:
        vals = [str(getattr(c, k)) for c in cfgs]
        mark = "" if len(set(vals)) == 1 else "  ←"
        print(f"{k:<26}" + "".join(f"{v:>18}" for v in vals) + mark)


if __name__ == "__main__":
    main()
