"""第十三章 · 把权重读成一个对象，换一个模型也能读

第二章的`Weights`只认SmolLM2：每层9个张量，名字写死。Qwen3-0.6B每层多了两个，
注意力里的`q_norm`和`k_norm`，各是(128,)的RMSNorm权重。

这里的办法是：必有的9个照旧，可有可无的那两个，文件里有就读进来，没有就不读。
`Weights`只管数据，不管这两个张量该怎么用，那是`Model`看着`Config`决定的。

模型放在`code/models/`下面，一个模型一个目录。默认还是SmolLM2，前十二章的
代码一行都不用改。

运行：
    cd code
    uv run python book/ch13/weights.py
    uv run python book/ch13/weights.py Qwen3-0.6B-Base
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open

# code/book/ch13/weights.py → 上三层就是 code/
MODELS = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_DIR = MODELS / "SmolLM2-135M"

# 每层必有的9个，和第二章一样，按数据流的顺序排。
REQUIRED = {
    "input_norm"    : "input_layernorm",
    "q_proj"        : "self_attn.q_proj",
    "k_proj"        : "self_attn.k_proj",
    "v_proj"        : "self_attn.v_proj",
    "o_proj"        : "self_attn.o_proj",
    "post_attn_norm": "post_attention_layernorm",
    "gate_proj"     : "mlp.gate_proj",
    "up_proj"       : "mlp.up_proj",
    "down_proj"     : "mlp.down_proj",
}
# 有的模型才有。Qwen3的QK-Norm。
OPTIONAL = {
    "q_norm": "self_attn.q_norm",
    "k_norm": "self_attn.k_norm",
}


def count_layers(names):
    """层数是数出来的：看看`model.layers.N.`里的N最大到几。"""
    ids = [int(n.split(".")[2]) for n in names if n.startswith("model.layers.")]
    return max(ids) + 1


class Weights:
    """一个模型的全部权重。"""

    def __init__(self, model_dir=MODEL_DIR, dtype=torch.float32):
        path = Path(model_dir) / "model.safetensors"
        with safe_open(path, framework="pt") as f:
            names = set(f.keys())

            def get(name): # 读一个张量，顺便从bf16升成fp32。
                return f.get_tensor(name).to(dtype)

            self.embed_tokens = get("model.embed_tokens.weight")
            self.n_layers = count_layers(names)

            self.layers = []
            for i in range(self.n_layers):
                p = f"model.layers.{i}"
                layer = {k: get(f"{p}.{v}.weight") for k, v in REQUIRED.items()}
                for k, v in OPTIONAL.items():
                    if f"{p}.{v}.weight" in names:
                        layer[k] = get(f"{p}.{v}.weight")
                self.layers.append(layer)

            self.norm = get("model.norm.weight")
            # 权重共享的模型，本来用不着lm_head。可Qwen3-0.6B的对话版文件里偏偏
            # 存了一份，和词嵌入矩阵逐位相同，白占300多MB。有就核对一下，不读进来。
            # 要是对不上，说明这个模型真没共享，本书的to_logits不处理这种情况。
            if "lm_head.weight" in names:
                same = torch.equal(f.get_tensor("lm_head.weight"),
                                   f.get_tensor("model.embed_tokens.weight"))
                assert same, "lm_head和词嵌入不是同一份，这个模型没有共享权重"

    @property
    def n_tensors(self):
        return 2 + sum(len(layer) for layer in self.layers)

    @property
    def n_params(self):
        """参数总数。线性投影和词嵌入共用一份权重，所以只算一次。"""
        total = self.embed_tokens.numel() + self.norm.numel()
        return total + sum(t.numel() for layer in self.layers for t in layer.values())


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M"
    w = Weights(MODELS / name)

    print(name)
    print(f"{'embed_tokens':<18}{str(tuple(w.embed_tokens.shape)):>14}{w.embed_tokens.numel():>14,}")
    print("layers[0]")
    for k, t in w.layers[0].items():
        print(f"  {k:<16}{str(tuple(t.shape)):>14}{t.numel():>14,}")
    print(f"{'norm':<18}{str(tuple(w.norm.shape)):>14}")

    per_layer = sum(t.numel() for t in w.layers[0].values())
    print()
    print(f"张量           {w.n_tensors} 块")
    print(f"层数           {w.n_layers}，每层{len(w.layers[0])}个张量")
    print(f"一层合计       {per_layer:,}")
    print(f"全部参数       {w.n_params:,}")
    print(f"词嵌入占比     {w.embed_tokens.numel() / w.n_params:.1%}")


if __name__ == "__main__":
    main()
