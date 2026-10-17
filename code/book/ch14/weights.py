"""第十四章 · 把权重读成一个对象，放到指定的设备上

和第十三章一样，只是多了两个参数：`dtype`和`device`。

    dtype    读进来换成什么精度。默认fp32，和前十三章一样；GPU上常用bf16
    device   放在哪。默认"cpu"；"cuda"是NVIDIA的GPU，"mps"是苹果的GPU

文件里存的是bf16。fp32是把它升上去，bf16是原样用，fp16是换一种16位的格式，
第二章说过，fp16的指数位只有5位，数值范围窄得多。

运行：
    cd code
    uv run python book/ch14/weights.py
    uv run python book/ch14/weights.py Qwen3-0.6B-Base
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

    def __init__(self, model_dir=MODEL_DIR, dtype=torch.float32, device="cpu"):
        path = Path(model_dir) / "model.safetensors"
        with safe_open(path, framework="pt") as f:
            names = set(f.keys())

            def get(name): # 读一个张量，换成要的精度，搬到要的设备上。
                return f.get_tensor(name).to(device=device, dtype=dtype)

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
