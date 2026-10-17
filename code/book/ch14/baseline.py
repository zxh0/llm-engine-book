"""第十三章 · 给新模型也存一份基准（第十四章原样搬过来）

和第一章的`baseline.py`做的是同一件事：用transformers跑一遍，把逐层的中间结果、
logits和贪婪生成的结果存成文件，之后拿我们自己的引擎回来对拍。

第一章只给SmolLM2存过一份（`reference/baseline.pt`）。这一章换成Qwen3-0.6B，
要重新存一份，文件名跟着模型走：`reference/Qwen3-0.6B-Base.pt`。

对拍用基座版：它和对话版的架构、形状一模一样，只是权重不同，
而基座版的续写结果更好验证。

运行：
    cd code
    uv run python book/ch14/baseline.py                      # 默认Qwen3-0.6B-Base
    uv run python book/ch14/baseline.py Qwen3-0.6B
"""

import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from weights import MODELS

PROMPT = "Once upon a time"
MAX_NEW_TOKENS = 20


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-0.6B-Base"
    model_dir = MODELS / name
    out_path = MODELS.parent / "reference" / f"{name}.pt"

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    ids = tok(PROMPT, return_tensors="pt", add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        res = model(ids, output_hidden_states=True)
        greedy = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                temperature=None, top_p=None, top_k=None)
    hidden = torch.stack(res.hidden_states)            # (层数+1, 1, T, 隐层)
    text = tok.decode(greedy[0])

    print(f"模型           {name}")
    print(f"torch          {torch.__version__}")
    print(f"transformers   {transformers.__version__}")
    print(f"参数量         {n_params:,}")
    print(f"ids            {ids[0].tolist()}")
    print(f"hidden_states  {tuple(hidden.shape)}")
    print(f"logits         {tuple(res.logits.shape)}")
    print(f"贪婪生成       {text!r}")

    payload = {
        "prompt": PROMPT,
        "input_ids": ids,
        "logits": res.logits,
        "hidden_states": hidden,
        "greedy": greedy,
        "greedy_text": text,
        "n_params": n_params,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    out_path.parent.mkdir(exist_ok=True)
    torch.save(payload, out_path)
    print(f"已写入         reference/{out_path.name}")


if __name__ == "__main__":
    main()
