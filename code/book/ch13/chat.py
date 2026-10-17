"""第十三章 · 和对话版的Qwen3-0.6B说几句话

第十章说过：基座模型只会续写，想要对话，得换一份做过后训练的权重，再按对话模板
把问题包起来。这一章就这么做，引擎一行不改：

  - 权重换成`Qwen3-0.6B`，和`Qwen3-0.6B-Base`架构、形状一模一样；
  - 问题按ChatML格式包起来，和第三章、第十章见过的`<|im_start|>`、`<|im_end|>`一样；
  - 采样参数照`generation_config.json`里推荐的来：温度0.6，top-k 20，top-p 0.95；
  - EOS是两个：`<|im_end|>`（151645）和`<|endoftext|>`（151643），抽到哪个都停。

Qwen3还有一个“思考模式”：回答之前先在`<think>`和`</think>`之间想一段。
不想让它想，就在助手那一轮的开头先替它写上一对空的`<think></think>`，
transformers的对话模板`enable_thinking=False`干的就是这件事。

运行：
    cd code
    REPO=Qwen/Qwen3-0.6B ./download_model.sh          # 先下对话版的权重
    uv run python book/ch13/chat.py
    uv run python book/ch13/chat.py "Why is the sky blue?" --think
"""

import argparse
import time

from detokenizer import StreamDecoder
from engine import Engine
from sampler import Sampler
from weights import MODELS

SEED = 20261017


def chatml(question, think):
    """按Qwen3的对话模板把一个问题包起来，等助手开口。"""
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    if not think:
        prompt += "<think>\n\n</think>\n\n"  # 替它想完了：一段空的思考
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default="What is the capital of France?")
    ap.add_argument("--think", action="store_true", help="让它先想一想")
    ap.add_argument("--greedy", action="store_true", help="不用推荐的采样参数，取最高分")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--model", default="Qwen3-0.6B")
    args = ap.parse_args()

    engine = Engine(MODELS / args.model)
    gen = engine.model.config.gen
    sampler = (Sampler.build(temperature=0) if args.greedy else
               Sampler.build(gen["temperature"], gen["top_k"], gen["top_p"], SEED))

    prompt = chatml(args.question, args.think)
    ids = engine.tok.encode(prompt, add_special_tokens=False).ids
    print(f"模型     {args.model}")
    print(f"采样器   {sampler!r}")
    print(f"EOS      {sorted(engine.eos)}")
    print(f"提示词   {prompt!r}")
    print(f"         {len(ids)}个词元\n")

    stream = StreamDecoder(engine.tok)
    n = 0
    start = time.perf_counter()
    for i in engine.stream(ids, sampler, args.max_new_tokens):
        n += 1
        print(stream.push(i), end="", flush=True)
    print(stream.flush())
    seconds = time.perf_counter() - start
    why = "抽到EOS" if n < args.max_new_tokens else "到了上限"
    print(f"\n{n}个新词元，{why}停下，用时{seconds:.1f}秒")


if __name__ == "__main__":
    main()
