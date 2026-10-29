"""第十章 · 引擎第一次成为一个循环

    文字 -> 词元 -> ID -> [ 词嵌入 -> 30个Block -> 出口norm -> 线性投影 ] -> 分数 -> 新ID -> 文字
                    ^                                                               |
                    +------------------------------ append to ----------------------+

和第八章比，总览图上那条`append to`虚线真正接上了：

  - 挑下一个ID不再写死取最高分，交给一个采样器，见`sampler.py`；
  - 圈数不再写死，抽到EOS就停，最多转`max_new_tokens`圈兜底；
  - 来一个词元印一个，字节先进缓冲，凑齐了才印，见`detokenizer.py`。

到这里引擎算是能用了。

运行：
    cd code
    uv run python book/ch10/engine.py
    uv run python book/ch10/engine.py "你想试的任意一段话"
    uv run python book/ch10/engine.py "Once upon a time" --temperature 0.7 --top-p 0.9 --seed 42
"""

import argparse
import time

from tokenizers import Tokenizer

from detokenizer import StreamDecoder
from model import Model
from sampler import GREEDY, Sampler
from weights import MODEL_DIR

MAX_NEW_TOKENS = 20  # 兜底：模型一直不说EOS，也最多转这么多圈
WIDTH = 78           # 印到屏幕上时的折行宽度，书里的代码块按80列排


class Engine:
    """把四个模块凑在一起：Tokenizer、LLM、Sampler、Detokenizer。"""

    def __init__(self, model_dir=MODEL_DIR):
        self.model = Model(model_dir)
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.eos = self.model.config.eos_token_id          # 0，`<|endoftext|>`

    def stream(self, ids, sampler=GREEDY, max_new_tokens=MAX_NEW_TOKENS):
        """一串词元ID进去，新ID一个一个吐出来。

        每圈：整串forward一遍 → 取最后一个位置的logits → 采样 → 追加 → 再来。
        抽到EOS就停，EOS本身不吐出来。

        循环里只有一句`sampler(logits)`，不问它是贪婪还是随机，那是构造时定的。
        """
        ids = list(ids)  # 复制一份，不去改调用方手里那个list
        for _ in range(max_new_tokens):
            next_id = self.infer(ids, sampler)
            if next_id == self.eos: # 模型自己认为说完了
                return
            ids.append(next_id)
            yield next_id

    def generate(self, ids, sampler=GREEDY, max_new_tokens=MAX_NEW_TOKENS):
        """一串词元ID进去，一串更长的ID出来。`stream`收齐了再交出去。"""
        return list(ids) + list(self.stream(ids, sampler, max_new_tokens))

    def infer(self, ids, sampler=GREEDY):
        """转一圈：一串词元ID进去，下一个词元ID出来。

        对应总览图中间那两个框：LLM算出分数，采样器挑一个。
        `model.forward`只返回最后一个位置的logits：前面每个位置也都预测了
        「它的下一个词元」，但那些位置的下一个词元我们已经知道了。
        """
        logits = self.model.forward(ids) # (T,) → (49152,)
        return sampler(logits)           # (49152,) → 一个ID


class LineWrap:
    """边生成边折行：只在词元开头的空格处换行，模型生成的文字一个字不改。

    模型生成的是不带换行的一整段，折行只是为了印在书里好看。
    """

    def __init__(self, width=WIDTH):
        self.width = width
        self.col = 0  # 当前这一行已经印了几个字符

    def __call__(self, text):
        lines = text.split("\n")
        out = []
        for n, seg in enumerate(lines):
            if n > 0:  # 文字里本来就有的换行，照印，列数归零
                out.append("\n")
                self.col = 0
            if seg.startswith(" ") and self.col > 0 and self.col + len(seg) > self.width:
                out.append("\n")
                seg = seg[1:]  # 换行顶替了那个空格
                self.col = 0
            out.append(seg)
            self.col += len(seg)
        return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="Once upon a time")
    ap.add_argument("--temperature", type=float, default=0.0, help="0就是贪婪")
    ap.add_argument("--top-k", type=int, default=0, help="0表示不截断")
    ap.add_argument("--top-p", type=float, default=1.0, help="1表示不截断")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    engine = Engine()
    tok = engine.tok
    sampler = Sampler.build(args.temperature, args.top_k, args.top_p, args.seed)

    ids = tok.encode(args.text, add_special_tokens=False).ids  # Tokenizer：文字到ID
    print(f"输入     {args.text!r}   {ids}")
    print(f"采样器   {sampler!r}")
    print(f"最多     {args.max_new_tokens}个新词元，抽到EOS（ID {engine.eos}）就停\n")

    # ---- 转起来，来一个印一个 -------------------------------------------------
    stream = StreamDecoder(tok)                            # Detokenizer：ID回到文字
    wrap = LineWrap()
    out = list(ids)
    start = time.perf_counter()
    print(wrap(args.text), end="", flush=True)
    for i in engine.stream(ids, sampler, args.max_new_tokens):
        out.append(i)
        print(wrap(stream.push(i)), end="", flush=True)
    print(wrap(stream.flush()))
    seconds = time.perf_counter() - start

    new = out[len(ids):]
    why = "抽到EOS" if len(new) < args.max_new_tokens else "到了上限"
    each = f"，每个{seconds / len(new):.2f}秒" if new else "" # 0个就不谈每个
    print(f"\n{len(new)}个新词元，{why}停下，用时{seconds:.1f}秒{each}")
    print(f"ID       {out}")


if __name__ == "__main__":
    main()
