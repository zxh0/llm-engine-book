"""第十三章 · 把输出排成书上放得下的宽度

这一章的脚本要打印中文、词元列表和长长的提示词，原样打出来，一行能有两三百列，
印在书上放不下。这里几个小函数只管折行，打印的内容一个字不改：

    LineWrap     边生成边折行。英文只在空格处断，中文每个字后面都能断
    wrap_repr    一个字符串的repr太长，拆成几段repr，一段一行，拼起来还是原来那个
    wrap_items   一个列表太长，在逗号后面折行，续行对齐到第一个元素

宽度按终端的列数算：汉字和全角标点占2列，其余占1列。第十章`engine.py`
里的`LineWrap`只管英文，这里是它的推广。
"""

import unicodedata

COLS = 78  # 书上一行大约放得下80列，留一点余地

# 这些标点不放在行首，宁可让上一行超出一两列。
NO_LINE_START = set("，。、；：？！）》」』】”’,.;:?!)]}")
# 这些标点不放在行尾，跟着后面那一块一起走。
NO_LINE_END = set("（《「『【“‘([{")


def is_wide(c):
    return unicodedata.east_asian_width(c) in "WF"


def width(s):
    """终端上占几列：汉字、全角标点2列，其余1列。"""
    return sum(2 if is_wide(c) else 1 for c in s)


class LineWrap:
    """边生成边折行。每次喂进一段文字，还回可以马上印出来的那部分。

    英文按单词折：一个单词还没写完（后面还没跟空格、换行或汉字），就先扣着。
    折行的地方，换行顶替了那里的空格；文字里本来就有的换行，照印。
    """

    def __init__(self, cols=COLS, indent="", col=0):
        self.cols = cols
        self.indent = indent    # 续行前面垫的空格
        self.col = col          # 当前这一行已经印了几列
        self.spaces = ""        # 扣着的空格：下一块放得下就照印，放不下就被换行顶替
        self.word = ""          # 扣着的半个英文单词
        self.opener = ""        # 扣着的左括号、左引号，要和后面那一块粘在一起

    def __call__(self, text):
        out = []
        for c in text:
            if c == "\n":
                out.append(self._word() + self.spaces + "\n" + self.indent)
                self.spaces, self.col = "", width(self.indent)
            elif c == " ":
                out.append(self._word())
                if self.opener:
                    out.append(self._put(""))
                self.spaces += c
            elif c in NO_LINE_END:
                out.append(self._word())
                self.opener += c
            elif is_wide(c):
                out.append(self._word())
                out.append(self._put(c))
            else:
                self.word += c
        return "".join(out)

    def flush(self):
        out = self._word()
        if self.opener:
            out += self._put("")
        out += self.spaces
        self.spaces = ""
        return out

    def _word(self):
        if not self.word:
            return ""
        w, self.word = self.word, ""
        return self._put(w)

    def _put(self, w):
        """印一个不可再分的块：放不下就先换行，不能放在行首的标点除外。"""
        w, self.opener = self.opener + w, ""
        start = width(self.indent)
        need = width(self.spaces) + width(w)
        if self.col > start and self.col + need > self.cols and w[0] not in NO_LINE_START:
            out = "\n" + self.indent + w
            self.col = start + width(w)
        else:
            out = self.spaces + w
            self.col += need
        self.spaces = ""
        return out


def segments(text):
    """把text切成不可再分的小段：断在换行符后、空格前、汉字后。"""
    segs, cur = [], ""
    for i, c in enumerate(text):
        cur += c
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if nxt and c not in NO_LINE_END and nxt not in NO_LINE_START and (
                c == "\n" or nxt == " " or is_wide(c) or is_wide(nxt) or nxt in NO_LINE_END):
            segs.append(cur)
            cur = ""
    if cur:
        segs.append(cur)
    return segs


def wrap_repr(text, cols=COLS, indent=""):
    """把repr(text)拆成几段，一段一行，像Python里相邻的字符串那样拼起来还是text。

    换行符后面总是断开（连着的几个换行算一处），这样提示词的结构一眼就能看清。
    """
    segs = segments(text)
    pieces, cur = [], ""
    for k, seg in enumerate(segs):
        if cur and width(indent + repr(cur + seg)) > cols:
            pieces.append(cur)
            cur = ""
        cur += seg
        nxt = segs[k + 1] if k + 1 < len(segs) else ""
        if cur.endswith("\n") and not nxt.startswith("\n"):
            pieces.append(cur)
            cur = ""
    if cur or not pieces:
        pieces.append(cur)
    return "\n".join(indent + repr(p) for p in pieces)


def wrap_items(items, cols=COLS, indent=""):
    """把一个列表印成`[a, b, ...]`，太长就在逗号后面折行，续行对齐到第一个元素。"""
    head = indent + "["
    pad = " " * width(head)
    lines, cur = [], head
    for k, x in enumerate(items):
        piece = repr(x) + ("]" if k == len(items) - 1 else ",")
        if cur != head and width(cur) + 1 + width(piece) > cols:
            lines.append(cur)
            cur = pad + piece
        else:
            cur += ("" if cur == head else " ") + piece
    if not items:
        cur += "]"
    lines.append(cur)
    return "\n".join(lines)
