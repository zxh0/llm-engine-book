#!/usr/bin/env python3
"""Check manuscript body text (正文) for prose rule violations.

Usage:
    python3 check.py <file.md | dir> [more...]

Rules live in the RULES registry below. To add a rule:

    def rule_my_rule(lines, masked, path):
        issues = []
        for i, text in enumerate(masked, 1):
            if <violation in text>:
                issues.append((i, "规则名", "问题描述", lines[i - 1].rstrip()))
        return issues

    RULES = [rule_my_rule, ...]

`lines` is the raw file, `masked` is the same lines with code blocks, formulas,
URLs, HTML and frontmatter blanked out to spaces (offsets preserved). Match
against `masked`, report from `lines`.
"""

import re
import sys

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lintscope import disabled_rules, keep, excluded

from prose import (IMG_STYLE_PAT, IMG_WIDTH_PAT, INLINE_CODE_PAT,
                   block_is_exempt, has_exempt_marker, img_want_width,
                   iter_code_blocks, iter_formulas, iter_img_tags, iter_targets,
                   is_runnable_example, is_toy, logical_lines, mask_lines,
                   read_lines)

# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
# 每条规则实现为一个函数，签名 (lines, masked, path)，
# 返回 [(行号, 规则名, 问题描述, 原文), ...]，然后注册到 RULES 列表中。

# 规则1 符号记法：标量=小写，向量=粗体小写，矩阵=大写
BOLD_UPPER_PAT = re.compile(r'\\(mathbf|boldsymbol|bm)\{\s*([A-Z])')
VEC_PAT = re.compile(r'\\vec\{\s*([^}]*)\}')
BOLDSYM_LATIN_PAT = re.compile(r'\\boldsymbol\{\s*([a-zA-Z][a-zA-Z0-9]*)\s*\}')


def rule_notation(lines, masked, path):
    """粗体只用于向量（粗体小写）；矩阵用非粗体大写；不用 \\vec。

    机械层只能查排版层面的硬性违规。一个不加粗的字母到底是标量、向量
    还是矩阵，需要结合上下文人工判读，见 SKILL.md 的语义层步骤。
    """
    issues = []
    seen = set()

    def add(lineno, problem, rule="符号记法"):
        key = (lineno, problem)
        if key not in seen:
            seen.add(key)
            issues.append((lineno, rule, problem, lines[lineno - 1].rstrip()))

    for lineno, content, _kind in iter_formulas(lines):
        for m in BOLD_UPPER_PAT.finditer(content):
            # 子规则名，好让 lint-scope.json 单独关掉它：这本书用 Goodfellow
            # 那一套（粗体大写=矩阵），所以 lint-scope.json 里对全书关掉了它。
            add(lineno, f"粗体大写 `\\{m.group(1)}{{{m.group(2)}...}}`："
                        f"粗体只表示向量（小写），矩阵应用非粗体大写 `{m.group(2)}`",
                rule="符号记法/粗体大写")
        for m in VEC_PAT.finditer(content):
            add(lineno, f"箭头向量 `\\vec{{{m.group(1)}}}`："
                        f"向量统一用粗体小写 `\\mathbf{{{m.group(1).lower()}}}`")
        for m in BOLDSYM_LATIN_PAT.finditer(content):
            if m.group(1)[0].islower():
                add(lineno, f"`\\boldsymbol{{{m.group(1)}}}`：拉丁字母的粗体"
                            f"统一用 `\\mathbf{{{m.group(1)}}}`"
                            f"（`\\boldsymbol` 只留给希腊字母）")

    return issues


# 规则3 少用破折号：破折号密集是典型的 AI 文风
EM_DASH_PAT = re.compile(r'\s*—+\s*')

# 规则4 公式用 $...$，代码用反引号
GREEK = 'αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ'
BACKTICK_GREEK_PAT = re.compile(r'`([^`]*[' + GREEK + r'][^`]*)`')
BACKTICK_TIMES_PAT = re.compile(r'`([^`]*×[^`]*)`')
BARE_GREEK_PAT = re.compile(r'[' + GREEK + r']')


def rule_math_vs_code(lines, masked, path):
    """公式相关用 $...$，代码相关用反引号。

    机械层只查高置信度的三类。「单个拉丁字母套反引号」看着像数学变量，
    但本书第五章讲 one-hot 时用 `a`、`b`、`z` 指字母本身，属于正当用法，
    所以那一类交给语义层人工判断，不在这里报。
    """
    issues = []
    for i, line in enumerate(lines, 1):
        for m in BACKTICK_GREEK_PAT.finditer(line):
            issues.append((i, "公式/代码", f"希腊字母 `{m.group(1)}` 用了反引号："
                                          f"数学符号应写成 $...$", line.rstrip()))
        for m in BACKTICK_TIMES_PAT.finditer(line):
            issues.append((i, "公式/代码", f"算式 `{m.group(1)}` 用了反引号："
                                          f"含 × 的算式应写成 $...$", line.rstrip()))
    # 屏蔽掉公式后仍出现的希腊字母，说明它裸露在正文里
    for i, text in enumerate(masked, 1):
        stripped = INLINE_CODE_PAT.sub('', text)
        for m in BARE_GREEK_PAT.finditer(stripped):
            issues.append((i, "公式/代码", f"正文里裸露的希腊字母 `{m.group(0)}`："
                                          f"应放进 $...$ 并用 LaTeX 命令（如 \\sigma）",
                           lines[i - 1].rstrip()))
    return issues


def rule_no_em_dash(lines, masked, path):
    """尽量不用破折号。改写为逗号、冒号、括号，或干脆断成两句。

    只报告，不自动修复：破折号承担的语义每处不同（补充说明、话锋一转、
    举例、同位语），换成什么标点取决于上下文，机械替换会写坏句子。
    """
    issues = []
    for i, text in enumerate(masked, 1):
        for m in EM_DASH_PAT.finditer(text):
            if '—' not in m.group(0):
                continue
            issues.append((i, "破折号",
                           "尽量不用破折号（AI 文风明显）："
                           "改用逗号、冒号、括号，或断成两句",
                           lines[i - 1].rstrip()))
    return issues


# 规则5 论文引用：全书第一次出现写《标题》（arXiv:编号），之后只写标题
TITLE_PAT = re.compile(r'《([A-Za-z][^》]*)》')
ARXIV_TAIL_PAT = re.compile(r'\s*（arXiv:\d{4}\.\d{4,5}）')
PAPER_WINDOW = 30  # 书名号前后多少个字符内出现「论文」，就当它是论文而不是书


def _book_order(path):
    """同目录下按文件名排好的书稿：_front、_preface、ch01、ch02……

    只对 chapters/ 这种「一章一个文件」的目录有意义；目录里没有 chNN_*.md
    时返回只含自己的列表，规则退化成单文件内检查。
    """
    d = os.path.dirname(os.path.abspath(path))
    names = sorted(n for n in os.listdir(d) if n.endswith('.md'))
    if not any(re.match(r'ch\d{2}_', n) for n in names):
        return [os.path.abspath(path)]
    return [os.path.join(d, n) for n in names
            if (n.startswith('_') or re.match(r'ch\d{2}_', n))]


def _citations(lines, masked):
    """[(行号, 标题, 有没有编号, 是不是论文), ...]，按出现顺序。"""
    out = []
    for i, text in enumerate(masked, 1):
        raw = lines[i - 1]
        for m in TITLE_PAT.finditer(text):
            tagged = bool(ARXIV_TAIL_PAT.match(raw, m.end()))
            near = text[max(0, m.start() - PAPER_WINDOW):m.end() + PAPER_WINDOW]
            out.append((i, m.group(1), tagged, tagged or '论文' in near))
    return out


def rule_paper_citation(lines, masked, path):
    """论文在全书第一次出现时写《标题》（arXiv:编号），后面各章只写标题。

    「全书第一次」按文件名顺序算，所以要把整本书一起扫，单查一章也会先读
    它前面的各章。书名号里是英文、前后 30 个字符内有「论文」或带了编号的，
    才当成论文；像《Build a Large Language Model (From Scratch)》这样的书不管。
    """
    seen = set()
    me = os.path.abspath(path)
    for p in _book_order(path):
        if p == me:
            break
        if excluded(p):
            continue
        prev = read_lines(p)
        for _i, title, _tagged, _paper in _citations(prev, mask_lines(prev)):
            seen.add(title)

    issues = []
    for i, title, tagged, paper in _citations(lines, masked):
        if title in seen:
            if tagged:
                issues.append((i, "论文引用",
                               f"《{title}》前面已经出现过，arXiv编号只在全书第一次出现时写",
                               lines[i - 1].rstrip()))
            continue
        if paper:
            seen.add(title)
            if not tagged:
                issues.append((i, "论文引用",
                               f"《{title}》在全书第一次出现，要写成《标题》（arXiv:编号）",
                               lines[i - 1].rstrip()))
    return issues



# 规则6 图片宽度：用 width 属性，值是图片像素宽度的一半
def rule_img_width(lines, masked, path):
    """<img> 用 width="像素宽度的一半"，不用 style="zoom:50%"。

    插图都是两倍像素（Retina 截图、export_drawio.py 的 SCALE = 2），要按一半显示。
    `zoom` 只有 Typora 认，GitHub 会把整个 style 属性删掉；width 是标准 HTML 属性，
    到处都认。图片文件不在（images/ 不进 git）时只查 style，不查宽度。
    """
    issues = []
    for i, m in iter_img_tags(lines):
        tag = m.group(0)
        want = img_want_width(tag, path)
        have = IMG_WIDTH_PAT.search(tag)
        if IMG_STYLE_PAT.search(tag):
            hint = f'width="{want}"' if want else 'width="像素宽度的一半"'
            issues.append((i, "图片宽度", f"不要用 style 缩放（GitHub 会删掉它），改成 {hint}",
                           lines[i - 1].rstrip()))
        elif want is not None and (not have or int(have.group(1)) != want):
            now = f'现在是 {have.group(1)}' if have else '现在没写'
            issues.append((i, "图片宽度", f'width 应为图片像素宽度的一半 {want}（{now}）',
                           lines[i - 1].rstrip()))
    return issues


RULES = [
    rule_notation,
    rule_no_em_dash,
    rule_math_vs_code,
    rule_paper_citation,
    rule_img_width,
    # rule_md_toycode is appended below, after the toy-code helpers it reuses
]


# --------------------------------------------------------------------------
# Toy-code rules -- apply to book/code/**/toy_*.py only.
# Signature is (lines, path) -> [(行号, 规则名, 问题描述, 原文), ...]
# --------------------------------------------------------------------------

# 规则2 玩具代码：统一用 numpy array，向量名 vec 开头、矩阵名 mat 开头
NUMPY_IMPORT_PAT = re.compile(r'^\s*(import\s+numpy|from\s+numpy\s+import)')
LIST_ANNOT_PAT = re.compile(r'\blist\s*\[\s*(list\s*\[)?\s*(float|int)\s*\]')

# 赋值语句左侧的简单标识符
ASSIGN_PAT = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?!=)(.*)$')

# 纯数字的列表字面量：一维是向量，嵌套是矩阵。要求至少含一个数字，
# 否则 `acc = []`（后续 append 矩阵的空容器）会被误判成向量。
NUM_LIST_PAT = re.compile(r'^\[[\s\d.,+\-eE\[\]]*\d[\s\d.,+\-eE\[\]]*\]$')
NESTED_LIST_PAT = re.compile(r'^\[\s*\[')

# 右侧构造出的是矩阵（二维及以上）还是向量（一维）
# Anchored at the start of the right-hand side on purpose: the assignment must
# BE the array, not merely mention one. `layer = new_fc_layer(np.random.randn(
# 100, 300), ...)` binds a closure, and `kernels = [(np.array([[..]]), b), ..]`
# binds a list of tuples -- neither is a matrix.
MATRIX_RHS = [
    re.compile(r'^np\.(array|asarray)\s*\(\s*\[\s*\['),          # np.array([[..
    re.compile(r'^np\.[\w.]+\s*\(\s*\(\s*[\w.]+\s*,'),           # np.zeros((n, m))
    re.compile(r'^np\.[\w.]+\s*\(\s*[\w.]+\s*,\s*[\w.]+\s*\)'),  # np.random.rand(n, m)
    re.compile(r'^[\w.]+\.reshape\s*\(\s*[^)]*,'),               # a.reshape(n, m)
]
VECTOR_RHS = [
    re.compile(r'^np\.(array|asarray)\s*\(\s*\[(?!\s*\[)'),      # np.array([1, 2])
    re.compile(r'^np\.[\w.]+\s*\(\s*[\w.]+\s*\)'),               # np.zeros(n)
]


def _rhs_kind(rhs):
    """'matrix' / 'vector' / None -- what this right-hand side constructs.

    Recognises both numpy constructors and bare numeric list literals (the
    latter matter because representing an array as a list is itself a rule-2
    violation). Returns None whenever the shape is not mechanically obvious;
    the rule only reports what it can prove.
    """
    rhs = rhs.strip()
    for pat in MATRIX_RHS:
        if pat.match(rhs):
            return 'matrix'
    for pat in VECTOR_RHS:
        if pat.match(rhs):
            return 'vector'
    if NUM_LIST_PAT.match(rhs):
        return 'matrix' if NESTED_LIST_PAT.match(rhs) else 'vector'
    return None


def _numpy_issues(body, offset, tag, exempt):
    """Rule 2.1 -- vectors/matrices go in numpy arrays, not Python lists.

    `exempt` suppresses this check for hand-written demos that avoid numpy on
    purpose (see the toy-exempt marker). Only representing an array as a list
    is reported; scalar-only code needs no numpy at all and is never flagged.
    """
    if exempt:
        return []
    issues = []
    for k, line in enumerate(body):
        m = LIST_ANNOT_PAT.search(line)
        if m:
            issues.append((offset + k, f"{tag}-numpy",
                           f"用 `{m.group(0)}` 表示向量／矩阵："
                           f"玩具代码统一改用 numpy 的 array",
                           line.rstrip()))
    for rel, text in logical_lines(body):
        m = ASSIGN_PAT.match(text)
        if m and NUM_LIST_PAT.match(m.group(2).strip()):
            issues.append((offset + rel - 1, f"{tag}-numpy",
                           f"`{m.group(1)}` 用 Python 列表字面量表示向量／矩阵："
                           f"改用 numpy 的 array",
                           body[rel - 1].rstrip()))
    return issues


def _naming_issues(body, offset, tag):
    """Rule 2.2 / 2.3 -- vectors named vec*, matrices named mat*.

    Applies even to toy-exempt code: the marker excuses not using numpy, it
    does not excuse a name that hides the variable's shape.
    """
    issues = []
    for rel, text in logical_lines(body):
        m = ASSIGN_PAT.match(text)
        if not m:
            continue
        name, rhs = m.group(1), m.group(2)
        kind = _rhs_kind(rhs)
        if kind is None:
            continue
        want = 'vec' if kind == 'vector' else 'mat'
        bare = name.lstrip('_')
        if bare.startswith(want):
            continue
        other = 'mat' if want == 'vec' else 'vec'
        label = '向量' if kind == 'vector' else '矩阵'
        hint = (f"（现在是 `{other}` 开头，但右侧构造的是{label}）"
                if bare.startswith(other) else '')
        issues.append((offset + rel - 1, f"{tag}-命名",
                       f"`{name}` 是{label}，变量名应以 `{want}` 开头{hint}",
                       body[rel - 1].rstrip()))
    return issues


def rule_toy_numpy(lines, path):
    """玩具代码统一使用 numpy 的 array，不用 Python 原生列表。"""
    exempt = has_exempt_marker(''.join(lines))
    return _numpy_issues(lines, 1, "玩具代码", exempt)


def rule_toy_naming(lines, path):
    """向量变量名统一 vec 开头，矩阵变量名统一 mat 开头。"""
    return _naming_issues(lines, 1, "玩具代码")


TOY_RULES = [
    rule_toy_numpy,
    rule_toy_naming,
]


def rule_md_toycode(lines, masked, path):
    """章节里的玩具代码块同样遵守规则2（PyTorch 示例块跳过）。"""
    issues = []
    for offset, body in iter_code_blocks(lines, 'python'):
        if is_runnable_example(body):
            continue
        exempt = block_is_exempt(lines, offset) or has_exempt_marker(''.join(body))
        issues.extend(_numpy_issues(body, offset, "正文代码", exempt))
        issues.extend(_naming_issues(body, offset, "正文代码"))
    return issues


RULES.append(rule_md_toycode)


# --------------------------------------------------------------------------


def check(path):
    lines = read_lines(path)
    issues = []
    if is_toy(path):
        for rule in TOY_RULES:
            issues.extend(rule(lines, path))
    else:
        masked = mask_lines(lines)
        for rule in RULES:
            issues.extend(rule(lines, masked, path))
    issues.sort(key=lambda x: x[0])
    return issues


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    include_generated = '--include-generated' in sys.argv
    if not args:
        print(f"Usage: {sys.argv[0]} [--all-rules] <file.md | dir> [more...]")
        sys.exit(1)

    if not RULES and not TOY_RULES:
        print("（尚未定义任何规则，见 SKILL.md 的 Rules 小节）")

    all_rules = '--all-rules' in sys.argv
    total = skipped = 0
    for path in iter_targets(args, include_generated):
        if excluded(path):
            continue
        issues = check(path)
        before = len(issues)
        issues = keep(issues, path, all_rules=all_rules)
        skipped += before - len(issues)
        if not issues:
            continue
        total += len(issues)
        print(f"=== {path} ===")
        for lineno, rule, problem, text in issues:
            print(f"L{lineno} [{rule}] {problem}")
            print(f"  原文: {text[:100]}")
            print()

    note = f"（另有 {skipped} 处被 .claude/lint-scope.json 豁免）" if skipped else ""
    if total == 0:
        print("✓ 未发现问题" + note)
    else:
        print(f"共发现 {total} 处违规{note}")
        sys.exit(1)


if __name__ == "__main__":
    main()
