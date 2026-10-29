# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese-language book in progress, 《自己动手写LLM推理引擎》 (title still provisional) — teaching how LLM **inference** works by hand-writing an engine for it. Target model is SmolLM2-135M on PyTorch 2.2 / CPU.

The thesis, stated in `chapters/_preface.md`: 从训练入手会比较难，从推理入手就比较简单；从 GPU 入手会比较难，从 CPU 入手就比较简单. Everything follows from that — the model choice, the hardware, the chapter order.

Three standing conventions from the preface that the manuscript must honour:

1. **No backpropagation.** The target reader is 略懂 AI/DL but 害怕微积分和反向传播. Anything whose real explanation lives in gradients gets the result only, plus its consequence for inference code — never the derivation. (E.g. pre-norm vs post-norm is argued from "is there a rescale on the trunk", not from gradient flow.)
2. **Every chapter diffs against `transformers`.** Chapter 1 dumps a baseline; every later chapter compares to it. The strictness varies deliberately: `==` on ints (ch03) → bit-identical (ch05) → relative error < 1e-5 (ch07 onward, once RoPE's cos table introduces a 1-ulp gap) → back to `torch.equal` (ch10, because argmax chains amplify).
3. **Shapes are the subject.** Most bugs here are shape bugs, not math bugs, and they fail silently. Chapters spell out `(T, 576) → (T, 9, 64) → (T, 576)` at every transformation.

## Layout

| Path | Role |
|------|------|
| `chapters/_front.md` | 扉页 — title, English subtitle, version stamp, cover image |
| `chapters/todo.md` | scratch list of open decisions; **not part of the book**, to be deleted. `merge.py` skips it |
| `chapters/_preface.md` | 前言 — motivation, target reader, reading conventions, tech choices, TOC. Also absorbed the old `ch00.md` (准备). Sorts before `ch01` thanks to the `_` prefix. |
| `chapters/chNN_slug.md` | manuscript source, one file per chapter, `chNN` matches 第N章. Fourteen chapters, `ch01`–`ch14`. |
| `images/chNN/` | exported PNGs, referenced from chapters as `../images/chNN/...` inside `<img ... width="N" />` tags, where N is half the PNG's pixel width (every image is 2x: Retina screenshots, and `export_drawio.py` exports at `SCALE = 2`). Never `style="zoom:50%"`: GitHub strips `style` and `zoom` is non-standard CSS. `check-prose` checks the width (rule `图片宽度`) and its `fix.py` rewrites it after an image is re-exported |
| `draw/chNN.drawio` | draw.io sources for those PNGs. A plotted figure (a function curve, not a diagram) has no drawio page: its source is a `plot_*.py` in that chapter's code directory, e.g. `code/book/ch08/plot_af.py` → `images/ch08/af.png`. matplotlib is not a project dependency; the script declares it with PEP 723, so run it as `uv run book/ch08/plot_af.py` (no `python`). A plot that needs the model (real logits) cannot use PEP 723, which drops the project's torch; `code/book/ch10/plot_sampling.py` → `images/ch10/filters.png`, `temperature.png` runs in the project env instead, as `uv run --with matplotlib --with "numpy<2" python book/ch10/plot_sampling.py` (without the numpy pin, the overlay pulls numpy 2 and torch 2.2 breaks) |
| `Book.md` | **generated** single-file manuscript, gitignored, never edit by hand — see `merge.py` below |
| `merge.py` | the generator: `python3 merge.py` (regenerate), `--check` (exit 1 if `Book.md` is out of sync with `chapters/`, version stamp ignored), `-o` (write elsewhere) |
| `code/` | one uv project holding two code trees — see `code/CLAUDE.md` |
| `code/book/` | the book's code: one directory per chapter, `ch01/`…`ch12/`. **New work goes here.** `ch01/`–`ch14/` are all written. Like ch09, `ch12/` adds no engine code: its engine files are byte-identical copies of `ch11/`, plus `online_softmax.py` and `flash_attn.py` (a pure-PyTorch tiled attention; on CPU it is *faster* than the one-shot version at block sizes 128–256, because it skips future blocks and fits in cache). `ch14/` adds `device` / `dtype` / `attn_impl` to `Model`; with the defaults it is bit-identical to `ch13/`. The author's machine has no CUDA, so ch14's numbers come from CPU and MPS (torch 2.2's MPS has no bf16). ch09 adds no engine code: its four engine files are byte-identical copies of `ch08/` (an untouched file keeps the header of the chapter that last changed it, as `weights.py` does everywhere), plus three inspection scripts (`residual_stream.py`, `logit_lens.py`, `sinkhorn.py`), so `ch10/` still builds on `ch08/`. In `ch11/`, `NO_CACHE` must reproduce ch10 bit for bit, and the engine feeds `ids[cache.length:]` instead of branching on the cache. |
| `code/poc/` | the original proof of concept. **Frozen reference, not the book's code.** |
| `code/models/`, `code/reference/` | weights and baseline tensors, shared by both trees, gitignored |

| `tools/` | standalone HTML tools the chapters point readers to (`sinpe-heatmap/`, `rope-angles/`, `safetensors-viewer/`): one `index.html` each, no build step, plus a `README.md` |
| `export_drawio.py` | exports one chapter's `draw/chNN.drawio` page by page into `images/chNN/`, then shrinks the PNGs |
| `.claude/` | this book's own skills (`check-formulas`, `check-prose`, `polish-prose`, `shrink-images`) and `lint-scope.json` |

**The book directory is self-contained.** Everything it needs, the skills and lint config included, lives under this directory. Never reference, import or run anything from a parent directory or a sibling book; if a rule or script is needed, keep a copy here.

`merge.py` concatenates `_front.md`, `_preface.md` and `chNN_*.md` sorted by filename (`todo.md` is skipped) and, on the way through: demotes every heading one level outside code fences (`#` → `##`, so the book title in `_front.md` stays the only H1), inserts the page-break `<div>` between chapters, rewrites every relative asset path `../` → `./`, inserts the 自动生成 banner under the H1, and stamps `版本：vYYYY.MM.DD` with today's date. So **chapter files keep their H1 chapter titles, must not contain the page-break div or the banner**, and asset paths in `chapters/` are always written `../images/...` / `../aigc/...`. Run it after editing any chapter; the linters skip `Book.md` by name. The table of contents lives in `_preface.md` under 本书结构; chapter order is otherwise carried by the filenames.

**Read `code/CLAUDE.md` before writing any code.** It records the environment pins, the verified SmolLM2 config values, and the tokenizer / RoPE / KV-cache gotchas that were found the hard way. Reuse those findings rather than rediscovering them — but do not copy `poc/`'s file layout or API, since `book/` is deliberately a different take (one self-contained directory per chapter). Extending `poc/` is not the task.

## Chapters

Order follows the data flow of one forward pass, split into two parts.

**跑通 (1–10)** — walk the pipeline left to right, one component per chapter:

| Ch | Topic | PoC reference |
|---|---|---|
| ch01 | LLM推理引擎概览 — run it with `transformers` first, dump the baseline | `poc/steps/00_reference.py` |
| ch02 | 模型文件与权重 | `poc/toyllm/config.py`, `weights.py`, `poc/steps/01_load_and_inspect.py` |
| ch03 | 分词（Tokenization） | `poc/toyllm/tokenizer.py`, `poc/steps/02_tokenizer.py` |
| ch04 | 词嵌入（Embedding） — one matrix read both ways | `poc/toyllm/weights.py`, `model.py` |
| ch05 | 归一化（Normalization） — RMSNorm, pre-norm vs post-norm | `layers.RMSNorm`, `poc/steps/03_rmsnorm.py` |
| ch06 | 注意力机制（Attention） — GQA + causal mask | `layers.Attention`, `poc/steps/06_attention.py` |
| ch07 | 位置编码（Positional Encoding） — classic PE, then RoPE | `layers.RotaryEmbedding`, `poc/steps/05_rope.py` |
| ch08 | 前馈网络（FFN） — SwiGLU, with a brief look at MoE | `layers.SwiGLU`, `poc/steps/04_swiglu.py` |
| ch09 | 残差连接 — wraps both sublayers; the Block is complete here | `poc/toyllm/model.py`, `poc/steps/07_forward.py` |
| ch10 | 采样（Sampling） — plus the generate loop | `poc/toyllm/sampling.py`, `poc/steps/09_sampling.py` |

**跑快 (11–12)** — same data flow, different execution:

| Ch | Topic | PoC reference |
|---|---|---|
| ch11 | KV Cache — prefill / decode | `layers.KVCache` / `NoCache`, `poc/steps/08_kv_cache.py` |
| ch12 | FlashAttention — online softmax + tiling | none (`code/book/ch12/flash_attn.py`) |

**走出去 (13–14)** — same engine, another model, another device:

| Ch | Topic | PoC reference |
|---|---|---|
| ch13 | 换一个模型 — Qwen3-0.6B, chapter-by-chapter review of what changes; `--model` picks the directory under `code/models/` | none |
| ch14 | GPU — device, BF16, batching, FlashAttention measured | none |

Note the chapter order **differs from the order the PoC was built in**, so a chapter's material usually lives in a differently-numbered step script. Three chapters have no single PoC counterpart: ch04 (the embedding/lm_head pair was never its own step), ch09 (the PoC never separated residuals from the Block), and ch12.

Each unfinished chapter opens with a `> **本章代码**` blockquote stating what to build, the tensor shapes, the current engine state, what to diff against, and the known traps. **This block is scaffolding, not book content** — the author deletes it once the chapter's prose is written (ch01–ch05 no longer have one). Treat it as the chapter's build contract while drafting, and never as text a reader will see.

A consequence: the diffing criteria live only in those blocks, in the code, and in this file. **The prose never mentions the baseline file or the tolerance table.** When a chapter needs to refer to the reference output, it points at something the reader has actually seen, e.g. 「和第一章那句话一字不差」 rather than 「和第一章那份基准逐字节相同」.

**When writing a chapter, run its code first and quote real output.** The numbers in ch02 (272 tensors, 269.1 MB, 134,515,008 params, 3,540,096 per block) come from `poc/steps/01_load_and_inspect.py` and must keep matching it.

## Commands

Everything runnable lives under `code/`, and must go through `uv run` (system `python3` is 3.9 with no torch):

```bash
cd code
uv sync --group dev                                # .venv + deps + pytest oracles
uv run python book/ch01/hello_5l.py              # run one chapter's code (book/ is where new work goes)
uv run python poc/steps/07_forward.py              # run one PoC milestone script
uv run pytest                                      # all tests (testpaths = poc/tests)
uv run pytest poc/tests/test_rope.py -x            # one test file
```

Downloads (weights, `uv sync`) must bypass Claude Code's proxy — prefix with
`env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy`. See `code/CLAUDE.md`.

## Writing conventions

- **Language and punctuation.** Prose is Simplified Chinese; code, identifiers, filenames and diagram names are English. No space between CJK and Latin/digits (`SmolLM2模型`, not `SmolLM2 模型`). Full-width Chinese quotes `“”`, never `""`. No em-dashes (`——`): use a comma, a colon, or split the sentence.
- **Notation: matrices are bold uppercase.** This book follows Goodfellow's scheme: bold lowercase = vector ($\mathbf{q}_m$, $\mathbf{x}$), **bold uppercase = matrix** ($\mathbf{Q}$, $\mathbf{W}_Q$, $\mathbf{R}_m$), unbolded = a plain count or scalar ($d$, $T$, $m\theta_i$). ch06 and ch07 are written this way; do not "fix" bold matrices to plain. `.claude/lint-scope.json` disables `符号记法/粗体大写` for that reason. Greek vectors still need `\boldsymbol`, and `\vec{}` is still wrong.

- **Em-dash cleanup is in progress**: 110 `——` remain in ch06–ch12 and `_preface.md`, from before the author's voice pass. Clear them as each chapter gets rewritten rather than in one sweep. (ch01–ch05 are clean.)
- **The model is always `SmolLM2`, never `SmolLM`.** This typo has recurred several times; grep for `SmolLM[^2]` after editing.
- **Large integers fall into three classes**, and ch02/ch03 already follow this:
  - *Identifiers* never take a separator, because they say *which one*, not *how many*: token ids (46119), line numbers (第49347行), merge ranks (45867), code points (`U+0120`).
  - *Spec values* a reader can look up in `config.json` / `tokenizer.json` stay bare, so prose and file read the same: 49152, 576, 8192, 30.
  - *Quantities* take a comma every three digits from four digits up: 134,515,008 params, 269,060,552 bytes, 30,536 bytes, 35,136 params.
  - One exception: a quantity being reconciled against a spec value in the same passage follows the spec value and stays bare. ch03's 48900 merge rules sits inside `49152 - 17 - 235 = 48900`, so both stay bare rather than mixing `49,152` with `48,900`.
  - Numbers inside code blocks, JSON excerpts and quoted script output are verbatim and never reformatted.
- **Paper citations**: the first time a paper appears in the book (file-name order: `_preface`, `ch01`, …) it is written 《Title》（arXiv:NNNN.NNNNN）, with the title exactly as on arXiv; every later mention, in any chapter, is just 《Title》. A paper mentioned only by author and year gets its title added. `check-prose` enforces both halves (rule `论文引用`).
- Bold marks a term's first formal definition, given as **中文**（English）: **词元**（Token）, **前馈网络**（Feed-Forward Network，简称FFN）. Sampling parameters keep their conventional lowercase-hyphenated spelling in prose (`top-k`, `top-p`) and snake_case in code (`top_k`, `top_p`).
- Chapter titles are 中文（English）where a standard English term exists — 分词（Tokenization）, 归一化（RMSNorm） — and plain English only where no settled Chinese term exists (KV Cache, FlashAttention).
- Chapters end with a `## 本章小结` section (小结 = summary; 小节 would mean subsection — the whole book was corrected from that typo, so do not reintroduce it).
- Chapter files may still hold **raw pasted terminal or AI output as scratch material**, and `_preface.md` keeps its open questions in a trailing HTML comment. Do not assume everything in a chapter file is intended final text.
- The preface is the author's own voice, first person. Preserve their wording when editing it; add rather than rewrite.

### Linters

Both live in this book's own `.claude/skills/`, and read their per-file rule scoping from `.claude/lint-scope.json`. Run them from the book root (this directory):

```bash
python3 .claude/skills/check-formulas/scripts/check.py chapters/chNN_slug.md
python3 .claude/skills/check-prose/scripts/check.py    chapters/chNN_slug.md
```

`check-formulas` rules: inline `$...$` needs surrounding spaces, blank lines around block formulas, `\begin{aligned}` for multi-line, `\tag{}` after `\end{aligned}`, no `\operatorname`, multi-letter function names in `\mathrm{}`. `check-prose` rules: 符号记法统一 (notation), 尽量不用破折号 (em-dash), 公式用 `$`/代码用反引号 (math-vs-code). Per-file exemptions live in `.claude/lint-scope.json`: `CLAUDE.md`, `todo.md`, `node_modules/` and the generated `docs/` are excluded outright, and 符号记法/粗体大写 is off everywhere (bold matrices are this book's convention). Exempted counts are reported, never silently dropped; `--all-rules` puts them back for one run.

Both accept a directory (`chapters/`) or the whole repo, and both have a `fix.py` beside them that takes `--dry-run`. Run `check-formulas` after editing any chapter — inline `$...$` needs surrounding spaces, and it is easy to miss by hand. `check-prose` is useful here mainly for the em-dash sweep above, but note it also reports on the `> **本章代码**` scaffolding blockquotes, which are not book content — ignore those hits rather than editing the scaffolding to satisfy the linter. Its toy-code rules dispatch on the `toy_*.py` filename, so they do not touch `code/`.

### Images

`.claude/skills/shrink-images/` compresses the manuscript's PNGs in place (flatten alpha onto white, strip metadata, and quantize to an 8-bit palette only when the measured error says the image is line art). `export_drawio.py` calls it automatically after exporting a chapter's draw.io pages; run it by hand after a new 章首图 lands in `aigc/`:

```bash
uv run --no-project .claude/skills/shrink-images/scripts/shrink.py --dry-run .
```

It needs Pillow, declared inline (PEP 723) so `uv run` builds the env — the system `python3` is 3.9 with no Pillow.

## Site

Published at https://zxh0.github.io/llm-engine-book/ (full-text search and formula rendering). `.github/workflows/deploy.yml` builds and deploys on every push to `main`.

```bash
npm ci                   # once
npm run dev              # prepare:site + vitepress dev
npm run build            # prepare:site + vitepress build -> docs/.vitepress/dist
```

`scripts/build-site.mjs` is the only content pipeline. `chapters/*.md` and `tools/*/` are the sources; `docs/chapters/`, `docs/public/` and `docs/.vitepress/sidebar.json` are regenerated on every run and gitignored. `docs/index.md` (进度和目录), `docs/tools.md` and `docs/.vitepress/` are hand-written.

- **Chapters go up one at a time.** `PUBLISHED` in `build-site.mjs` lists the chapter files that get a page; the rest only show as grey titles on the index page. Publishing a chapter means adding its filename there.
- Pages keep the `chapters/` filenames and are mapped to clean URLs (`/ch01-overview`) through VitePress `rewrites`, so the edit link resolves straight back to the source file.
- `<img src="../images/chNN/x.png" style="zoom:50%">` tags are rewritten to Markdown images under `/images/chNN/`; `../aigc/` likewise. `aigc/front.png` is copied as the site's `cover.png`.
- **Adding a tool** means dropping a directory with an `index.html` into `tools/` and adding one entry (title, blurb, note) to `TOOLS`; the `/tools` page reads it from `sidebar.json`. Tool pages live outside VitePress, so links to them need `withBase()` in an explicit `<a target="_blank">`. Keep `README.md` in sync.
