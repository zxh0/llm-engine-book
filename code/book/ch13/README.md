# 第十三章 · 换一个模型（Qwen3）

同一套代码，跑两个模型：SmolLM2-135M和Qwen3-0.6B。从第十一章复制过来，
把几处“恰好对SmolLM2成立”的写法改成从配置里读，再加上Qwen3的QK-Norm。
`--model`指定`code/models/`下的目录名，不指定就是SmolLM2，行为和第十一章一样。

## 运行

```bash
cd code
REPO=Qwen/Qwen3-0.6B-Base ./download_model.sh      # 基座版，对拍用，约1.2 GB
REPO=Qwen/Qwen3-0.6B ./download_model.sh           # 对话版，聊天用，约1.5 GB
REPO=HuggingFaceTB/SmolLM2-135M-Instruct ./download_model.sh   # 可选：SmolLM2的对话版，约270 MB
uv run python book/ch13/config.py                  # 三个模型的配置并排比
uv run python book/ch13/weights.py Qwen3-0.6B-Base # 每层11个张量
uv run python book/ch13/tokenizers_diff.py         # 两个分词器切同一句话
uv run python book/ch13/baseline.py                # 用transformers给Qwen3存一份基准
uv run python book/ch13/check_forward.py Qwen3-0.6B-Base   # 四级验收
uv run python book/ch13/check_forward.py           # SmolLM2，和第十一章一样
uv run python book/ch13/assumptions.py             # 把SmolLM2的假设套到Qwen3上会怎样
uv run python book/ch13/engine.py "Once upon a time" --model Qwen3-0.6B-Base
uv run python book/ch13/chat.py                    # 和对话版说几句话
uv run python book/ch13/chat.py "What is 17 + 25?" --think --max-new-tokens 1200
uv run python book/ch13/chat.py --model SmolLM2-135M-Instruct --greedy   # 它没有推荐的采样参数，要加--greedy
```

## 文件

| 文件 | 作用 |
|---|---|
| `config.py` | `head_dim`优先读文件；`rope_interleaved`没写就是false；EOS读`generation_config.json`，统一成集合；`qk_norm`按`model_type`认 |
| `weights.py` | 每层张量分成必有的9个和可有的`q_norm`、`k_norm`；对话版多存的`lm_head`核对和词嵌入相同后不读 |
| `model.py` | 每个头的宽度读`config.head_dim`；拆头之后、RoPE之前多了QK-Norm，没有就是`no_norm` |
| `kv_cache.py` | 每个头的宽度读`config.head_dim` |
| `engine.py` | `--model`选模型；EOS判断改成“在不在集合里” |
| `sampler.py`、`detokenizer.py` | 第十章那两个，**一字未改** |
| `baseline.py` | **这一章新增**。导入transformers，给新模型存一份`reference/<模型名>.pt` |
| `check_forward.py` | 第十一章那个，模型和基准文件从命令行来 |
| `assumptions.py` | **这一章新增**。把`Config`的字段一个个改回SmolLM2的样子 |
| `tokenizers_diff.py` | **这一章新增**。两个分词器的文件结构和切分结果 |
| `chat.py` | **这一章新增**。ChatML模板、思考模式、`generation_config.json`推荐的采样参数 |

## 几个数

```
Qwen3-0.6B-Base   310个张量   596,049,920个参数   词嵌入占26.1%
四级验收          29份逐层比，最大相对误差8.0e-7，贪婪20个词元和transformers一字不差
KV Cache          每个词元229,376字节（SmolLM2是46,080）
速度              贪婪100个词元，每个0.166秒（SmolLM2是0.048）
```
