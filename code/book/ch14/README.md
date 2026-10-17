# 第十四章 · GPU

同一个引擎，换一个地方算。从第十三章复制过来，`Model`和`Engine`多了三个构造参数：
`device`（cpu / cuda / mps）、`dtype`（float32 / bfloat16 / float16）、`attn_impl`
（manual / sdpa）。默认还是CPU、fp32、手写注意力，和第十三章逐位相同。

另外两处：张量前面可以多一维批次，(T, 576)可以是(B, T, 576)；RMSNorm先升到fp32算。

**作者的电脑没有CUDA。** 下面的数字来自一台Intel芯片的MacBook，CPU和MPS（苹果的GPU接口）。
MPS在torch 2.2上不支持bf16。代码在CUDA上也能跑，CUDA的数字等作者在云上跑完再补。

## 运行

```bash
cd code
uv run python book/ch14/check_forward.py                  # CPU上的四级验收，和第十三章一样
uv run python book/ch14/engine.py --device auto           # 有什么设备用什么
uv run python book/ch14/check_device.py                   # 换设备、换精度以后还对不对
uv run python book/ch14/speed.py                          # prefill和decode各花多久
uv run python book/ch14/batch.py                          # 批处理的吞吐量
uv run python book/ch14/flash.py                          # 手写注意力 vs F.scaled_dot_product_attention
```

在CUDA上：

```bash
uv run python book/ch14/check_device.py --device cuda
uv run python book/ch14/speed.py --device cuda
uv run python book/ch14/batch.py --device cuda --dtype bfloat16
uv run python book/ch14/flash.py --device cuda --max-t 16384
```

## 文件

| 文件 | 作用 |
|---|---|
| `model.py` | **这一章的重点**。`device`、`dtype`、`attn_impl`；维度从后往前数，可以带批次；RMSNorm先升fp32；`attend_manual` / `attend_sdpa` |
| `weights.py` | 读的时候换精度、搬设备 |
| `kv_cache.py` | 缓存放在设备上，前面可以多一维批次 |
| `engine.py` | `--device`、`--dtype`、`--attn`；分数搬回CPU再采样 |
| `check_device.py` | **这一章新增**。以CPU、fp32为基准，换设备、精度、注意力实现再对拍 |
| `speed.py` | **这一章新增**。同步、热身之后量prefill和decode |
| `batch.py` | **这一章新增**。B条序列一起生成：正确性和吞吐量 |
| `flash.py` | **这一章新增**。只拿注意力那一段，比耗时和峰值显存（峰值只有CUDA量得到） |
| `config.py`、`sampler.py`、`detokenizer.py` | 第十三章、第十章那几个，**一字未改** |
| `check_forward.py`、`baseline.py` | 第十三章那两个，只改了文件头 |

## 几个数（Intel Mac，2026-10-17）

```
check_device   SmolLM2，MPS fp32：逐层相对误差2.9e-6，贪婪100个全部相同
               MPS fp16：8.4e-3，第94个起分叉；CPU bf16：1.1e-1，第76个起分叉
speed          decode每个词元：CPU 45.0ms，MPS 223.8ms（MPS反而慢）
batch          CPU上B=16，每秒词元是B=1的16.5倍；MPS上10.3倍
flash          T=4096：CPU上手写756.8ms，sdpa 534.3ms
```
