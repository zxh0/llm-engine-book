# 第十二章 · FlashAttention

和第九章一样，这一章不改引擎：`config.py`、`weights.py`、`model.py`、`engine.py`、
`kv_cache.py`、`sampler.py`、`detokenizer.py`都是从第十一章原样复制过来的，一字未改。
decode时分数表只有一行，分块帮不上忙，所以不把它接进生成循环。两个脚本把原理验证清楚。

## 运行

```bash
cd code
uv run python book/ch12/online_softmax.py   # 正文那个x=[1,3,2,5]的例子，一步一步打印
uv run python book/ch12/flash_attn.py       # 分块注意力：对拍、接进模型、内存、耗时、块大小
```

`flash_attn.py`在CPU上大约十秒。

## 文件

| 文件 | 作用 |
|---|---|
| `online_softmax.py` | **这一章新增**。维护最大值、分母、分子，分块算softmax(x) · v，和torch.softmax比 |
| `flash_attn.py` | **这一章新增**。`flash_attention()`：外层循环Q块，内层循环K、V块，整块在未来的跳过；`attn_flash()`把它接进`Model.attn` |
| 其余 | 第十一章那几个，**一字未改** |

## 几个数（Intel Mac，CPU，2026-10-17）

```
对拍      随机q、k、v，相对误差1e-7上下；换进30层，逐层9.5e-7，贪婪20个一字不差
内存      T=8192：原来那张表2,415,919,104字节，分块版最大的临时张量589,824字节
跳过      4096块里跳过2016块（整块都在未来）
耗时      T=2048：一次算完227.3ms，分块（块大小128）53.1ms
块大小    8是一次算完的15.8倍，128是0.24倍，2048是1.39倍
```

快的原因一半是跳过了未来的块，一半是小块装得进CPU的缓存；块太小会被Python循环的开销拖垮。
