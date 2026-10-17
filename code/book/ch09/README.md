# 第九章 · 残差连接（Residual Connections）

第九章没有新代码。`config.py`、`model.py`、`engine.py`、`weights.py`都是从第八章
原样复制过来的，一字未改，`block()`里那两个加号从第五章起就在了。这一章的三个脚本只是把
那两个加号拆开来看，外加一个和模型无关的小演示。

## 运行

```bash
cd code
./download_model.sh                          # 还没下权重的话，先跑这个
uv run python book/ch09/residual_stream.py   # 每层加了多少；跳过一层会怎样；去掉残差会怎样
uv run python book/ch09/logit_lens.py        # 在每一层后面提前问一句“下一个词是什么”
uv run python book/ch09/sinkhorn.py          # mHC的双随机矩阵，以及30层乘下来会怎样
```

`residual_stream.py`要把30种“跳过一层”各生成20个词元，CPU上大约半分钟。

## 文件

| 文件 | 作用 |
|---|---|
| `config.py` | 第八章那个，**一字未改**。残差连接只是一次加法，不需要任何配置 |
| `model.py` | 第八章那个，**一字未改** |
| `engine.py` | 第八章那个，**一字未改** |
| `weights.py` | 第二章那个，**一字未改** |
| `residual_stream.py` | **这一章新增**。拆开`block()`里的两个加号，打印每层注意力、FFN各往主干上加了多少，前后方向转了多少；再逐层跳过、整体去掉残差，看生成的话怎么变 |
| `logit_lens.py` | **这一章新增**。把每一层之后的主干直接送去出口，看最终答案在第几层冒出来。默认输入是“The Eiffel Tower is located in the city of” |
| `sinkhorn.py` | **这一章新增**。不碰模型，用随机数演示Sinkhorn-Knopp算法，以及不加约束和双随机两种混合矩阵乘30层的结果 |

## 几个数

“Once upon a time”最后一个位置，每层前后主干的余弦大多在0.94 ~ 0.98之间，例外是
第0层（0.092，词嵌入模长只有2.89，第0层的FFN一下子加了37.0）、第24层（0.791，
注意力加了189.0）和第29层（0.834，FFN加了359.9）。

跳过一层：第17层只差2个词元（18/20相同），第0、11、28层一跳就成了乱码。
去掉残差：`'Once upon a timeomenclomenclomencl...'`。

logit lens：`'ĠParis'`在第0 ~ 22层一直排在几千名开外，第23层后142名，第24层后17名，
第26层后第2名，第27层后排到第一，最后概率0.881。
