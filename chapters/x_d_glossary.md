# 附录D：术语表

正文里正式定义过的术语，按第一次正式定义的先后排列，最后一列是定义它的章节，前面先提了一句、后面再详细讲的，两章都写上。有简称的，简称写在英文后面。

<!-- 草稿：条目取自正文里“**中文**（English）”形式的定义，外加几个正文常用、但没有加粗定义过的词（标了*）。是否按拼音重排、要不要加一句话的解释，由作者定。 -->



| 中文 | 英文 | 说明 | 章节 |
| --- | --- | --- | --- |
| 词元 | Token | LLM处理文字的单位 | 第一章、第三章 |
| 分数 | Logits | LLM给词表里每个词元打的分，还不是概率 | 第一章 |
| 预填充 | Prefill | 把整段提示词一次性送进模型 | 第一章、第十一章 |
| 解码 | Decode | 每次只送进刚生成的那1个词元 | 第一章、第十一章 |
| Transformer* | Transformer | 2017年提出的架构，现在的LLM大多只用它的Decoder部分 | 第一章 |
| 张量 | Tensor | 多维数组，向量和矩阵都是它的特例 | 第二章 |
| 元数据 | Metadata | Safetensors文件头里的JSON，记录每个张量的类型、形状和位置 | 第二章 |
| 小端序 | Little-Endian | 多字节整数低位在前的存法 | 第二章 |
| 文件头 | Header | Safetensors文件开头的8字节长度加元数据 | 第二章 |
| BF16* | BrainFloat16 | 8位指数、7位尾数的16位浮点格式 | 第二章 |
| 词表 | Vocabulary | 词元到ID的对照表，SmolLM2有49152个词元 | 第三章 |
| BPE* | Byte Pair Encoding | 按合并规则把字节一步步合成词元的分词算法 | 第三章 |
| BOS | Beginning of Sequence | 标记序列开头的特殊词元 | 第三章 |
| EOS | End of Sequence | 标记序列结束的特殊词元，生成循环靠它停下 | 第三章 |
| 词嵌入 | Embedding | 代表一个词元的向量，SmolLM2里是576维 | 第四章 |
| 特征 | Feature | 向量每一维上的数值所代表的东西 | 第四章 |
| 特征向量 | Feature Vector | 词嵌入的另一个叫法 | 第四章 |
| 相似度 | Similarity | 衡量两个向量有多接近 | 第四章 |
| 余弦相似度 | Cosine Similarity | 只看夹角的相似度 | 第四章 |
| 线性变换 | Linear Transformation | 乘一个矩阵 | 第四章 |
| 线性投影 | Linear Projection | 同上；出口那一步把576维变成49152个分数 | 第四章 |
| 权重共享 | Weight Tying | 入口和出口共用词嵌入矩阵 | 第四章 |
| 归一化 | Normalization，简称Norm | 把一堆数值拉到同一个尺度上 | 第五章 |
| 批归一化 | Batch Normalization，简称BatchNorm | 跨样本统计的归一化 | 第五章 |
| 层归一化 | Layer Normalization，简称LayerNorm | 每个词元用自己的数统计的归一化 | 第五章 |
| 均值 | Mean | | 第五章 |
| 方差 | Variance | | 第五章 |
| 标准差 | Standard Deviation | 方差的平方根 | 第五章 |
| 增益 | Gain | LayerNorm、RMSNorm里的 $\boldsymbol{\gamma}$ | 第五章 |
| 偏置 | Bias | LayerNorm里的 $\boldsymbol{\beta}$ ；SmolLM2没有任何偏置 | 第五章 |
| 哈达玛积 | Hadamard Product | 逐元素相乘 | 第五章 |
| 均方根 | Root Mean Square，简称RMS | RMSNorm用它代替标准差 | 第五章 |
| post-norm* | Post-Norm | 先相加再归一化，2017年的做法 | 第五章 |
| pre-norm* | Pre-Norm | 先归一化再送进子层，SmolLM2的做法 | 第五章 |
| 注意力 | Attention | 唯一让词元之间互相看的模块，本质是加权求和 | 第六章 |
| 缩放点积注意力 | Scaled Dot-Product Attention | 标准的注意力公式 | 第六章 |
| 查询 | Query | Q，“我想找什么” | 第六章 |
| 键 | Key | K，“我是什么” | 第六章 |
| 值 | Value | V，“匹配上了，我能提供什么” | 第六章 |
| 自注意力 | Self-Attention | Q、K、V都来自同一个输入 | 第六章 |
| 头 | Head | 注意力并行算的那几份 | 第六章 |
| 超参数 | Hyperparameter | 层数、头数这类架构上的数 | 第六章 |
| 多头注意力 | Multi-Head Attention，简称MHA | 每个Q头都有自己的K和V | 第六章 |
| 多查询注意力 | Multi-Query Attention，简称MQA | 所有Q头共用一组K和V | 第六章 |
| 分组查询注意力 | Grouped-Query Attention，简称GQA | 几个Q头一组，每组共用一组K和V | 第六章 |
| 自监督学习 | Self-Supervised Learning | 拿原文的下一个词当答案来训练 | 第六章 |
| 因果掩码 | Causal Mask | 把未来位置的分数换成负无穷 | 第六章 |
| Softmax* | Softmax | 归一化指数函数，把一组分数变成加起来等于1的概率 | 第六章 |
| 位置编码 | Positional Encoding，简称PE | 把位置信息告诉模型 | 第七章 |
| 正弦位置编码 | Sinusoidal PE | 2017年的方案，加在词嵌入上 | 第七章 |
| 旋转位置编码 | Rotary Position Embedding，简称RoPE | 按位置旋转Q和K | 第七章 |
| 底数 | Base | RoPE公式里的10000，SmolLM2是100000 | 第七章 |
| 前馈网络 | Feed-Forward Network，简称FFN | 逐词元的非线性加工 | 第八章 |
| 多层感知机 | Multi-Layer Perceptron，简称MLP | FFN的另一个名字 | 第八章 |
| 激活函数 | Activation Function | 逐元素的非线性函数 | 第八章 |
| ReLU | Rectified Linear Unit | $\max(0, x)$ | 第八章 |
| GELU | Gaussian Error Linear Unit | $x \cdot \Phi(x)$ | 第八章 |
| SiLU | Sigmoid Linear Unit | $x \cdot \sigma(x)$ ，也叫Swish | 第八章 |
| 门控线性单元 | Gated Linear Unit，简称GLU | 一路投影当门，调节另一路 | 第八章 |
| SwiGLU* | SwiGLU | 门那一路用SiLU的GLU，SmolLM2的FFN | 第八章 |
| 注意力汇聚点 | Attention Sink | 大量注意力落在序列开头的现象 | 第八章 |
| 混合专家 | Mixture of Experts，简称MoE | 每个词元只挑几个小FFN来算 | 第八章 |
| 专家 | Expert | MoE里的每个小FFN | 第八章 |
| 路由器 | Router | 给每个词元挑专家 | 第八章 |
| 共享专家 | Shared Expert | 每个词元都要算的专家 | 第八章 |
| 残差连接 | Residual Connection | 子层的结果加回到它自己的输入上 | 第九章 |
| 残差流 | Residual Stream | 从词嵌入一路通到出口、不断累加的主干 | 第九章 |
| 反向传播 | Backpropagation | 训练时把误差信号从出口往回传 | 第九章 |
| 梯度 | Gradient | 往回传的那个信号 | 第九章 |
| 梯度消失 | Vanishing Gradient | 信号越传越小 | 第九章 |
| 梯度爆炸 | Exploding Gradient | 信号越传越大 | 第九章 |
| logit lens* | Logit Lens | 把中间层的残差流直接送去出口，看它预测什么 | 第九章 |
| 超连接 | Hyper-Connections，简称HC | 把主干扩成好几条 | 第九章 |
| 流形约束超连接 | Manifold-Constrained Hyper-Connections，简称mHC | 要求主干上的混合矩阵是双随机矩阵 | 第九章 |
| 双随机矩阵 | Doubly Stochastic Matrix | 行和、列和都是1的非负方阵 | 第九章 |
| 注意力残差 | Attention Residuals，简称AttnRes | 用注意力给前面各层的贡献加权 | 第九章 |
| 采样 | Sampling | 从概率清单里挑一个词元 | 第十章 |
| 采样器 | Sampler | 负责采样的模块 | 第十章 |
| 安全Softmax | Safe Softmax | 先减去最大值再取指数，不会溢出 | 第十章 |
| 贪婪采样 | Greedy Sampling | 每次挑概率最高的 | 第十章 |
| 重复惩罚 | Repetition Penalty | 给上文出现过的词元打折 | 第十章 |
| 束搜索 | Beam Search | 同时保留好几条候选路线 | 第十章 |
| 随机采样 | Random Sampling | 按概率随机抽 | 第十章 |
| 随机数种子 | Random Seed | 伪随机数的起点，种子相同结果就相同 | 第十章 |
| 幻觉 | Hallucination | 一本正经地胡说八道 | 第十章 |
| 过滤器 | Filter | 抽之前改分数的预处理 | 第十章 |
| 温度 | Temperature | 分数除以的那个数，越大分布越平 | 第十章 |
| top-k* | Top-k | 只留分数最高的 $k$ 个 | 第十章 |
| 核采样 | Nucleus Sampling | 即top-p，只留累计概率够 $p$ 的那一撮 | 第十章 |
| 基座模型 | Base Model | 只学过续写的模型 | 第十章 |
| 对话模型* | Instruct Model | 在基座模型上再用对话数据训练过的模型 | 第十章 |
| KV缓存 | KV Cache | 存下每一层历史词元的K和V | 第十一章 |
| 计算密集型 | Compute-bound | 时间主要花在计算上 | 第十一章 |
| 内存带宽密集型 | Memory-bound | 时间主要花在搬数据上 | 第十一章 |
| 批处理 | Batching | 把很多条序列凑成一批一起算 | 第十一章 |
| PD分离 | Prefill-Decode Disaggregation | 把prefill和decode放到不同的机器上 | 第十一章 |
| 多头潜在注意力 | Multi-head Latent Attention，简称MLA | 把K和V压缩后再存 | 第十一章 |
| 量化 | Quantization | 把数映射成很小的整数再配一个缩放系数，用更少的位数来存 | 第十一章 |
| HBM | High Bandwidth Memory | GPU的显存，容量大，比SRAM慢 | 第十二章 |
| SRAM* | Static Random-Access Memory | GPU芯片上的高速存储，容量小 | 第十二章 |
| 在线Softmax | Online Softmax | 分块计算Softmax、边走边修正的办法 | 第十二章 |
| FlashAttention* | FlashAttention | 分块计算注意力，不把完整的分数表存下来 | 第十二章 |
| QK归一化 | QK-Norm | RoPE之前，Q和K的每个头各过一次RMSNorm | 第十三章 |
