# 附录B：引擎公式汇总

这个附录把推理引擎用到的公式，按数据流动的方向从头到尾排一遍，每一步都标上SmolLM2里的形状和数字。它对应第一章那张总览图，也就是本书的路线图。公式的来历和推导，请翻回注明的章节。

下面用到的符号：

| 符号 | 含义 | SmolLM2里的值 |
| --- | --- | --- |
| $T$ | 这一次送进模型的词元数 | 随输入变化 |
| $V$ | 词表大小，`vocab_size` | 49152 |
| $d$ | 词嵌入维度，`hidden_size` | 576 |
| $h$ | Q头数，`num_attention_heads` | 9 |
| $n_\mathrm{kv}$ | KV头数，`num_key_value_heads` | 3 |
| $g$ | 每组几个Q头共用一组K和V， $h / n_\mathrm{kv}$ | 3 |
| $d_k$ | 每个头的维度， $d / h$ | 64 |
| $d_\mathrm{ff}$ | FFN中间层的宽度，`intermediate_size` | 1536 |
| $n_\mathrm{layer}$ | Decoder块的个数，`num_hidden_layers` | 30 |
| $\epsilon$ | RMSNorm里防止除零的小数，`rms_norm_eps` | 1e-05 |
| $b$ | RoPE的底数，`rope_theta` | 100000 |



## 一、入口：词嵌入（第四章）

$$
\mathbf{X} = \mathbf{E}[\mathrm{ids}]
$$

$\mathbf{E}$ 是 $[V \times d]$ 的词嵌入矩阵，按词元ID取出对应的行，得到 $[T \times d]$ 的 $\mathbf{X}$ 。这一步只是查表，没有计算。

形状： $(T,) \to (T, 576)$



## 二、归一化：RMSNorm（第五章）

$$
\mathrm{RMSNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}
$$

对每个词元的 $d$ 个数单独做，和别的词元无关。 $\epsilon$ 在根号**里面**。 $\boldsymbol{\gamma}$ 是训练出来的 $d$ 维权重，每个Decoder块两个，出口一个，一共 $2 n_\mathrm{layer} + 1 = 61$ 个。

形状： $(T, 576) \to (T, 576)$



## 三、注意力（第六章、第七章）

### 投影

$$
\mathbf{Q} = \mathbf{X} \times \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X} \times \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X} \times \mathbf{W}_V
$$

$\mathbf{W}_Q$ 是 $[576 \times 576]$ ， $\mathbf{W}_K$ 和 $\mathbf{W}_V$ 都是 $[576 \times 192]$ 。权重文件里存的是它们的转置。

形状： $\mathbf{Q}$ 是 $(T, 576)$ ， $\mathbf{K}$ 、 $\mathbf{V}$ 是 $(T, 192)$

### 拆头

$$
[\mathbf{Q}_1, \ldots, \mathbf{Q}_h] = \mathbf{Q}, \quad
[\mathbf{K}_1, \ldots, \mathbf{K}_{h/g}] = \mathbf{K}, \quad
[\mathbf{V}_1, \ldots, \mathbf{V}_{h/g}] = \mathbf{V}
$$

沿着列切，每个头 $d_k = 64$ 维。

形状： $(T, 576) \to (9, T, 64)$ ， $(T, 192) \to (3, T, 64)$

### QK归一化（只有Qwen3等模型有，第十三章）

$$
\mathbf{q} \leftarrow \mathrm{RMSNorm}(\mathbf{q}), \quad \mathbf{k} \leftarrow \mathrm{RMSNorm}(\mathbf{k})
$$

拆完头之后、RoPE之前，Q和K的每个头各自过一次RMSNorm，归一化的是每个头的 $d_k$ 维。SmolLM2没有这一步。

### RoPE

第 $i$ 个平面的转速（ $i = 1, \ldots, d_k/2$ ）：

$$
\theta_i = b^{-2(i-1)/d_k}
$$

第 $m$ 个位置上，第 $i$ 个平面转 $m\theta_i$ 。按half-split配对（第 $j$ 维和第 $j + d_k/2$ 维一组），用两次哈达玛积来算：

$$
\mathrm{RoPE}(\mathbf{x}, m) = \mathbf{x} \odot \cos(m\boldsymbol{\theta}) + \mathrm{rotate\_half}(\mathbf{x}) \odot \sin(m\boldsymbol{\theta})
$$

其中 $\mathrm{rotate\_half}([\mathbf{a}, \mathbf{b}]) = [-\mathbf{b}, \mathbf{a}]$ ， $\mathbf{a}$ 和 $\mathbf{b}$ 是前后两半； $\cos(m\boldsymbol{\theta})$ 是把32个角度的余弦抄两遍拼成的64维向量， $\sin$ 同理。只转Q和K，不转V。转完之后，第 $m$ 个位置的Q和第 $n$ 个位置的K的点积只和 $n - m$ 有关：

$$
(\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_{n-m} \mathbf{k}
$$

形状不变。

### 每个头的注意力

第 $i$ 个Q头用第 $\lceil i/g \rceil$ 组K和V：

$$
\mathbf{O}_i = \mathrm{softmax}\left(\mathrm{mask}\left(\frac{\mathbf{Q}_i \times \mathbf{K}^\top_{\lceil i/g \rceil}}{\sqrt{d_k}}\right)\right) \mathbf{V}_{\lceil i/g \rceil}
$$

因果掩码把未来的位置换成负无穷：

$$
\mathrm{mask}(s_{ab}) =
\begin{cases}
s_{ab}, & b \le a \\
-\infty, & b > a
\end{cases}
$$

Softmax按行做。把一行展开，第 $a$ 个词元的输出是：

$$
\mathbf{o}_a = \sum_{b=1}^{a}
\frac{\exp(\mathbf{q}_a \cdot \mathbf{k}_b / \sqrt{d_k})}{\sum_{c=1}^{a} \exp(\mathbf{q}_a \cdot \mathbf{k}_c / \sqrt{d_k})}
\mathbf{v}_b
$$

形状：分数表 $(9, T, T)$ ，输出 $(9, T, 64)$

### 合并与输出投影

$$
\mathbf{O} = [\mathbf{O}_1, \ldots, \mathbf{O}_h], \quad \mathbf{Y} = \mathbf{O} \times \mathbf{W}_O
$$

$\mathbf{W}_O$ 是 $[576 \times 576]$ 。

形状： $(9, T, 64) \to (T, 576) \to (T, 576)$



## 四、前馈网络：SwiGLU（第八章）

$$
\mathrm{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}
$$

$$
\mathrm{FFN}(\mathbf{x}) = \big(\mathrm{SiLU}(\mathbf{x} \times \mathbf{W}_\mathrm{gate}) \odot (\mathbf{x} \times \mathbf{W}_\mathrm{up})\big) \times \mathbf{W}_\mathrm{down}
$$

$\mathbf{W}_\mathrm{gate}$ 和 $\mathbf{W}_\mathrm{up}$ 都是 $[576 \times 1536]$ ， $\mathbf{W}_\mathrm{down}$ 是 $[1536 \times 576]$ 。SiLU只套在门那一路上。 $d_\mathrm{ff} = \frac{8}{3} d = 1536$ ，让三个矩阵的参数和2017年两个4倍宽的矩阵一样多。

形状： $(T, 576) \to (T, 1536) \to (T, 576)$



## 五、Decoder块：残差连接（第五章、第九章）

pre-norm，每个子层的结果加回到它自己的输入上：

$$
\begin{aligned}
\mathbf{H}' &= \mathbf{H} + \mathrm{Attn}(\mathrm{RMSNorm}(\mathbf{H})) \\
\mathbf{H}_\mathrm{out} &= \mathbf{H}' + \mathrm{FFN}(\mathrm{RMSNorm}(\mathbf{H}'))
\end{aligned}
$$

两个RMSNorm各有各的 $\boldsymbol{\gamma}$ 。重复 $n_\mathrm{layer} = 30$ 次。把递推展开，残差流上的每一处都是词嵌入加上前面各个子层的贡献：

$$
\mathbf{h}_{l+1} = \mathbf{h}_1 + f_1(\mathbf{h}_1) + f_2(\mathbf{h}_2) + \cdots + f_l(\mathbf{h}_l)
$$

形状始终是 $(T, 576)$



## 六、出口：线性投影（第四章、第五章）

30层走完，先过出口那个RMSNorm，再取最后一个位置，乘上词嵌入矩阵的转置（权重共享）：

$$
\mathbf{z} = \mathrm{RMSNorm}(\mathbf{h}_T) \times \mathbf{E}^\top
$$

第 $j$ 个分数，就是这个向量和第 $j$ 个词元的词嵌入的点积。

形状： $(576,) \to (49152,)$



## 七、采样（第十章）

### Softmax

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}} = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{V} e^{z_j - \max(\mathbf{z})}}
$$

后一种写法是安全Softmax，结果和前一种完全一样，只是不会溢出。

### 三个过滤器

按温度、top-k、top-p的顺序，分数进，分数出：

$$
z_i \leftarrow z_i / T_\mathrm{temp}
$$

$$
z_i \leftarrow
\begin{cases}
z_i, & z_i \ge z_{(k)} \\
-\infty, & z_i < z_{(k)}
\end{cases}
$$

$$
z_i \leftarrow
\begin{cases}
z_i, & i \notin S \\
-\infty, & i \in S
\end{cases}
$$

$T_\mathrm{temp}$ 是温度（为了和序列长度 $T$ 区分，这里加了下标）； $z_{(k)}$ 是第 $k$ 大的分数； $S$ 是把概率从小到大排好之后，从最小的开始累加、累加和不超过 $1 - p$ 的那些词元，概率最大的那个永远不在 $S$ 里。

### 抽取

$$
\mathrm{id} =
\begin{cases}
\arg\max_i z_i, & \text{贪婪采样，} T_\mathrm{temp} = 0 \\
\text{按 } \mathrm{softmax}(\mathbf{z}) \text{ 随机抽一个}, & \text{随机采样}
\end{cases}
$$

抽到EOS（SmolLM2里是0号词元）就停。



## 八、KV缓存（第十一章）

缓存里已经存了 $L$ 个位置时，新来的 $T$ 个词元在第 $L$ 到第 $L + T - 1$ 个位置，RoPE从第 $L$ 个位置起算；它们的K和V（RoPE之后的）追加到缓存末尾，分数表变成 $(9, T, L + T)$ 。decode时 $T = 1$ ，不需要掩码：

$$
\mathbf{o} = \mathrm{softmax}\left(\frac{\mathbf{q} \times \mathbf{K}_\mathrm{cache}^\top}{\sqrt{d_k}}\right) \mathbf{V}_\mathrm{cache}
$$

缓存占的内存：

$$
\text{缓存字节数} = 2 \times n_\mathrm{kv} \times d_k \times n_\mathrm{layer} \times \text{每个数的字节数} \times \text{词元数}
$$

SmolLM2按FP32算，每个词元46,080字节，8192个位置是377,487,360字节。



## 九、在线Softmax（第十二章）

把一行分数 $\mathbf{x}$ 和对应的 $\mathbf{v}$ 切成两块，每块各自记下最大值 $m_k$ 、分母 $\ell_k$ 和分子 $y_k$ ：

$$
m_k = \max_{i \in \text{第}k\text{块}} x_i, \quad
\ell_k = \sum_{i \in \text{第}k\text{块}} e^{x_i - m_k}, \quad
y_k = \sum_{i \in \text{第}k\text{块}} e^{x_i - m_k} v_i
$$

合并时换到同一个基准 $m = \max(m_1, m_2)$ ：

$$
a = \frac{y_1 e^{m_1 - m} + y_2 e^{m_2 - m}}{\ell_1 e^{m_1 - m} + \ell_2 e^{m_2 - m}}
$$

和整行一次算完严格相等，不是近似。切成多少块都一样，每来一块就合并一次。



## 十、参数量（第二章、第六章、第八章）

| 部分 | 公式 | SmolLM2 |
| --- | --- | --: |
| 词嵌入 | $V \times d$ | 28,311,552 |
| 每块的注意力 | $d \times (d + 2 n_\mathrm{kv} d_k + d)$ | 884,736 |
| 每块的FFN | $3 \times d \times d_\mathrm{ff}$ | 2,654,208 |
| 每块的两个RMSNorm | $2 \times d$ | 1,152 |
| 出口的RMSNorm | $d$ | 576 |
| 合计 | 词嵌入 + $n_\mathrm{layer}$ × 每块 + 出口 | 134,515,008 |

出口的线性投影和词嵌入共用一个矩阵，所以不另算。权重文件用BF16存，每个参数2字节；读进内存升成FP32，每个参数4字节，大约538MB。
