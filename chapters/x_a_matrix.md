# 附录A：矩阵相关公式

这本书用到的数学，基本上就是向量、矩阵，以及它们之间的几种运算。这些内容散落在各章里，用到的时候才讲。这个附录把它们收在一起，方便查阅。每一条后面都注明了它第一次出现在哪一章。



## 记号

本书沿用Goodfellow《深度学习》的那一套记号：

| 写法 | 含义 | 例子 |
| --- | --- | --- |
| 粗体小写 | 向量 | $\mathbf{x}$ 、 $\mathbf{q}_m$ 、 $\boldsymbol{\gamma}$ |
| 粗体大写 | 矩阵 | $\mathbf{Q}$ 、 $\mathbf{W}_Q$ 、 $\mathbf{R}_m$ |
| 不加粗 | 标量，或者一个数量 | $d$ 、 $T$ 、 $x_i$ 、 $m\theta_i$ |

希腊字母表示的向量，要用 $\boldsymbol{\gamma}$ 这种写法才能加粗。

矩阵的形状写作 $[M \times N]$ ，也就是 $M$ 行 $N$ 列；代码里对应的写法是`(M, N)`。向量的下标表示它的第几个分量，比如 $x_i$ 是 $\mathbf{x}$ 的第 $i$ 个数。公式里的下标从1开始，代码里从0开始（第六章）。

本书大部分地方把一个词元的向量看成**行向量**，矩阵从右边乘上去，写成 $\mathbf{x} \times \mathbf{W}$ ；把T个词元的向量一行一个摞起来，就是一个 $[T \times d]$ 的矩阵（第六章）。只有第七章讲RoPE时，为了和论文一致，把单个向量写成了列向量，矩阵从左边乘上去。



## 向量

### 模长

$$
\|\mathbf{a}\| = \sqrt{\sum_{i=1}^{d} a_i^2}
$$

向量的长度。第四章用它算余弦相似度。

### 均方根

$$
\mathrm{RMS}(\mathbf{a}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} a_i^2} = \frac{\|\mathbf{a}\|}{\sqrt{d}}
$$

和模长只差一个 $\sqrt{d}$ 。所以RMSNorm把均方根拉到1之后，576维向量的模长是 $\sqrt{576}=24$ ，而不是1（第五章）。

### 点积

$$
\mathbf{a}\cdot\mathbf{b} = \sum_{i=1}^{d} a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta
$$

两个维度相同的向量对应位置相乘再加起来，结果是一个数。后一个等号说明，点积同时受模长和夹角影响，模长大的向量“占便宜”（第四章）。注意力分数（第六章）和出口的分数（第四章），算的都是点积。

### 余弦相似度

$$
\mathrm{cos\_sim}(\mathbf{a},\mathbf{b}) = \cos\theta = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
$$

只看夹角，不看模长。两个向量的模长都是1时，它就等于点积（第四章）。

### 哈达玛积

$$
\mathbf{a} \odot \mathbf{b} = [a_1 b_1, a_2 b_2, \ldots, a_d b_d]
$$

逐元素相乘，结果还是一个向量，别和点积搞混（第五章）。RMSNorm的 $\boldsymbol{\gamma}$ 、SwiGLU的门、RoPE的快速算法，用的都是它。PyTorch里对张量直接写`*`就是哈达玛积。

### 向量和标量

向量加减一个标量，意思是每个分量都加减它，比如LayerNorm里的 $\mathbf{x} - \mu$ （第五章）。PyTorch把这种做法推广成了**广播**：形状对不上的那一维，只要有一边是1，就自动复制到另一边的大小，比如第五章`(T, 576)`除以`(T, 1)`，第七章`(T, 64)`的cos表乘到`(9, T, 64)`的每个头上。



## 矩阵

### 矩阵乘法

$$
\mathbf{C} = \mathbf{A} \times \mathbf{B}, \quad c_{ij} = \sum_{k=1}^{K} a_{ik} b_{kj}
$$

$[M \times K]$ 乘 $[K \times N]$ ，得到 $[M \times N]$ ：左边的列数必须等于右边的行数。 $\mathbf{C}$ 第 $i$ 行第 $j$ 列的数，是 $\mathbf{A}$ 的第 $i$ 行和 $\mathbf{B}$ 的第 $j$ 列做点积（第四章）。

两个推论在书里反复用到：

- **输出的第 $i$ 行只取决于输入的第 $i$ 行。** 所以一个 $[T \times 576]$ 的矩阵乘上权重矩阵，就是T个词元各算各的，这就是“逐词元”（第八章）。
- **线性投影可以看成挨个做点积。** 出口的 $\mathbf{l} = \mathbf{x} \times \mathbf{W}$ ，第 $j$ 个分数就是 $\mathbf{x}$ 和 $\mathbf{W}$ 第 $j$ 列的点积（第四章）。

PyTorch的`@`对三维张量也能用：只拿最后两维做矩阵乘法，前面的维度当成“有几组”，每组各算各的（第六章）。

### 结合律

$$
(\mathbf{A} \times \mathbf{B}) \times \mathbf{C} = \mathbf{A} \times (\mathbf{B} \times \mathbf{C})
$$

所以两个矩阵中间没有激活函数，就等于一个矩阵（第八章）；MLA也是靠它，把还原矩阵提前合并进别的权重（第十一章）。

矩阵乘法**不满足交换律**， $\mathbf{A} \times \mathbf{B}$ 一般不等于 $\mathbf{B} \times \mathbf{A}$ 。

### 转置

$$
(\mathbf{A}^\top)_{ij} = a_{ji}, \qquad (\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top \mathbf{A}^\top
$$

行列对调。权重文件里的矩阵按“输出×输入”存放，公式里要的是“输入×输出”，所以代码里写`x @ W.T`（第六章）。词嵌入矩阵 $[49152 \times 576]$ 转置之后，正好就是出口要的 $[576 \times 49152]$ ，这是权重共享的基础（第四章）。后一个公式用在RoPE的推导里（第七章）。

### 外积

$$
(\mathbf{u}\mathbf{v}^\top)_{ij} = u_i v_j
$$

两个向量两两相乘，得到一张表。第七章用`torch.outer()`把8192个位置和32个转速两两相乘，得到RoPE的角度表。

### 拆分与拼接

把一个 $[T \times 576]$ 的矩阵沿着列切成9段，每段 $[T \times 64]$ ，就是注意力的拆头；反过来左右并排拼回去，就是合并（第六章）。代码里对应`view()`、`transpose()`和`reshape()`。



## 旋转矩阵

二维平面上，把一个向量逆时针转 $\theta$ 的矩阵是：

$$
\mathbf{R}(\theta) =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

它有三个性质，第七章都用到了：

$$
\begin{aligned}
\mathbf{R}(\theta)^\top &= \mathbf{R}(-\theta) \\
\mathbf{R}(\alpha)\,\mathbf{R}(\beta) &= \mathbf{R}(\alpha + \beta) \\
\|\mathbf{R}(\theta)\,\mathbf{a}\| &= \|\mathbf{a}\|
\end{aligned}
$$

转置等于反着转；先后转两次等于一次转两个角度之和；旋转不改变模长。前两条合起来，就是RoPE“只依赖相对位置”的来源。

把很多个 $2 \times 2$ 的旋转矩阵沿着对角线排起来，其余位置都是0，就得到一个分块对角矩阵，每一块各转各的平面，RoPE论文里的 $\mathbf{R}_m$ 就是这样（第七章）。这种矩阵绝大部分是0，实际计算时用两次哈达玛积代替矩阵乘法，见附录B。

推导正弦位置编码时还用到了中学的两角和公式（第七章）：

$$
\begin{aligned}
\sin(\alpha + \beta) &= \sin\alpha \cos\beta + \cos\alpha \sin\beta \\
\cos(\alpha + \beta) &= \cos\alpha \cos\beta - \sin\alpha \sin\beta
\end{aligned}
$$



## 双随机矩阵

每个数都不小于0，每一行加起来等于1，每一列加起来也等于1的方阵：

$$
\mathbf{M}\mathbf{1}_n = \mathbf{1}_n, \quad \mathbf{1}_n^\top \mathbf{M} = \mathbf{1}_n^\top, \quad \mathbf{M} \ge 0
$$

其中 $\mathbf{1}_n$ 是 $n$ 个1组成的列向量。两个双随机矩阵相乘，结果还是双随机矩阵，所以连乘多少次都不会变大或变小（第九章）。



## 指数

Softmax、温度和在线Softmax，靠的都是指数的这几条性质：

$$
e^{a+b} = e^a e^b, \qquad e^{-\infty} = 0, \qquad e^0 = 1
$$

第一条让安全Softmax和在线Softmax可以“换基准”（第十章、第十二章）；第二条让掩码、top-k、top-p都可以用负无穷来“砍掉”一个位置（第六章、第十章）。指数函数是单调递增的，所以Softmax不改变大小顺序，贪婪采样可以直接对分数取argmax（第十章）。
