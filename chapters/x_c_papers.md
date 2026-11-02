# 附录C：论文列表

正文里提到的论文，按第一次出现的先后排列，最后一列是它出现在哪一章。每一篇都附上arXiv编号，在arXiv网站上搜索编号就能找到原文。年份取的是论文第一次发布在arXiv上的时间，和正式发表的年份可能差一年。

<!-- TODO 第十二章的FlashAttention、在线Softmax、FlashAttention-2已在正文补上引用并收进下表（2026-11-01）。
     另外几篇正文提到但没有正式引用的，要不要补，由作者定：
     PagedAttention《Efficient Memory Management for Large Language Model Serving with PagedAttention》（arXiv:2309.06180），第十一章
     《Qwen3 Technical Report》（arXiv:2505.09388），第十三章 -->



| 年份 | 论文 | 说了什么 | 章节 |
| --- | --- | --- | --- |
| 2025 | 《SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model》（arXiv:2502.02737） | 本书用的模型 | 前言 |
| 2017 | 《Attention Is All You Need》（arXiv:1706.03762） | Transformer架构、缩放点积注意力、多头注意力、正弦位置编码 | 第一章 |
| 2015 | 《Neural Machine Translation of Rare Words with Subword Units》（arXiv:1508.07909） | 把BPE用于分词 | 第三章 |
| 2015 | 《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》（arXiv:1502.03167） | BatchNorm | 第五章 |
| 2016 | 《Layer Normalization》（arXiv:1607.06450） | LayerNorm | 第五章 |
| 2019 | 《Root Mean Square Layer Normalization》（arXiv:1910.07467） | RMSNorm | 第五章 |
| 2019 | 《Fast Transformer Decoding: One Write-Head is All You Need》（arXiv:1911.02150） | MQA | 第六章 |
| 2023 | 《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》（arXiv:2305.13245） | GQA | 第六章 |
| 2021 | 《RoFormer: Enhanced Transformer with Rotary Position Embedding》（arXiv:2104.09864） | RoPE | 第七章 |
| 2021 | 《Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth》（arXiv:2103.03404） | 只有注意力，各个位置会越来越像 | 第八章 |
| 2020 | 《Transformer Feed-Forward Layers Are Key-Value Memories》（arXiv:2012.14913） | 把FFN看成键值记忆 | 第八章 |
| 2016 | 《Gaussian Error Linear Units (GELUs)》（arXiv:1606.08415） | GELU | 第八章 |
| 2017 | 《Searching for Activation Functions》（arXiv:1710.05941） | Swish，也就是SiLU | 第八章 |
| 2016 | 《Language Modeling with Gated Convolutional Networks》（arXiv:1612.08083） | GLU | 第八章 |
| 2020 | 《GLU Variants Improve Transformer》（arXiv:2002.05202） | SwiGLU | 第八章 |
| 2023 | 《Efficient Streaming Language Models with Attention Sinks》（arXiv:2309.17453） | 注意力汇聚点；第十一章用它压缩KV缓存 | 第八章、第十一章 |
| 2024 | 《Mixtral of Experts》（arXiv:2401.04088） | MoE的路由公式 | 第八章 |
| 2015 | 《Deep Residual Learning for Image Recognition》（arXiv:1512.03385） | ResNet，残差连接 | 第九章 |
| 2016 | 《Residual Networks Behave Like Ensembles of Relatively Shallow Networks》（arXiv:1605.06431） | 残差网络像许多浅网络的集合 | 第九章 |
| 2024 | 《Hyper-Connections》（arXiv:2409.19606） | HC | 第九章 |
| 2025 | 《mHC: Manifold-Constrained Hyper-Connections》（arXiv:2512.24880） | mHC | 第九章 |
| 2026 | 《Attention Residuals》（arXiv:2603.15031） | AttnRes | 第九章 |
| 2019 | 《The Curious Case of Neural Text Degeneration》（arXiv:1904.09751） | top-p（核采样） | 第十章 |
| 2020 | 《Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention》（arXiv:2006.16236） | 线性注意力 | 第十一章 |
| 2024 | 《DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model》（arXiv:2405.04434） | MLA | 第十一章 |
| 2022 | 《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》（arXiv:2205.14135） | 分块加在线Softmax，不把 $T \times T$ 的大表写进显存 | 第十二章 |
| 2018 | 《Online normalizer calculation for softmax》（arXiv:1805.02867） | 在线Softmax | 第十二章 |
| 2023 | 《FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning》（arXiv:2307.08691） | 外层循环改成Q | 第十二章 |



## 其他资料

不是论文，或者不在arXiv上，但正文提到过的：

- Philip Gage，1994年：BPE最初是一个数据压缩算法（第三章）
- Rosenblatt，1958年：感知机（第八章）
- nostalgebraist，2020年的博客文章“interpreting GPT: the logit lens”（第九章）
- Karpathy的[nanoGPT](https://github.com/karpathy/nanoGPT)，以及Raschka的《Build a Large Language Model (From Scratch)》：从训练的角度手写LLM（第一章）
