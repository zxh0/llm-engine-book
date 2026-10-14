# 自己动手写LLM推理引擎

<script setup>
import { withBase } from 'vitepress'
import { book } from './.vitepress/sidebar.json'
// 全书十二章，分成两段：一到十章「跑通」，十一、十二章「跑快」
const toc = book.toc
const part1 = toc.slice(0, 10)
const part2 = toc.slice(10)
</script>

> Build a Toy LLM Inference Engine (from Scratch)

::: warning 施工中（WIP）
书名待定，正文还在很早期的草稿阶段。写得差不多的章节会陆续放到这个站点上，目录里能点开的就是已经发布的。还没发布的章节可以直接读仓库里的
[chapters/](https://github.com/zxh0/llm-engine-book/tree/main/chapters)，不过那些内容更早期，可能有很多逻辑混乱、甚至胡言乱语的地方。
:::

![自己动手写LLM推理引擎](/cover.png)

## 这本书想做什么

用 Python 和 PyTorch，从零手写一个能跑的 LLM 推理引擎：加载真实的模型权重，把一段文字变成词元（Token），算完三十层 Transformer，采样出下一个词，再把它变回文字，然后循环。

全程不使用 `transformers` 的建模代码。分词、权重加载、RMSNorm、RoPE、GQA、SwiGLU、KV Cache、采样、生成循环，全部自己写。目标模型是 **SmolLM2-135M**，跑在一台**普通笔记本的 CPU** 上。慢，但每一个数字都能和 HuggingFace 对上。

主张很简单：

> 从训练入手，会比较难；从推理入手，就比较简单，然后再去学训练。
>
> 从 GPU 入手，会比较难；从 CPU 入手，就比较简单，然后再去学 GPU。

「从零手写理解 LLM」这个赛道其实很挤，但已有的资料几乎全是**训练视角**：搭一个模型，然后训练它。这本书换一个视角：只讲推理，于是梯度、数据、漫长的等待全都不用管，只剩下模型结构这一件事。而且每一步都可验证，自己写的每个部件，当场就能和 `transformers` 对答案。

## 目录

十二章，分成两段。章节标题已经定下来了，正文还在填坑。已经发布的章节可以点开读：

**跑通**：沿数据流从左走到右，每章造一个部件。

<ul class="book-toc">
  <li v-for="t in part1" :key="t.text" :class="{ published: t.link }">
    <a v-if="t.link" :href="withBase(t.link)">{{ t.text }}</a>
    <template v-else>{{ t.text }}</template>
  </li>
</ul>

**跑快**：不改变数据流，改变执行方式。

<ul class="book-toc">
  <li v-for="t in part2" :key="t.text" :class="{ published: t.link }">
    <a v-if="t.link" :href="withBase(t.link)">{{ t.text }}</a>
    <template v-else>{{ t.text }}</template>
  </li>
</ul>

## 随书代码

随书代码在仓库的 [code/](https://github.com/zxh0/llm-engine-book/tree/main/code) 下，一章一个目录。另外 `code/poc/` 里是最早的概念验证，已经跑通全流程并有一批测试，作为正确性参照保留。

## 小工具

写书时顺手做的几个纯前端单页工具，打开就能用，数据不出浏览器。见[工具](/tools)页。

## 反馈贡献

我还在慢慢学习中，书里难免有错误和疏漏。如果你发现任何问题，或者有优化建议，非常欢迎[提交 Issue](https://github.com/zxh0/llm-engine-book/issues)，具体包括但不限于：

- 错别字、语病或表述模糊的内容
- 公式推导和概念解释里的错误、逻辑漏洞
- 认为应当补充的重要知识点
- 能提升阅读体验的排版、配图建议

你的每一份反馈，都是让这本书更完善、更易读的重要帮助。
