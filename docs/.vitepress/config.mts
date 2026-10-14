import { defineConfig } from 'vitepress'
import { book, rewrites } from './sidebar.json'

// sidebar.json 由 scripts/build-site.mjs 从 chapters/ 和 tools/ 生成
const firstChapter = book.sidebar[0].link

export default defineConfig({
  title: '自己动手写LLM推理引擎',
  description:
    '一本还在写的书：用 Python + PyTorch 从零手写一个 LLM 推理引擎，目标模型 SmolLM2-135M，跑在笔记本 CPU 上。',
  lang: 'zh-CN',
  base: '/llm-engine-book/',
  cleanUrls: true,
  lastUpdated: true,
  // docs/chapters/ 下的页面文件名沿用 chapters/ 的原名，这里映射成干净的地址
  rewrites,
  head: [
    ['meta', { property: 'og:title', content: '自己动手写LLM推理引擎' }],
    ['meta', { property: 'og:image', content: 'https://zxh0.github.io/llm-engine-book/cover.png' }],
  ],

  markdown: {
    lineNumbers: false,
    image: { lazyLoading: true },
    // 正文里有 $...$ 和 $$...$$ 公式，需要 markdown-it-mathjax3
    math: true,
  },

  themeConfig: {
    outline: { level: [2, 3], label: '本页目录' },

    nav: [
      { text: '目录', link: '/' },
      { text: '开始阅读', link: firstChapter },
      { text: '工具', link: '/tools', activeMatch: '^/tools' },
      { text: '提问纠错', link: 'https://github.com/zxh0/llm-engine-book/issues' },
    ],

    // 章节页和首页共用一个侧边栏；/tools 页没有侧边栏
    sidebar: {
      '/tools': [],
      '/': [
        {
          text: '自己动手写LLM推理引擎',
          items: [{ text: '进度和目录', link: '/' }, ...book.sidebar],
        },
      ],
    },

    socialLinks: [{ icon: 'github', link: 'https://github.com/zxh0/llm-engine-book' }],

    editLink: {
      // 章节页面文件名保留了 chapters/ 的原名，所以站点里的 filePath 就是仓库里的源文件。
      // 注意：这个函数和下面的分词器一样，会被序列化后送到浏览器执行，必须自包含。
      pattern: ({ filePath }) => {
        if (filePath.startsWith('chapters/'))
          return `https://github.com/zxh0/llm-engine-book/edit/main/${filePath}`
        return 'https://github.com/zxh0/llm-engine-book/issues'
      },
      text: '在 GitHub 上纠错',
    },

    search: {
      provider: 'local',
      options: {
        // 默认分词器按空白切词，对中文几乎无效。这里把中文按字拆开建索引，
        // 配合 AND 组合，搜「注意力」= 同时包含这三个字的页面。
        // 注意：这个函数会被序列化后送到浏览器执行，必须自包含。
        miniSearch: {
          options: {
            tokenize: (text: string) =>
              text
                .split(/[^\p{L}\p{N}_]+/u)
                .flatMap((w) => (/[一-鿿]/.test(w) ? w.split('') : [w]))
                .filter(Boolean),
            processTerm: (term: string) => term.toLowerCase(),
          },
          searchOptions: {
            combineWith: 'AND',
            prefix: true,
            boost: { title: 4, text: 2, titles: 1 },
          },
        },
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            displayDetails: '显示详情',
            resetButtonTitle: '清除',
            noResultsText: '没有找到',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },

    docFooter: { prev: '上一章', next: '下一章' },
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    sidebarMenuLabel: '目录',
    returnToTopLabel: '回到顶部',
    lastUpdated: { text: '最后更新' },
    outlineTitle: '本页目录',

    footer: {
      message: '文字和插图均为原创，转载请注明出处',
      copyright: 'Copyright © 2026 zxh0',
    },
  },
})
