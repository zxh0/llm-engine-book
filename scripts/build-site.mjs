// 从 chapters/、images/、aigc/ 和 tools/ 生成 VitePress 站点内容。
// chapters/*.md 和 tools/*/ 是唯一的内容来源，
// docs/chapters/、docs/public/、sidebar.json 都是生成物。
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const chaptersDir = path.join(root, 'chapters')
const toolsDir = path.join(root, 'tools')
const docsDir = path.join(root, 'docs')
const chaptersOutDir = path.join(docsDir, 'chapters')
const publicDir = path.join(docsDir, 'public')

// 这本书「写完一章发一章」：往 published 里加一行文件名，那一章就上站，
// 没列进来的章节只在目录页里露个标题，正文继续待在仓库里。
// 分章文件名是 chNN_slug.md，章标题本来就是 #；
// _front.md / _preface.md / todo.md 不匹配 CHAPTER_RE，自然被挡在外面。
const CHAPTER_RE = /^(ch\d+_.+)\.md$/
const COVER = { from: path.join('aigc', 'front.png'), name: 'cover.png' }
const PUBLISHED = new Set([
  'ch01_overview.md',
  'ch02_weights.md',
  'ch03_tokenizer.md',
  'ch04_embedding.md',
  'ch05_norm.md',
  'ch06_attn.md',
  'ch07_rope.md',
])

// 工具：tools/<目录>/ 下的纯静态单页，整个目录原样拷到 docs/public/tools/<目录>/，
// 站点地址就是 /tools/<目录>/。加一个工具 = 放一个目录进 tools/，再往这里加一行。
// README.md 是写给仓库读者的，不上站。
const TOOLS = [
  {
    dir: 'safetensors-viewer',
    title: 'safetensors 参数布局查看器',
    blurb:
      '给它一个 HF 模型仓库名或者 model.safetensors 的地址，它把模型里每个张量在文件中的字节排布画出来：' +
      '内存布局、结构树、逐层对比、张量表、原始 JSON 头五个视图，前三个能导出 SVG 和 PNG。',
    note: '只下载文件头，通常几十 KB。纯前端，模型数据不经过任何服务器。配合第二章看。',
  },
  {
    dir: 'sinpe-heatmap',
    title: '正弦位置编码热力图',
    blurb:
      '把《Attention Is All You Need》3.5 节的位置编码表画成一张热力图：行是位置，列是维度，颜色是编码的值。' +
      '鼠标放到任意一格上，显示这一格的位置、组号、用的是 sin 还是 cos，以及完整的计算过程。',
    note: '配合第七章「正弦位置编码」一节看。',
  },
  {
    dir: 'rope-angles',
    title: 'RoPE 旋转角度',
    blurb:
      '把 RoPE 每个位置、每个平面转了多少画出来，再用两个可以拖动的向量演示：' +
      '旋转之后的点积，只和两个位置的差有关。默认是 SmolLM2 的参数。',
    note: '配合第七章「旋转位置编码」一节看。',
  },
]

/** 把 <img> 标签转成标准 Markdown 图片语法，并把仓库里的相对路径改成站点绝对路径。
 *  rebases 是一串 [前缀, 换成什么]，第一个匹配上的生效：
 *  插图 ../images/ -> /images/ 和 AI 生成的章首图 ../aigc/ -> /aigc/。
 *  转成 Markdown 语法是为了让 VitePress 自动加上 base 前缀（裸 HTML 它不管）。
 *  Typora 写的 style="zoom:50%" 直接丢掉，图片宽度由正文栏宽兜住。 */
function normalizeImages(md, rebases) {
  const rebase = (url) => {
    for (const [from, to] of rebases) {
      if (url.startsWith(from)) return to + url.slice(from.length)
    }
    return url
  }
  return md
    .replace(/<img\s+([^>]*?)\/?>/g, (whole, attrs) => {
      const src = attrs.match(/src="([^"]+)"/)
      if (!src) return whole
      const alt = attrs.match(/alt="([^"]*)"/)
      return `![${alt ? alt[1] : ''}](${rebase(src[1])})`
    })
    .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (whole, alt, url) => `![${alt}](${rebase(url)})`)
}

function cleanup(md, rebases) {
  return normalizeImages(md, rebases)
    .replace(/<div style="page-break-after: always;"><\/div>/g, '') // 只对 PDF 有意义
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

/** 收集页面内的 ## 小节，用于侧边栏展开。 */
function subsectionsOf(md) {
  const items = []
  for (const line of md.split('\n')) {
    const h2 = line.match(/^## (.+?)\s*$/)
    if (h2) items.push(h2[1])
  }
  return items
}

function firstHeadingOf(md) {
  return md.match(/^#{1,2} (.+?)\s*$/m)?.[1]
}

// 复刻 VitePress 默认的标题锚点规则（来自 @mdit-vue/shared 的 slugify，
// 它被打包进 vitepress 里了，没法直接 import，所以照抄一份）。
// 注意 NFKD 会把全角字符归一成半角，所以「分词（Tokenization）」
// 里的（）会跟半角括号一样变成连字符，不能想当然地当成普通中文字符留着。
const rControl = /[\u0000-\u001f]/g
const rSpecial = /[\s~`!@#$%^&*()\-_+=[\]{}|\;:"'“”‘’<>,.?/]+/g
const rCombining = /[̀-ͯ]/g

function anchor(text) {
  return text
    .normalize('NFKD')
    .replace(rCombining, '')
    .replace(rControl, '')
    .replace(rSpecial, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/^(\d)/, '_$1')
    .toLowerCase()
}

function copyDir(src, dest) {
  fs.rmSync(dest, { recursive: true, force: true })
  fs.cpSync(src, dest, { recursive: true, filter: (f) => !f.endsWith('.DS_Store') })
}

// ---- 章节 ----

// 按章上站：PUBLISHED 里的章节生成页面，其余的只在目录页里列个标题。
// 页面文件保留 chapters/ 的原始文件名，再用 VitePress 的 rewrites 映射到干净的
// 地址（/ch01-overview），这样 editLink 拿到的 filePath 还能直接对回 chapters/ 下的源文件。
fs.rmSync(chaptersOutDir, { recursive: true, force: true })
fs.mkdirSync(chaptersOutDir, { recursive: true })

const rewrites = {}
const chapters = []

for (const name of fs.readdirSync(chaptersDir).sort()) {
  if (!CHAPTER_RE.test(name)) continue

  const raw = fs.readFileSync(path.join(chaptersDir, name), 'utf8')
  const title = firstHeadingOf(raw)
  if (!title) continue

  if (!PUBLISHED.has(name)) {
    chapters.push({ title })
    continue
  }

  const slug = name.replace(CHAPTER_RE, '$1').replace(/_/g, '-')
  const body = cleanup(raw, [
    ['../images/', '/images/'],
    ['../aigc/', '/aigc/'],
  ])
  fs.writeFileSync(path.join(chaptersOutDir, name), body + '\n')
  rewrites[`chapters/${name}`] = `${slug}.md`

  const link = `/${slug}`
  const subs = subsectionsOf(body)
  chapters.push({
    title,
    link,
    collapsed: subs.length > 0 ? true : undefined,
    items: subs.map((t) => ({ text: t, link: `${link}#${anchor(t)}` })),
  })
}

if (chapters.length === 0) throw new Error(`${chaptersDir} 下没有找到分章文件`)

const book = {
  // 目录页要列全部章节，侧边栏只放已经上站的
  toc: chapters.map(({ title, link }) => ({ text: title, link })),
  sidebar: chapters
    .filter((c) => c.link)
    .map(({ title, link, collapsed, items }) => ({ text: title, link, collapsed, items })),
}

// ---- 输出 ----

fs.mkdirSync(path.join(docsDir, '.vitepress'), { recursive: true })
fs.writeFileSync(
  path.join(docsDir, '.vitepress', 'sidebar.json'),
  JSON.stringify(
    {
      book,
      // 首页和 /tools 页要用
      tools: TOOLS.map(({ dir, title, blurb, note }) => ({
        title,
        blurb,
        note,
        link: `/tools/${dir}/`,
      })),
      rewrites,
    },
    null,
    2
  ) + '\n'
)

fs.mkdirSync(publicDir, { recursive: true })
copyDir(path.join(root, 'images'), path.join(publicDir, 'images'))
copyDir(path.join(root, 'aigc'), path.join(publicDir, 'aigc'))
fs.copyFileSync(path.join(root, COVER.from), path.join(publicDir, COVER.name))

const toolsOutDir = path.join(publicDir, 'tools')
fs.rmSync(toolsOutDir, { recursive: true, force: true })
for (const tool of TOOLS) {
  const src = path.join(toolsDir, tool.dir)
  if (!fs.existsSync(path.join(src, 'index.html')))
    throw new Error(`tools/${tool.dir}/ 下没有 index.html`)
  fs.cpSync(src, path.join(toolsOutDir, tool.dir), {
    recursive: true,
    filter: (f) => !f.endsWith('.DS_Store') && !f.endsWith('README.md'),
  })
}

console.log(
  `生成 ${book.sidebar.length}/${book.toc.length} 章正文 -> docs/chapters/，` +
    `${TOOLS.length} 个工具 -> docs/public/tools/`
)
