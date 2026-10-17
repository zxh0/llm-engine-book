"""第十二章 · 用纯PyTorch写一个FlashAttention

第十二章那段伪代码，照着写成代码：外层循环每次拿一块Q，内层循环把K、V一块一块
过一遍，用在线Softmax维护每一行的最大值m、分母l和分子O。整块都在“未来”的，
直接跳过；正好压在对角线上的，块里右上角换成-inf。那张T×T的分数表从头到尾不出现。

这个脚本不改引擎：`model.py`和第十一章逐字节相同。它做四件事：

    一  对拍       随机的q、k、v，分块版和一次算完的版本比
    二  接进模型   把30层的注意力都换成分块版，逐层比，再贪婪生成20个
    三  内存       运行时最大的临时张量有多大，和T×T那张表比
    四  耗时       在CPU上，块大小合适时它反而更快
    五  块大小     块太小，Python循环的开销压垮一切；块太大，又回到了一次算完

运行：
    cd code
    uv run python book/ch12/flash_attn.py
"""

import time

import torch

from engine import Engine
from kv_cache import NO_CACHE

BLOCK = 128  # 每块多少行


def flash_attention(q, k, v, block=BLOCK, stats=None):
    """带因果掩码的分块注意力。q、k、v各(头数, T, 64) → (头数, T, 64)。

    stats是个dict，给了就往里记两样东西：最大的临时张量有多少字节，
    以及一共算了多少块、跳过了多少块。
    """
    H, T, d = q.shape
    out = torch.empty_like(q)
    scale = d ** -0.5
    for qs in range(0, T, block):                      # 外层：一块Q
        qe = min(qs + block, T)
        qi = q[:, qs:qe]                               # (H, Bq, d)
        m = torch.full((H, qe - qs, 1), float("-inf"))  # 这一块Q每一行的最大值
        l = torch.zeros(H, qe - qs, 1)                 # 以m为基准的分母
        o = torch.zeros(H, qe - qs, d)                 # 以m为基准的分子
        for ks in range(0, T, block):                  # 内层：一块K、V
            if ks >= qe:                               # 整块都在未来，跳过
                if stats is not None:
                    stats["skipped"] = stats.get("skipped", 0) + 1
                continue
            ke = min(ks + block, T)
            s = qi @ k[:, ks:ke].transpose(-1, -2) * scale  # (H, Bq, Bk)，只有一小块
            if ke > qs:                                # 压在对角线上：块里也有未来
                rows = torch.arange(qs, qe)[:, None]
                cols = torch.arange(ks, ke)[None, :]
                s = s.masked_fill(cols > rows, float("-inf"))
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            p = torch.exp(s - m_new)                   # (H, Bq, Bk)
            fix = torch.exp(m - m_new)                 # 旧的基准换成新的
            l = l * fix + p.sum(-1, keepdim=True)
            o = o * fix + p @ v[:, ks:ke]
            m = m_new
            if stats is not None:
                stats["computed"] = stats.get("computed", 0) + 1
                big = max(t.numel() * t.element_size() for t in (s, p, o))
                stats["peak"] = max(stats.get("peak", 0), big)
        out[:, qs:qe] = o / l                          # 最后除一次分母
    return out


def reference(q, k, v):
    """第六章那一套：一次算出整张T×T的表。"""
    T = q.shape[1]
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    future = torch.ones(T, T, dtype=torch.bool).triu(1)
    return torch.softmax(scores.masked_fill(future, float("-inf")), -1) @ v


def attn_flash(model, x, layer, i, cache=NO_CACHE):
    """`Model.attn`前后都照抄，只把中间“分数表、掩码、softmax、加权求和”换成分块版。"""
    T = x.shape[0]
    cfg = model.config
    h, h_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    d = cfg.hidden_size // h
    q = (x @ layer["q_proj"].T).view(T, h, d).transpose(0, 1)
    k = (x @ layer["k_proj"].T).view(T, h_kv, d).transpose(0, 1)
    v = (x @ layer["v_proj"].T).view(T, h_kv, d).transpose(0, 1)
    q, k = model.rope(q), model.rope(k)
    k = k.repeat_interleave(h // h_kv, dim=0)
    v = v.repeat_interleave(h // h_kv, dim=0)
    out = flash_attention(q, k, v)
    return out.transpose(0, 1).reshape(T, h * d) @ layer["o_proj"].T


def hidden_states(model, ids):
    x = model.to_embeddings(ids)
    states = [x]
    for i, layer in enumerate(model.weights.layers):
        x = model.block(x, layer, i)
        states.append(x)
    return states


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def main():
    g = torch.Generator().manual_seed(0)

    # ---- 一：对拍 -------------------------------------------------------------
    print(f"一  随机的q、k、v，9个头，每头64维，块大小{BLOCK}")
    print(f"    {'T':>6}{'相对误差':>12}")
    for T in (100, 128, 1000, 2048):
        q, k, v = (torch.randn(9, T, 64, generator=g) for _ in range(3))
        print(f"    {T:>6}{rel(flash_attention(q, k, v), reference(q, k, v)):>12.1e}")

    # ---- 二：接进模型 ---------------------------------------------------------
    engine = Engine()
    model = engine.model
    ids = engine.tok.encode("Once upon a time", add_special_tokens=False).ids
    with torch.no_grad():
        ref_states = hidden_states(model, ids)
        ref_out = engine.generate(ids, cache=False)
        model.attn = lambda x, layer, i, cache=NO_CACHE: attn_flash(model, x, layer, i, cache)
        states = hidden_states(model, ids)
        out = engine.generate(ids, cache=False)        # 不带缓存：每圈整串重算
    worst = max(rel(s, r) for s, r in zip(states, ref_states))
    print(f"\n二  30层的注意力都换成分块版，逐层最大相对误差 {worst:.1e}")
    print(f"    贪婪生成  {engine.tok.decode(out)!r}")
    print(f"    和原来的{'一字不差' if out == ref_out else '不同'}")

    # ---- 三：内存 -------------------------------------------------------------
    T = 8192
    q, k, v = (torch.randn(9, T, 64, generator=g) for _ in range(3))
    stats = {}
    start = time.perf_counter()
    flash_attention(q, k, v, stats=stats)
    seconds = time.perf_counter() - start
    table = 9 * T * T * 4
    total = stats["computed"] + stats["skipped"]
    print(f"\n三  T={T}，一层9个头")
    print(f"    原来那张分数表      {table:>15,}字节")
    print(f"    分块版最大的临时张量 {stats['peak']:>15,}字节")
    print(f"    {total}块里算了{stats['computed']}块，跳过{stats['skipped']}块（整块都在未来）")
    print(f"    用时 {seconds:.2f}秒")

    # ---- 四：耗时 -------------------------------------------------------------
    print(f"\n四  耗时，单位毫秒")
    print(f"    {'T':>6}{'一次算完':>10}{'分块':>10}{'是几倍':>10}")
    for T in (512, 1024, 2048, 4096):
        q, k, v = (torch.randn(9, T, 64, generator=g) for _ in range(3))
        times = []
        for fn in (reference, flash_attention):
            fn(q, k, v)                                # 热身
            start = time.perf_counter()
            fn(q, k, v)
            times.append(time.perf_counter() - start)
        print(f"    {T:>6}{times[0] * 1000:>10.1f}{times[1] * 1000:>10.1f}"
              f"{times[1] / times[0]:>10.1f}")

    # ---- 五：块大小 -----------------------------------------------------------
    T = 2048
    q, k, v = (torch.randn(9, T, 64, generator=g) for _ in range(3))
    reference(q, k, v)
    start = time.perf_counter()
    reference(q, k, v)
    base = time.perf_counter() - start
    print(f"\n五  T={T}，换不同的块大小，一次算完是{base * 1000:.1f}毫秒")
    print(f"    {'块大小':>6}{'一块分数表':>12}{'耗时':>10}{'是一次算完的几倍':>16}")
    for b in (8, 16, 32, 64, 128, 256, 512, 2048):
        flash_attention(q, k, v, block=b)
        start = time.perf_counter()
        flash_attention(q, k, v, block=b)
        t = time.perf_counter() - start
        print(f"    {b:>6}{9 * b * b * 4 / 1e3:>10.0f}KB{t * 1000:>8.1f}ms{t / base:>16.2f}")


if __name__ == "__main__":
    main()
