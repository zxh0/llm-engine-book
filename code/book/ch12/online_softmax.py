"""第十二章 · 在线Softmax：边走边修正

第十二章正文里那个手算的例子，在这里一步一步打印出来：

    x = [1, 3, 2, 5]      注意力分数的一行，已经除过√d
    v = [10, 20, 30, 40]  V的一列

一次看完整行，a = softmax(x) · v。切成两块，每块只看得见自己那一半，就维护三样东西：

    m   目前见过的最大值
    l   以m为基准的分母，sum(exp(x - m))
    y   以m为基准的分子，sum(exp(x - m) * v)

每来一块，先看最大值要不要更新；更新了，就给手里的l和y乘一个修正系数exp(m_旧 - m_新)。
最后a = y / l。后半段再拿1000个随机数、块大小7，看它和torch.softmax差多少。

运行：
    cd code
    uv run python book/ch12/online_softmax.py
"""

import math

import torch


def online(x, v, block, show=False):
    """分块算softmax(x) · v。x、v各是一串数，block是每块几个。"""
    m, l, y = float("-inf"), 0.0, 0.0
    for start in range(0, len(x), block):
        xs, vs = x[start:start + block], v[start:start + block]
        m_new = max(m, max(xs))
        fix = math.exp(m - m_new)                      # 旧的基准换成新的
        l = l * fix + sum(math.exp(t - m_new) for t in xs)
        y = y * fix + sum(math.exp(t - m_new) * u for t, u in zip(xs, vs))
        if show:
            print(f"  块{xs}   最大值 {m} → {m_new}   修正系数 {fix:.4f}"
                  f"   分母 {l:.4f}   分子 {y:.4f}")
        m = m_new
    return y / l


def main():
    x, v = [1.0, 3.0, 2.0, 5.0], [10.0, 20.0, 30.0, 40.0]
    w = torch.softmax(torch.tensor(x), dim=0)
    print("一次看完整行：")
    print(f"  权重 {[round(t, 4) for t in w.tolist()]}")
    print(f"  a = {float(w @ torch.tensor(v)):.4f}\n")

    print("切成两块，每块2个：")
    a = online(x, v, block=2, show=True)
    print(f"  a = 分子 / 分母 = {a:.4f}\n")

    # ---- 再拿一大串随机数试一下 ------------------------------------------------
    g = torch.Generator().manual_seed(0)
    xs = (torch.randn(1000, generator=g, dtype=torch.float64) * 5).tolist()
    vs = torch.randn(1000, generator=g, dtype=torch.float64).tolist()
    # Python的浮点数是64位的，参照也用64位算，比的才是算法本身，不是精度。
    ref = float(torch.softmax(torch.tensor(xs, dtype=torch.float64), 0)
                @ torch.tensor(vs, dtype=torch.float64))
    got = online(xs, vs, block=7)
    print(f"1000个随机数，块大小7：torch.softmax {ref:.12f}   在线 {got:.12f}"
          f"   相对误差 {abs(got - ref) / abs(ref):.1e}")


if __name__ == "__main__":
    main()
