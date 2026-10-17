"""第九章 · mHC里的双随机矩阵：Sinkhorn-Knopp算法，以及30层乘下来会怎样

SmolLM2不用HC，也不用mHC，所以这个脚本不碰模型，只用随机数演示两件事：

1. Sinkhorn-Knopp：先取exp让每个数都变成正数，然后交替地“每行除以行和”、
   “每列除以列和”。来回做几次，行和、列和就都变成了1，这就是双随机矩阵。
2. 主干每过一层，都要乘上一个n×n的混合矩阵B。30层乘下来：
   不加约束的B，乘积要么越滚越大，要么越缩越小；双随机的B，乘积还是双随机的。

n取4，和DeepSeek-V4的技术报告里一样。随机数的种子写死在SEED里，可以原样复现。

运行：
    cd code
    uv run python book/ch09/sinkhorn.py
"""

import torch

SEED = 0
N = 4        # 主干扩成几条
LAYERS = 30  # 乘多少层


def sinkhorn(b, steps):
    """把任意n×n矩阵变成（近似）双随机矩阵：每个数非负，行和、列和都是1。"""
    m = b.exp() # 先变成正数
    for _ in range(steps):
        m = m / m.sum(dim=1, keepdim=True) # 每行除以行和
        m = m / m.sum(dim=0, keepdim=True) # 每列除以列和
    return m


def sums(m):
    rows = " ".join(f"{float(v):.4f}" for v in m.sum(dim=1))
    cols = " ".join(f"{float(v):.4f}" for v in m.sum(dim=0))
    return f"行和 [{rows}]   列和 [{cols}]"


def main():
    g = torch.Generator().manual_seed(SEED)

    # ---- 1. Sinkhorn-Knopp --------------------------------------------------
    b = torch.randn(N, N, generator=g)
    print("1. Sinkhorn-Knopp，n=4")
    for steps in (0, 1, 3, 20):
        print(f"   {steps:2d}次   {sums(sinkhorn(b, steps))}")
    print("   20次之后的矩阵：")
    for row in sinkhorn(b, 20):
        print("     ", "  ".join(f"{float(v):.4f}" for v in row))

    # ---- 2. 30层乘下来 ------------------------------------------------------
    # 不加约束：单位矩阵加一点扰动，每一层的B看上去都很“温和”。
    free = torch.eye(N)
    ds = torch.eye(N)
    print(f"\n2. 每层乘一个随机的{N}×{N}矩阵，乘{LAYERS}层，看乘积里最大的数")
    print(f"   {'层数':>4}{'不加约束':>14}{'双随机':>12}")
    for i in range(1, LAYERS + 1):
        free = (torch.eye(N) + 0.5 * torch.randn(N, N, generator=g)) @ free
        ds = sinkhorn(torch.randn(N, N, generator=g), 20) @ ds
        if i in (1, 5, 10, 20, 30):
            print(f"   {i:>4}{float(free.abs().max()):14.3e}"
                  f"{float(ds.abs().max()):12.4f}")
    print(f"   30层的双随机乘积   {sums(ds)}")


if __name__ == "__main__":
    main()
