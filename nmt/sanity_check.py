"""多头注意力自检（题 (a)-3）。

运行 ``python nmt/sanity_check.py``：
  * 检查输出与注意力权重的形状；
  * 检查每行权重之和为 1、被屏蔽位置的权重为 0；
  * 在玩具输入上画出编码器自注意力、解码器带掩码的自注意力、
    cross-attention 三种权重（``nmt/figures/sanity_attention.png``）。

玩具输入假设：编码器输入的最后 1 个 token、解码器输入的最后 2 个
token 是 ``<pad>``。
"""

from __future__ import annotations

import torch

from common import FIGURES_DIR, PAD_ID, setup_cjk_font
from data import combine_masks, make_causal_mask, make_pad_mask
from model import MultiHeadAttention

HIDDEN_SIZE = 32
NUM_HEADS = 4
NUM_ENC_TOKENS = 6
NUM_DEC_TOKENS = 6
SRC_TOKENS = ["the", "cat", "sat", "on", "mat", "<pad>"]
TGT_TOKENS = ["<s>", "le", "chat", "s'assit", "<pad>", "<pad>"]


def _toy_embeddings() -> torch.Tensor:
    """构造近似 one-hot 的玩具嵌入，使自注意力的对角线清晰可见。"""
    vocab = 16
    emb = torch.zeros(vocab, HIDDEN_SIZE)
    emb[:, :vocab] = torch.eye(vocab)
    return emb


def _check(name: str, condition: bool, detail: str = "") -> None:
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"：{detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"{name} 检查未通过 {detail}")


def run_checks(seed: int = 42) -> None:
    """对多头注意力做形状、归一化与掩码自检，并保存示意图。"""
    torch.manual_seed(seed)
    attn = MultiHeadAttention(HIDDEN_SIZE, NUM_HEADS)
    # 为便于观察，把四个投影都初始化为恒等映射（真实训练时用默认初始化）。
    with torch.no_grad():
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj):
            proj.weight.copy_(torch.eye(HIDDEN_SIZE))
            proj.bias.zero_()

    embeddings = _toy_embeddings()

    # 非特殊 token 使用 id 4..8，pad 使用真实的 PAD_ID=3
    src_ids = torch.tensor([[4, 5, 6, 7, 8, PAD_ID]])          # (1, 6)
    tgt_ids = torch.tensor([[1, 4, 5, 6, PAD_ID, PAD_ID]])      # (1, 6)

    src_pad_mask = make_pad_mask(src_ids)                       # (1, 1, 6)
    tgt_pad_mask = make_pad_mask(tgt_ids)                       # (1, 1, 6)
    causal = make_causal_mask(tgt_ids.size(1), device=tgt_ids.device)  # (1, 6, 6)
    tgt_mask = combine_masks(tgt_pad_mask, causal)              # (1, 6, 6)

    src_emb = embeddings[src_ids]                               # (1, 6, 32)
    tgt_emb = embeddings[tgt_ids]                               # (1, 6, 32)

    enc_out, enc_weights = attn(src_emb, src_emb, src_emb, src_pad_mask)
    dec_out, dec_weights = attn(tgt_emb, tgt_emb, tgt_emb, tgt_mask)
    cross_out, cross_weights = attn(tgt_emb, src_emb, src_emb, src_pad_mask)

    batch = src_ids.size(0)

    # ---------------------------------------------------------- 形状检查
    _check(
        "编码器自注意力输出形状",
        tuple(enc_out.shape) == (batch, NUM_ENC_TOKENS, HIDDEN_SIZE),
        str(tuple(enc_out.shape)),
    )
    _check(
        "编码器自注意力权重形状",
        tuple(enc_weights.shape) == (batch, NUM_HEADS, NUM_ENC_TOKENS, NUM_ENC_TOKENS),
        str(tuple(enc_weights.shape)),
    )
    _check(
        "解码器自注意力权重形状",
        tuple(dec_weights.shape) == (batch, NUM_HEADS, NUM_DEC_TOKENS, NUM_DEC_TOKENS),
        str(tuple(dec_weights.shape)),
    )
    _check(
        "cross-attention 权重形状",
        tuple(cross_weights.shape) == (batch, NUM_HEADS, NUM_DEC_TOKENS, NUM_ENC_TOKENS),
        str(tuple(cross_weights.shape)),
    )

    # ---------------------------------------------------------- 归一化检查
    for name, weights in (
        ("编码器自注意力", enc_weights),
        ("解码器自注意力", dec_weights),
        ("cross-attention", cross_weights),
    ):
        sums = weights.sum(dim=-1)
        _check(
            f"{name} 每行权重之和为 1",
            bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-5)),
            f"最小 {sums.min():.6f}，最大 {sums.max():.6f}",
        )

    # ---------------------------------------------------------- 掩码检查
    _check(
        "编码器自注意力：<pad> 列权重为 0",
        torch.count_nonzero(enc_weights[..., -1]).item() == 0,
        f"非零元素 {torch.count_nonzero(enc_weights[..., -1]).item()} 个",
    )
    _check(
        "解码器自注意力：因果掩码上三角权重为 0",
        torch.count_nonzero(torch.triu(dec_weights[0], diagonal=1)).item() == 0,
    )
    _check(
        "解码器自注意力：<pad> 列权重为 0",
        torch.count_nonzero(dec_weights[..., -2:]).item() == 0,
    )
    _check(
        "cross-attention：<pad> 列权重为 0",
        torch.count_nonzero(cross_weights[..., -1]).item() == 0,
    )

    # ---------------------------------------------------------- 画图
    setup_cjk_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    def _plot(ax, weights, ylabels, xlabels, title):
        matrix = weights[0].mean(dim=0).detach().cpu().numpy()  # 对头取平均
        image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("key（被注意的位置）")
        ax.set_ylabel("query（发起注意的位置）")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if value <= 1e-6:
                    ax.add_patch(
                        Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            facecolor="#e6e6e6", edgecolor="white", linewidth=0.8,
                        )
                    )
                    ax.text(
                        j, i, "0", ha="center", va="center",
                        fontsize=8, color="#c00000", fontweight="bold",
                    )
                else:
                    ax.text(
                        j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value > 0.5 else "black",
                    )
        return image

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 10.5))
    image = _plot(
        axes[0], enc_weights, SRC_TOKENS, SRC_TOKENS,
        "编码器自注意力（最后 1 个 <pad>）",
    )
    _plot(
        axes[1], dec_weights, TGT_TOKENS, TGT_TOKENS,
        "解码器自注意力（因果掩码 + 2 个 <pad>）",
    )
    _plot(
        axes[2], cross_weights, TGT_TOKENS, SRC_TOKENS,
        "cross-attention（源端 1 个 <pad>）",
    )
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    fig.suptitle(
        "多头注意力自检：对 4 个头取平均后的权重（灰色格子为被屏蔽位置）",
        fontsize=13, y=0.985,
    )
    fig.subplots_adjust(top=0.93, hspace=0.55, bottom=0.05)
    output_path = FIGURES_DIR / "sanity_attention.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {output_path}")
    print("全部检查通过。")


if __name__ == "__main__":
    run_checks()
