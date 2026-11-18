"""Cross-attention 可视化（题 (d)）。

运行::

    python nmt/visualize_attention.py

对作业要求的两个中文句子（也可用 ``--sentence`` 追加）画热力图：
行是生成的英文 token，列是中文源 token，权重取最后一个解码器层、
对所有注意力头取平均。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import (
    BOS_ID,
    DATA_DIR,
    EOS_ID,
    FIGURES_DIR,
    OUTPUT_DIR,
    PAD_ID,
    RESULTS_DIR,
    resolve_device,
    set_seed,
    setup_cjk_font,
)
from data import make_pad_mask
from evaluate import load_model

DEFAULT_SENTENCES = [
    "黑猫坐在红色的地毯上。",
    "她还没有对自己的未来做出重要的决定。",
]


@torch.no_grad()
def translate_with_attention(
    model,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    tok_tgt,
    max_length: int = 32,
):
    """贪心解码并收集最后一层 cross-attention。

    Returns:
        ``(token_ids, attention)``：``token_ids`` 不含 ``<s>``/``</s>``；
        ``attention`` 形状为 (生成 token 数, 源 token 数)，已对头取平均。
    """
    device = src_ids.device
    memory = model.forward_encoder(src_ids, src_mask)
    generated = [BOS_ID]
    rows = []

    for _ in range(max_length - 1):
        tgt = torch.tensor([generated], dtype=torch.long, device=device)
        logits, cross_weights = model.forward_decoder(
            tgt, memory, src_mask, return_attention_weights=True
        )
        # 最后一层 cross-attention：(1, H, T_q, T_s) -> 对头平均
        last_layer = cross_weights[-1][0].mean(dim=0)  # (T_q, T_s)
        next_id = int(logits[0, -1].argmax().item())
        if next_id == EOS_ID:
            break
        generated.append(next_id)
        rows.append(last_layer[-1].detach().cpu())

    token_ids = generated[1:]
    attention = (
        torch.stack(rows, dim=0) if rows else torch.zeros(0, src_ids.size(1))
    )
    return token_ids, attention


def plot_attention(
    src_tokens,
    tgt_tokens,
    attention,
    output_path: Path,
    title: str,
) -> None:
    setup_cjk_font()
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = attention.numpy()
    fig, ax = plt.subplots(
        figsize=(max(5.0, 0.55 * len(src_tokens) + 2.0),
                 max(3.0, 0.40 * len(tgt_tokens) + 1.6))
    )
    image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0)
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=9)
    ax.set_xlabel("源语言（中文）token")
    ax.set_ylabel("生成的目标语言（英文）token")
    ax.set_title(title, fontsize=11)

    if matrix.size and matrix.shape[0] <= 20 and matrix.shape[1] <= 20:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if value >= 0.01:
                    ax.text(
                        j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if value > 0.55 else "black",
                    )
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="注意力权重（4 头平均）")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 cross-attention")
    parser.add_argument("--model-path", type=Path, default=OUTPUT_DIR / "model.pt")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--sentence", action="append", default=None,
                        help="追加要可视化的中文句子，可重复使用")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if not args.model_path.exists():
        raise SystemExit(f"找不到模型 {args.model_path}，请先运行 python nmt/train.py")

    from transformers import AutoTokenizer

    tok_src = AutoTokenizer.from_pretrained(str(args.data_dir / "tokenizer_zh"))
    tok_tgt = AutoTokenizer.from_pretrained(str(args.data_dir / "tokenizer_en"))
    model = load_model(args.model_path, device)

    sentences = list(DEFAULT_SENTENCES)
    if args.sentence:
        sentences.extend(args.sentence)

    setup_cjk_font()
    for index, sentence in enumerate(sentences, 1):
        ids = tok_src.encode(sentence)
        src_ids = torch.tensor([ids], dtype=torch.long, device=device)
        src_mask = make_pad_mask(src_ids, PAD_ID).to(device)
        token_ids, attention = translate_with_attention(
            model, src_ids, src_mask, tok_tgt, args.max_length
        )
        translation = tok_tgt.decode(token_ids, skip_special_tokens=True).strip()
        src_tokens = tok_src.convert_ids_to_tokens(ids, skip_special_tokens=True)
        tgt_tokens = tok_tgt.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        # pad 列不画
        keep = [i for i, t in enumerate(src_tokens) if t != tok_src.pad_token]
        src_tokens = [src_tokens[i] for i in keep]
        attention = attention[:, keep]

        print(f"\n例 {index}：{sentence}")
        print(f"译文：{translation}")

        output_path = FIGURES_DIR / f"attention_{index}.png"
        plot_attention(src_tokens, tgt_tokens, attention, output_path,
                       title=f"例 {index}：{sentence}")

        with open(RESULTS_DIR / f"attention_{index}.txt", "w", encoding="utf-8") as f:
            f.write(f"Sentence: {sentence}\nTranslation: {translation}\n")
            f.write("Source tokens: " + " ".join(src_tokens) + "\n")
            f.write("Target tokens: " + " ".join(tgt_tokens) + "\n")


if __name__ == "__main__":
    main()
