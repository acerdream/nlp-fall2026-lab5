"""词表与 batch 统计（题 (b)）。

运行::

    python nmt/batch_stats.py

回答：
  (i) 源语言与目标语言的词表大小（含特殊符号）；
  (ii) 一个训练 batch 中平均有多少源/目标 token，``<pad>`` 占多大比例。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from common import DATA_DIR, PAD_ID, RESULTS_DIR, save_json
from data import collate_batch, load_split, load_tokenizers, tokenize_dataset
from model import ModelConfig


def batch_statistics(
    samples: list[dict],
    batch_size: int = 32,
    pad_token_id: int = PAD_ID,
) -> dict:
    """统计训练 batch 的 token 数与 ``<pad>`` 比例。

    每个 batch 先按最长序列补齐（与 :func:`collate_batch` 一致），再统计
    非 ``<pad>`` 的 token 数；填充比例按整个训练集累计，而不是对 batch
    比例再取平均。
    """
    total_src_real = 0
    total_tgt_real = 0
    total_src_slots = 0
    total_tgt_slots = 0
    num_batches = math.ceil(len(samples) / batch_size)

    for start in range(0, len(samples), batch_size):
        batch = collate_batch(samples[start : start + batch_size])
        src_ids, tgt_ids = batch["src_ids"], batch["tgt_ids"]
        total_src_real += src_ids.ne(pad_token_id).sum().item()
        total_tgt_real += tgt_ids.ne(pad_token_id).sum().item()
        total_src_slots += src_ids.numel()
        total_tgt_slots += tgt_ids.numel()

    stats = {
        "num_samples": len(samples),
        "batch_size": batch_size,
        "num_batches": num_batches,
        "avg_src_tokens_per_batch": total_src_real / num_batches,
        "avg_tgt_tokens_per_batch": total_tgt_real / num_batches,
        "avg_src_slots_per_batch": total_src_slots / num_batches,
        "avg_tgt_slots_per_batch": total_tgt_slots / num_batches,
        "src_pad_ratio": 1.0 - total_src_real / total_src_slots,
        "tgt_pad_ratio": 1.0 - total_tgt_real / total_tgt_slots,
        "total_src_real": total_src_real,
        "total_tgt_real": total_tgt_real,
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="统计词表与训练 batch")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    tok_src, tok_tgt = load_tokenizers(args.data_dir)
    print(f"(i) 源语言（中文）词表大小：{len(tok_src)}")
    print(f"    目标语言（英文）词表大小：{len(tok_tgt)}")

    config = ModelConfig()
    samples = tokenize_dataset(
        load_split("train", args.data_dir), tok_src, tok_tgt, config.max_sequence_length
    )
    stats = batch_statistics(samples, batch_size=32)
    stats["src_vocab_size"] = len(tok_src)
    stats["tgt_vocab_size"] = len(tok_tgt)
    save_json(stats, RESULTS_DIR / "batch_stats.json")

    print(f"(ii) 训练集 {stats['num_samples']} 对，共 {stats['num_batches']} 个 batch（batch=32）")
    print(
        f"     平均每个 batch：源语言 {stats['avg_src_tokens_per_batch']:.2f} 个真实 token"
        f"（含 pad 共 {stats['avg_src_slots_per_batch']:.2f} 个槽位），"
        f"目标语言 {stats['avg_tgt_tokens_per_batch']:.2f} 个真实 token"
        f"（含 pad 共 {stats['avg_tgt_slots_per_batch']:.2f} 个槽位）"
    )
    print(
        f"     <pad> 比例：源语言 {stats['src_pad_ratio'] * 100:.2f}%，"
        f"目标语言 {stats['tgt_pad_ratio'] * 100:.2f}%"
    )
    print(f"已保存 {RESULTS_DIR / 'batch_stats.json'}")


if __name__ == "__main__":
    main()
