"""数据加载与分词（题 (a)-1）。

约定：
  * 每条样本是 dict，含原始文本 ``zh`` / ``en`` 以及分词后的
    ``zh_ids`` / ``en_ids``（列表，含两端的 ``<s>``、``</s>``）；
  * 掩码一律用 BoolTensor，``True`` 表示“可以注意”，``False`` 表示
    ``<pad>`` 或被因果掩码屏蔽的位置。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from common import DATA_DIR, EOS_ID, PAD_ID, read_jsonl


def load_tokenizers(data_dir: str | Path = DATA_DIR):
    """加载中英两个 BPE 分词器。

    Returns:
        (tok_src, tok_tgt)：中文（源语言）与英文（目标语言）分词器。
    """
    data_dir = Path(data_dir)
    tok_src = AutoTokenizer.from_pretrained(str(data_dir / "tokenizer_zh"))
    tok_tgt = AutoTokenizer.from_pretrained(str(data_dir / "tokenizer_en"))
    return tok_src, tok_tgt


def load_split(split: str, data_dir: str | Path = DATA_DIR) -> list[dict]:
    """读取 ``zh_en/{split}.jsonl``，返回原始记录列表。"""
    return read_jsonl(Path(data_dir) / "zh_en" / f"{split}.jsonl")


def tokenize_dataset(
    records: Iterable[dict],
    tok_src,
    tok_tgt,
    max_sequence_length: int = 32,
) -> list[dict]:
    """把原始句对编码为 token id 序列。

    超长序列截断到 ``max_sequence_length``，并保证末尾仍是 ``</s>``。
    """
    tokenized = []
    for record in records:
        src_ids = list(tok_src.encode(record["zh"]))
        tgt_ids = list(tok_tgt.encode(record["en"]))
        if len(src_ids) > max_sequence_length:
            src_ids = src_ids[: max_sequence_length - 1] + [EOS_ID]
        if len(tgt_ids) > max_sequence_length:
            tgt_ids = tgt_ids[: max_sequence_length - 1] + [EOS_ID]
        tokenized.append(
            {
                "zh": record["zh"],
                "en": record["en"],
                "zh_ids": src_ids,
                "en_ids": tgt_ids,
            }
        )
    return tokenized


def make_pad_mask(ids: torch.Tensor, pad_token_id: int = PAD_ID) -> torch.Tensor:
    """构造 ``<pad>`` 掩码。

    Args:
        ids: (B, T) 的 token id。

    Returns:
        (B, 1, T) 的 BoolTensor，``True`` 表示该位置不是 ``<pad>``。
    """
    return ids.ne(pad_token_id).unsqueeze(1)


def make_causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    """构造因果掩码，位置 i 只能看到 j <= i。

    Returns:
        (1, length, length) 的 BoolTensor。
    """
    ones = torch.ones(length, length, dtype=torch.bool, device=device)
    return torch.tril(ones).unsqueeze(0)


def combine_masks(pad_mask: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
    """把 ``<pad>`` 掩码与因果掩码合并（逻辑与）。

    Args:
        pad_mask: (B, 1, T_k) 或 (B, T_k)。
        causal_mask: (1, T_q, T_k)。

    Returns:
        (B, T_q, T_k) 的 BoolTensor。
    """
    if pad_mask.dim() == 3:
        pad_mask = pad_mask[:, 0, :]
    return pad_mask.unsqueeze(1) & causal_mask.to(pad_mask.device)


def collate_batch(samples: Sequence[dict]) -> dict:
    """把若干条样本补齐成一个 batch，并给出两种掩码。

    Returns:
        dict：``src_ids`` (B, T_s)、``tgt_ids`` (B, T_t)、
        ``src_mask`` (B, 1, T_s)、``tgt_mask`` (B, T_t, T_t)。
    """
    src_ids = pad_sequence(
        [torch.tensor(s["zh_ids"], dtype=torch.long) for s in samples],
        batch_first=True,
        padding_value=PAD_ID,
    )
    tgt_ids = pad_sequence(
        [torch.tensor(s["en_ids"], dtype=torch.long) for s in samples],
        batch_first=True,
        padding_value=PAD_ID,
    )
    src_mask = make_pad_mask(src_ids)
    tgt_mask = combine_masks(
        make_pad_mask(tgt_ids),
        make_causal_mask(tgt_ids.size(1), device=tgt_ids.device),
    )
    return {
        "src_ids": src_ids,
        "tgt_ids": tgt_ids,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
    }
