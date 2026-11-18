"""Beam search 解码与 BLEU 评测（题 (c)）。

运行::

    python nmt/evaluate.py            # beam search（宽度 5）
    python nmt/evaluate.py --greedy   # 贪心解码，用于对比
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from common import (
    BOS_ID,
    DATA_DIR,
    EOS_ID,
    OUTPUT_DIR,
    PAD_ID,
    RESULTS_DIR,
    format_seconds,
    resolve_device,
    save_json,
    set_seed,
)
from data import load_split, load_tokenizers, make_pad_mask, tokenize_dataset
from model import EncoderDecoderModel, ModelConfig


def _prepare_source(sample: dict, device: torch.device):
    src_ids = torch.tensor([sample["zh_ids"]], dtype=torch.long, device=device)
    return src_ids, make_pad_mask(src_ids, PAD_ID).to(device)


@torch.no_grad()
def greedy_decode(
    model: EncoderDecoderModel,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    max_length: int = 32,
) -> List[int]:
    """单句贪心解码，返回不含 ``<s>`` 的 token id 列表。"""
    device = src_ids.device
    memory = model.forward_encoder(src_ids, src_mask)
    generated = [BOS_ID]
    for _ in range(max_length - 1):
        tgt = torch.tensor([generated], dtype=torch.long, device=device)
        logits = model.forward_decoder(tgt, memory, src_mask)
        next_id = int(logits[0, -1].argmax().item())
        generated.append(next_id)
        if next_id == EOS_ID:
            break
    return generated[1:]


@torch.no_grad()
def beam_search(
    model: EncoderDecoderModel,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    beam_width: int = 5,
    max_length: int = 32,
) -> Tuple[List[int], float]:
    """单句 beam search。

    每个假设记录 ``(token 列表, 累积 log 概率)``；所有当前 beam 拼成一个
    batch 一次送进解码器，编码器输出用 ``expand`` 复制到 beam 大小。
    最终按“平均每个 token 的 log 概率”选句，避免偏向过短译文。

    Returns:
        ``(最佳 token id 列表（不含 <s>）, 归一化得分)``。
    """
    device = src_ids.device
    memory = model.forward_encoder(src_ids, src_mask)
    memory = memory.expand(beam_width, -1, -1)
    src_mask_beam = src_mask.expand(beam_width, -1, -1)

    beams: List[Tuple[List[int], float]] = [([BOS_ID], 0.0)]
    beam_scores = torch.zeros(1, device=device)
    completed: List[Tuple[List[int], float]] = []

    for _ in range(max_length - 1):
        if not beams:
            break
        tgt = torch.tensor([seq for seq, _ in beams], dtype=torch.long, device=device)
        logits = model.forward_decoder(
            tgt, memory[: len(beams)], src_mask_beam[: len(beams)]
        )
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
        vocab_size = log_probs.size(-1)

        # (beam, V) 的打分展平后一次 topk，避免 Python 双重循环
        candidates = (log_probs + beam_scores[: len(beams), None]).view(-1)
        top_scores, top_indices = torch.topk(
            candidates, k=min(2 * beam_width, candidates.numel())
        )

        new_beams: List[Tuple[List[int], float]] = []
        new_scores: List[float] = []
        for score, flat_index in zip(top_scores.tolist(), top_indices.tolist()):
            i, token_id = divmod(flat_index, vocab_size)
            seq = beams[i][0]
            if token_id == EOS_ID:
                completed.append((seq + [EOS_ID], score))
            else:
                new_beams.append((seq + [token_id], score))
                new_scores.append(score)
            if len(new_beams) >= beam_width:
                break
        beams = new_beams
        beam_scores = torch.tensor(new_scores, device=device)
        if len(completed) >= beam_width:
            break

    pool = completed if completed else beams
    best_seq, best_score = max(
        pool, key=lambda item: item[1] / max(len(item[0]) - 1, 1)
    )
    normalized = best_score / max(len(best_seq) - 1, 1)
    return best_seq[1:], normalized


def compute_bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """用 ``nltk.translate.bleu_score.corpus_bleu`` 计算语料级 BLEU。

    输入是已经 detokenize 的英文句子（空格分词），不是 BPE token。
    每句的参考译文按 list 传入，以支持多参考。
    """
    from nltk.translate.bleu_score import corpus_bleu

    hyp_tokens = [hyp.split() for hyp in hypotheses]
    ref_tokens = [[ref.split()] for ref in references]
    return float(corpus_bleu(ref_tokens, hyp_tokens))


def load_model(model_path: Path, device: torch.device) -> EncoderDecoderModel:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["config"])
    model = EncoderDecoderModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beam search + BLEU")
    parser.add_argument("--model-path", type=Path, default=OUTPUT_DIR / "model.pt")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 句（0 表示全部）")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if not args.model_path.exists():
        raise SystemExit(f"找不到模型 {args.model_path}，请先运行 python nmt/train.py")
    tok_src, tok_tgt = load_tokenizers(args.data_dir)
    model = load_model(args.model_path, device)

    records = load_split(args.split, args.data_dir)
    if args.limit:
        records = records[: args.limit]
    samples = tokenize_dataset(records, tok_src, tok_tgt, model.config.max_sequence_length)

    mode = "greedy" if args.greedy else "beam"
    print(f"解码方式：{mode}（beam width={args.beam_width}，max length={args.max_length}）")
    print(f"测试句数：{len(samples)}")

    hypotheses, references = [], []
    start = time.time()
    for index, sample in enumerate(samples, 1):
        src_ids, src_mask = _prepare_source(sample, device)
        if args.greedy:
            output_ids = greedy_decode(model, src_ids, src_mask, args.max_length)
        else:
            output_ids, _ = beam_search(
                model, src_ids, src_mask, args.beam_width, args.max_length
            )
        hypothesis = tok_tgt.decode(output_ids, skip_special_tokens=True).strip()
        hypotheses.append(hypothesis)
        references.append(sample["en"])
        if index % 50 == 0:
            print(f"  已解码 {index}/{len(samples)}")

    elapsed = time.time() - start
    bleu = compute_bleu(hypotheses, references)
    print(f"语料级 BLEU：{bleu:.4f}（耗时 {format_seconds(elapsed)}）")

    result = {
        "mode": mode,
        "beam_width": None if args.greedy else args.beam_width,
        "max_length": args.max_length,
        "split": args.split,
        "num_sentences": len(samples),
        "bleu": bleu,
        "seconds": elapsed,
    }
    save_json(result, RESULTS_DIR / f"bleu_{mode}.json")

    translation_path = RESULTS_DIR / f"translations_{mode}.txt"
    with open(translation_path, "w", encoding="utf-8") as f:
        for i, (sample, hyp) in enumerate(zip(samples, hypotheses), 1):
            f.write(f"# {i}\n")
            f.write(f"ZH:  {sample['zh']}\n")
            f.write(f"HYP: {hyp}\n")
            f.write(f"REF: {sample['en']}\n\n")

    print("\n示例译文：")
    for i in range(min(8, len(samples))):
        print(f"  ZH : {samples[i]['zh']}")
        print(f"  HYP: {hypotheses[i]}")
        print(f"  REF: {references[i]}")
        print()
    print(f"结果已写入 {RESULTS_DIR / f'bleu_{mode}.json'} 与 {translation_path}")


if __name__ == "__main__":
    main()
