"""训练脚本（题 (b)）。

用法（GPU 服务器）::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        python nmt/train.py

日志写入 ``nmt/results/train.log``，验证集困惑度最低的模型保存为
``output/model.pt``。
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from common import (
    DATA_DIR,
    OUTPUT_DIR,
    PAD_ID,
    RESULTS_DIR,
    AverageMeter,
    Tee,
    describe_device,
    format_seconds,
    resolve_device,
    set_seed,
    timestamp,
)
from data import collate_batch, load_split, load_tokenizers, tokenize_dataset
from model import EncoderDecoderModel, ModelConfig


def compute_loss(
    model: EncoderDecoderModel,
    batch: dict,
    pad_token_id: int = PAD_ID,
) -> torch.Tensor:
    """Teacher forcing 下的逐 token 平均交叉熵（忽略 ``<pad>``）。

    解码器输入是 ``tgt_ids[:, :-1]``，预测目标是 ``tgt_ids[:, 1:]``。
    """
    src_ids = batch["src_ids"]
    tgt_ids = batch["tgt_ids"]
    src_mask = batch.get("src_mask")
    tgt_mask = batch.get("tgt_mask")

    memory = model.forward_encoder(src_ids, src_mask)
    decoder_input = tgt_ids[:, :-1]
    labels = tgt_ids[:, 1:]
    if tgt_mask is not None:
        tgt_mask = tgt_mask[:, :-1, :-1]

    logits = model.forward_decoder(decoder_input, memory, src_mask, tgt_mask)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_token_id,
    )


@torch.no_grad()
def evaluate_ppl(
    model: EncoderDecoderModel,
    batches: list[dict],
    device: torch.device,
    pad_token_id: int = PAD_ID,
) -> Tuple[float, float]:
    """在给定批次上计算平均损失与困惑度（按真实 token 数加权）。

    Returns:
        ``(loss, perplexity)``。
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = compute_loss(model, batch, pad_token_id)
        n_tokens = batch["tgt_ids"][:, 1:].ne(pad_token_id).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    return mean_loss, math.exp(min(mean_loss, 100.0))


def make_batches(samples: list[dict], batch_size: int, shuffle: bool, seed: int = 42) -> list[dict]:
    """把样本切成 batch；每个 epoch 用固定种子打乱一次。"""
    order = list(range(len(samples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    batches = []
    for start in range(0, len(order), batch_size):
        chunk = [samples[i] for i in order[start : start + batch_size]]
        batches.append(collate_batch(chunk))
    return batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练中译英 Transformer")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0, help="仅用于调试，0 表示不限制")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "train.log"
    if log_path.exists():
        log_path.unlink()
    import sys

    sys.stdout = Tee(log_path)
    print(f"[{timestamp()}] 开始训练")
    print(describe_device(device))

    tok_src, tok_tgt = load_tokenizers(args.data_dir)
    config = ModelConfig(
        src_vocab_size=len(tok_src),
        tgt_vocab_size=len(tok_tgt),
    )
    print(f"源语言词表 {config.src_vocab_size}，目标语言词表 {config.tgt_vocab_size}")

    t0 = time.time()
    train_samples = tokenize_dataset(
        load_split("train", args.data_dir), tok_src, tok_tgt, config.max_sequence_length
    )
    valid_samples = tokenize_dataset(
        load_split("validation", args.data_dir), tok_src, tok_tgt, config.max_sequence_length
    )
    print(f"训练集 {len(train_samples)} 对，验证集 {len(valid_samples)} 对")

    valid_batches = make_batches(valid_samples, args.batch_size, shuffle=False)
    model = EncoderDecoderModel(config).to(device)
    print(f"模型参数量 {model.num_parameters():,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"数据加载耗时 {format_seconds(time.time() - t0)}")

    steps_per_epoch = math.ceil(len(train_samples) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    best_ppl = float("inf")
    best_step = 0
    global_step = 0
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        batches = make_batches(train_samples, args.batch_size, shuffle=True, seed=args.seed + epoch)
        meter = AverageMeter()
        epoch_start = time.time()
        model.train()
        for batch in batches:
            if args.max_steps and global_step >= args.max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss(model, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            meter.update(loss.item())
            if global_step % args.log_every == 0:
                print(
                    f"[{timestamp()}] step {global_step}/{total_steps} | epoch {epoch} | "
                    f"train loss {meter.avg:.4f}"
                )
            if global_step % args.eval_every == 0:
                val_loss, val_ppl = evaluate_ppl(model, valid_batches, device)
                print(
                    f"[{timestamp()}] step {global_step} | val loss {val_loss:.4f} | "
                    f"val ppl {val_ppl:.2f}"
                )
                if val_ppl < best_ppl:
                    best_ppl, best_step = val_ppl, global_step
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "config": config.__dict__,
                        "step": global_step,
                        "epoch": epoch,
                        "val_ppl": val_ppl,
                    }
                    torch.save(checkpoint, args.output_dir / "model.pt")
                    print(f"[{timestamp()}] 保存最佳模型：step {global_step}，ppl {val_ppl:.2f}")

        print(
            f"[{timestamp()}] epoch {epoch} 完成 | 平均 loss {meter.avg:.4f} | "
            f"耗时 {format_seconds(time.time() - epoch_start)}"
        )
        if args.max_steps and global_step >= args.max_steps:
            break

    total_time = time.time() - train_start
    print(f"[{timestamp()}] 训练结束：{global_step} 步，总耗时 {format_seconds(total_time)}")
    print(f"最佳验证困惑度 {best_ppl:.2f}（step {best_step}）")
    with open(RESULTS_DIR / "train_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"steps={global_step}\n")
        f.write(f"best_val_ppl={best_ppl:.4f}\n")
        f.write(f"best_step={best_step}\n")
        f.write(f"train_seconds={total_time:.2f}\n")
        f.write(f"parameters={model.num_parameters()}\n")
        f.write(f"device={describe_device(device)}\n")


if __name__ == "__main__":
    main()
