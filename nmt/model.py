"""多头注意力 / 嵌入 / Transformer 块 / 整体模型（题 (a)-2,4,5,6）。

全部模块从零实现：没有调用 ``nn.MultiheadAttention``、
``nn.functional.scaled_dot_product_attention`` 或 ``nn.Transformer`` 系列。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import PAD_ID


@dataclass
class ModelConfig:
    """模型超参数（表 1），默认值不要修改。"""

    src_vocab_size: int = 3200
    tgt_vocab_size: int = 3200
    hidden_size: int = 32
    intermediate_size: int = 128
    num_attention_heads: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    max_sequence_length: int = 32
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.0
    src_pad_token_id: int = PAD_ID
    tgt_pad_token_id: int = PAD_ID
    activation: str = "relu"
    norm_position: str = "post"
    tie_word_embeddings: bool = False


class MultiHeadAttention(nn.Module):
    """缩放点积多头注意力，一次矩阵乘法处理所有头。

    ``forward`` 返回 ``(输出, 注意力权重)``；权重形状为
    ``(batch_size, num_heads, num_query_tokens, num_key_tokens)``。
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d) -> (B, H, T, d_H)。"""
        batch, length, _ = x.shape
        x = x.view(batch, length, self.num_attention_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, T_q, d)
            key: (B, T_k, d)
            value: (B, T_k, d)
            mask: BoolTensor，``True`` 表示可以注意；可为 (B, T_k)、
                (B, 1, T_k)、(B, T_q, T_k) 或 (B, 1, T_q, T_k)。

        Returns:
            (out, attention_weights)，形状分别为 (B, T_q, d) 与
            (B, H, T_q, T_k)。
        """
        batch, tgt_len, _ = query.shape
        q = self._split_heads(self.q_proj(query))    # (B, H, T_q, d_H)
        k = self._split_heads(self.k_proj(key))      # (B, H, T_k, d_H)
        v = self._split_heads(self.v_proj(value))    # (B, H, T_k, d_H)

        # 一次 @ 得到所有头的打分，不对头做 Python 循环
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T_q,T_k)

        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=scores.device)
            if mask.dim() == 2:
                mask = mask[:, None, None, :]       # (B, 1, 1, T_k)
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]          # (B, 1, T_q, T_k)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)     # (B, H, T_q, d_H)
        context = context.transpose(1, 2).contiguous().view(batch, tgt_len, self.hidden_size)
        return self.out_proj(context), attn_weights


class Embedding(nn.Module):
    """词嵌入与可学习的位置嵌入相加。"""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_sequence_length: int,
        pad_token_id: int = PAD_ID,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_size)
        self.max_sequence_length = max_sequence_length
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.max_sequence_length:
            raise ValueError(
                f"序列长度 {seq_len} 超过 max_sequence_length={self.max_sequence_length}"
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        return self.dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )


class TransformerBlock(nn.Module):
    """一个编码器层或解码器层（post-norm）。

    解码器层比编码器层多一个 cross-attention 子层。每个子层都做
    ``LayerNorm(x + Dropout(Sublayer(x)))``。
    """

    def __init__(self, config: ModelConfig, is_decoder: bool = False):
        super().__init__()
        d = config.hidden_size
        self.is_decoder = is_decoder
        self.self_attn = MultiHeadAttention(
            d, config.num_attention_heads, config.attention_probs_dropout_prob
        )
        self.cross_attn = (
            MultiHeadAttention(d, config.num_attention_heads, config.attention_probs_dropout_prob)
            if is_decoder
            else None
        )
        self.ffn = nn.Sequential(
            nn.Linear(d, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, d),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.self_attn_norm = nn.LayerNorm(d)
        self.cross_attn_norm = nn.LayerNorm(d) if is_decoder else None
        self.ffn_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            (hidden_states, self_attention_weights, cross_attention_weights)；
            编码器层的第三个元素为 ``None``。
        """
        attn_out, self_weights = self.self_attn(
            hidden_states, hidden_states, hidden_states, attention_mask
        )
        hidden_states = self.self_attn_norm(hidden_states + self.dropout(attn_out))

        cross_weights = None
        if self.is_decoder:
            if encoder_hidden_states is None:
                raise ValueError("解码器层必须提供 encoder_hidden_states")
            cross_out, cross_weights = self.cross_attn(
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states,
                encoder_attention_mask,
            )
            hidden_states = self.cross_attn_norm(hidden_states + self.dropout(cross_out))

        hidden_states = self.ffn_norm(hidden_states + self.ffn(hidden_states))
        return hidden_states, self_weights, cross_weights


class Encoder(nn.Module):
    """编码器：嵌入 + N 个 Transformer 块。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = Embedding(
            config.src_vocab_size,
            config.hidden_size,
            config.max_sequence_length,
            config.src_pad_token_id,
            dropout=0.0,
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config, is_decoder=False) for _ in range(config.num_encoder_layers)]
        )

    def forward(
        self,
        src_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ):
        hidden_states = self.embedding(src_ids)
        all_weights: List[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, weights, _ = layer(hidden_states, src_mask)
            all_weights.append(weights)
        if return_attention_weights:
            return hidden_states, all_weights
        return hidden_states


class Decoder(nn.Module):
    """解码器：嵌入 + N 个带因果掩码与 cross-attention 的块。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = Embedding(
            config.tgt_vocab_size,
            config.hidden_size,
            config.max_sequence_length,
            config.tgt_pad_token_id,
            dropout=0.0,
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config, is_decoder=True) for _ in range(config.num_decoder_layers)]
        )

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ):
        hidden_states = self.embedding(tgt_ids)
        cross_weights: List[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, _, weights = layer(hidden_states, tgt_mask, memory, src_mask)
            if weights is not None:
                cross_weights.append(weights)
        if return_attention_weights:
            return hidden_states, cross_weights
        return hidden_states


class EncoderDecoderModel(nn.Module):
    """编码器—解码器 Transformer。词嵌入与输出投影不共享权重。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.output_layer = nn.Linear(config.hidden_size, config.tgt_vocab_size, bias=False)

    def encode(self, src_ids: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        return self.encoder(src_ids, src_mask)

    def forward_encoder(self, src_ids: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        """编码源句，返回 (B, T_s, d) 的 memory。"""
        return self.encode(src_ids, src_mask)

    def forward_decoder(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ):
        """解码。

        Args:
            tgt_ids: (B, T_t) 解码器输入（训练时为 ``tgt_ids[:, :-1]``）。
            memory: (B, T_s, d) 编码器输出。
            src_mask: (B, 1, T_s) 源端 ``<pad>`` 掩码。
            tgt_mask: (B, T_t, T_t)；为 ``None`` 时按 ``tgt_ids`` 自动构造。
            return_attention_weights: 为 ``True`` 时额外返回每层的
                cross-attention 权重（list，长度等于解码器层数）。

        Returns:
            logits (B, T_t, V)，或 ``(logits, attention_weights)``。
        """
        if tgt_mask is None:
            from data import combine_masks, make_causal_mask, make_pad_mask

            tgt_mask = combine_masks(
                make_pad_mask(tgt_ids, self.config.tgt_pad_token_id),
                make_causal_mask(tgt_ids.size(1), device=tgt_ids.device),
            ).to(tgt_ids.device)

        output = self.decoder(
            tgt_ids,
            memory,
            src_mask,
            tgt_mask,
            return_attention_weights=return_attention_weights,
        )
        if return_attention_weights:
            hidden_states, cross_weights = output
            return self.output_layer(hidden_states), cross_weights
        return self.output_layer(output)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.forward_encoder(src_ids, src_mask)
        return self.forward_decoder(tgt_ids, memory, src_mask, tgt_mask)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        return asdict(self.config)
