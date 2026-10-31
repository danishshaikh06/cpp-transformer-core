
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache, DynamicCache

from cpp_transformer.configs.attention import AttentionConfig
from cpp_transformer.models.attention.rope import apply_rotary_pos_emb


class GroupedQueryAttention(nn.Module):
    """
    Causal Grouped Query Attention.

    Supports:
        - MHA, GQA and MQA
        - Reference and SDPA backends
        - DynamicCache during inference
        - Boolean attention masks
        - Training-time dropout

    The caller owns the KV cache.
    No cache is stored as mutable model state.
    """

    def __init__(
        self,
        config: AttentionConfig,
        *,
        layer_idx: int,
    ) -> None:
        super().__init__()

        if (
            isinstance(layer_idx, bool)
            or not isinstance(layer_idx, int)
            or layer_idx < 0
        ):
            raise ValueError(
                "layer_idx must be a nonnegative integer"
            )

        self.config = config
        self.layer_idx = layer_idx

        d = config.d_model
        kv_dim = config.num_key_value_heads * config.head_dim

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, kv_dim, bias=False)
        self.v_proj = nn.Linear(d, kv_dim, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """ 
        Args:
            hidden_states:
                (batch, new_tokens, d_model)

            attention_mask:
                Boolean key-padding mask (B, total_kv_tokens),
                or explicit 4D attention mask.
                True means attention is allowed.

            past_key_values:
                Request-owned DynamicCache.

            cache_position:
                Absolute positions of the incoming tokens.
                Must be consecutive and consistent with cache.

            use_cache:
                Whether to update the supplied cache.

        Returns:
            Attention output: (B, new_tokens, d_model)

        Note:
            The supplied cache is mutated in place.
        """

        # Input Validation
        if (
            hidden_states.ndim != 3
            or hidden_states.shape[-1] != self.config.d_model
        ):
            raise ValueError(
                "hidden_states must have shape "
                "(batch, tokens, d_model)"
            )

        if not hidden_states.is_floating_point():
            raise TypeError(
                "hidden_states must be floating point"
            )

        batch, q_len, _ = hidden_states.shape

        if batch == 0 or q_len == 0:
            raise ValueError(
                "Empty batches or sequences are unsupported"
            )

        if use_cache and past_key_values is None:
            raise ValueError(
                "Pass a request-owned DynamicCache "
                "when use_cache=True"
            )

        if past_key_values is not None and not use_cache:
            raise ValueError(
                "past_key_values supplied with use_cache=False"
            )

        if (
            past_key_values is not None
            and not isinstance(past_key_values, DynamicCache)
        ):
            raise NotImplementedError(
                "Only DynamicCache is supported "
                "in this implementation"
            )

        # Cache positions
        past_len = (
            past_key_values.get_seq_length(self.layer_idx)
            if past_key_values is not None
            else 0
        )

        kv_len = past_len + q_len

        if cache_position is not None:
            if (
                cache_position.ndim != 1
                or cache_position.numel() != q_len
                or cache_position.dtype != torch.long
                or cache_position.device != hidden_states.device
            ):
                raise ValueError(
                    "cache_position must be a LongTensor "
                    "of shape (tokens,) on the input device"
                )
        else:
            cache_position = torch.arange(
                past_len,
                kv_len,
                device=hidden_states.device,
                dtype=torch.long,
            )

        # Validate masks before cache mutation
        if attention_mask is not None:

            if (
                attention_mask.dtype != torch.bool
                or attention_mask.device != hidden_states.device
            ):
                raise ValueError(
                    "attention_mask must be boolean "
                    "and on the input device"
                )

            allowed_shapes = (
                (batch, kv_len),
                (batch, 1, q_len, kv_len), 
                (
                    batch,
                    self.config.num_attention_heads,
                    q_len,
                    kv_len,
                ),
            )

            if tuple(attention_mask.shape) not in allowed_shapes:
                raise ValueError(
                    "Expected a 2D key-padding or 4D "
                    "query/key mask, got "
                    f"{tuple(attention_mask.shape)}"
                )

        Hq = self.config.num_attention_heads
        Hkv = self.config.num_key_value_heads
        d = self.config.head_dim

        # Q, K, V projections
        query = self.q_proj(hidden_states) # (B, T, D)
        key = self.k_proj(hidden_states)   # (B,T,kv_dim)
        value = self.v_proj(hidden_states) # (B,T,kv_dim)

        # (B, Hq, Tq, d)
        query = query.reshape(
            batch, q_len, Hq, d
        ).transpose(1, 2)

        # (B, Hkv, Tq, d)
        key = key.reshape(
            batch, q_len, Hkv, d
        ).transpose(1, 2)

        value = value.reshape(
            batch, q_len, Hkv, d
        ).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings

            query, key = apply_rotary_pos_emb(
                query,
                key,
                cos,
                sin,
            )

        # KV cache update
        if past_key_values is not None:
            key, value = past_key_values.update(
                key,
                value,
                layer_idx=self.layer_idx,
                cache_kwargs={
                    "cache_position": cache_position
                },
            )

        # Attention mask
        # PyTorch's non-square is_causal alignment
        # is not suitable for cached decode path.
        square_fast_path = (
            past_len == 0
            and attention_mask is None
        )

        if square_fast_path:
            mask = None
        else:
            key_positions = torch.arange(
                key.shape[-2],
                device=hidden_states.device,
            )

            # Query i attends to keys at positions <= i.
            mask = (
                key_positions[None, :]
                <= cache_position[:, None] # -> this results in shape [Tq,Tk]
            )[None, None, :, :] # (B, Tq, Hq, Tk)

            if attention_mask is not None:
                if attention_mask.ndim == 2:
                    attention_mask = attention_mask[
                        :, None, None, :
                    ]

                mask = mask & attention_mask

        # SDPA backend 
        if self.config.backend == "sdpa":

            # Native GQA acceleration is intended for CUDA.
            # Keep a portable non-CUDA fallback.
    
            native_gqa = Hq != Hkv and query.is_cuda

            if Hq != Hkv and not native_gqa:
                group = self.config.group_size

                #key heads:   [K0, K0, K1, K1] ---> example if query group is 2 
                #value heads: [V0, V0, V1, V1]
                key = key.repeat_interleave(
                    group, dim=1
                )

                value = value.repeat_interleave(
                    group, dim=1
                )

            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                is_causal=square_fast_path,
                dropout_p=(
                    float(self.config.dropout)
                    if self.training
                    else 0.0
                ),
                enable_gqa=native_gqa,
            )

            if mask is not None:
                # Fully masked query rows produce zero output.
                valid = mask.any(
                    dim=-1, keepdim=True
                )

                context = torch.where(
                    valid, context, 0.0
                )

        # Reference backend 
        else:
            # Explicit repetition for numerical verification.
            # Not intended as the optimized GPU path.
            group = self.config.group_size

            expanded_k = key.repeat_interleave(
                group, dim=1
            )

            expanded_v = value.repeat_interleave(
                group, dim=1
            )

            scores = (
                query @ expanded_k.transpose(-2, -1)
            ) / math.sqrt(d)

            if square_fast_path:
                causal = torch.ones(
                    q_len,
                    q_len,
                    dtype=torch.bool,
                    device=query.device,
                ).tril()

                mask = causal[None, None, :, :]

            # Compute softmax in FP32.
            scores = scores.float().masked_fill(
                ~mask,
                torch.finfo(torch.float32).min,
            )

            weights = torch.softmax(
                scores, dim=-1
            )

            # Ensure fully masked rows contribute zero.
            weights = torch.where(
                mask.any(dim=-1, keepdim=True),
                weights,
                0.0,
            )

            weights = F.dropout(
                weights,
                p=float(self.config.dropout),
                training=self.training,
            )

            context = (
                weights.to(query.dtype) @ expanded_v
            )

        # Output projection
        context = context.transpose(1, 2).contiguous()

        context = context.reshape(
            batch,
            q_len,
            Hq * d,
        )

        return self.o_proj(context)
