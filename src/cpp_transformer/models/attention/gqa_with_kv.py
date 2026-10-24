
import math

import torch
from torch import nn


class GroupedQueryAttention(nn.Module):
    """
    Causal Grouped Query Attention with optional KV caching.

    kv_heads == h : MHA
    kv_heads == 1 : MQA
    otherwise     : GQA
    """

    def __init__(
        self,
        d_model: int,
        h: int,
        kv_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if d_model <= 0 or h <= 0 or kv_heads <= 0:
            raise ValueError("Dimensions must be positive")

        if d_model % h != 0:
            raise ValueError("d_model must be divisible by h")

        if h % kv_heads != 0:
            raise ValueError("h must be divisible by kv_heads")

        self.h = h
        self.kv_heads = kv_heads
        self.group_size = h // kv_heads
        self.d_k = d_model // h

        kv_dim = kv_heads * self.d_k

        # Independent query heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)

        # Shared key/value heads
        self.w_k = nn.Linear(d_model, kv_dim, bias=False)
        self.w_v = nn.Linear(d_model, kv_dim, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        """
        Causal self-attention.

        Input Q/K/V: (B, Tnew, D)

        Cached K/V:
            (B, Hkv, Tpast, d)

        Output:
            (B, Tnew, D)

        If use_cache=True, also return updated K/V.
        """

        if q.ndim != 3 or q.shape != k.shape or k.shape != v.shape:
            raise ValueError("Expected matching (B, Tnew, D) inputs")

        B, Tq, _ = q.shape

        query = self.w_q(q) # (B, T, D)
        key = self.w_k(k)   # (B,T,kv_dim)
        value = self.w_v(v) # (B,T,kv_dim)

        #Split into heads
        query = query.reshape(
            B, Tq, self.h, self.d_k
        ).transpose(1, 2)

        key = key.reshape(
            B, Tq, self.kv_heads, self.d_k
        ).transpose(1, 2)

        value = value.reshape(
            B, Tq, self.kv_heads, self.d_k
        ).transpose(1, 2)

        # Append previously cached K/V
        past_len = 0

        if past_key_value is not None:
            past_k, past_v = past_key_value

            if (
                past_k.shape != past_v.shape
                or past_k.shape[:2] != (B, self.kv_heads)
                or past_k.shape[-1] != self.d_k
            ):
                raise ValueError("Invalid cached K/V dimensions")

            past_len = past_k.shape[-2]

            key = torch.cat((past_k, key), dim=-2)
            value = torch.cat((past_v, value), dim=-2)

        Tk = key.shape[-2]

        # Group query heads
        # (B, Hkv, groups, Tq, d)
        query = query.reshape(
            B,
            self.kv_heads,
            self.group_size,
            Tq,
            self.d_k,
        )

        # Shared KV attention scores
        # (B, Hkv, groups, Tq, Tk)
        scores = (
            query @ key.unsqueeze(2).transpose(-2, -1) # the unsqueeze(2) is used for broadcasting to match the shape of query  
        ) / math.sqrt(self.d_k)

        # Causal mask accounting for past tokens
        q_positions = past_len + torch.arange(
            Tq, device=q.device
        )

        k_positions = torch.arange(
            Tk, device=q.device
        )

        # broadcasting adds extra dimension 
        allowed = (
            k_positions[None, :] <= q_positions[:, None] # -> this results in shape [Tq,Tk]
        )[None, None, None, :, :] # -> this results in shape (B, H, Tq, Tk)

        # Optional padding / external attention mask
        if mask is not None:

            if mask.dtype != torch.bool:
                raise ValueError("Attention mask must be boolean")

            if mask.shape[-2:] != (Tq, Tk):
                raise ValueError("Incorrect attention mask shape")

            if mask.ndim == 2:
                mask = mask[None, None, None, :, :]

            elif (
                mask.ndim == 4
                and mask.shape[0] == B
                and mask.shape[1] == 1
            ):
                mask = mask.unsqueeze(2)

            elif (
                mask.ndim == 4
                and mask.shape[:2] == (B, self.h)
            ):
                mask = mask.reshape(
                    B, self.kv_heads, self.group_size, Tq, Tk
                )

            else:
                raise ValueError("Unsupported mask dimensions")

            allowed = allowed & mask

        scores = scores.masked_fill(
            ~allowed, float("-inf")
        )

        # 7. Numerically stable softmax
        weights = torch.softmax(
            scores.float(), dim=-1
        ).to(query.dtype)

        weights = self.dropout(weights)

        # 8. Attention output
        output = weights @ value.unsqueeze(2)

        # 9. Merge query groups into heads
        output = output.reshape(
            B, self.h, Tq, self.d_k
        )

        # (B, Tq, D)
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(B, Tq, self.h * self.d_k)

        output = self.W_o(output)

        # Return per-query-head attention probabilities
        weights = weights.reshape(B, self.h, Tq, Tk)

        if use_cache:
            return output, weights, (key, value)

        return output, weights
