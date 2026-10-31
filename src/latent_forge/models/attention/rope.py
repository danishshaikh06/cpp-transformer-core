
from __future__ import annotations

import math

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """
    Llama-style Rotary Position Embedding.

    Supports:
        - 1D and batched position IDs
        - Full or partial rotary dimensions
        - FP32 frequency calculations
        - FP16/BF16 model execution
    """

    def __init__(
        self,
        head_dim: int,
        theta: float = 10000.0,
        rotary_dim: int | None = None,
    ):
        super().__init__()

        if (
            isinstance(head_dim, bool)
            or not isinstance(head_dim, int)
            or head_dim <= 0
        ):
            raise ValueError("head_dim must be a positive integer")

        rotary_dim = head_dim if rotary_dim is None else rotary_dim

        if (
            isinstance(rotary_dim, bool)
            or not isinstance(rotary_dim, int)
            or rotary_dim < 2
            or rotary_dim > head_dim
            or rotary_dim % 2 != 0
        ):
            raise ValueError(
                "rotary_dim must be even and between 2 and head_dim"
            )

        if (
            isinstance(theta, bool)
            or not isinstance(theta, (float, int))
            or not math.isfinite(theta)
            or theta <= 1
        ):
            raise ValueError("theta must be finite and > 1")

        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.theta = float(theta)

        inv_freq = 1.0 / (
            self.theta ** (
                torch.arange(
                    0, rotary_dim, 2, dtype=torch.float32
                ) / rotary_dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    def _apply(self, fn):
        """
        Intercepts PyTorch's internal tensor transformation (e.g., .half(), 
        .bfloat16(), .to(device)) to ensure inverse frequencies remain in FP32.
        
        Maintaining FP32 precision for `inv_freq` is critical to prevent 
        catastrophic underflow, overflow, or rounding errors during high-precision 
        trigonometric calculations across long context windows.
        """
        super()._apply(fn)
        self.inv_freq = self.inv_freq.float()
        return self

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Accept (T,) and (B, T).
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

        if x.ndim not in (3, 4):
            raise ValueError("x must be (B,T,D) or (B,H,T,D)")

        if (
            position_ids.ndim != 2
            or position_ids.dtype != torch.long
            or position_ids.device != x.device
            or position_ids.shape[0] not in (1, x.shape[0])
            or position_ids.shape[1] != x.shape[-2]
        ):
            raise ValueError(
                "Invalid position_ids shape, dtype or device"
            )

        # (B, T, 1) * (1, 1, rotary_dim/2)
        freqs = (
            position_ids.float().unsqueeze(-1)
            * self.inv_freq[None, None, :]
        )

        # Llama-style half-split convention
        emb = torch.cat((freqs, freqs), dim=-1)

        target_dtype = kwargs.get("dtype", x.dtype)
        
        cos = emb.cos().to(dtype=target_dtype)
        sin = emb.sin().to(dtype=target_dtype)

        return cos, sin
    
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Using complex number technique that if you wnat to rotate any vector/point then swap the two points and add negative sign at the start of the swap co-ordinates
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError("Rotary dimension must be even")

    x1, x2 = x.chunk(2, dim=-1)

    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Q:   (B, Hq, T, head_dim)
    K:   (B, Hkv, T, head_dim)

    cos/sin: (B or 1, T, rotary_dim)
    """

    if (
        q.ndim != 4
        or k.ndim != 4
        or q.shape[0] != k.shape[0]
        or q.shape[-2:] != k.shape[-2:]
    ):
        raise ValueError("Invalid Q/K shapes")

    rotary_dim = cos.shape[-1]

    if (
        cos.ndim != 3
        or cos.shape != sin.shape
        or cos.shape[0] not in (1, q.shape[0])
        or cos.shape[1] != q.shape[-2]
        or rotary_dim < 2
        or rotary_dim % 2 != 0
        or rotary_dim > q.shape[-1]
    ):
        raise ValueError("Invalid RoPE frequencies")

    if (
        cos.device != q.device
        or sin.device != q.device
        or k.device != q.device
    ):
        raise ValueError("Q, K, cos and sin must share a device")

    # (B, T, D) -> (B, 1, T, D)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def apply(x: torch.Tensor) -> torch.Tensor:

        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]

        # Rotating it partially that is 50 percent and rest 50 percent kept to ensure semantic meaning 
        # Llama rotates 100%, but models like GPT-NeoX, Phi, and some variants of Qwen rotate only a fraction, like 25% or 40%.
        rotated = (
            x_rot * cos.to(dtype=x.dtype)
            + rotate_half(x_rot) * sin.to(dtype=x.dtype)
        )

        if rotary_dim == x.shape[-1]:
            return rotated

        return torch.cat((rotated, x_pass), dim=-1)

    return apply(q), apply(k)
