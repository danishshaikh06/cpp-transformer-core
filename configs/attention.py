from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


AttentionBackend = Literal["eager", "sdpa"]


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    """Configuration class for multi-head attention and grouped-query attention.

    Attributes:
        d_model: The hidden dimension of the model.
        num_attention_heads: The number of query attention heads.
        num_key_value_heads: The number of key-value attention heads (for GQA/MQA).
        dropout: The dropout probability, must be in [0, 1).
        backend: The attention backend to use ("eager" or "sdpa").
    """
    d_model: int
    num_attention_heads: int
    num_key_value_heads: int
    dropout: float = 0.0
    backend: AttentionBackend = "sdpa"

    def __post_init__(self) -> None:
        """Validates configuration parameters upon initialization."""
        for name in (
            "d_model",
            "num_attention_heads",
            "num_key_value_heads",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer"
                )

        if self.d_model % self.num_attention_heads:
            raise ValueError(
                "d_model must be divisible by num_attention_heads"
            )

        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible "
                "by num_key_value_heads"
            )

        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (float, int))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError(
                "dropout must be finite and in [0, 1)"
            )

        if self.backend not in ("eager", "sdpa"):
            raise ValueError(
                "backend must be 'eager' or 'sdpa'"
            )

    @property
    def head_dim(self) -> int:
        """Returns the dimension size of each attention head."""
        return self.d_model // self.num_attention_heads

    @property
    def group_size(self) -> int:
        """Returns the number of query heads sharing each key-value head."""
        return (
            self.num_attention_heads
            // self.num_key_value_heads
        )