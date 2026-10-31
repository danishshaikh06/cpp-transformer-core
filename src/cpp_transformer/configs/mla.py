"""Configuration for DeepSeek-style Multi-Head Latent Attention."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

MLAImplementation = Literal["naive", "absorbed"]


@dataclass(frozen=True, slots=True)
class MLAConfig:
    hidden_size: int
    num_attention_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    q_lora_rank: int | None = None
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    implementation: MLAImplementation = "naive"

    def __post_init__(self) -> None:
        dims = (
            "hidden_size", "num_attention_heads", "kv_lora_rank",
            "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
        )
        if self.q_lora_rank is not None:
            dims += ("q_lora_rank",)
        for name in dims:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be even")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 1:
            raise ValueError("rope_theta must be finite and > 1")
        if not math.isfinite(self.attention_dropout) or not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.implementation not in ("naive", "absorbed"):
            raise ValueError("implementation must be 'naive' or 'absorbed'")

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def cached_elements_per_token(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim
