from __future__ import annotations

import pytest
import torch

from transformers.cache_utils import DynamicCache

from cpp_transformer.configs.attention import AttentionConfig
from cpp_transformer.models.attention.gqa import (
    GroupedQueryAttention,
)


def make_attention(
    kv_heads: int,
    backend: str = "sdpa",
    layer_idx: int = 0,
):
    """Creates and returns a GroupedQueryAttention module for testing."""
    return GroupedQueryAttention(
        AttentionConfig(
            d_model=64,
            num_attention_heads=8,
            num_key_value_heads=kv_heads,
            dropout=0.0,
            backend=backend,
        ),
        layer_idx=layer_idx,
    ).eval()


# 1. SDPA must match the mathematical reference.
@pytest.mark.parametrize("kv_heads", [1, 2, 8])
def test_sdpa_matches_reference(kv_heads):
    """Tests that the SDPA attention backend produces outputs matching the eager reference implementation."""
    torch.manual_seed(42)

    reference = make_attention(kv_heads, "eager")
    optimized = make_attention(kv_heads, "sdpa")

    optimized.load_state_dict(reference.state_dict())

    x = torch.randn(2, 6, 64)

    padding_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    for mask in (None, padding_mask):
        expected = reference(
            x, attention_mask=mask
        )

        actual = optimized(
            x, attention_mask=mask
        )

        torch.testing.assert_close(
            actual,
            expected,
            atol=1e-5,
            rtol=1e-5,
        )


# 2. Cached decode must match full-sequence attention.
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("kv_heads", [1, 2, 8])
def test_cached_equals_full(backend, kv_heads):
    """Tests that attention with cached decoding matches full-sequence forward pass outputs."""
    torch.manual_seed(42)

    attention = make_attention(
        kv_heads, backend
    )

    x = torch.randn(2, 7, 64)

    full_output = attention(x)

    cache = DynamicCache()
    outputs = []

    # Prefill 3 tokens, decode 1, then process 3.
    for start, end in ((0, 3), (3, 4), (4, 7)):

        output = attention(
            x[:, start:end],
            past_key_values=cache,
            cache_position=torch.arange(start, end),
            use_cache=True,
        )

        outputs.append(output)

    cached_output = torch.cat(outputs, dim=1)

    torch.testing.assert_close(
        full_output,
        cached_output,
        atol=1e-5,
        rtol=1e-5,
    )

    assert cache.get_seq_length(0) == 7

    assert cache[0][0].shape == (
        2, kv_heads, 7, 8
    )


# 3. Both backends must support backpropagation.
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_backward(backend):
    """Tests that both eager and SDPA attention backends support valid backpropagation."""
    attention = GroupedQueryAttention(
        AttentionConfig(
            d_model=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            backend=backend,
        ),
        layer_idx=0,
    )

    x = torch.randn(
        2, 5, 32, requires_grad=True
    )

    output = attention(x)

    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in attention.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


# 4. Fully masked rows must not produce NaNs.
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_fully_masked_rows(backend):
    """Tests that fully masked rows do not introduce NaN values and output zeros correctly."""
    attention = make_attention(
        kv_heads=2,
        backend=backend,
    )

    x = torch.randn(2, 3, 64)

    mask = torch.tensor(
        [
            [False, False, False],
            [True, True, True],
        ]
    )

    output = attention(
        x, attention_mask=mask
    )

    assert torch.isfinite(output).all()

    torch.testing.assert_close(
        output[0],
        torch.zeros_like(output[0]),
    )


# 5. Invalid inputs must not corrupt the cache.
def test_invalid_mask_preserves_cache():
    """Tests that providing an invalid attention mask preserves the cache state without corruption."""
    attention = make_attention(2)
    cache = DynamicCache()

    with pytest.raises(ValueError):
        attention(
            torch.randn(1, 4, 64),
            attention_mask=torch.ones(
                1, 99, dtype=torch.bool
            ),
            past_key_values=cache,
            use_cache=True,
        )

    assert cache.get_seq_length(0) == 0


# 6. Each Transformer layer must own its cache entry.
def test_layer_cache_isolation():
    """Tests that each Transformer layer properly isolates and manages its own cache entry."""
    first = make_attention(2, layer_idx=0)
    second = make_attention(2, layer_idx=1)

    cache = DynamicCache()
    x = torch.randn(1, 4, 64)

    first(x, past_key_values=cache, use_cache=True)
    second(x, past_key_values=cache, use_cache=True)

    assert cache.get_seq_length(0) == 4
    assert cache.get_seq_length(1) == 4

    first(
        x[:, :1],
        past_key_values=cache,
        use_cache=True,
    )

    assert cache.get_seq_length(0) == 5
    assert cache.get_seq_length(1) == 4


# 7. Requests must not share cache state.
def test_request_cache_isolation():
    """Tests that different requests maintain independent and isolated cache states."""
    attention = make_attention(2)

    cache_a = DynamicCache()
    cache_b = DynamicCache()

    attention(
        torch.randn(1, 5, 64),
        past_key_values=cache_a,
        use_cache=True,
    )

    attention(
        torch.randn(1, 2, 64),
        past_key_values=cache_b,
        use_cache=True,
    )

    assert cache_a.get_seq_length(0) == 5
    assert cache_b.get_seq_length(0) == 2


# 8. Invalid cache positions must be rejected.
def test_invalid_cache_position():
    """Tests that providing invalid cache positions results in a raised ValueError."""
    attention = make_attention(2)
    cache = DynamicCache()

    with pytest.raises(ValueError):
        attention(
            torch.randn(1, 3, 64),
            past_key_values=cache,
            cache_position=torch.tensor([0, 1]),
            use_cache=True,
        )

    assert cache.get_seq_length(0) == 0


# 9. Invalid model configurations must fail early.
def test_configuration_validation():
    """Tests that invalid AttentionConfig parameters trigger early validation errors."""
    invalid = [
        (31, 8, 2),
        (32, 6, 2),
        (32, 4, 3),
        (32, 4, 0),
    ]

    for d_model, heads, kv_heads in invalid:
        with pytest.raises(ValueError):
            AttentionConfig(
                d_model=d_model,
                num_attention_heads=heads,
                num_key_value_heads=kv_heads,
            )

    with pytest.raises(ValueError):
        AttentionConfig(
            d_model=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            dropout=1.5,
        )