import pytest
import torch
from transformers.cache_utils import DynamicCache

from cpp_transformer.models.attention.rope import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)
from cpp_transformer.models.decoder import build_transformer


def make_model(kv_heads=2, backend="sdpa"):
    return build_transformer(
        tgt_vocab_size=101,
        tgt_seq_len=32,
        d_model=32,
        N=3,
        h=4,
        kv_heads=kv_heads,
        attention_backend=backend,
        position_encoding="rope",
        rope_theta=10000.0,
        dropout=0.0,
        d_ff=64,
    )


def test_rope_preserves_norm():
    torch.manual_seed(42)

    rope = RotaryEmbedding(head_dim=8)

    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 2, 6, 8)

    positions = torch.arange(6)

    cos_sin = rope(
        positions,
        dtype=q.dtype,
    )

    q_rot, k_rot = apply_rotary_pos_emb(
        q, k, cos_sin
    )

    torch.testing.assert_close(
        q.norm(dim=-1),
        q_rot.norm(dim=-1),
        atol=1e-6,
        rtol=1e-6,
    )

    torch.testing.assert_close(
        k.norm(dim=-1),
        k_rot.norm(dim=-1),
        atol=1e-6,
        rtol=1e-6,
    )


def test_relative_position_property():
    torch.manual_seed(42)

    rope = RotaryEmbedding(head_dim=8)

    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)

    def rotated_dot(query_pos, key_pos):
        q_freq = rope(
            torch.tensor([query_pos]),
            dtype=q.dtype,
        )
        k_freq = rope(
            torch.tensor([key_pos]),
            dtype=k.dtype,
        )

        q_rot, _ = apply_rotary_pos_emb(q, q, q_freq)
        _, k_rot = apply_rotary_pos_emb(k, k, k_freq)

        return (q_rot * k_rot).sum()

    # Both position pairs have distance 4.
    torch.testing.assert_close(
        rotated_dot(3, 7),
        rotated_dot(0, 4),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_full_vs_cached_logits(kv_heads, backend):
    torch.manual_seed(42)

    model = make_model(kv_heads, backend).eval()

    tokens = torch.randint(
        1, 101, (2, 9)
    )

    with torch.inference_mode():

        mask = model.create_tgt_mask(
            tokens,
            pad_token_id=0,
        )

        full_logits = model(tokens, mask)

        cache = DynamicCache()
        chunks = []

        for start, end in (
            (0, 4),
            (4, 5),
            (5, 7),
            (7, 9),
        ):
            logits, returned_cache = model(
                tokens[:, start:end],
                past_key_values=cache,
                use_cache=True,
            )

            assert returned_cache is cache
            chunks.append(logits)

        cached_logits = torch.cat(
            chunks,
            dim=1,
        )

    torch.testing.assert_close(
        full_logits,
        cached_logits,
        atol=1e-4,
        rtol=1e-4,
    )

    for layer_idx in range(3):
        assert cache.get_seq_length(layer_idx) == 9

        assert cache[layer_idx][0].shape == (
            2, kv_heads, 9, 8
        )


def test_rope_gradients():
    torch.manual_seed(42)

    model = make_model().train()

    tokens = torch.randint(
        1, 101, (2, 9)
    )

    logits = model(tokens)

    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 101),
        tokens[:, 1:].reshape(-1),
    )

    loss.backward()

    for layer in model.decoder.layers:
        attention = layer.self_attention_block

        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.o_proj,
        ):
            assert projection.weight.grad is not None
            assert torch.isfinite(
                projection.weight.grad
            ).all()


def test_bfloat16_frequencies():
    rope = RotaryEmbedding(head_dim=8)
    rope = rope.to(dtype=torch.bfloat16)

    assert rope.inv_freq.dtype == torch.float32

    cos, sin = rope(
        torch.arange(4),
        dtype=torch.bfloat16,
    )

    assert cos.dtype == torch.bfloat16
    assert sin.dtype == torch.bfloat16


def test_position_encoding_is_exclusive():
    model = make_model()

    assert model.tgt_pos is None
    assert model.rotary_emb is not None


def test_invalid_rotary_dimension():
    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim=7)

    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim=8, rotary_dim=9)
