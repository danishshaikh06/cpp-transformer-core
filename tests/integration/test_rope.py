
import pytest
import torch
from transformers.cache_utils import DynamicCache

from cpp_transformer.models.decoder import build_transformer


def create_model(
    kv_heads: int = 2,
    backend: str = "sdpa",
):
    return build_transformer(
        tgt_vocab_size=128,
        tgt_seq_len=64,
        d_model=64,
        N=3,
        h=8,
        kv_heads=kv_heads,
        dropout=0.0,
        d_ff=128,
        attention_backend=backend,
        position_encoding="rope",
        rope_theta=10000.0,
    )


@pytest.mark.parametrize("kv_heads", [1, 2, 8])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_full_vs_cached(kv_heads, backend):

    torch.manual_seed(42)

    model = create_model(
        kv_heads=kv_heads,
        backend=backend,
    ).eval()

    tokens = torch.randint(1, 128, (2, 12))

    with torch.inference_mode():

        full_logits = model(tokens)

        cache = DynamicCache()
        chunks = []

        for start, end in (
            (0, 4),
            (4, 5),
            (5, 8),
            (8, 12),
        ):
            logits, returned_cache = model(
                tokens[:, start:end],
                past_key_values=cache,
                use_cache=True,
            )

            assert returned_cache is cache

            chunks.append(logits)

        cached_logits = torch.cat(chunks, dim=1)

    torch.testing.assert_close(
        full_logits,
        cached_logits,
        atol=1e-4,
        rtol=1e-4,
    )

    for layer_idx in range(3):

        assert cache.get_seq_length(layer_idx) == 12

        assert cache[layer_idx][0].shape == (
            2, kv_heads, 12, 8
        )


def test_rope_gradients():

    torch.manual_seed(42)

    model = create_model().train()

    tokens = torch.randint(1, 128, (2, 10))

    logits = model(tokens)

    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 128),
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
            grad = projection.weight.grad

            assert grad is not None
            assert torch.isfinite(grad).all()


def test_rope_and_sinusoidal_are_exclusive():

    rope_model = create_model()

    assert rope_model.rotary_emb is not None
    assert rope_model.tgt_pos is None

    legacy_model = build_transformer(
        tgt_vocab_size=128,
        tgt_seq_len=64,
        d_model=64,
        N=2,
        h=8,
        kv_heads=2,
        dropout=0.0,
        d_ff=128,
        position_encoding="sinusoidal",
    )

    assert legacy_model.rotary_emb is None
    assert legacy_model.tgt_pos is not None


def test_rotary_configuration_validation():

    with pytest.raises(ValueError):
        build_transformer(
            tgt_vocab_size=128,
            tgt_seq_len=64,
            d_model=64,
            N=2,
            h=8,
            dropout=0.0,
            kv_heads=2,
            position_encoding="rope",
            rotary_dim=7,
        )
