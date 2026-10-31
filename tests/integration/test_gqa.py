
import pytest
import torch

from transformers.cache_utils import DynamicCache

from latent_forge.models.decoder import build_transformer


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_cached_logits_match_full_model(kv_heads, backend):

    torch.manual_seed(42)

    model = build_transformer(
        tgt_vocab_size=101,
        tgt_seq_len=32,
        d_model=32,
        N=3,
        h=4,
        kv_heads=kv_heads,
        attention_backend=backend,
        dropout=0.0,
        d_ff=64,
    ).eval()

    tokens = torch.randint(
        1, 101, (2, 9)
    )

    with torch.inference_mode():

        # Full-sequence inference
        mask = model.create_tgt_mask(
            tokens,
            pad_token_id=0,
        )

        full_logits = model(tokens, mask)

        # Cached inference, same token sequence.
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

    assert cache.get_seq_length(0) == 9
    assert cache.get_seq_length(1) == 9
    assert cache.get_seq_length(2) == 9

    # Batch=2, KV heads, tokens=9, head_dim=8
    assert cache[0][0].shape == (
        2, kv_heads, 9, 8
    )


def test_full_model_backward():

    torch.manual_seed(42)

    model = build_transformer(
        tgt_vocab_size=101,
        tgt_seq_len=16,
        d_model=32,
        N=2,
        h=4,
        kv_heads=2,
        dropout=0.0,
        d_ff=64,
    )

    tokens = torch.randint(
        1, 101, (2, 8)
    )

    logits = model(tokens)

    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 101),
        tokens[:, 1:].reshape(-1),
    )

    loss.backward()

    for layer in model.decoder.layers:
        attention = layer.self_attention_block

        assert attention.k_proj.weight.grad is not None
        assert attention.v_proj.weight.grad is not None

        assert torch.isfinite(
            attention.k_proj.weight.grad
        ).all()
