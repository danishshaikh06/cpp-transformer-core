
import math

import pytest
import torch

from cpp_transformer.models.attention.gqa_with_kv import (
    GroupedQueryAttention,
)
from cpp_transformer.models.decoder import (
    MultiHeadAttention,
    build_transformer,
)


def explicit_reference(attn, x, mask):
    """
    Reference using physically repeated KV heads.
    """

    B, T, D = x.shape

    q = attn.w_q(x).reshape(
        B, T, attn.h, attn.d_k
    ).transpose(1, 2)

    k = attn.w_k(x).reshape(
        B, T, attn.kv_heads, attn.d_k
    ).transpose(1, 2)

    v = attn.w_v(x).reshape(
        B, T, attn.kv_heads, attn.d_k
    ).transpose(1, 2)

    k = k.repeat_interleave(
        attn.group_size, dim=1
    )

    v = v.repeat_interleave(
        attn.group_size, dim=1
    )

    scores = (
        q @ k.transpose(-2, -1)
    ) / math.sqrt(attn.d_k)

    scores = scores.masked_fill(
        ~mask, float("-inf")
    )

    weights = torch.softmax(scores, dim=-1)

    output = weights @ v

    output = output.transpose(1, 2).contiguous()
    output = output.reshape(B, T, D)

    return attn.W_o(output)


@pytest.mark.parametrize("kv_heads", [1, 2, 8])
def test_gqa_matches_explicit_reference(kv_heads):

    torch.manual_seed(42)

    attn = GroupedQueryAttention(
        d_model=64,
        h=8,
        kv_heads=kv_heads,
        dropout=0.0,
    ).eval()

    x = torch.randn(2, 7, 64)

    mask = torch.tril(
        torch.ones(7, 7, dtype=torch.bool)
    )[None, None, :, :]

    output, weights = attn(x, x, x, mask)

    expected = explicit_reference(attn, x, mask)

    torch.testing.assert_close(
        output, expected,
        rtol=1e-5, atol=1e-6,
    )

    assert output.shape == (2, 7, 64)
    assert weights.shape == (2, 8, 7, 7)


def test_gqa_matches_existing_mha():

    torch.manual_seed(42)

    mha = MultiHeadAttention(64, 8, 0.0).eval()

    gqa = GroupedQueryAttention(
        64, 8, 8, 0.0
    ).eval()

    # Identical weights, because Hkv == Hq
    gqa.load_state_dict(mha.state_dict())

    x = torch.randn(2, 6, 64)

    mask = torch.tril(
        torch.ones(6, 6, dtype=torch.bool)
    )[None, None, :, :]

    mha_out, _ = mha(x, x, x, mask)
    gqa_out, _ = gqa(x, x, x, mask)

    torch.testing.assert_close(
        mha_out, gqa_out,
        rtol=1e-5, atol=1e-6,
    )


def test_gqa_gradients():

    torch.manual_seed(42)

    attn = GroupedQueryAttention(
        64, 8, 2, 0.0
    )

    x = torch.randn(2, 6, 64)

    output, _ = attn(x, x, x)

    loss = output.square().mean()

    loss.backward()

    for projection in (
        attn.w_q,
        attn.w_k,
        attn.w_v,
        attn.W_o,
    ):
        assert projection.weight.grad is not None
        assert torch.isfinite(
            projection.weight.grad
        ).all()


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_full_model_matches_cached_decode(kv_heads):

    torch.manual_seed(42)

    model = build_transformer(
        tgt_vocab_size=101,
        tgt_seq_len=24,
        d_model=32,
        N=2,
        h=4,
        kv_heads=kv_heads,
        dropout=0.0,
        d_ff=64,
    ).eval()

    tokens = torch.randint(
        1, 101, (2, 9)
    )

    # Full-sequence inference
    full_mask = model.create_tgt_mask(
        tokens, pad_token_id=0
    )

    full_logits = model(tokens, full_mask)

    # Prefill first four tokens
    first_logits, cache = model(
        tokens[:, :4],
        use_cache=True,
    )

    outputs = [first_logits]

    # Decode remaining tokens individually
    for i in range(4, 9):

        step_logits, cache = model(
            tokens[:, i:i + 1],
            past_key_values=cache,
            use_cache=True,
        )

        outputs.append(step_logits)

    cached_logits = torch.cat(
        outputs, dim=1
    )

    torch.testing.assert_close(
        full_logits,
        cached_logits,
        rtol=1e-5,
        atol=1e-6,
    )

    # Cache: (batch, KV heads, tokens, head dimension)
    assert cache[0][0].shape == (
        2, kv_heads, 9, 8
    )


def test_gqa_training_step():

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
        1, 101, (2, 9)
    )

    mask = model.create_tgt_mask(
        tokens, pad_token_id=0
    )

    logits = model(tokens, mask)

    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 101),
        tokens[:, 1:].reshape(-1),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    optimizer.zero_grad()
    loss.backward()

    assert (
        model.decoder.layers[0]
        .self_attention_block
        .w_k.weight.grad.abs().sum() > 0
    )

    optimizer.step()
