import pytest
import torch
from transformers.cache_utils import DynamicCache

from latent_forge.configs.mla import MLAConfig
from latent_forge.models.attention.mla import MultiHeadLatentAttention


def make_module(mode="naive", q_lora_rank=32, layer_idx=0):
    cfg = MLAConfig(
        hidden_size=64,
        num_attention_heads=4,
        kv_lora_rank=24,
        qk_nope_head_dim=12,
        qk_rope_head_dim=8,
        v_head_dim=16,
        q_lora_rank=q_lora_rank,
        attention_dropout=0,
        implementation=mode,
    )
    return MultiHeadLatentAttention(cfg, layer_idx=layer_idx)


@pytest.mark.parametrize("q_rank", [None, 32])
def test_naive_and_absorbed_match(q_rank):
    torch.manual_seed(7)
    m = make_module(q_lora_rank=q_rank).eval()
    x = torch.randn(2, 9, 64)
    mask = torch.tensor([[True]*9, [False, False] + [True]*7])
    with torch.inference_mode():
        naive = m(x, attention_mask=mask, implementation="naive")
        absorbed = m(x, attention_mask=mask, implementation="absorbed")
    torch.testing.assert_close(naive, absorbed, rtol=1e-5, atol=1e-6)
    assert naive.shape == (2, 9, 64)


@pytest.mark.parametrize("mode", ["naive", "absorbed"])
@pytest.mark.parametrize("q_rank", [None, 32])
def test_cached_matches_full(mode, q_rank):
    torch.manual_seed(8)
    m = make_module(q_lora_rank=q_rank).eval()
    x = torch.randn(2, 11, 64)
    with torch.no_grad():
        full = m(x, implementation=mode)
        cache = DynamicCache()
        pieces = []
        for lo, hi in [(0, 4), (4, 5), (5, 8), (8, 11)]:
            pieces.append(
                m(x[:, lo:hi], past_key_values=cache,
                  use_cache=True, implementation=mode)
            )
        cached = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full, cached, rtol=1e-5, atol=1e-6)
    assert cache.get_seq_length(0) == 11
    assert cache[0][0].shape == (2, 1, 11, 24)
    assert cache[0][1].shape == (2, 1, 11, 8)


def test_naive_prefill_then_absorbed_decode():
    torch.manual_seed(9)
    m = make_module().eval()
    x = torch.randn(2, 9, 64)
    with torch.no_grad():
        expected = m(x, implementation="naive")
        cache = DynamicCache()
        first = m(x[:, :5], past_key_values=cache, use_cache=True, implementation="naive")
        last = m(x[:, 5:], past_key_values=cache, use_cache=True, implementation="absorbed")
    torch.testing.assert_close(torch.cat([first, last], dim=1), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", ["naive", "absorbed"])
def test_backward(mode):
    torch.manual_seed(10)
    m = make_module().train()
    x = torch.randn(2, 7, 64, requires_grad=True)
    out = m(x, implementation=mode)
    loss = out.square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for n, p in m.named_parameters():
        assert p.grad is not None, n
        assert torch.isfinite(p.grad).all(), n


@pytest.mark.parametrize("mode", ["naive", "absorbed"])
def test_fully_masked_row(mode):
    torch.manual_seed(11)
    m = make_module().eval()
    x = torch.randn(2, 5, 64)
    mask = torch.tensor([[False]*5, [True]*5])
    with torch.no_grad():
        out = m(x, attention_mask=mask, implementation=mode)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[0], torch.zeros_like(out[0]))


def test_invalid_mask_does_not_mutate_cache():
    m = make_module().eval()
    cache = DynamicCache()
    with pytest.raises(ValueError):
        m(torch.randn(1, 3, 64), attention_mask=torch.ones(1, 4, dtype=torch.bool),
          past_key_values=cache, use_cache=True)
    assert cache.get_seq_length(0) == 0


def test_independent_layer_cache():
    m0, m1 = make_module(layer_idx=0), make_module(layer_idx=1)
    cache = DynamicCache()
    x = torch.randn(1, 4, 64)
    m0(x, past_key_values=cache, use_cache=True)
    m1(x, past_key_values=cache, use_cache=True)
    assert cache.get_seq_length(0) == cache.get_seq_length(1) == 4


def test_config_validation():
    with pytest.raises(ValueError):
        MLAConfig(hidden_size=64, num_attention_heads=4, kv_lora_rank=24,
                  qk_nope_head_dim=12, qk_rope_head_dim=7, v_head_dim=16)


def test_naive_absorbed_gradients_match():
    torch.manual_seed(15)
    a, b = make_module().train(), make_module().train()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 6, 64)
    # Use same deterministic forward; dropout is disabled in configuration.
    ya = a(x, implementation="naive").square().mean()
    yb = b(x, implementation="absorbed").square().mean()
    ya.backward()
    yb.backward()
    for (name_a, pa), (name_b, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(pa.grad, pb.grad, atol=2e-6, rtol=2e-5)


def test_request_cache_isolation():
    m = make_module().eval()
    a, b = DynamicCache(), DynamicCache()
    m(torch.randn(1, 2, 64), past_key_values=a, use_cache=True)
    m(torch.randn(1, 5, 64), past_key_values=b, use_cache=True)
    assert a.get_seq_length(0) == 2
    assert b.get_seq_length(0) == 5
