import torch

from src.shakespeare_transformer import (
    TransformerConfig,
    ShakespeareTransformer,
    apply_rope,
    build_causal_mask,
    estimate_flops_by_layer,
    theoretical_parameter_count,
)


def test_rope_preserves_shape_and_changes_values():
    x = torch.randn(2, 4, 8, 12)
    out = apply_rope(x)
    assert out.shape == x.shape
    assert not torch.allclose(out[:, :, 1:], x[:, :, 1:])


def test_causal_mask_blocks_future_tokens():
    mask = build_causal_mask(4, device=torch.device("cpu"))
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 0].item() == 0
    assert torch.isneginf(mask[0, 0, 0, 1]).item()
    assert mask[0, 0, 3, 0].item() == 0


def test_model_forward_shape_and_loss():
    config = TransformerConfig(
        vocab_size=128,
        max_seq_len=16,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )
    model = ShakespeareTransformer(config)
    idx = torch.randint(0, config.vocab_size, (3, 10))
    targets = torch.randint(0, config.vocab_size, (3, 10))
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 10, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)


def test_theoretical_parameter_count_matches_model_parameters():
    config = TransformerConfig(
        vocab_size=64,
        max_seq_len=8,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )
    model = ShakespeareTransformer(config)
    actual = sum(p.numel() for p in model.parameters())
    theoretical = theoretical_parameter_count(config)["total"]
    assert theoretical == actual


def test_flop_estimate_has_expected_layer_entries():
    config = TransformerConfig(vocab_size=100, max_seq_len=32, n_layers=3, n_heads=4, d_model=64)
    table = estimate_flops_by_layer(config, seq_len=16, batch_size=2)
    names = [row["layer"] for row in table]
    assert "token_embedding_lookup" in names
    assert "transformer_block_1" in names
    assert "transformer_block_3" in names
    assert "lm_head" in names
    assert all(row["flops"] >= 0 for row in table)
