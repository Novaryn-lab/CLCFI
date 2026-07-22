"""CPU-only shape/gradient/leakage smoke test; no dataset or BERT download required."""

import torch

from .config import CLCFIConfig
from .model import CLCFI


def main() -> None:
    torch.manual_seed(7)
    config = CLCFIConfig(
        precomputed_text_dim=12,
        model_dim=16,
        num_heads=4,
        temporal_layers=1,
        cross_layers=1,
        fusion_layers=1,
        prototype_size=4,
        dropout=0.0,
    )
    model = CLCFI(config, audio_dim=5, vision_dim=7)
    batch_size, text_length = 3, 6
    inputs = {
        "text_features": torch.randn(batch_size, text_length, 12),
        "text_padding_mask": torch.tensor([
            [False, False, False, False, False, True],
            [False, False, False, False, True, True],
            [False, False, False, False, False, False],
        ]),
        "audio": torch.randn(batch_size, 8, 5),
        "vision": torch.randn(batch_size, 7, 7),
        "audio_padding_mask": torch.tensor([
            [False, False, False, False, False, False, True, True],
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, False, False, False],
        ]),
        "vision_padding_mask": torch.tensor([
            [False, False, False, False, False, True, True],
            [False, False, False, False, True, True, True],
            [False, False, False, False, False, False, False],
        ]),
        "labels": torch.tensor([[0.8], [-1.0], [0.2]]),
    }
    model.train()
    output = model(**inputs)
    assert output.predictions.shape == (batch_size, 1)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert torch.allclose(output.weight_ta + output.weight_tv, torch.ones(batch_size), atol=1e-6)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    selector_grad = model.cue_selection.ta_distribution.score.weight.grad
    assert selector_grad is not None and torch.isfinite(selector_grad).all()

    model.eval()
    with torch.no_grad():
        first = model(**inputs).predictions
        changed = dict(inputs)
        changed["labels"] = -inputs["labels"]
        second = model(**changed).predictions
    max_difference = (first - second).abs().max().item()
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-5), (
        f"Evaluation prediction leaked ground-truth labels (max diff={max_difference})"
    )
    print(f"label leakage check max diff: {max_difference:.3e}")
    print("CLCFI smoke test passed")


if __name__ == "__main__":
    main()
