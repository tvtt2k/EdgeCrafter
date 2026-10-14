"""CPU smoke test of real ECDet-S training with synthetic images and targets.

No COCO dataset, pretrained weights, CUDA, or distributed launch required.
Run from ecdetseg/: python -m pytest -v tests/test_ecdet_full_train_smoke.py
"""

import pytest
import torch

from engine.core import YAMLConfig


@pytest.fixture(scope="module")
def training_setup():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    cfg = YAMLConfig("configs/ecdet/ecdet_s.yml")

    # Preserve the real ECDet-S backbone, encoder, transformer and criterion.
    # Reduce resolution/decoder workload so a hosted CPU runner can execute it.
    cfg.yaml_cfg["eval_spatial_size"] = [128, 128]
    cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    cfg.yaml_cfg["ECTransformer"]["num_queries"] = 32
    cfg.yaml_cfg["ECTransformer"]["num_layers"] = 2
    cfg.yaml_cfg["ECTransformer"]["num_denoising"] = 8

    model = cfg.model.train()
    criterion = cfg.criterion.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    return model, criterion, optimizer


@pytest.mark.parametrize(
    "num_denoising,num_objects",
    [(8, 1), (0, 1), (8, 0)],
    ids=["normal", "denoising-disabled", "empty-targets"],
)
def test_real_model_forward_backward_and_optimizer(
    training_setup, num_denoising, num_objects
):
    model, criterion, optimizer = training_setup
    model.decoder.num_denoising = num_denoising
    targets = [{
        "labels": torch.zeros(num_objects, dtype=torch.long),
        "boxes": torch.tensor(
            [[0.5, 0.5, 0.2, 0.2]] * num_objects,
            dtype=torch.float32,
        ).reshape(-1, 4),
    }]

    optimizer.zero_grad(set_to_none=True)
    images = torch.randn(1, 3, 128, 128)
    outputs = model(images, targets=targets)
    assert "pred_boxes" in outputs
    assert "pred_logits" in outputs
    assert ("dn_outputs" in outputs) == (num_denoising > 0)

    losses = criterion(outputs, targets)
    assert losses, "Criterion produced no losses"
    for name, value in losses.items():
        assert torch.isfinite(value).all(), f"{name}: {value}"

    total_loss = sum(losses.values())
    assert torch.isfinite(total_loss), f"Nonfinite total loss: {total_loss}"
    total_loss.backward()

    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients, "No gradients computed"
    assert all(torch.isfinite(g).all() for g in gradients), "Nonfinite gradient"
    optimizer.step()
