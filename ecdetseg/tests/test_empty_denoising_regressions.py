"""Regressions for empty denoising losses and disabled denoising.

Run from ecdetseg/: python -m pytest tests/test_empty_denoising_regressions.py -q
Requires normal EdgeCrafter dependencies; no GPU or dataset required.
"""
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from engine.edgecrafter.criterion import ECCriterion
from engine.edgecrafter.decoder import ECTransformer


def _criterion():
    return ECCriterion(
        matcher=SimpleNamespace(mask_point_sample_ratio=None),
        weight_dict={},
        losses=[],
        num_classes=3,
        alpha=0.2,
        gamma=1.5,
    )


def _case(query_count):
    logits = torch.tensor(
        [[[1.0, -0.4, -0.7], [0.3, 0.8, -0.2]]],
        requires_grad=True,
    )[:, :query_count, :]
    outputs = {
        "pred_logits": logits,
        "pred_boxes": torch.tensor(
            [[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]]
        )[:, :query_count],
    }
    n = 0 if query_count == 0 else 1
    targets = [{
        "labels": torch.zeros((n,), dtype=torch.long),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])[:n],
    }]
    indices = [(torch.arange(n), torch.arange(n))]
    return outputs, targets, indices


@pytest.mark.parametrize("loss_name", ["vfl", "mal"])
def test_zero_queries_produce_finite_raw_loss_and_backward(loss_name):
    outputs, targets, indices = _case(0)
    loss_fn = getattr(_criterion(), "loss_labels_" + loss_name)
    loss = next(iter(loss_fn(
        outputs, targets, indices, num_boxes=1.0,
        values=torch.empty(0),
    ).values()))
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    (gradient,) = torch.autograd.grad(loss, outputs["pred_logits"])
    assert gradient.numel() == 0


@pytest.mark.parametrize("loss_name", ["vfl", "mal"])
def test_nonempty_queries_remain_finite(loss_name):
    outputs, targets, indices = _case(2)
    loss_fn = getattr(_criterion(), "loss_labels_" + loss_name)
    loss = next(iter(loss_fn(
        outputs, targets, indices, num_boxes=2.0,
        values=torch.tensor([0.75]),
    ).values()))
    assert torch.isfinite(loss)
    assert loss.item() > 0
    (gradient,) = torch.autograd.grad(loss, outputs["pred_logits"])
    assert torch.isfinite(gradient).all()


def test_normal_reduction_value_and_gradient_equivalence():
    torch.manual_seed(1)
    x = torch.randn(3, 7, 4, dtype=torch.float64, requires_grad=True)
    old = x.mean(1).sum() * x.shape[1] / 2.5
    new = x.sum() / 2.5
    torch.testing.assert_close(new, old, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(
        torch.autograd.grad(new, x, retain_graph=True)[0],
        torch.autograd.grad(old, x)[0],
        rtol=1e-13, atol=1e-13,
    )


class _StubDecoder(nn.Module):
    def __init__(self, with_masks):
        super().__init__()
        self.with_masks = with_masks
        self.pre_segs = None

    def forward(self, *args, **kwargs):
        layers, batch, queries, classes = 2, 1, 3, 3
        bboxes = torch.zeros(layers, batch, queries, 4)
        logits = torch.zeros(layers, batch, queries, classes)
        corners = torch.zeros(layers, batch, queries, 8)
        refs = torch.zeros(layers, batch, queries, 4)
        masks = torch.zeros(layers, batch, queries, 2, 2) if self.with_masks else None
        pre_bboxes = torch.zeros(batch, queries, 4)
        pre_logits = torch.zeros(batch, queries, classes)
        pre_segs = (
            torch.full((batch, queries, 2, 2), 3.0)
            if self.with_masks else None
        )
        self.pre_segs = pre_segs
        return bboxes, logits, corners, refs, masks, pre_bboxes, pre_logits, pre_segs


def _transformer_without_denoising(with_masks):
    # Exercise the actual ECTransformer.forward without constructing a ViT.
    model = ECTransformer.__new__(ECTransformer)
    nn.Module.__init__(model)
    model.num_denoising = 0
    model.aux_loss = True
    model.query_select_method = "default"
    model.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
    model.reg_scale = nn.Parameter(torch.tensor([4.0]), requires_grad=False)
    model.decoder = _StubDecoder(with_masks)
    for name in ("dec_bbox_head", "dec_score_head", "query_pos_head",
                 "pre_bbox_head", "integral"):
        setattr(model, name, nn.Identity())

    def fake_encoder_input(self, feats):
        return torch.zeros(1, 4, 3), [[2, 2]]

    def fake_decoder_input(self, memory, shapes, denoising_logits, denoising_bbox_unact):
        assert denoising_logits is None and denoising_bbox_unact is None
        return (
            torch.zeros(1, 3, 3), torch.zeros(1, 3, 4),
            [torch.zeros(1, 3, 4)], [torch.zeros(1, 3, 3)],
        )

    model._get_encoder_input = MethodType(fake_encoder_input, model)
    model._get_decoder_input = MethodType(fake_decoder_input, model)
    model.train()
    return model


@pytest.mark.parametrize("with_masks", [False, True])
def test_train_without_denoising_preserves_segmentation_outputs(with_masks):
    model = _transformer_without_denoising(with_masks)
    outputs = model([torch.zeros(1, 3, 2, 2)], targets=None)
    assert "dn_outputs" not in outputs
    assert "dn_meta" not in outputs
    assert "pre_outputs" in outputs
    if with_masks:
        torch.testing.assert_close(
            outputs["pre_outputs"]["pred_masks"], model.decoder.pre_segs
        )
        assert outputs["pred_masks"] is not None
    else:
        assert outputs["pre_outputs"]["pred_masks"] is None
        assert outputs["pred_masks"] is None
