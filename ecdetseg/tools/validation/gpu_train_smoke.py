"""Synthetic end-to-end ECDet-S CUDA/AMP and 2-rank NCCL validation.

From ecdetseg/:
  python tools/validation/gpu_train_smoke.py --mode single --precision both
  torchrun --standalone --nproc_per_node=2 tools/validation/gpu_train_smoke.py --mode ddp --precision both

CPU-only checks (Gloo for DDP):
  python tools/validation/gpu_train_smoke.py --mode single --device cpu --precision fp32
  torchrun --standalone --nproc_per_node=2 tools/validation/gpu_train_smoke.py --mode ddp --device cpu --precision fp32

All samples and annotations are synthetic. Nothing is downloaded or uploaded.
"""
import argparse
import contextlib
import math
import os
import sys
from datetime import timedelta
from pathlib import Path

# Support launching this file directly from tools/validation/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from engine.core import YAMLConfig


def build_model(device, full_config, num_denoising):
    # Identical initialization on all ranks; DDP additionally synchronizes params.
    torch.manual_seed(31415)
    cfg = YAMLConfig("configs/ecdet/ecdet_s.yml")
    cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True

    if not full_config:
        cfg.yaml_cfg["eval_spatial_size"] = [128, 128]
        cfg.yaml_cfg["ECTransformer"]["num_queries"] = 32
        cfg.yaml_cfg["ECTransformer"]["num_layers"] = 2

    cfg.yaml_cfg["ECTransformer"]["num_denoising"] = num_denoising
    model = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    return model, criterion, optimizer


def targets_for(num_objects, device):
    return [{
        "labels": torch.zeros(num_objects, dtype=torch.long, device=device),
        "boxes": torch.tensor(
            [[0.5, 0.5, 0.2, 0.2]] * num_objects,
            dtype=torch.float32,
            device=device,
        ).reshape(-1, 4),
    }]


def step(model, criterion, optimizer, scaler, device, num_objects, amp, size,
         rank, step_id):
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(11000 + step_id * 17 + rank)
    image = torch.randn(1, 3, size, size, device=device)
    targets = targets_for(num_objects, device)

    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp else contextlib.nullcontext()
    )
    with ctx:
        outputs = model(image, targets=targets)

    meta = outputs.get("dn_meta")
    dn_groups = meta["dn_num_group"] if meta is not None else None
    print(f"[rank={rank} step={step_id}] forward OK, objects={num_objects}, "
          f"dn_groups={dn_groups}", flush=True)

    # Criterion normally executes outside autocast in the training pipeline.
    losses = criterion(outputs, targets)
    assert losses, "Criterion returned no losses"
    keys = sorted(losses)

    if dist.is_available() and dist.is_initialized():
        per_rank_keys = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(per_rank_keys, keys)
        assert all(x == keys for x in per_rank_keys), (
            f"Loss dictionary keys differ across ranks: {per_rank_keys}"
        )

    for name, value in losses.items():
        assert torch.isfinite(value).all().item(), (
            f"Rank {rank}: nonfinite {name}={value}"
        )
    total = sum(losses.values())
    assert torch.isfinite(total).all().item(), f"nonfinite total loss: {total}"

    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    grads = [(name, p.grad) for name, p in model.named_parameters()
             if p.requires_grad and p.grad is not None]
    assert grads, "No trainable parameter received a gradient"
    for name, grad in grads:
        assert torch.isfinite(grad).all().item(), (
            f"Rank {rank}: nonfinite gradient in {name}"
        )

    missing = [name for name, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    scaler.step(optimizer)
    scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"[rank={rank} step={step_id}] PASS: loss={total.item():.6f}, "
          f"grads={len(grads)}, missing_grads={len(missing)}, amp={amp}",
          flush=True)


def precisions(choice):
    return [False, True] if choice == "both" else [choice == "amp"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "ddp"], default="single")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=["fp32", "amp", "both"], default="both")
    parser.add_argument("--full-config", action="store_true",
                        help="Use standard 640px, 300 queries, and 4 decoder layers")
    parser.add_argument("--steps", type=int, default=6,
                        help="Maximum number of DDP iterations")
    args = parser.parse_args()

    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA GPU is not available"
    if args.device == "cpu":
        assert args.precision == "fp32", "AMP test requires CUDA"
    torch.set_num_threads(2)

    rank = 0
    if args.mode == "ddp":
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        assert world_size == 2, "DDP scenarios require exactly 2 ranks"
        if args.device == "cuda":
            assert torch.cuda.device_count() >= 2, "Two visible CUDA GPUs required"
            torch.cuda.set_device(local_rank)
        device = torch.device(
            f"cuda:{local_rank}" if args.device == "cuda" else "cpu"
        )
        dist.init_process_group(
            backend="nccl" if args.device == "cuda" else "gloo",
            timeout=timedelta(seconds=150),
        )
    else:
        device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
        if args.device == "cuda":
            torch.cuda.set_device(device)

    size = 640 if args.full_config else 128
    try:
        if args.mode == "single":
            scenarios = [("normal", 8, 1), ("denoising-off", 0, 1),
                         ("empty-targets", 8, 0)]
            for amp in precisions(args.precision):
                for i, (label, dn, count) in enumerate(scenarios):
                    print(f"[single] scenario={label}, amp={amp}, size={size}",
                          flush=True)
                    model, criterion, optimizer = build_model(
                        device, args.full_config,
                        num_denoising=(100 if args.full_config and dn else dn),
                    )
                    scaler = torch.amp.GradScaler("cuda", enabled=amp)
                    step(model, criterion, optimizer, scaler, device,
                         count, amp, size, rank, i)
                    del model, criterion, optimizer, scaler

        else:
            scenarios = [(1, 1), (0, 1), (1, 0), (0, 0), (1, 1), (0, 1)]
            for amp in precisions(args.precision):
                model, criterion, optimizer = build_model(
                    device, args.full_config,
                    num_denoising=100 if args.full_config else 8,
                )
                model = DDP(
                    model,
                    device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=False,
                )
                scaler = torch.amp.GradScaler("cuda", enabled=amp)
                for i, counts in enumerate(scenarios[:args.steps]):
                    step(model, criterion, optimizer, scaler, device,
                         counts[rank], amp, size, rank, i)
                dist.barrier()
                del model, criterion, optimizer, scaler

        if rank == 0:
            print("ALL GPU VALIDATION SCENARIOS PASSED"
                  if args.device == "cuda" else "ALL CPU CHECK SCENARIOS PASSED",
                  flush=True)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
