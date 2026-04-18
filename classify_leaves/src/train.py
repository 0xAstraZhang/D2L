import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision.models as models
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the leaves classifier with single-process or DDP.",
    )
    parser.add_argument("--data-dir", default="data", help="Path to the dataset directory.")
    parser.add_argument("--output-dir", default="model", help="Directory to save checkpoints and plots.")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-process batch size. Under DDP this is the batch size on each GPU.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=35, help="Maximum training epochs.")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for stratified K-fold.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count per process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for reproducibility.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "none"], help="Learning-rate scheduler.")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for cosine annealing.")
    return parser.parse_args()  


def _create_model(num_classes: int) -> nn.Module:
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(num_classes: int, context: utils.DistributedContext) -> nn.Module:
    if not context.distributed:
        return _create_model(num_classes)

    # Let rank0 populate the pretrained-weight cache first, then let the other ranks load it.
    if utils.is_main_process(context):
        model = _create_model(num_classes)
        utils.barrier_if_distributed(context)
    else:
        utils.barrier_if_distributed(context)
        model = _create_model(num_classes)

    utils.barrier_if_distributed(context)
    return model


def run_fold(
    args: argparse.Namespace,
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    context: utils.DistributedContext,
    train_dataset: utils.LeavesDataset,
    val_dataset: utils.LeavesDataset,
    num_classes: int,
    output_dir: Path,
) -> Dict[str, List[float]]:
    if utils.is_main_process(context):
        print(f"\n===== Fold {fold + 1}/{args.k_folds} =====")

    utils.seed_everything(args.seed + fold)
    model = build_model(num_classes, context)
    model.to(context.device)

    if context.distributed:
        model = DDP(model, device_ids=[context.local_rank], output_device=context.local_rank)

    loss_fn = nn.CrossEntropyLoss().to(context.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )

    train_loader, val_loader, train_sampler = utils.build_fold_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_context=context,
        seed=args.seed + fold,
    )

    if utils.is_main_process(context):
        val_size = len(val_idx)
        print(
            f"训练集大小: {len(train_idx)}, 验证集大小: {val_size}, "
            f"device: {context.device}, world_size: {context.world_size}"
        )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(args.epochs):
        epoch_start_time = time.perf_counter()

        if train_sampler is not None:
            # DistributedSampler must change its shuffle order every epoch.
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = utils.train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=context.device,
            distributed_context=context,
        )

        val_loss, val_acc = utils.validate(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=context.device,
            distributed_context=context,
        )

        should_stop = False
        if utils.is_main_process(context):
            assert val_loss is not None and val_acc is not None
            epoch_duration = time.perf_counter() - epoch_start_time
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"LR: {current_lr:.6g}, Time: {epoch_duration:.2f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                utils.save_checkpoint(model=model, output_dir=output_dir, fold=fold)
            else:
                early_stop_counter += 1
                should_stop = early_stop_counter >= args.patience
                if should_stop:
                    print(f"Early stopping at epoch {epoch + 1}")

        if scheduler is not None:
            scheduler.step()

        should_stop = utils.broadcast_stop_flag(
            should_stop=should_stop,
            device=context.device,
            distributed_context=context,
        )
        if should_stop:
            break

    if utils.is_main_process(context):
        utils.plot_history(
            train_loss=history["train_loss"],
            train_acc=history["train_acc"],
            val_loss=history["val_loss"],
            val_acc=history["val_acc"],
            fold=fold,
            output_dir=output_dir,
        )
        print(f"Fold {fold + 1} best val loss: {best_val_loss:.4f}")

    utils.barrier_if_distributed(context)
    
    return history


def main() -> None:
    args = parse_args()
    context = utils.setup_distributed()

    try:
        data_dir = utils.resolve_project_path(args.data_dir)
        output_dir = utils.resolve_project_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset, train_dataset, val_dataset = utils.build_datasets(data_dir)
        num_classes = len(dataset.classes)

        skf = StratifiedKFold(
            n_splits=args.k_folds,
            shuffle=True,
            random_state=args.seed,
        )

        fold_splits = skf.split(np.arange(len(dataset)), dataset.labels)
        for fold, (train_idx, val_idx) in enumerate(fold_splits):
            run_fold(
                args=args,
                fold=fold,
                train_idx=train_idx,
                val_idx=val_idx,
                context=context,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                num_classes=num_classes,
                output_dir=output_dir,
            )
    finally:
        utils.cleanup_distributed()


if __name__ == "__main__":
    main()
