import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as T
from torch.backends import mps
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


class LeavesDataset(Dataset):
    def __init__(self, csv_file: str | Path, data_dir: str | Path, transform=None):
        super().__init__()
        data_dir = Path(data_dir)
        csv_path = data_dir / csv_file
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(self.df.iloc[:, 1].unique())
        self.cls_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_cls = {idx: cls_name for idx, cls_name in enumerate(self.classes)}
        self.labels = np.array(self.df.iloc[:, 1].map(self.cls_to_idx).values, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        image_name = str(self.df.iloc[idx, 0])
        label = int(self.labels[idx])

        image_path = self.data_dir / image_name
        image = torchvision.io.read_image(str(image_path))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class LeavesTestDataset(Dataset):
    def __init__(self, csv_file: str | Path, data_dir: str | Path, transform=None):
        super().__init__()
        data_dir = Path(data_dir)
        csv_path = data_dir / csv_file
        self.df = pd.read_csv(csv_path)
        if "image" not in self.df.columns:
            raise ValueError(f"Expected an 'image' column in {csv_path}")
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        image_name = str(self.df.iloc[idx]["image"])
        image_path = self.data_dir / image_name
        image = torchvision.io.read_image(str(image_path))

        if self.transform is not None:
            image = self.transform(image)

        return image, image_name

def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed() -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA. Please launch with GPUs available.")

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return DistributedContext(
            distributed=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return DistributedContext(
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=device,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(distributed_context: DistributedContext) -> bool:
    return distributed_context.rank == 0


def barrier_if_distributed(distributed_context: DistributedContext) -> None:
    if not distributed_context.distributed:
        return

    if distributed_context.device.type == "cuda":
        dist.barrier(device_ids=[distributed_context.local_rank])
        return

    dist.barrier()


def broadcast_stop_flag(
    should_stop: bool,
    device: torch.device,
    distributed_context: DistributedContext,
) -> bool:
    if not distributed_context.distributed:
        return should_stop

    stop_tensor = torch.tensor([int(should_stop)], device=device, dtype=torch.int32)
    dist.broadcast(stop_tensor, src=0)
    return bool(stop_tensor.item())


def all_reduce_sum(values: torch.Tensor, distributed_context: DistributedContext) -> torch.Tensor:
    if distributed_context.distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def build_transforms() -> Tuple[T.Compose, T.Compose]:
    train_transforms = T.Compose(
        [
            T.Resize(300, antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transforms = T.Compose(
        [
            T.Resize(300, antialias=True),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transforms, val_transforms


def build_datasets(data_dir: str | Path) -> Tuple[LeavesDataset, LeavesDataset, LeavesDataset]:
    data_dir = resolve_project_path(data_dir)
    csv_path = data_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find training CSV at {csv_path}")

    train_transforms, val_transforms = build_transforms()
    dataset = LeavesDataset(csv_file="train.csv", data_dir=data_dir, transform=None)
    train_dataset = LeavesDataset(csv_file="train.csv", data_dir=data_dir, transform=train_transforms)
    val_dataset = LeavesDataset(csv_file="train.csv", data_dir=data_dir, transform=val_transforms)
    return dataset, train_dataset, val_dataset


def build_fold_loaders(
    train_dataset: LeavesDataset,
    val_dataset: LeavesDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
    distributed_context: DistributedContext,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DistributedSampler]]:
    train_subset = Subset(train_dataset, train_idx.tolist())
    val_subset = Subset(val_dataset, val_idx.tolist())

    pin_memory = distributed_context.device.type == "cuda"
    persistent_workers = num_workers > 0

    train_sampler: Optional[DistributedSampler] = None
    if distributed_context.distributed:
        train_sampler = DistributedSampler(
            train_subset,
            num_replicas=distributed_context.world_size,
            rank=distributed_context.rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader: Optional[DataLoader] = None
    if not distributed_context.distributed or is_main_process(distributed_context):
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    return train_loader, val_loader, train_sampler


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    distributed_context: DistributedContext,
) -> Tuple[float, float]:
    model.train()
    metric_sums = torch.zeros(3, device=device, dtype=torch.float64)
    use_non_blocking = device.type == "cuda"

    for images, labels in data_loader:
        images = images.to(device, non_blocking=use_non_blocking)
        labels = labels.to(device, non_blocking=use_non_blocking)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)
        batch_size = labels.size(0)

        # Cross-rank loss must be accumulated as a sample-weighted sum.
        metric_sums[0] += loss.detach() * batch_size
        metric_sums[1] += (predictions == labels).sum()
        metric_sums[2] += batch_size

    metric_sums = all_reduce_sum(metric_sums, distributed_context)
    train_loss = (metric_sums[0] / metric_sums[2]).item()
    train_acc = (metric_sums[1] / metric_sums[2]).item()
    return train_loss, train_acc


def validate(
    model: torch.nn.Module,
    data_loader: Optional[DataLoader],
    loss_fn: torch.nn.Module,
    device: torch.device,
    distributed_context: DistributedContext,
) -> Tuple[Optional[float], Optional[float]]:
    if distributed_context.distributed and not is_main_process(distributed_context):
        barrier_if_distributed(distributed_context)
        return None, None

    if data_loader is None:
        return None, None

    eval_model = _unwrap_model(model)
    eval_model.eval()
    metric_sums = torch.zeros(3, device=device, dtype=torch.float64)
    use_non_blocking = device.type == "cuda"

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=use_non_blocking)
            labels = labels.to(device, non_blocking=use_non_blocking)

            outputs = eval_model(images)
            loss = loss_fn(outputs, labels)
            predictions = outputs.argmax(dim=1)
            batch_size = labels.size(0)

            metric_sums[0] += loss.detach() * batch_size
            metric_sums[1] += (predictions == labels).sum()
            metric_sums[2] += batch_size

    val_loss = (metric_sums[0] / metric_sums[2]).item()
    val_acc = (metric_sums[1] / metric_sums[2]).item()

    barrier_if_distributed(distributed_context)
    return val_loss, val_acc


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DDP):
        return model.module
    return model


def save_checkpoint(model: torch.nn.Module, output_dir: str | Path, fold: int) -> Path:
    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"best_model_fold_{fold}.pth"
    torch.save(_unwrap_model(model).state_dict(), checkpoint_path)
    return checkpoint_path


def plot_history(
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    fold: int,
    output_dir: str | Path,
) -> Path:
    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, "b-", label="Train Loss")
    plt.plot(epochs, val_loss, "c--", label="Val Loss")
    plt.plot(epochs, train_acc, "g-.", label="Train Acc")
    plt.plot(epochs, val_acc, "r:", label="Val Acc")

    plt.title(f"Fold {fold + 1} Training Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)

    save_path = output_dir / f"learning_curve_fold_{fold}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"第 {fold + 1} 折的学习曲线已保存至: {save_path}")
    return save_path
