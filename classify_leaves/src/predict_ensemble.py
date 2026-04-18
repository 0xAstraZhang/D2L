import argparse
import re
from pathlib import Path

import pandas as pd
import torch
import torchvision.models as models
from torch import nn
from torch.backends import mps
from torch.utils.data import DataLoader

import utils


EXPECTED_FOLDS = 5
CHECKPOINT_PATTERN = re.compile(r"best_model_fold_(\d+)\.pth$")

    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ensemble inference with K-fold checkpoints for the leaves classifier.",
    )
    parser.add_argument("--data-dir", default="data", help="Path to the dataset directory.")
    parser.add_argument("--model-dir", default="model", help="Directory containing fold checkpoints.")
    parser.add_argument(
        "--output-path",
        default="model/submission_ensemble.csv",
        help="Path to write the ensemble submission CSV.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int) -> nn.Module:
    # Inference only needs the architecture definition because weights come from checkpoints.
    model = models.resnext50_32x4d(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def find_checkpoints(model_dir: Path) -> list[Path]:
    matched_paths: list[tuple[int, Path]] = []
    for path in model_dir.glob("best_model_fold_*.pth"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        matched_paths.append((int(match.group(1)), path))

    if len(matched_paths) != EXPECTED_FOLDS:
        raise FileNotFoundError(
            f"Expected {EXPECTED_FOLDS} checkpoints in {model_dir}, found {len(matched_paths)}."
        )

    matched_paths.sort(key=lambda item: item[0])
    fold_ids = [fold_id for fold_id, _ in matched_paths]
    expected_fold_ids = list(range(EXPECTED_FOLDS))
    if fold_ids != expected_fold_ids:
        raise ValueError(f"Expected fold ids {expected_fold_ids}, found {fold_ids}.")

    return [path for _, path in matched_paths]


def load_models(checkpoint_paths: list[Path], num_classes: int, device: torch.device) -> list[nn.Module]:
    models_list: list[nn.Module] = []
    for checkpoint_path in checkpoint_paths:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        fc_weight = state_dict.get("fc.weight")
        if fc_weight is None:
            raise KeyError(f"Checkpoint {checkpoint_path} is missing 'fc.weight'.")
        if fc_weight.shape[0] != num_classes:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has {fc_weight.shape[0]} classes, expected {num_classes}."
            )

        model = build_model(num_classes=num_classes)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models_list.append(model)

    return models_list


@torch.inference_mode()
def run_inference(
    models_list: list[nn.Module],
    data_loader: DataLoader,
    idx_to_cls: dict[int, str],
    device: torch.device,
) -> pd.DataFrame:
    image_names: list[str] = []
    predicted_labels: list[str] = []
    use_non_blocking = device.type == "cuda"

    for images, batch_image_names in data_loader:
        images = images.to(device, non_blocking=use_non_blocking)

        probability_sum = None
        for model in models_list:
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            if probability_sum is None:
                probability_sum = probabilities
            else:
                probability_sum += probabilities

        assert probability_sum is not None
        ensemble_probabilities = probability_sum / len(models_list)
        predictions = ensemble_probabilities.argmax(dim=1).cpu().tolist()

        image_names.extend(batch_image_names)
        predicted_labels.extend(idx_to_cls[prediction] for prediction in predictions)

    return pd.DataFrame({"image": image_names, "label": predicted_labels})


def main() -> None:
    args = parse_args()
    data_dir = utils.resolve_project_path(args.data_dir)
    model_dir = utils.resolve_project_path(args.model_dir)
    output_path = utils.resolve_project_path(args.output_path)
    device = pick_device()

    train_dataset, _, _ = utils.build_datasets(data_dir)
    num_classes = len(train_dataset.classes)

    _, val_transforms = utils.build_transforms()
    test_dataset = utils.LeavesTestDataset(csv_file="test.csv", data_dir=data_dir, transform=val_transforms)
    checkpoint_paths = find_checkpoints(model_dir)
    models_list = load_models(checkpoint_paths=checkpoint_paths, num_classes=num_classes, device=device)

    pin_memory = device.type == "cuda"
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    submission = run_inference(
        models_list=models_list,
        data_loader=test_loader,
        idx_to_cls=train_dataset.idx_to_cls,
        device=device,
    )

    if len(submission) != len(test_dataset):
        raise ValueError(
            f"Submission row count {len(submission)} does not match test size {len(test_dataset)}."
        )

    expected_images = test_dataset.df["image"].tolist()
    actual_images = submission["image"].tolist()
    if actual_images != expected_images:
        raise ValueError("Prediction output order does not match test.csv.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"Loaded {len(models_list)} checkpoints from {model_dir}")
    print(f"Using device: {device}")
    print(f"Saved ensemble submission to: {output_path}")


if __name__ == "__main__":
    main()
