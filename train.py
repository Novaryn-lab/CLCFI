import gc
import json
import os
import random
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from .config import CLCFIConfig, build_parser, config_from_args
from .data_loader import create_dataloaders, move_batch_to_device
from .metrics import regression_metrics
from .model import CLCFI

NUMBER_OF_SEEDS = 5
SEED_INTERVAL = 1111

def build_experiment_seeds(base_seed: int) -> Tuple[int, ...]:

    if not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer.")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative.")

    seeds = tuple(
        base_seed + index * SEED_INTERVAL
        for index in range(NUMBER_OF_SEEDS)
    )
    if len(set(seeds)) != NUMBER_OF_SEEDS:
        raise RuntimeError("The generated random seeds are not unique.")
    return seeds


def set_seed(seed: int) -> None:

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:

    if not isinstance(requested, str):
        raise TypeError("device must be a string.")

    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(normalized)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{requested}' was requested, but CUDA is unavailable."
            )
        index = 0 if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is invalid; "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")

    if device.type == "mps":
        mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if not mps_available:
            raise RuntimeError("MPS was requested, but it is unavailable.")

    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            f"Unsupported device type '{device.type}'. Use auto, cpu, mps or cuda."
        )

    return device


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        return f"{device} ({torch.cuda.get_device_name(index)})"
    return str(device)


def _jsonable(value: Any) -> Any:

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def create_run_directory(output_dir: str, dataset: str) -> Path:

    root = Path(output_dir).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = root / dataset / "five_seed_runs"
    candidate = parent / f"run_{timestamp}"

    suffix = 1
    while candidate.exists():
        candidate = parent / f"run_{timestamp}_{suffix:02d}"
        suffix += 1

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _canonical_sample_id(value: Any) -> str:

    if isinstance(value, Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", errors="replace")

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):
        return repr(value)


def validate_split_disjointness(loaders: Mapping[str, Any]) -> None:

    expected_splits = ("train", "valid", "test")
    missing = [split for split in expected_splits if split not in loaders]
    if missing:
        raise KeyError(f"Missing data loaders: {', '.join(missing)}.")

    split_ids: Dict[str, set] = {}
    for split in expected_splits:
        dataset = loaders[split].dataset
        ids = {
            _canonical_sample_id(dataset[index]["id"])
            for index in range(len(dataset))
        }
        if not ids:
            raise RuntimeError(f"The {split} split contains no sample IDs.")
        split_ids[split] = ids

    pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))
    for left, right in pairs:
        overlap = split_ids[left].intersection(split_ids[right])
        if overlap:
            examples = sorted(overlap)[:5]
            raise RuntimeError(
                f"Data leakage detected: {left} and {right} share "
                f"{len(overlap)} sample ID(s). Examples: {examples}"
            )


def _require_finite_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf values.")


def _validate_model_output(output: Any, stage: str) -> None:

    if output.loss is None:
        raise RuntimeError(f"{stage}: the model did not return total loss.")
    if output.prediction_loss is None:
        raise RuntimeError(f"{stage}: the model did not return prediction loss.")

    named_tensors = {
        "predictions": output.predictions,
        "total_loss": output.loss,
        "prediction_loss": output.prediction_loss,
        "ccl_loss": output.ccl_loss,
        "consistency_loss": output.consistency_loss,
        "weight_ta": output.weight_ta,
        "weight_tv": output.weight_tv,
        "sufficiency_ta": output.sufficiency_ta,
        "sufficiency_tv": output.sufficiency_tv,
        "necessity_ta": output.necessity_ta,
        "necessity_tv": output.necessity_tv,
    }
    for name, tensor in named_tensors.items():
        _require_finite_tensor(f"{stage}.{name}", tensor)

    for name, tensor in output.auxiliary_predictions.items():
        _require_finite_tensor(f"{stage}.auxiliary_predictions[{name}]", tensor)


def _loss_totals() -> Dict[str, float]:
    return {
        "total_loss": 0.0,
        "prediction_loss": 0.0,
        "ccl_loss": 0.0,
        "consistency_loss": 0.0,
    }


def train_epoch(
    model: CLCFI,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    seed: int,
) -> Dict[str, float]:

    model.train()
    totals = _loss_totals()
    sample_count = 0

    progress = tqdm(
        loader,
        desc=f"seed={seed} epoch={epoch:03d} train",
        leave=False,
    )
    for batch_index, batch in enumerate(progress, start=1):
        model_batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        output = model(**model_batch)
        _validate_model_output(
            output,
            stage=f"train(seed={seed},epoch={epoch},batch={batch_index})",
        )

        output.loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf"),
            error_if_nonfinite=True,
        )
        _require_finite_tensor(
            f"train(seed={seed},epoch={epoch},batch={batch_index}).gradient_norm",
            gradient_norm,
        )

        optimizer.step()

        batch_size = int(model_batch["labels"].size(0))
        sample_count += batch_size
        totals["total_loss"] += float(output.loss.detach().item()) * batch_size
        totals["prediction_loss"] += (
            float(output.prediction_loss.detach().item()) * batch_size
        )
        totals["ccl_loss"] += float(output.ccl_loss.detach().item()) * batch_size
        totals["consistency_loss"] += (
            float(output.consistency_loss.detach().item()) * batch_size
        )

    if sample_count == 0:
        raise RuntimeError("The training DataLoader produced no samples.")

    return {
        name: value / sample_count
        for name, value in totals.items()
    }


@torch.inference_mode()
def evaluate(
    model: CLCFI,
    loader: Any,
    device: torch.device,
    stage: str,
    seed: int,
) -> Tuple[Dict[str, float], Tensor, Tensor]:

    model.eval()
    totals = _loss_totals()
    sample_count = 0
    predictions: List[Tensor] = []
    targets: List[Tensor] = []

    progress = tqdm(
        loader,
        desc=f"seed={seed} {stage}",
        leave=False,
    )
    for batch_index, batch in enumerate(progress, start=1):
        model_batch = move_batch_to_device(batch, device)
        output = model(**model_batch)
        _validate_model_output(
            output,
            stage=f"{stage}(seed={seed},batch={batch_index})",
        )

        batch_size = int(model_batch["labels"].size(0))
        sample_count += batch_size
        totals["total_loss"] += float(output.loss.item()) * batch_size
        totals["prediction_loss"] += (
            float(output.prediction_loss.item()) * batch_size
        )
        totals["ccl_loss"] += float(output.ccl_loss.item()) * batch_size
        totals["consistency_loss"] += (
            float(output.consistency_loss.item()) * batch_size
        )

        predictions.append(output.predictions.detach().cpu())
        targets.append(model_batch["labels"].detach().cpu())

    if sample_count == 0 or not predictions or not targets:
        raise RuntimeError(f"The {stage} DataLoader produced no samples.")

    statistics = {
        name: value / sample_count
        for name, value in totals.items()
    }
    return statistics, torch.cat(predictions, dim=0), torch.cat(targets, dim=0)


def _paper_metrics(metrics: Mapping[str, float]) -> Dict[str, float]:

    formatted: Dict[str, float] = {}
    for name, value in metrics.items():
        numeric = float(value)
        if name.startswith("acc") or name.startswith("f1"):
            formatted[name] = numeric * 100.0
        else:
            formatted[name] = numeric
    return formatted


def aggregate_metrics(
    runs: Sequence[Mapping[str, float]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if len(runs) != NUMBER_OF_SEEDS:
        raise ValueError(
            f"Expected {NUMBER_OF_SEEDS} metric dictionaries, found {len(runs)}."
        )

    expected_keys = set(runs[0])
    if not expected_keys:
        raise ValueError("Metric dictionaries must not be empty.")

    for index, metrics in enumerate(runs):
        if set(metrics) != expected_keys:
            raise ValueError(
                f"Run {index} does not contain the same metric keys as run 0."
            )

    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for key in sorted(expected_keys):
        values = np.asarray(
            [float(metrics[key]) for metrics in runs],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Metric '{key}' contains NaN or Inf.")
        mean[key] = float(values.mean())
        std[key] = float(values.std(ddof=1))

    return mean, std


def run_one_seed(
    base_config: CLCFIConfig,
    seed: int,
    all_seeds: Sequence[int],
    run_root: Path,
    device: torch.device,
    check_splits: bool,
) -> Dict[str, Any]:
    config = replace(base_config, seed=seed)
    config.validate()
    set_seed(seed)

    seed_dir = run_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)

    loaders, (audio_dim, vision_dim) = create_dataloaders(config)
    if check_splits:
        validate_split_disjointness(loaders)

    model = CLCFI(
        config,
        audio_dim=audio_dim,
        vision_dim=vision_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    effective_configuration = {
        "config": config.to_dict(),
        "experiment_seeds": list(all_seeds),
        "optimizer": "Adam",
        "learning_rate_schedule": "fixed",
        "weight_decay_used": 0.0,
        "gradient_clipping_used": False,
        "selection_metric": "validation_mae",
        "audio_feature_dim": audio_dim,
        "vision_feature_dim": vision_dim,
        "device": describe_device(device),
    }
    atomic_write_json(seed_dir / "config.json", effective_configuration)

    history: List[Dict[str, Any]] = []
    checkpoint_path = seed_dir / "best.pt"
    best_valid_mae = float("inf")
    best_epoch = 0
    remaining_patience = config.early_stop

    for epoch in range(1, config.epochs + 1):
        train_statistics = train_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            seed=seed,
        )

        valid_statistics, valid_prediction, valid_target = evaluate(
            model=model,
            loader=loaders["valid"],
            device=device,
            stage="valid",
            seed=seed,
        )
        valid_metrics = regression_metrics(
            valid_prediction.numpy(),
            valid_target.numpy(),
            config.dataset,
        )
        valid_mae = float(valid_metrics["mae"])

        epoch_record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_statistics,
            "valid_losses": valid_statistics,
            "valid_metrics": valid_metrics,
        }
        history.append(epoch_record)
        atomic_write_json(seed_dir / "history.json", {"history": history})

        print(
            f"seed={seed} epoch={epoch:03d} "
            f"train_total={train_statistics['total_loss']:.6f} "
            f"train_mae={train_statistics['prediction_loss']:.6f} "
            f"valid_mae={valid_mae:.6f} "
            f"best_valid_mae={min(best_valid_mae, valid_mae):.6f}"
        )

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            best_epoch = epoch
            remaining_patience = config.early_stop

            atomic_torch_save(
                checkpoint_path,
                {
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "best_valid_mae": best_valid_mae,
                    "dataset": config.dataset,
                    "audio_feature_dim": audio_dim,
                    "vision_feature_dim": vision_dim,
                },
            )
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                print(
                    f"seed={seed} early stopping at epoch {epoch}; "
                    f"best epoch={best_epoch}"
                )
                break

    if best_epoch == 0 or not checkpoint_path.is_file():
        raise RuntimeError(
            f"Seed {seed} did not produce a valid checkpoint."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)

    test_statistics, test_prediction, test_target = evaluate(
        model=model,
        loader=loaders["test"],
        device=device,
        stage="test",
        seed=seed,
    )
    test_metrics = regression_metrics(
        test_prediction.numpy(),
        test_target.numpy(),
        config.dataset,
    )
    test_metrics["prediction_loss"] = float(
        test_statistics["prediction_loss"]
    )
    test_metrics["total_loss"] = float(test_statistics["total_loss"])

    np.savez_compressed(
        seed_dir / "test_predictions.npz",
        predictions=test_prediction.numpy(),
        targets=test_target.numpy(),
    )

    result = {
        "dataset": config.dataset,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_valid_mae": best_valid_mae,
        "test_metrics": test_metrics,
        "paper_table_metrics": _paper_metrics(test_metrics),
    }
    atomic_write_json(seed_dir / "metrics.json", result)

    del model
    del optimizer
    del loaders
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_config = config_from_args(args)

    seeds = build_experiment_seeds(base_config.seed)
    device = resolve_device(base_config.device)
    run_root = create_run_directory(
        output_dir=base_config.output_dir,
        dataset=base_config.dataset,
    )

    print(f"dataset: {base_config.dataset}")
    print(f"device: {describe_device(device)}")
    print(f"five seeds: {list(seeds)}")
    print(f"output directory: {run_root}")

    manifest = {
        "dataset": base_config.dataset,
        "seeds": list(seeds),
        "number_of_runs": NUMBER_OF_SEEDS,
        "device": describe_device(device),
        "output_directory": str(run_root),
        "selection_metric": "validation_mae",
        "result_protocol": "mean_and_sample_standard_deviation",
    }
    atomic_write_json(run_root / "manifest.json", manifest)

    seed_results: List[Dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        print(
            f"\n=== Run {index + 1}/{NUMBER_OF_SEEDS}: seed={seed} ==="
        )
        result = run_one_seed(
            base_config=base_config,
            seed=seed,
            all_seeds=seeds,
            run_root=run_root,
            device=device,
            check_splits=(index == 0),
        )
        seed_results.append(result)

        atomic_write_json(
            run_root / "progress.json",
            {
                "completed_runs": len(seed_results),
                "expected_runs": NUMBER_OF_SEEDS,
                "results": seed_results,
            },
        )

    metric_runs = [
        result["test_metrics"]
        for result in seed_results
    ]
    mean_metrics, std_metrics = aggregate_metrics(metric_runs)

    summary = {
        "dataset": base_config.dataset,
        "seeds": list(seeds),
        "number_of_runs": NUMBER_OF_SEEDS,
        "selection_metric": "validation_mae",
        "metric_scale": {
            "mae_corr_losses": "native",
            "acc_and_f1": "proportion_in_raw_results",
            "paper_table_acc_and_f1": "percentage_points",
        },
        "runs": seed_results,
        "mean": mean_metrics,
        "sample_standard_deviation": std_metrics,
        "paper_table_mean": _paper_metrics(mean_metrics),
        "paper_table_sample_standard_deviation": _paper_metrics(std_metrics),
    }
    atomic_write_json(run_root / "summary.json", summary)

    print("\n=== Five-seed mean ± sample standard deviation ===")
    for name in sorted(mean_metrics):
        mean_value = mean_metrics[name]
        std_value = std_metrics[name]
        if name.startswith("acc") or name.startswith("f1"):
            print(
                f"{name}: {mean_value * 100.0:.2f} "
                f"± {std_value * 100.0:.2f}"
            )
        else:
            print(f"{name}: {mean_value:.6f} ± {std_value:.6f}")

    print(f"\nSaved all results to: {run_root}")


if __name__ == "__main__":
    main()
