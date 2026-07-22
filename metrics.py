from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


SUPPORTED_DATASETS = {"mosi", "mosei", "sims"}


def _normalize_dataset_name(dataset: str) -> str:

    if not isinstance(dataset, str):
        raise TypeError(
            f"dataset must be a string, but found {type(dataset).__name__}."
        )

    normalized = dataset.strip().lower().replace("_", "-")
    aliases = {
        "ch-sims": "sims",
        "chsims": "sims",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized not in SUPPORTED_DATASETS:
        choices = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Expected one of: {choices}, ch-sims."
        )
    return normalized


def _prepare_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    try:
        prediction_array = np.asarray(prediction, dtype=np.float64).reshape(-1)
        target_array = np.asarray(target, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "prediction and target must be numeric array-like objects."
        ) from exc

    if prediction_array.size == 0 or target_array.size == 0:
        raise ValueError("prediction and target must not be empty.")

    if prediction_array.size != target_array.size:
        raise ValueError(
            "prediction and target must contain the same number of values, "
            f"but found {prediction_array.size} and {target_array.size}."
        )

    if not np.isfinite(prediction_array).all():
        raise ValueError("prediction contains NaN or Inf values.")
    if not np.isfinite(target_array).all():
        raise ValueError("target contains NaN or Inf values.")

    return prediction_array, target_array


def _safe_corr(prediction: np.ndarray, target: np.ndarray) -> float:

    if prediction.size < 2:
        return 0.0
    if np.allclose(prediction, prediction[0]):
        return 0.0
    if np.allclose(target, target[0]):
        return 0.0

    correlation = float(np.corrcoef(prediction, target)[0, 1])
    return correlation if np.isfinite(correlation) else 0.0


def _weighted_binary_metrics(
    predicted_class: np.ndarray,
    target_class: np.ndarray,
) -> Dict[str, float]:

    predicted_class = np.asarray(predicted_class, dtype=np.int64).reshape(-1)
    target_class = np.asarray(target_class, dtype=np.int64).reshape(-1)

    if predicted_class.size == 0:
        raise ValueError("Binary metrics cannot be computed on an empty subset.")
    if predicted_class.size != target_class.size:
        raise ValueError(
            "Binary prediction and target arrays must have the same length."
        )

    return {
        "accuracy": float(accuracy_score(target_class, predicted_class)),
        "f1": float(
            f1_score(
                target_class,
                predicted_class,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def _mosi_mosei_binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float]]:

    has0 = _weighted_binary_metrics(
        predicted_class=(prediction >= 0),
        target_class=(target >= 0),
    )

    nonzero_indices = target != 0
    if not np.any(nonzero_indices):
        raise ValueError(
            "The Non0 metric is undefined because every ground-truth label is 0."
        )

    non0 = _weighted_binary_metrics(
        predicted_class=(prediction[nonzero_indices] > 0),
        target_class=(target[nonzero_indices] > 0),
    )
    return has0, non0


def _rounded_accuracy(
    prediction: np.ndarray,
    target: np.ndarray,
    minimum: float,
    maximum: float,
) -> float:

    predicted_class = np.rint(np.clip(prediction, minimum, maximum))
    target_class = np.rint(np.clip(target, minimum, maximum))
    return float(accuracy_score(target_class, predicted_class))


def _sims_five_class(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:

    boundaries = np.asarray([-0.7, -0.1, 0.1, 0.7], dtype=np.float64)
    predicted_class = np.digitize(prediction, boundaries, right=True)
    target_class = np.digitize(target, boundaries, right=True)
    return float(accuracy_score(target_class, predicted_class))


def _mosi_mosei_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:

    has0, non0 = _mosi_mosei_binary_metrics(prediction, target)

    return {
        "mae": float(np.mean(np.abs(prediction - target))),
        "corr": _safe_corr(prediction, target),

        "acc2": has0["accuracy"],
        "f1": has0["f1"],

        "acc2_nonzero": non0["accuracy"],
        "f1_nonzero": non0["f1"],
        "acc7": _rounded_accuracy(prediction, target, -3.0, 3.0),
    }


def _sims_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:

    clipped_prediction = np.clip(prediction, -1.0, 1.0)
    clipped_target = np.clip(target, -1.0, 1.0)

    binary = _weighted_binary_metrics(
        predicted_class=(clipped_prediction > 0),
        target_class=(clipped_target > 0),
    )

    return {
        "mae": float(np.mean(np.abs(clipped_prediction - clipped_target))),
        "corr": _safe_corr(clipped_prediction, clipped_target),
        "acc2": binary["accuracy"],
        "f1": binary["f1"],
        "acc5": _sims_five_class(clipped_prediction, clipped_target),
    }


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    dataset: str,
) -> Dict[str, float]:

    normalized_dataset = _normalize_dataset_name(dataset)
    prediction_array, target_array = _prepare_arrays(prediction, target)

    if normalized_dataset == "sims":
        return _sims_metrics(prediction_array, target_array)

    return _mosi_mosei_metrics(prediction_array, target_array)
