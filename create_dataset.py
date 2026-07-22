import argparse
import csv
import os
import pickle
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


_SPLIT_ALIASES = {
    "train": ("train",),
    "valid": ("valid", "validation", "dev", "val"),
    "test": ("test",),
}

_FIELD_ALIASES = {
    "audio": ("audio", "acoustic", "acoustics", "audio_features"),
    "vision": ("vision", "visual", "video", "visual_features", "vision_features"),
    "text": ("raw_text", "raw_words", "words", "sentence", "sentences", "text"),
    "label": (
        "regression_labels",
        "multimodal_labels",
        "multimodal_label",
        "labels",
        "label",
    ),
    "id": ("id", "ids", "sample_id", "sample_ids", "segment_id", "segment_ids"),
}

_EXPECTED_COUNTS = {
    "mosi": {"train": 1284, "valid": 229, "test": 686},
    "mosei": {"train": 16326, "valid": 1871, "test": 4659},
    "sims": {"train": 1368, "valid": 456, "test": 457},
}

_OUTPUT_DIR_NAMES = {
    "mosi": "MOSI",
    "mosei": "MOSEI",
    "sims": "SIMS",
}


def _normalize_dataset_name(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "cmu-mosi": "mosi",
        "cmu-mosei": "mosei",
        "ch-sims": "sims",
        "chsims": "sims",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"mosi", "mosei", "sims"}:
        raise ValueError("dataset must be one of: mosi, mosei, sims, ch-sims")
    return normalized


def _load_pickle(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Source pickle does not exist: {path}")
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise RuntimeError(f"Failed to read source pickle: {path}") from exc


def _atomic_pickle_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _case_insensitive_key(mapping: Mapping[str, Any], name: str) -> Optional[str]:
    target = name.casefold()
    for key in mapping:
        if str(key).casefold() == target:
            return str(key)
    return None


def _select_key(
    mapping: Mapping[str, Any],
    field: str,
    override: Optional[str],
) -> Optional[str]:
    if override:
        matched = _case_insensitive_key(mapping, override)
        if matched is None:
            raise KeyError(
                f"Requested {field} key '{override}' was not found. "
                f"Available keys: {sorted(map(str, mapping.keys()))}"
            )
        return matched

    for alias in _FIELD_ALIASES[field]:
        matched = _case_insensitive_key(mapping, alias)
        if matched is not None:
            return matched
    return None


def _select_split(source: Mapping[str, Any], split: str) -> Any:
    for alias in _SPLIT_ALIASES[split]:
        matched = _case_insensitive_key(source, alias)
        if matched is not None:
            return source[matched]
    raise KeyError(
        f"Source pickle has no '{split}' split. "
        f"Available top-level keys: {sorted(map(str, source.keys()))}"
    )


def _sequence_length(value: Any, field: str) -> int:
    try:
        length = len(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be an indexable sample collection.") from exc
    if length < 1:
        raise ValueError(f"{field} is empty.")
    return int(length)


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return value
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="strict")
    return value


def _normalize_id(value: Any, split: str, index: int) -> str:
    value = _decode_scalar(value)
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        if flat.size != 1:
            raise ValueError(
                f"{split} sample {index}: sample ID must contain one value."
            )
        value = _decode_scalar(flat[0])

    sample_id = str(value).strip()
    if not sample_id:
        raise ValueError(f"{split} sample {index}: sample ID is empty.")
    return sample_id


def _extract_float_label(value: Any, split: str, index: int) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size < 1:
        raise ValueError(f"{split} sample {index}: label is empty.")
    try:
        label = float(array[0])
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{split} sample {index}: label cannot be converted to float."
        ) from exc
    if not np.isfinite(label):
        raise ValueError(f"{split} sample {index}: label contains NaN or Inf.")
    return label


def _as_feature_matrix(
    value: Any,
    modality: str,
    split: str,
    index: int,
) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{split} sample {index}: {modality} features cannot be "
            "converted to float32."
        ) from exc

    if matrix.ndim != 2:
        raise ValueError(
            f"{split} sample {index}: {modality} features must have shape "
            f"[sequence, feature], but found {matrix.shape}."
        )
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(
            f"{split} sample {index}: {modality} features must have "
            "positive dimensions."
        )
    if not np.isfinite(matrix).all():
        raise ValueError(
            f"{split} sample {index}: {modality} features contain NaN or Inf."
        )
    return np.ascontiguousarray(matrix)


def _trim_outer_zero_padding(
    matrix: np.ndarray,
    modality: str,
    split: str,
    index: int,
    tolerance: float,
) -> np.ndarray:
    active_rows = np.any(np.abs(matrix) > tolerance, axis=1)
    if not bool(active_rows.any()):
        raise ValueError(
            f"{split} sample {index}: {modality} contains no non-zero frame."
        )

    valid_indices = np.flatnonzero(active_rows)
    start = int(valid_indices[0])
    stop = int(valid_indices[-1]) + 1
    trimmed = matrix[start:stop]

    if trimmed.shape[0] < 1:
        raise RuntimeError(
            f"{split} sample {index}: failed to trim {modality} padding."
        )
    return np.ascontiguousarray(trimmed)


def _is_text_scalar(value: Any) -> bool:
    value = _decode_scalar(value)
    return isinstance(value, str)


def _normalize_text(
    value: Any,
    dataset: str,
    split: str,
    index: int,
) -> List[str]:
    value = _decode_scalar(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{split} sample {index}: text is empty.")
        if dataset == "sims":
            return [text]
        tokens = text.split()
        if not tokens:
            raise ValueError(f"{split} sample {index}: text is empty.")
        return tokens

    if isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        raise TypeError(
            f"{split} sample {index}: text must be a string or token sequence."
        )

    tokens: List[str] = []
    for token in values:
        token = _decode_scalar(token)
        if not isinstance(token, str):
            raise TypeError(
                f"{split} sample {index}: text field contains non-text values."
            )
        token = token.strip()
        if token and token.casefold() not in {"<pad>", "[pad]", "pad"}:
            tokens.append(token)

    if not tokens:
        raise ValueError(f"{split} sample {index}: text is empty.")
    return tokens


def _text_collection_is_usable(values: Any, sample_count: int) -> bool:
    try:
        if len(values) != sample_count:
            return False
        first = values[0]
    except (TypeError, KeyError, IndexError):
        return False

    first = _decode_scalar(first)
    if isinstance(first, str):
        return True
    if isinstance(first, np.ndarray):
        items = first.reshape(-1).tolist()
    elif isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        items = list(first)
    else:
        return False
    return bool(items) and all(_is_text_scalar(item) for item in items)


def _compact_id(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", value.casefold())


def _id_lookup_candidates(value: str) -> Tuple[str, ...]:
    stripped = value.strip()
    candidates = {stripped.casefold(), _compact_id(stripped)}
    return tuple(candidate for candidate in candidates if candidate)


def _find_column(
    fieldnames: Iterable[str],
    aliases: Iterable[str],
) -> Optional[str]:
    names = list(fieldnames)
    for alias in aliases:
        for name in names:
            if name.casefold() == alias.casefold():
                return name
    return None


def _load_text_lookup(csv_path: Optional[Path]) -> Dict[str, str]:
    if csv_path is None:
        return {}
    if not csv_path.is_file():
        raise FileNotFoundError(f"Label CSV does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        text_column = _find_column(
            reader.fieldnames,
            ("text", "sentence", "raw_text", "transcript"),
        )
        if text_column is None:
            raise KeyError(
                f"CSV must contain a text column. Available columns: "
                f"{reader.fieldnames}"
            )

        direct_id_column = _find_column(
            reader.fieldnames,
            ("id", "sample_id", "segment_id", "key"),
        )
        video_column = _find_column(
            reader.fieldnames,
            ("video_id", "video", "videoid"),
        )
        clip_column = _find_column(
            reader.fieldnames,
            ("clip_id", "clip", "clipid", "segment"),
        )

        if direct_id_column is None and (
            video_column is None or clip_column is None
        ):
            raise KeyError(
                "CSV must contain either an ID column or both video_id and clip_id."
            )

        lookup: Dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            text = (row.get(text_column) or "").strip()
            if not text:
                raise ValueError(
                    f"{csv_path}, row {row_number}: text is empty."
                )

            raw_ids: List[str] = []
            if direct_id_column is not None:
                direct_id = (row.get(direct_id_column) or "").strip()
                if direct_id:
                    raw_ids.append(direct_id)

            if video_column is not None and clip_column is not None:
                video = (row.get(video_column) or "").strip()
                clip = (row.get(clip_column) or "").strip()
                if video and clip:
                    raw_ids.extend(
                        [
                            f"{video}_{clip}",
                            f"{video}[{clip}]",
                            f"{video}$$_${clip}",
                            f"{video}-{clip}",
                            f"{video}/{clip}",
                        ]
                    )

            for raw_id in raw_ids:
                for candidate in _id_lookup_candidates(raw_id):
                    previous = lookup.get(candidate)
                    if previous is not None and previous != text:
                        raise ValueError(
                            f"{csv_path}: duplicate identifier maps to "
                            "different text values."
                        )
                    lookup[candidate] = text

    if not lookup:
        raise ValueError(f"No usable text records were found in {csv_path}.")
    return lookup


def _lookup_text(
    sample_id: str,
    text_lookup: Mapping[str, str],
    split: str,
    index: int,
) -> str:
    for candidate in _id_lookup_candidates(sample_id):
        if candidate in text_lookup:
            return text_lookup[candidate]
    raise KeyError(
        f"{split} sample {index}: no text was found for sample ID '{sample_id}'."
    )


def _mapping_to_samples(
    split_data: Mapping[str, Any],
    split: str,
    dataset: str,
    text_lookup: Mapping[str, str],
    audio_key: Optional[str],
    vision_key: Optional[str],
    text_key: Optional[str],
    label_key: Optional[str],
    id_key: Optional[str],
    zero_tolerance: float,
) -> List[Tuple[Any, float, str]]:
    selected_audio_key = _select_key(split_data, "audio", audio_key)
    selected_vision_key = _select_key(split_data, "vision", vision_key)
    selected_label_key = _select_key(split_data, "label", label_key)
    selected_id_key = _select_key(split_data, "id", id_key)
    selected_text_key = _select_key(split_data, "text", text_key)

    required = {
        "audio": selected_audio_key,
        "vision": selected_vision_key,
        "label": selected_label_key,
        "id": selected_id_key,
    }
    missing = [field for field, key in required.items() if key is None]
    if missing:
        raise KeyError(
            f"{split} split is missing fields: {', '.join(missing)}. "
            f"Available keys: {sorted(map(str, split_data.keys()))}"
        )

    audio_values = split_data[selected_audio_key]
    vision_values = split_data[selected_vision_key]
    label_values = split_data[selected_label_key]
    id_values = split_data[selected_id_key]

    sample_count = _sequence_length(audio_values, f"{split}.{selected_audio_key}")
    lengths = {
        "vision": _sequence_length(
            vision_values, f"{split}.{selected_vision_key}"
        ),
        "label": _sequence_length(
            label_values, f"{split}.{selected_label_key}"
        ),
        "id": _sequence_length(id_values, f"{split}.{selected_id_key}"),
    }
    for field, length in lengths.items():
        if length != sample_count:
            raise ValueError(
                f"{split}: {field} contains {length} samples, "
                f"but audio contains {sample_count}."
            )

    text_values = None
    if selected_text_key is not None:
        candidate_values = split_data[selected_text_key]
        if _text_collection_is_usable(candidate_values, sample_count):
            text_values = candidate_values
        elif text_key is not None:
            raise TypeError(
                f"{split}.{selected_text_key} is not a textual sample collection."
            )

    if text_values is None and not text_lookup:
        raise KeyError(
            f"{split}: source pickle contains no usable raw text. "
            "Provide --label-csv with sample identifiers and text."
        )

    samples: List[Tuple[Any, float, str]] = []
    for index in range(sample_count):
        sample_id = _normalize_id(id_values[index], split, index)
        raw_text = (
            text_values[index]
            if text_values is not None
            else _lookup_text(sample_id, text_lookup, split, index)
        )
        words = _normalize_text(raw_text, dataset, split, index)

        audio = _as_feature_matrix(
            audio_values[index], "audio", split, index
        )
        vision = _as_feature_matrix(
            vision_values[index], "vision", split, index
        )
        audio = _trim_outer_zero_padding(
            audio, "audio", split, index, zero_tolerance
        )
        vision = _trim_outer_zero_padding(
            vision, "vision", split, index, zero_tolerance
        )

        label = _extract_float_label(label_values[index], split, index)
        features = (
            np.empty((0,), dtype=np.int64),
            vision,
            audio,
            words,
            int(vision.shape[0]),
            int(audio.shape[0]),
        )
        samples.append((features, label, sample_id))

    return samples


def _record_to_mapping(record: Mapping[str, Any]) -> Dict[str, List[Any]]:
    result: Dict[str, List[Any]] = {}
    for key, value in record.items():
        result[str(key)] = [value]
    return result


def _sequence_to_samples(
    split_data: Sequence[Any],
    split: str,
    dataset: str,
    text_lookup: Mapping[str, str],
    audio_key: Optional[str],
    vision_key: Optional[str],
    text_key: Optional[str],
    label_key: Optional[str],
    id_key: Optional[str],
    zero_tolerance: float,
) -> List[Tuple[Any, float, str]]:
    if len(split_data) < 1:
        raise ValueError(f"{split} split is empty.")

    samples: List[Tuple[Any, float, str]] = []
    for index, record in enumerate(split_data):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"{split} sample {index}: expected a mapping, "
                f"but found {type(record).__name__}."
            )
        converted = _mapping_to_samples(
            _record_to_mapping(record),
            split=split,
            dataset=dataset,
            text_lookup=text_lookup,
            audio_key=audio_key,
            vision_key=vision_key,
            text_key=text_key,
            label_key=label_key,
            id_key=id_key,
            zero_tolerance=zero_tolerance,
        )
        samples.extend(converted)
    return samples


def _convert_split(
    split_data: Any,
    split: str,
    dataset: str,
    text_lookup: Mapping[str, str],
    audio_key: Optional[str],
    vision_key: Optional[str],
    text_key: Optional[str],
    label_key: Optional[str],
    id_key: Optional[str],
    zero_tolerance: float,
) -> List[Tuple[Any, float, str]]:
    if isinstance(split_data, Mapping):
        return _mapping_to_samples(
            split_data=split_data,
            split=split,
            dataset=dataset,
            text_lookup=text_lookup,
            audio_key=audio_key,
            vision_key=vision_key,
            text_key=text_key,
            label_key=label_key,
            id_key=id_key,
            zero_tolerance=zero_tolerance,
        )
    if isinstance(split_data, Sequence) and not isinstance(
        split_data, (str, bytes)
    ):
        return _sequence_to_samples(
            split_data=split_data,
            split=split,
            dataset=dataset,
            text_lookup=text_lookup,
            audio_key=audio_key,
            vision_key=vision_key,
            text_key=text_key,
            label_key=label_key,
            id_key=id_key,
            zero_tolerance=zero_tolerance,
        )
    raise TypeError(
        f"{split} split must be a mapping of arrays or a sequence of mappings."
    )


def _feature_dimensions(samples: Sequence[Tuple[Any, float, str]]) -> Tuple[int, int]:
    if not samples:
        raise ValueError("Cannot inspect feature dimensions of an empty split.")
    features = samples[0][0]
    vision_dim = int(np.asarray(features[1]).shape[1])
    audio_dim = int(np.asarray(features[2]).shape[1])
    return audio_dim, vision_dim


def _validate_converted_data(
    converted: Mapping[str, Sequence[Tuple[Any, float, str]]],
    dataset: str,
    strict_counts: bool,
) -> None:
    dimensions = {
        split: _feature_dimensions(samples)
        for split, samples in converted.items()
    }
    if len(set(dimensions.values())) != 1:
        raise ValueError(
            f"Feature dimensions differ across splits: {dimensions}"
        )

    ids_by_split: Dict[str, set[str]] = {}
    for split, samples in converted.items():
        sample_ids = [sample[-1] for sample in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"{split} split contains duplicate sample IDs.")
        ids_by_split[split] = set(sample_ids)

    split_names = ("train", "valid", "test")
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1:]:
            overlap = ids_by_split[left] & ids_by_split[right]
            if overlap:
                example = sorted(overlap)[:5]
                raise ValueError(
                    f"{left} and {right} contain overlapping sample IDs: {example}"
                )

    if strict_counts:
        expected = _EXPECTED_COUNTS[dataset]
        actual = {
            split: len(converted[split])
            for split in split_names
        }
        if actual != expected:
            raise ValueError(
                f"Dataset split counts do not match the paper. "
                f"Expected {expected}, found {actual}."
            )


def create_dataset(
    dataset: str,
    source_pickle: str,
    output_dir: Optional[str] = None,
    label_csv: Optional[str] = None,
    audio_key: Optional[str] = None,
    vision_key: Optional[str] = None,
    text_key: Optional[str] = None,
    label_key: Optional[str] = None,
    id_key: Optional[str] = None,
    zero_tolerance: float = 0.0,
    strict_counts: bool = True,
    overwrite: bool = False,
) -> Dict[str, Path]:
    dataset_name = _normalize_dataset_name(dataset)
    source_path = Path(source_pickle).expanduser().resolve()
    csv_path = (
        Path(label_csv).expanduser().resolve()
        if label_csv is not None
        else None
    )
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path("/openbayes/home/datasets")
        / _OUTPUT_DIR_NAMES[dataset_name]
    )

    if zero_tolerance < 0:
        raise ValueError("zero_tolerance must be non-negative.")

    source = _load_pickle(source_path)
    if not isinstance(source, Mapping):
        raise TypeError(
            "Source pickle must contain a mapping with train, valid/dev and test."
        )

    text_lookup = _load_text_lookup(csv_path)
    converted: Dict[str, List[Tuple[Any, float, str]]] = {}
    for split in ("train", "valid", "test"):
        split_data = _select_split(source, split)
        converted[split] = _convert_split(
            split_data=split_data,
            split=split,
            dataset=dataset_name,
            text_lookup=text_lookup,
            audio_key=audio_key,
            vision_key=vision_key,
            text_key=text_key,
            label_key=label_key,
            id_key=id_key,
            zero_tolerance=zero_tolerance,
        )

    _validate_converted_data(
        converted=converted,
        dataset=dataset_name,
        strict_counts=strict_counts,
    )

    output_paths = {
        "train": destination / "train.pkl",
        "valid": destination / "dev.pkl",
        "test": destination / "test.pkl",
    }
    existing = [
        str(path) for path in output_paths.values() if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them: "
            + ", ".join(existing)
        )

    for split, path in output_paths.items():
        _atomic_pickle_dump(converted[split], path)

    audio_dim, vision_dim = _feature_dimensions(converted["train"])
    print(f"Dataset: {dataset_name}")
    print(
        "Samples: "
        f"train={len(converted['train'])}, "
        f"valid={len(converted['valid'])}, "
        f"test={len(converted['test'])}"
    )
    print(f"Feature dimensions: audio={audio_dim}, vision={vision_dim}")
    print(f"Output directory: {destination}")

    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert preprocessed multimodal data into the pickle format "
            "required by CLCFI."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mosi", "mosei", "sims", "ch-sims"],
    )
    parser.add_argument("--source-pkl", required=True)
    parser.add_argument("--label-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--audio-key", default=None)
    parser.add_argument("--vision-key", default=None)
    parser.add_argument("--text-key", default=None)
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--id-key", default=None)
    parser.add_argument("--zero-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--skip-count-check",
        action="store_true",
        help="Do not enforce the dataset split counts reported in the paper.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    create_dataset(
        dataset=args.dataset,
        source_pickle=args.source_pkl,
        output_dir=args.output_dir,
        label_csv=args.label_csv,
        audio_key=args.audio_key,
        vision_key=args.vision_key,
        text_key=args.text_key,
        label_key=args.label_key,
        id_key=args.id_key,
        zero_tolerance=args.zero_tolerance,
        strict_counts=not args.skip_count_check,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
