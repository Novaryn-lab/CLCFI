import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .config import CLCFIConfig


class CLCFIPickleDataset(Dataset):


    FILES = {
        "train": "train.pkl",
        "valid": "dev.pkl",
        "test": "test.pkl",
    }

    _PADDING_TOKENS = {"", "<pad>", "[pad]", "pad"}

    def __init__(self, data_dir: str, split: str, dataset_name: str) -> None:
        if split not in self.FILES:
            valid_splits = ", ".join(self.FILES)
            raise ValueError(
                f"Unknown split '{split}'. Expected one of: {valid_splits}."
            )

        normalized_dataset = dataset_name.strip().lower().replace("_", "-")
        if normalized_dataset == "ch-sims":
            normalized_dataset = "sims"
        if normalized_dataset not in {"mosi", "mosei", "sims"}:
            raise ValueError(
                "dataset_name must be one of: mosi, mosei, sims, ch-sims."
            )

        self.split = split
        self.dataset_name = normalized_dataset
        self.path = Path(data_dir).expanduser() / self.FILES[split]

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Missing dataset cache: {self.path}. Expected train.pkl, "
                "dev.pkl and test.pkl under the configured data directory."
            )

        try:
            with self.path.open("rb") as handle:
                samples = pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError) as exc:
            raise RuntimeError(f"Failed to read pickle cache: {self.path}") from exc

        if not isinstance(samples, (list, tuple)):
            raise TypeError(
                f"{self.path} must contain a list or tuple of samples, "
                f"but found {type(samples).__name__}."
            )
        if len(samples) == 0:
            raise ValueError(f"Dataset split is empty: {self.path}")

        self.samples = samples

        first = self._parse_sample(0, check_feature_dims=False)
        self._audio_dim = int(first["audio"].shape[1])
        self._vision_dim = int(first["vision"].shape[1])

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _decode_token(token: Any) -> str:
        if isinstance(token, (bytes, np.bytes_)):
            return token.decode("utf-8", errors="replace").strip()
        return str(token).strip()

    def _words_to_text(self, words: Any, index: int) -> str:
        if isinstance(words, (str, bytes, np.bytes_)):
            tokens = [words]
        elif isinstance(words, np.ndarray):
            tokens = words.reshape(-1).tolist()
        elif isinstance(words, Sequence):
            tokens = list(words)
        else:
            raise TypeError(
                f"{self.path}, sample {index}: words must be a string or token "
                f"sequence, but found {type(words).__name__}."
            )

        clean_tokens: List[str] = []
        for token in tokens:
            decoded = self._decode_token(token)
            if decoded.lower() in self._PADDING_TOKENS:
                continue
            clean_tokens.append(decoded)

        if not clean_tokens:
            raise ValueError(
                f"{self.path}, sample {index}: text is empty after removing "
                "padding tokens."
            )


        separator = "" if self.dataset_name == "sims" else " "
        return separator.join(clean_tokens)

    def _as_feature_matrix(
        self,
        value: Any,
        modality: str,
        index: int,
    ) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{self.path}, sample {index}: {modality} features cannot be "
                "converted to float32."
            ) from exc

        if array.ndim != 2:
            raise ValueError(
                f"{self.path}, sample {index}: {modality} features must have "
                f"shape [sequence_length, feature_dim], but found {array.shape}."
            )
        if array.shape[0] < 1:
            raise ValueError(
                f"{self.path}, sample {index}: {modality} sequence is empty."
            )
        if array.shape[1] < 1:
            raise ValueError(
                f"{self.path}, sample {index}: {modality} feature dimension "
                "must be positive."
            )
        if not np.isfinite(array).all():
            raise ValueError(
                f"{self.path}, sample {index}: {modality} features contain "
                "NaN or Inf values."
            )

        return np.ascontiguousarray(array)

    def _read_length(
        self,
        features: Sequence[Any],
        position: int,
        available_length: int,
        modality: str,
        index: int,
    ) -> int:
        if len(features) <= position:
            return available_length

        raw = np.asarray(features[position]).reshape(-1)
        if raw.size != 1:
            raise ValueError(
                f"{self.path}, sample {index}: {modality}_length must contain "
                "exactly one scalar value."
            )

        try:
            numeric = float(raw[0])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{self.path}, sample {index}: {modality}_length is not numeric."
            ) from exc

        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(
                f"{self.path}, sample {index}: {modality}_length must be a "
                f"finite integer, but found {numeric}."
            )

        length = int(numeric)
        if length < 1:
            raise ValueError(
                f"{self.path}, sample {index}: {modality}_length must be "
                "positive."
            )
        if length > available_length:
            raise ValueError(
                f"{self.path}, sample {index}: {modality}_length={length} "
                f"exceeds the stored sequence length {available_length}."
            )
        return length

    def _read_label(self, value: Any, index: int) -> float:
        raw = np.asarray(value).reshape(-1)
        if raw.size == 0:
            raise ValueError(f"{self.path}, sample {index}: label is empty.")

        try:
            label = float(raw[0])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{self.path}, sample {index}: label is not numeric."
            ) from exc

        if not np.isfinite(label):
            raise ValueError(
                f"{self.path}, sample {index}: label contains NaN or Inf."
            )
        return label

    def _read_sample_id(self, sample: Sequence[Any], index: int) -> Any:

        if len(sample) >= 3:
            sample_id = sample[-1]
            if isinstance(sample_id, np.ndarray) and sample_id.size == 1:
                return sample_id.reshape(-1)[0].item()
            return sample_id
        return f"{self.split}:{index}"

    def _parse_sample(
        self,
        index: int,
        check_feature_dims: bool = True,
    ) -> Dict[str, Any]:
        sample = self.samples[index]
        if not isinstance(sample, (list, tuple)):
            raise TypeError(
                f"{self.path}, sample {index}: expected a list or tuple, "
                f"but found {type(sample).__name__}."
            )
        if len(sample) < 2:
            raise ValueError(
                f"{self.path}, sample {index}: expected at least "
                "(features, label)."
            )

        features = sample[0]
        if not isinstance(features, (list, tuple)):
            raise TypeError(
                f"{self.path}, sample {index}: features must be a list or tuple."
            )
        if len(features) < 4:
            raise ValueError(
                f"{self.path}, sample {index}: expected features in the order "
                "(word_ids, vision, audio, words, ...)."
            )

        vision = self._as_feature_matrix(features[1], "vision", index)
        audio = self._as_feature_matrix(features[2], "audio", index)

        vision_length = self._read_length(
            features, 4, vision.shape[0], "vision", index
        )
        audio_length = self._read_length(
            features, 5, audio.shape[0], "audio", index
        )

        # Remove any cache-level trailing padding. Batch-level padding is added
        # later by CLCFICollator and is accompanied by an explicit padding mask.
        vision = np.ascontiguousarray(vision[:vision_length])
        audio = np.ascontiguousarray(audio[:audio_length])

        if check_feature_dims:
            if audio.shape[1] != self._audio_dim:
                raise ValueError(
                    f"{self.path}, sample {index}: audio feature dimension "
                    f"{audio.shape[1]} does not match expected "
                    f"{self._audio_dim}."
                )
            if vision.shape[1] != self._vision_dim:
                raise ValueError(
                    f"{self.path}, sample {index}: vision feature dimension "
                    f"{vision.shape[1]} does not match expected "
                    f"{self._vision_dim}."
                )

        return {
            "text": self._words_to_text(features[3], index),
            "vision": vision,
            "audio": audio,
            "vision_length": vision_length,
            "audio_length": audio_length,
            "label": self._read_label(sample[1], index),
            "id": self._read_sample_id(sample, index),
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._parse_sample(index)

    @property
    def feature_dims(self) -> Tuple[int, int]:
        """Return (audio_feature_dim, vision_feature_dim)."""
        return self._audio_dim, self._vision_dim


class CLCFICollator:
    """Convert variable-length multimodal samples into one model batch."""

    def __init__(self, tokenizer: Any, max_text_length: int, task_type: str) -> None:
        if max_text_length < 2:
            raise ValueError("max_text_length must be at least 2.")
        if task_type != "regression":
            raise ValueError(
                "The paper implementation requires task_type='regression'."
            )

        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    @staticmethod
    def _padding_mask(lengths: Tensor, max_length: int) -> Tensor:
        if lengths.ndim != 1:
            raise ValueError("lengths must be a one-dimensional tensor.")
        if max_length < 1:
            raise ValueError("max_length must be positive.")

        positions = torch.arange(max_length, dtype=torch.long).unsqueeze(0)
        return positions >= lengths.unsqueeze(1)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty sample list.")

        audio_tensors = [
            torch.as_tensor(sample["audio"], dtype=torch.float32)
            for sample in samples
        ]
        vision_tensors = [
            torch.as_tensor(sample["vision"], dtype=torch.float32)
            for sample in samples
        ]

        audio_dim = audio_tensors[0].shape[1]
        vision_dim = vision_tensors[0].shape[1]
        if any(tensor.ndim != 2 or tensor.shape[1] != audio_dim
               for tensor in audio_tensors):
            raise ValueError("All audio features in a batch must share one dimension.")
        if any(tensor.ndim != 2 or tensor.shape[1] != vision_dim
               for tensor in vision_tensors):
            raise ValueError("All vision features in a batch must share one dimension.")

        audio = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)
        vision = pad_sequence(vision_tensors, batch_first=True, padding_value=0.0)

        audio_lengths = torch.tensor(
            [sample["audio_length"] for sample in samples],
            dtype=torch.long,
        )
        vision_lengths = torch.tensor(
            [sample["vision_length"] for sample in samples],
            dtype=torch.long,
        )

        tokenized = self.tokenizer(
            [sample["text"] for sample in samples],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        if "input_ids" not in tokenized or "attention_mask" not in tokenized:
            raise KeyError(
                "Tokenizer output must contain input_ids and attention_mask."
            )

        labels = torch.tensor(
            [sample["label"] for sample in samples],
            dtype=torch.float32,
        ).unsqueeze(-1)

        batch: Dict[str, Any] = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "audio": audio,
            "vision": vision,
            "audio_padding_mask": self._padding_mask(
                audio_lengths, audio.size(1)
            ),
            "vision_padding_mask": self._padding_mask(
                vision_lengths, vision.size(1)
            ),
            "labels": labels,
            "ids": [sample["id"] for sample in samples],
        }

        if "token_type_ids" in tokenized:
            batch["token_type_ids"] = tokenized["token_type_ids"]

        return batch


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    config: CLCFIConfig,
) -> Tuple[Dict[str, DataLoader], Tuple[int, int]]:
    """Create train/validation/test loaders and return feature dimensions."""

    config.validate()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The transformers package is required to load the BERT tokenizer."
        ) from exc

    model_path = Path(config.text_model).expanduser()
    local_files_only = model_path.is_absolute() or model_path.exists()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.text_model,
            use_fast=True,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        mode = "local directory" if local_files_only else "model identifier"
        raise RuntimeError(
            f"Failed to load tokenizer from {mode}: {config.text_model}"
        ) from exc

    datasets = {
        split: CLCFIPickleDataset(
            data_dir=config.data_dir,
            split=split,
            dataset_name=config.dataset,
        )
        for split in ("train", "valid", "test")
    }

    expected_dims = datasets["train"].feature_dims
    for split in ("valid", "test"):
        if datasets[split].feature_dims != expected_dims:
            raise ValueError(
                f"{split} feature dimensions {datasets[split].feature_dims} "
                f"do not match training dimensions {expected_dims}."
            )

    collator = CLCFICollator(
        tokenizer=tokenizer,
        max_text_length=config.max_text_length,
        task_type=config.task_type,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            worker_init_fn=_seed_worker if config.num_workers > 0 else None,
            generator=generator if split == "train" else None,
            persistent_workers=config.num_workers > 0,
        )
        for split, dataset in datasets.items()
    }

    return loaders, expected_dims


def move_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:

    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
        if key != "ids"
    }
