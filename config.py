import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional


OPENBAYES_HOME = Path("/openbayes/home")
DATA_ROOT = OPENBAYES_HOME / "datasets"
OUTPUT_ROOT = OPENBAYES_HOME / "outputs" / "CLCFI"
PRETRAINED_ROOT = OPENBAYES_HOME / "pretrained_models"


@dataclass
class CLCFIConfig:

    dataset: str = "mosi"
    data_dir: str = str(DATA_ROOT / "MOSI")
    output_dir: str = str(OUTPUT_ROOT)
    text_model: str = str(PRETRAINED_ROOT / "bert-base-uncased")

    task_type: str = "regression"
    num_outputs: int = 1

    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stop: int = 15
    dropout: float = 0.3
    clip_grad_norm: float = 1.0

    model_dim: int = 128
    num_heads: int = 4
    temporal_layers: int = 2
    cross_layers: int = 1
    fusion_layers: int = 1
    lstm_layers: int = 1
    max_text_length: int = 50

    intervention_ratio: float = 0.1  # rho in Table 2
    prototype_size: int = 128  # K in Table 2
    temperature: float = 0.2  # tau in Table 2
    prototype_momentum: float = 0.1
    positive_cue_weight: float = 1.0
    lambda_ccl: float = 0.1
    lambda_consistency: float = 0.1

    apply_weight_twice: bool = False

    seed: int = 1111
    num_workers: int = 0
    device: str = "auto"
    precomputed_text_dim: Optional[int] = None

    @classmethod
    def for_dataset(cls, dataset: str, **overrides: Any) -> "CLCFIConfig":
        normalized = dataset.strip().lower().replace("_", "-")
        aliases = {"ch-sims": "sims", "chsims": "sims"}
        normalized = aliases.get(normalized, normalized)

        if normalized == "mosi":
            config = cls(
                dataset="mosi",
                data_dir=str(DATA_ROOT / "MOSI"),
                text_model=str(PRETRAINED_ROOT / "bert-base-uncased"),
                batch_size=32,
                learning_rate=1e-3,
                dropout=0.3,
            )
        elif normalized == "mosei":
            config = cls(
                dataset="mosei",
                data_dir=str(DATA_ROOT / "MOSEI"),
                text_model=str(PRETRAINED_ROOT / "bert-base-uncased"),
                batch_size=16,
                learning_rate=1e-3,
                dropout=0.3,
            )
        elif normalized == "sims":
            config = cls(
                dataset="sims",
                data_dir=str(DATA_ROOT / "SIMS"),
                text_model=str(PRETRAINED_ROOT / "bert-base-chinese"),
                batch_size=8,
                learning_rate=1e-4,
                dropout=0.2,
            )
        else:
            raise ValueError(
                f"Unsupported dataset: {dataset}. Choose mosi, mosei, sims, or ch-sims."
            )

        return replace(config, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.dataset not in {"mosi", "mosei", "sims"}:
            raise ValueError("dataset must be one of: mosi, mosei, sims")
        if self.task_type != "regression" or self.num_outputs != 1:
            raise ValueError(
                "The paper configuration requires task_type='regression' and num_outputs=1"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.early_stop < 1:
            raise ValueError("early_stop must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.clip_grad_norm <= 0:
            raise ValueError("clip_grad_norm must be positive")
        if self.model_dim < 1 or self.num_heads < 1:
            raise ValueError("model_dim and num_heads must be positive")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if min(
            self.temporal_layers,
            self.cross_layers,
            self.fusion_layers,
            self.lstm_layers,
        ) < 1:
            raise ValueError("all encoder layer counts must be positive")
        if self.max_text_length < 2:
            raise ValueError("max_text_length must be at least 2")
        if not 0.0 < self.intervention_ratio <= 1.0:
            raise ValueError("intervention_ratio must be in (0, 1]")
        if self.prototype_size < 1:
            raise ValueError("prototype_size must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.prototype_momentum <= 1.0:
            raise ValueError("prototype_momentum must be in [0, 1]")
        if self.positive_cue_weight < 0:
            raise ValueError("positive_cue_weight must be non-negative")
        if self.lambda_ccl < 0 or self.lambda_consistency < 0:
            raise ValueError("loss weights must be non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.precomputed_text_dim is not None and self.precomputed_text_dim < 1:
            raise ValueError("precomputed_text_dim must be positive when provided")
        if not (
            self.device == "auto"
            or self.device == "cpu"
            or self.device == "mps"
            or self.device.startswith("cuda")
        ):
            raise ValueError("device must be auto, cpu, mps, cuda, or cuda:<index>")

    def validate_runtime_paths(self) -> None:

        data_path = Path(self.data_dir)
        if not data_path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {data_path}")

        required_files = ("train.pkl", "dev.pkl", "test.pkl")
        missing = [name for name in required_files if not (data_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Dataset directory {data_path} is missing required files: {', '.join(missing)}"
            )

        model_path = Path(self.text_model)
        if model_path.is_absolute() and not model_path.is_dir():
            raise FileNotFoundError(f"Local text model directory does not exist: {model_path}")

        output_path = Path(self.output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"Cannot create output directory: {output_path}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CLCFI")
    parser.add_argument(
        "--dataset",
        choices=["mosi", "mosei", "sims", "ch-sims"],
        default="mosi",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--text-model",
        default=None,
        help="Hugging Face model name or a local BERT directory",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--early-stop", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--clip-grad-norm", type=float, default=None)
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--max-text-length", type=int, default=None)
    parser.add_argument("--rho", dest="intervention_ratio", type=float, default=None)
    parser.add_argument("--prototype-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--prototype-momentum", type=float, default=None)
    parser.add_argument("--positive-cue-weight", type=float, default=None)
    parser.add_argument("--lambda-ccl", type=float, default=None)
    parser.add_argument("--lambda-consistency", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser


def _normalize_path(value: str) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def config_from_args(args: argparse.Namespace) -> CLCFIConfig:
    values = vars(args).copy()
    dataset = values.pop("dataset")
    overrides = {key: value for key, value in values.items() if value is not None}
    config = CLCFIConfig.for_dataset(dataset, **overrides)

    config.data_dir = _normalize_path(config.data_dir)
    config.output_dir = _normalize_path(config.output_dir)

    model_ref = Path(config.text_model).expanduser()
    if model_ref.is_absolute() or config.text_model.startswith("."):
        config.text_model = str(model_ref.resolve(strict=False))

    config.validate()
    config.validate_runtime_paths()
    return config
