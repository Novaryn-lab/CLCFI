from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


def _require_finite(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf values.")


def _validate_dropout(dropout: float) -> float:
    dropout = float(dropout)
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    return dropout


def _validate_attention_configuration(
    model_dim: int,
    num_heads: int,
) -> None:
    if model_dim < 1:
        raise ValueError("model_dim must be positive.")
    if num_heads < 1:
        raise ValueError("num_heads must be positive.")
    if model_dim % num_heads != 0:
        raise ValueError("model_dim must be divisible by num_heads.")


def _validate_sequence(
    value: Tensor,
    name: str,
    feature_dim: Optional[int] = None,
) -> Tuple[int, int, int]:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 3:
        raise ValueError(
            f"{name} must have shape [batch, sequence, feature], "
            f"but found {tuple(value.shape)}."
        )

    batch_size, sequence_length, actual_dim = value.shape
    if batch_size < 1 or sequence_length < 1 or actual_dim < 1:
        raise ValueError(f"{name} must have positive dimensions.")
    if feature_dim is not None and actual_dim != feature_dim:
        raise ValueError(
            f"{name} feature dimension is {actual_dim}, "
            f"but expected {feature_dim}."
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype.")

    _require_finite(name, value)
    return batch_size, sequence_length, actual_dim


def _normalize_padding_mask(
    padding_mask: Optional[Tensor],
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    name: str,
) -> Optional[Tensor]:

    if padding_mask is None:
        return None
    if not isinstance(padding_mask, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None.")
    if padding_mask.ndim != 2:
        raise ValueError(
            f"{name} must have shape [batch, sequence], "
            f"but found {tuple(padding_mask.shape)}."
        )
    if tuple(padding_mask.shape) != (batch_size, sequence_length):
        raise ValueError(
            f"{name} must have shape {(batch_size, sequence_length)}, "
            f"but found {tuple(padding_mask.shape)}."
        )
    if padding_mask.device != device:
        raise ValueError(f"{name} and its sequence must be on the same device.")

    if padding_mask.dtype != torch.bool:
        binary = torch.logical_or(padding_mask == 0, padding_mask == 1)
        if not bool(binary.all()):
            raise ValueError(f"{name} must contain only bool or 0/1 values.")
        padding_mask = padding_mask.to(dtype=torch.bool)

    if bool(padding_mask.all(dim=1).any()):
        raise ValueError(
            f"{name} contains a sample whose entire sequence is padded."
        )
    return padding_mask


def _right_padded_lengths(
    padding_mask: Optional[Tensor],
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    name: str,
) -> Tuple[Optional[Tensor], Tensor]:
    normalized = _normalize_padding_mask(
        padding_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=device,
        name=name,
    )

    if normalized is None:
        lengths = torch.full(
            (batch_size,),
            sequence_length,
            dtype=torch.long,
            device=device,
        )
        return None, lengths

    seen_padding = normalized.cumsum(dim=1) > 0
    internal_valid = seen_padding & (~normalized)
    if bool(internal_valid.any()):
        raise ValueError(
            f"{name} must describe trailing right padding; "
            "valid positions cannot appear after a padded position."
        )

    lengths = (~normalized).sum(dim=1, dtype=torch.long)
    if bool((lengths <= 0).any()):
        raise ValueError(f"{name} contains an empty sequence.")
    return normalized, lengths


def masked_mean(
    x: Tensor,
    padding_mask: Optional[Tensor],
) -> Tensor:

    batch_size, sequence_length, _ = _validate_sequence(x, "x")
    mask = _normalize_padding_mask(
        padding_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=x.device,
        name="padding_mask",
    )

    if mask is None:
        result = x.mean(dim=1)
    else:
        valid = (~mask).unsqueeze(-1).to(dtype=x.dtype)
        counts = valid.sum(dim=1)
        if bool((counts <= 0).any()):
            raise ValueError(
                "Every sample must contain at least one valid sequence position."
            )
        result = (x * valid).sum(dim=1) / counts

    _require_finite("masked_mean_output", result)
    return result


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(
        self,
        model_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if model_dim < 1:
            raise ValueError("model_dim must be positive.")
        self.model_dim = int(model_dim)
        self.dropout = nn.Dropout(_validate_dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        _, sequence_length, model_dim = _validate_sequence(
            x,
            "position_input",
            self.model_dim,
        )

        position = torch.arange(
            sequence_length,
            device=x.device,
            dtype=torch.float32,
        ).unsqueeze(1)
        even_indices = torch.arange(
            0,
            model_dim,
            2,
            device=x.device,
            dtype=torch.float32,
        )
        div_term = torch.exp(
            even_indices * (-torch.log(torch.tensor(10000.0, device=x.device)) / model_dim)
        )

        encoding = torch.zeros(
            sequence_length,
            model_dim,
            device=x.device,
            dtype=torch.float32,
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        if model_dim > 1:
            cosine_width = encoding[:, 1::2].shape[1]
            encoding[:, 1::2] = torch.cos(
                position * div_term[:cosine_width]
            )

        output = x + encoding.to(dtype=x.dtype).unsqueeze(0)
        output = self.dropout(output)
        _require_finite("position_output", output)
        return output


class TextEncoder(nn.Module):

    def __init__(
        self,
        model_name: str,
        model_dim: int,
        dropout: float,
        precomputed_dim: Optional[int] = None,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()

        if model_dim < 1:
            raise ValueError("model_dim must be positive.")
        dropout = _validate_dropout(dropout)
        if precomputed_dim is not None and precomputed_dim < 1:
            raise ValueError("precomputed_dim must be positive when provided.")

        self.model_dim = int(model_dim)
        self.precomputed_dim = precomputed_dim
        self.freeze_bert = bool(freeze_bert)

        if precomputed_dim is None:
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError(
                    "model_name must be a non-empty local BERT path."
                )

            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "The transformers package is required for the BERT encoder."
                ) from exc

            model_reference = model_name.strip()
            model_path = Path(model_reference).expanduser()
            local_reference = model_path.is_absolute() or model_path.exists()

            if model_path.is_absolute() and not model_path.is_dir():
                raise FileNotFoundError(
                    f"Local BERT directory does not exist: {model_path}"
                )

            try:
                self.bert = AutoModel.from_pretrained(
                    model_reference,
                    local_files_only=local_reference,
                )
            except OSError as exc:
                mode = "local BERT directory" if local_reference else "BERT identifier"
                raise RuntimeError(
                    f"Failed to load {mode}: {model_reference}"
                ) from exc

            input_dim = int(self.bert.config.hidden_size)
            for parameter in self.bert.parameters():
                parameter.requires_grad = not self.freeze_bert
        else:
            self.bert = None
            input_dim = int(precomputed_dim)
            self.freeze_bert = False

        self.projection = nn.Sequential(
            nn.Linear(input_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.Dropout(dropout),
        )

    @property
    def bert_is_trainable(self) -> bool:
        if self.bert is None:
            return False
        return any(parameter.requires_grad for parameter in self.bert.parameters())

    def set_bert_trainable(self, trainable: bool) -> None:

        if self.bert is None:
            raise RuntimeError(
                "BERT trainability cannot be changed in precomputed-feature mode."
            )

        trainable = bool(trainable)
        for parameter in self.bert.parameters():
            parameter.requires_grad = trainable
        self.freeze_bert = not trainable

        if self.freeze_bert:
            self.bert.eval()
        else:
            self.bert.train(self.training)

    def train(self, mode: bool = True) -> "TextEncoder":
        super().train(mode)
        if self.bert is not None and self.freeze_bert:
            self.bert.eval()
        return self

    @staticmethod
    def _validate_token_inputs(
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor],
    ) -> None:
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, text_length].")
        if input_ids.size(0) < 1 or input_ids.size(1) < 1:
            raise ValueError("input_ids must not be empty.")
        if input_ids.dtype not in {
            torch.int32,
            torch.int64,
            torch.long,
        }:
            raise TypeError("input_ids must use an integer dtype.")

        if not isinstance(attention_mask, Tensor):
            raise TypeError("attention_mask must be a torch.Tensor.")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids."
            )
        if attention_mask.device != input_ids.device:
            raise ValueError(
                "attention_mask and input_ids must be on the same device."
            )
        if not bool(
            torch.logical_or(attention_mask == 0, attention_mask == 1).all()
        ):
            raise ValueError("attention_mask must contain only 0/1 values.")
        if bool(attention_mask.eq(0).all(dim=1).any()):
            raise ValueError(
                "Each text sample must contain at least one valid token."
            )

        if token_type_ids is not None:
            if not isinstance(token_type_ids, Tensor):
                raise TypeError("token_type_ids must be a torch.Tensor or None.")
            if token_type_ids.shape != input_ids.shape:
                raise ValueError(
                    "token_type_ids must have the same shape as input_ids."
                )
            if token_type_ids.device != input_ids.device:
                raise ValueError(
                    "token_type_ids and input_ids must be on the same device."
                )
            if token_type_ids.dtype not in {
                torch.int32,
                torch.int64,
                torch.long,
            }:
                raise TypeError("token_type_ids must use an integer dtype.")

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        text_features: Optional[Tensor] = None,
    ) -> Tensor:
        if self.bert is None:
            if text_features is None:
                raise ValueError(
                    "text_features are required in precomputed-feature mode."
                )
            _validate_sequence(
                text_features,
                "text_features",
                int(self.precomputed_dim),
            )
            if input_ids is not None or token_type_ids is not None:
                raise ValueError(
                    "Do not provide input_ids or token_type_ids in "
                    "precomputed-feature mode."
                )
            hidden = text_features
        else:
            if text_features is not None:
                raise ValueError(
                    "text_features cannot be provided when the BERT encoder is active."
                )
            if input_ids is None or attention_mask is None:
                raise ValueError(
                    "input_ids and attention_mask are required for BERT."
                )
            self._validate_token_inputs(
                input_ids,
                attention_mask,
                token_type_ids,
            )

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids

            if self.freeze_bert:
                with torch.no_grad():
                    hidden = self.bert(**kwargs).last_hidden_state
            else:
                hidden = self.bert(**kwargs).last_hidden_state

        _require_finite("text_hidden", hidden)
        output = self.projection(hidden)
        _require_finite("text_encoder_output", output)
        return output


class TemporalEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        transformer_layers: int,
        lstm_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        _validate_attention_configuration(model_dim, num_heads)
        if transformer_layers < 1:
            raise ValueError("transformer_layers must be positive.")
        if lstm_layers < 1:
            raise ValueError("lstm_layers must be positive.")
        dropout = _validate_dropout(dropout)

        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)

        self.input_projection = nn.Linear(self.input_dim, self.model_dim)
        self.position_encoding = SinusoidalPositionalEncoding(
            self.model_dim,
            dropout,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.lstm = nn.LSTM(
            input_size=self.model_dim,
            hidden_size=self.model_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.model_dim)

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor],
    ) -> Tensor:
        batch_size, sequence_length, _ = _validate_sequence(
            x,
            "temporal_input",
            self.input_dim,
        )
        normalized_mask, lengths = _right_padded_lengths(
            padding_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=x.device,
            name="temporal_padding_mask",
        )

        encoded = self.input_projection(x)
        encoded = self.position_encoding(encoded)

        if normalized_mask is not None:
            encoded = encoded.masked_fill(
                normalized_mask.unsqueeze(-1),
                0.0,
            )

        encoded = self.transformer(
            encoded,
            src_key_padding_mask=normalized_mask,
        )

        packed = pack_padded_sequence(
            encoded,
            lengths=lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        encoded, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence_length,
        )

        encoded = self.norm(encoded)
        if normalized_mask is not None:
            encoded = encoded.masked_fill(
                normalized_mask.unsqueeze(-1),
                0.0,
            )

        _require_finite("temporal_encoder_output", encoded)
        return encoded


class CrossModalLayer(nn.Module):

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        _validate_attention_configuration(model_dim, num_heads)
        dropout = _validate_dropout(dropout)
        self.model_dim = int(model_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(self.model_dim)
        self.norm2 = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(self.model_dim, 4 * self.model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.model_dim, self.model_dim),
        )

    def forward(
        self,
        text: Tensor,
        nonverbal: Tensor,
        text_padding_mask: Optional[Tensor],
        nonverbal_padding_mask: Optional[Tensor],
    ) -> Tensor:
        text_batch, text_length, _ = _validate_sequence(
            text,
            "cross_modal_text",
            self.model_dim,
        )
        nonverbal_batch, nonverbal_length, _ = _validate_sequence(
            nonverbal,
            "cross_modal_nonverbal",
            self.model_dim,
        )
        if text_batch != nonverbal_batch:
            raise ValueError(
                "Text and nonverbal sequences must share the same batch size."
            )
        if text.device != nonverbal.device:
            raise ValueError(
                "Text and nonverbal sequences must be on the same device."
            )
        if text.dtype != nonverbal.dtype:
            raise ValueError(
                "Text and nonverbal sequences must use the same dtype."
            )

        text_mask = _normalize_padding_mask(
            text_padding_mask,
            batch_size=text_batch,
            sequence_length=text_length,
            device=text.device,
            name="text_padding_mask",
        )
        nonverbal_mask = _normalize_padding_mask(
            nonverbal_padding_mask,
            batch_size=nonverbal_batch,
            sequence_length=nonverbal_length,
            device=nonverbal.device,
            name="nonverbal_padding_mask",
        )

        attended, _ = self.attention(
            query=text,
            key=nonverbal,
            value=nonverbal,
            key_padding_mask=nonverbal_mask,
            need_weights=False,
        )
        output = self.norm1(
            text + self.dropout(attended)
        )
        output = self.norm2(
            output + self.dropout(self.ffn(output))
        )

        if text_mask is not None:
            output = output.masked_fill(
                text_mask.unsqueeze(-1),
                0.0,
            )

        _require_finite("cross_modal_layer_output", output)
        return output


class CrossModalBranch(nn.Module):
    """Text-guided G_ta/G_tv interaction from Eq. (4)."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        _validate_attention_configuration(model_dim, num_heads)
        if layers < 1:
            raise ValueError("Cross-modal layer count must be positive.")
        _validate_dropout(dropout)

        self.model_dim = int(model_dim)
        self.layers = nn.ModuleList(
            [
                CrossModalLayer(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        text: Tensor,
        nonverbal: Tensor,
        text_mask: Optional[Tensor],
        nonverbal_mask: Optional[Tensor],
    ) -> Tensor:
        _validate_sequence(
            text,
            "branch_text",
            self.model_dim,
        )
        _validate_sequence(
            nonverbal,
            "branch_nonverbal",
            self.model_dim,
        )

        output = text
        for layer in self.layers:
            output = layer(
                text=output,
                nonverbal=nonverbal,
                text_padding_mask=text_mask,
                nonverbal_padding_mask=nonverbal_mask,
            )

        _require_finite("cross_modal_branch_output", output)
        return output


class FusionBlock(nn.Module):

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        _validate_attention_configuration(model_dim, num_heads)
        dropout = _validate_dropout(dropout)
        self.model_dim = int(model_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.model_dim, 4 * self.model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.model_dim, self.model_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.model_dim)

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor],
    ) -> Tensor:
        batch_size, sequence_length, _ = _validate_sequence(
            x,
            "fusion_input",
            self.model_dim,
        )
        mask = _normalize_padding_mask(
            padding_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=x.device,
            name="fusion_padding_mask",
        )

        attended, _ = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mask,
            need_weights=False,
        )
        transformed = self.ffn(attended)
        output = self.norm(
            x + self.dropout(transformed)
        )

        if mask is not None:
            output = output.masked_fill(
                mask.unsqueeze(-1),
                0.0,
            )

        _require_finite("fusion_block_output", output)
        return output


class FusionEncoder(nn.Module):

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        _validate_attention_configuration(model_dim, num_heads)
        if layers < 1:
            raise ValueError("Fusion layer count must be positive.")
        _validate_dropout(dropout)

        self.model_dim = int(model_dim)
        self.layers = nn.ModuleList(
            [
                FusionBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor],
    ) -> Tensor:
        _validate_sequence(
            x,
            "fusion_encoder_input",
            self.model_dim,
        )

        output = x
        for layer in self.layers:
            output = layer(output, padding_mask)

        _require_finite("fusion_encoder_output", output)
        return output
