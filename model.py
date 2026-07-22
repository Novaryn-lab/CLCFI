from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import CLCFIConfig
from .modules import (
    CounterfactualIntervention,
    CrossModalBranch,
    CueSelectionModule,
    FusionEncoder,
    SufficiencyNecessityEvaluator,
    TemporalEncoder,
    TextEncoder,
)
from .modules.encoders import masked_mean


@dataclass
class CLCFIOutput:


    predictions: Tensor
    loss: Optional[Tensor]
    prediction_loss: Optional[Tensor]
    ccl_loss: Tensor
    consistency_loss: Tensor
    weight_ta: Tensor
    weight_tv: Tensor
    sufficiency_ta: Tensor
    sufficiency_tv: Tensor
    necessity_ta: Tensor
    necessity_tv: Tensor
    cue_mask_ta: Tensor
    cue_mask_tv: Tensor
    auxiliary_predictions: Dict[str, Tensor]


class CLCFI(nn.Module):

    def __init__(
        self,
        config: CLCFIConfig,
        audio_dim: int,
        vision_dim: int,
    ) -> None:
        super().__init__()
        config.validate()

        if audio_dim < 1:
            raise ValueError("audio_dim must be positive.")
        if vision_dim < 1:
            raise ValueError("vision_dim must be positive.")
        if getattr(config, "apply_weight_twice", False):
            raise ValueError(
                "apply_weight_twice must be False. SNEOutput.corrected_ta and "
                "corrected_tv already contain the branch weights from Eq. (24)."
            )

        self.config = config
        self.audio_dim = int(audio_dim)
        self.vision_dim = int(vision_dim)
        self.model_dim = int(config.model_dim)

        self.text_encoder = TextEncoder(
            model_name=config.text_model,
            model_dim=self.model_dim,
            dropout=config.dropout,
            precomputed_dim=config.precomputed_text_dim,
        )
        self.audio_encoder = TemporalEncoder(
            input_dim=self.audio_dim,
            model_dim=self.model_dim,
            num_heads=config.num_heads,
            transformer_layers=config.temporal_layers,
            lstm_layers=config.lstm_layers,
            dropout=config.dropout,
        )
        self.vision_encoder = TemporalEncoder(
            input_dim=self.vision_dim,
            model_dim=self.model_dim,
            num_heads=config.num_heads,
            transformer_layers=config.temporal_layers,
            lstm_layers=config.lstm_layers,
            dropout=config.dropout,
        )

        self.ta_branch = CrossModalBranch(
            model_dim=self.model_dim,
            num_heads=config.num_heads,
            layers=config.cross_layers,
            dropout=config.dropout,
        )
        self.tv_branch = CrossModalBranch(
            model_dim=self.model_dim,
            num_heads=config.num_heads,
            layers=config.cross_layers,
            dropout=config.dropout,
        )

        self.cue_selection = CueSelectionModule(
            feature_dim=self.model_dim,
            prototype_size=config.prototype_size,
            temperature=config.temperature,
            prototype_momentum=config.prototype_momentum,
            positive_weight=config.positive_cue_weight,
            dropout=config.dropout,
        )
        self.ta_intervention = CounterfactualIntervention(
            feature_dim=self.model_dim,
            ratio=config.intervention_ratio,
        )
        self.tv_intervention = CounterfactualIntervention(
            feature_dim=self.model_dim,
            ratio=config.intervention_ratio,
        )

        self.evaluator = SufficiencyNecessityEvaluator(
            feature_dim=self.model_dim,
            num_outputs=1,
            task_type="regression",
            dropout=config.dropout,
        )
        self.fusion = FusionEncoder(
            model_dim=self.model_dim,
            num_heads=config.num_heads,
            layers=config.fusion_layers,
            dropout=config.dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.model_dim, 1),
        )

    @staticmethod
    def _validate_sequence(
        tensor: Tensor,
        name: str,
        expected_feature_dim: int,
    ) -> Tuple[int, int]:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.ndim != 3:
            raise ValueError(
                f"{name} must have shape [batch, sequence, feature], "
                f"but found {tuple(tensor.shape)}."
            )
        if tensor.size(0) < 1 or tensor.size(1) < 1:
            raise ValueError(f"{name} must contain at least one sample and one step.")
        if tensor.size(2) != expected_feature_dim:
            raise ValueError(
                f"{name} feature dimension is {tensor.size(2)}, but the model "
                f"expects {expected_feature_dim}."
            )
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype.")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf values.")
        return tensor.size(0), tensor.size(1)

    @staticmethod
    def _normalize_padding_mask(
        mask: Optional[Tensor],
        name: str,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        if mask is None:
            return None
        if not isinstance(mask, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None.")
        if mask.ndim != 2 or tuple(mask.shape) != (batch_size, sequence_length):
            raise ValueError(
                f"{name} must have shape {(batch_size, sequence_length)}, "
                f"but found {tuple(mask.shape)}."
            )
        if mask.device != device:
            raise ValueError(f"{name} and its input sequence must be on the same device.")

        if mask.dtype != torch.bool:
            valid_values = torch.logical_or(mask == 0, mask == 1)
            if not bool(valid_values.all()):
                raise ValueError(f"{name} must contain only boolean or 0/1 values.")
            mask = mask.to(dtype=torch.bool)

        if bool(mask.all(dim=1).any()):
            raise ValueError(f"{name} contains a sample whose entire sequence is padded.")
        return mask

    def _prepare_text_inputs(
        self,
        batch_size: int,
        input_ids: Optional[Tensor],
        attention_mask: Optional[Tensor],
        token_type_ids: Optional[Tensor],
        text_features: Optional[Tensor],
        text_padding_mask: Optional[Tensor],
        device: torch.device,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Tensor]:
        precomputed_mode = self.config.precomputed_text_dim is not None

        if precomputed_mode:
            if text_features is None:
                raise ValueError(
                    "text_features are required when precomputed_text_dim is set."
                )
            text_batch, text_length = self._validate_sequence(
                text_features,
                "text_features",
                int(self.config.precomputed_text_dim),
            )
            if text_batch != batch_size:
                raise ValueError(
                    "text_features, audio and vision must share the same batch size."
                )
            if text_features.device != device:
                raise ValueError("All modality tensors must be on the same device.")

            if input_ids is not None or token_type_ids is not None:
                raise ValueError(
                    "Do not provide input_ids or token_type_ids in precomputed text mode."
                )

            if attention_mask is not None:
                if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (
                    batch_size,
                    text_length,
                ):
                    raise ValueError(
                        "attention_mask must match text_features on batch and "
                        "sequence dimensions."
                    )
                if attention_mask.device != device:
                    raise ValueError(
                        "attention_mask and text_features must be on the same device."
                    )
                derived_padding_mask = attention_mask.eq(0)
                if text_padding_mask is not None:
                    provided_padding_mask = self._normalize_padding_mask(
                        text_padding_mask,
                        "text_padding_mask",
                        batch_size,
                        text_length,
                        device,
                    )
                    if not torch.equal(provided_padding_mask, derived_padding_mask):
                        raise ValueError(
                            "text_padding_mask is inconsistent with attention_mask."
                        )
                    text_padding_mask = provided_padding_mask
                else:
                    text_padding_mask = self._normalize_padding_mask(
                        derived_padding_mask,
                        "text_padding_mask",
                        batch_size,
                        text_length,
                        device,
                    )
            else:
                text_padding_mask = self._normalize_padding_mask(
                    text_padding_mask,
                    "text_padding_mask",
                    batch_size,
                    text_length,
                    device,
                )
                if text_padding_mask is None:
                    text_padding_mask = torch.zeros(
                        (batch_size, text_length),
                        dtype=torch.bool,
                        device=device,
                    )

            return None, attention_mask, None, text_features, text_padding_mask

        if text_features is not None:
            raise ValueError(
                "text_features can be used only when precomputed_text_dim is set."
            )
        if input_ids is None or attention_mask is None:
            raise ValueError(
                "input_ids and attention_mask are required for the BERT text encoder."
            )
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, text_length], but found "
                f"{tuple(input_ids.shape)}."
            )
        if input_ids.size(0) != batch_size or input_ids.size(1) < 1:
            raise ValueError(
                "input_ids must share the audio/vision batch size and contain "
                "at least one token."
            )
        if input_ids.device != device:
            raise ValueError("All modality tensors must be on the same device.")

        text_length = input_ids.size(1)
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (
            batch_size,
            text_length,
        ):
            raise ValueError("attention_mask must have the same shape as input_ids.")
        if attention_mask.device != device:
            raise ValueError("attention_mask and input_ids must be on the same device.")
        if not bool(torch.logical_or(attention_mask == 0, attention_mask == 1).all()):
            raise ValueError("attention_mask must contain only 0/1 values.")
        if bool(attention_mask.eq(0).all(dim=1).any()):
            raise ValueError("attention_mask contains a sample with no valid text token.")

        if token_type_ids is not None:
            if token_type_ids.ndim != 2 or token_type_ids.shape != input_ids.shape:
                raise ValueError("token_type_ids must have the same shape as input_ids.")
            if token_type_ids.device != device:
                raise ValueError(
                    "token_type_ids and input_ids must be on the same device."
                )

        derived_padding_mask = attention_mask.eq(0)
        if text_padding_mask is None:
            text_padding_mask = derived_padding_mask
        else:
            text_padding_mask = self._normalize_padding_mask(
                text_padding_mask,
                "text_padding_mask",
                batch_size,
                text_length,
                device,
            )
            if not torch.equal(text_padding_mask, derived_padding_mask):
                raise ValueError(
                    "text_padding_mask is inconsistent with attention_mask."
                )

        return input_ids, attention_mask, token_type_ids, None, text_padding_mask

    @staticmethod
    def _prepare_labels(
        labels: Optional[Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        if labels is None:
            return None
        if not isinstance(labels, Tensor):
            raise TypeError("labels must be a torch.Tensor or None.")
        if labels.device != device:
            raise ValueError("labels and model inputs must be on the same device.")
        if labels.ndim == 1:
            if labels.size(0) != batch_size:
                raise ValueError("One-dimensional labels must have shape [batch].")
            labels = labels.unsqueeze(-1)
        elif labels.ndim == 2:
            if tuple(labels.shape) != (batch_size, 1):
                raise ValueError("Two-dimensional labels must have shape [batch, 1].")
        else:
            raise ValueError("labels must have shape [batch] or [batch, 1].")

        labels = labels.to(dtype=torch.float32)
        if not torch.isfinite(labels).all():
            raise ValueError("labels contain NaN or Inf values.")
        return labels

    @staticmethod
    def _prediction_loss(prediction: Tensor, labels: Tensor) -> Tensor:
        if prediction.shape != labels.shape:
            raise ValueError(
                f"prediction shape {tuple(prediction.shape)} does not match "
                f"label shape {tuple(labels.shape)}."
            )
        return F.l1_loss(prediction, labels)

    def forward(
        self,
        audio: Tensor,
        vision: Tensor,
        audio_padding_mask: Optional[Tensor] = None,
        vision_padding_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        text_features: Optional[Tensor] = None,
        text_padding_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> CLCFIOutput:
        audio_batch, audio_length = self._validate_sequence(
            audio,
            "audio",
            self.audio_dim,
        )
        vision_batch, vision_length = self._validate_sequence(
            vision,
            "vision",
            self.vision_dim,
        )
        if audio_batch != vision_batch:
            raise ValueError("audio and vision must share the same batch size.")
        if audio.device != vision.device:
            raise ValueError("audio and vision must be on the same device.")

        device = audio.device
        batch_size = audio_batch
        audio_padding_mask = self._normalize_padding_mask(
            audio_padding_mask,
            "audio_padding_mask",
            batch_size,
            audio_length,
            device,
        )
        vision_padding_mask = self._normalize_padding_mask(
            vision_padding_mask,
            "vision_padding_mask",
            batch_size,
            vision_length,
            device,
        )

        (
            input_ids,
            attention_mask,
            token_type_ids,
            text_features,
            text_padding_mask,
        ) = self._prepare_text_inputs(
            batch_size=batch_size,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            text_features=text_features,
            text_padding_mask=text_padding_mask,
            device=device,
        )
        labels = self._prepare_labels(labels, batch_size, device)

        text_encoded = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            text_features=text_features,
        )
        audio_encoded = self.audio_encoder(audio, audio_padding_mask)
        vision_encoded = self.vision_encoder(vision, vision_padding_mask)

        f_ta = self.ta_branch(
            text_encoded,
            audio_encoded,
            text_padding_mask,
            audio_padding_mask,
        )
        f_tv = self.tv_branch(
            text_encoded,
            vision_encoded,
            text_padding_mask,
            vision_padding_mask,
        )

        cues = self.cue_selection(f_ta, f_tv, text_padding_mask)
        ta_intervened = self.ta_intervention(
            cues.ta.selected,
            cues.ta.negative,
            cues.ta.score,
            text_padding_mask,
        )
        tv_intervened = self.tv_intervention(
            cues.tv.selected,
            cues.tv.negative,
            cues.tv.score,
            text_padding_mask,
        )


        evaluator_target = labels if self.training else None
        sne = self.evaluator(
            ta_intervened,
            tv_intervened,
            text_padding_mask,
            evaluator_target,
        )


        fusion_input = torch.cat(
            [sne.corrected_tv, sne.corrected_ta],
            dim=1,
        )
        fusion_mask = torch.cat(
            [text_padding_mask, text_padding_mask],
            dim=1,
        )
        fused_sequence = self.fusion(fusion_input, fusion_mask)
        pooled_fusion = masked_mean(fused_sequence, fusion_mask)
        prediction = self.regressor(pooled_fusion)

        prediction_loss: Optional[Tensor] = None
        total_loss: Optional[Tensor] = None
        if labels is not None:
            prediction_loss = self._prediction_loss(prediction, labels)
            total_loss = (
                prediction_loss
                + self.config.lambda_ccl * cues.ccl_loss
                + self.config.lambda_consistency * cues.consistency_loss
            )

        return CLCFIOutput(
            predictions=prediction,
            loss=total_loss,
            prediction_loss=prediction_loss,
            ccl_loss=cues.ccl_loss,
            consistency_loss=cues.consistency_loss,
            weight_ta=sne.weight_ta.reshape(batch_size),
            weight_tv=sne.weight_tv.reshape(batch_size),
            sufficiency_ta=sne.sufficiency_ta,
            sufficiency_tv=sne.sufficiency_tv,
            necessity_ta=sne.necessity_ta,
            necessity_tv=sne.necessity_tv,
            cue_mask_ta=ta_intervened.cue_mask,
            cue_mask_tv=tv_intervened.cue_mask,
            auxiliary_predictions=sne.predictions,
        )
