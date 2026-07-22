import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .encoders import masked_mean


def _require_finite(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf values.")


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
    name: str = "padding_mask",
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
        raise ValueError(f"{name} and cue tensors must be on the same device.")

    if padding_mask.dtype != torch.bool:
        binary = torch.logical_or(padding_mask == 0, padding_mask == 1)
        if not bool(binary.all()):
            raise ValueError(f"{name} must contain only bool or 0/1 values.")
        padding_mask = padding_mask.to(dtype=torch.bool)

    if bool(padding_mask.all(dim=1).any()):
        raise ValueError(
            f"{name} contains a sample with no valid cue position."
        )
    return padding_mask


def _normalize_regression_target(
    target: Tensor,
    batch_size: int,
    device: torch.device,
    name: str,
) -> Tensor:
    if not isinstance(target, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if target.device != device:
        raise ValueError(f"{name} and predictions must be on the same device.")

    if target.ndim == 1:
        if target.size(0) != batch_size:
            raise ValueError(f"{name} must have shape [batch] or [batch, 1].")
        target = target.unsqueeze(-1)
    elif target.ndim == 2:
        if tuple(target.shape) != (batch_size, 1):
            raise ValueError(f"{name} must have shape [batch] or [batch, 1].")
    else:
        raise ValueError(f"{name} must have shape [batch] or [batch, 1].")

    if not target.is_floating_point():
        target = target.to(dtype=torch.float32)
    _require_finite(name, target)
    return target


@dataclass
class InterventionOutput:

    original: Tensor
    masked: Tensor
    replaced: Tensor
    kept: Tensor
    cue_mask: Tensor


class CounterfactualIntervention(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        ratio: float,
    ) -> None:
        super().__init__()

        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        if not 0.0 < ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1].")

        self.feature_dim = int(feature_dim)
        self.ratio = float(ratio)
        self.register_buffer(
            "neutral",
            torch.zeros(self.feature_dim),
        )

    def _topk_mask(
        self,
        score: Tensor,
        padding_mask: Optional[Tensor],
    ) -> Tensor:
        if not isinstance(score, Tensor):
            raise TypeError("score must be a torch.Tensor.")
        if score.ndim != 2:
            raise ValueError(
                "score must have shape [batch, sequence], "
                f"but found {tuple(score.shape)}."
            )
        if not score.is_floating_point():
            raise TypeError("score must use a floating-point dtype.")
        _require_finite("score", score)

        batch_size, sequence_length = score.shape
        normalized_mask = _normalize_padding_mask(
            padding_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=score.device,
        )

        valid = torch.ones_like(score, dtype=torch.bool)
        if normalized_mask is not None:
            valid = ~normalized_mask

        cue_mask = torch.zeros_like(valid)
        for sample_index in range(batch_size):
            valid_count = int(valid[sample_index].sum().item())
            if valid_count < 1:
                raise ValueError(
                    f"Sample {sample_index} has no valid cue position."
                )

            selected_count = max(
                1,
                math.ceil(valid_count * self.ratio),
            )
            selected_count = min(selected_count, valid_count)

            candidate_score = score[sample_index].masked_fill(
                ~valid[sample_index],
                torch.finfo(score.dtype).min,
            )
            chosen = torch.topk(
                candidate_score,
                k=selected_count,
                largest=True,
                sorted=False,
            ).indices
            cue_mask[sample_index, chosen] = True

        if normalized_mask is not None and bool(cue_mask[normalized_mask].any()):
            raise RuntimeError("A padded position was selected for intervention.")
        return cue_mask

    def forward(
        self,
        selected: Tensor,
        replacement: Tensor,
        score: Tensor,
        padding_mask: Optional[Tensor],
    ) -> InterventionOutput:
        batch_size, sequence_length, _ = _validate_sequence(
            selected,
            "selected",
            self.feature_dim,
        )
        replacement_shape = _validate_sequence(
            replacement,
            "replacement",
            self.feature_dim,
        )
        if replacement_shape != (
            batch_size,
            sequence_length,
            self.feature_dim,
        ):
            raise ValueError(
                "selected and replacement must have identical shapes."
            )
        if replacement.device != selected.device:
            raise ValueError(
                "selected and replacement must be on the same device."
            )
        if replacement.dtype != selected.dtype:
            raise ValueError(
                "selected and replacement must use the same dtype."
            )

        if not isinstance(score, Tensor) or score.ndim != 2:
            raise ValueError("score must have shape [batch, sequence].")
        if tuple(score.shape) != (batch_size, sequence_length):
            raise ValueError(
                f"score must have shape {(batch_size, sequence_length)}, "
                f"but found {tuple(score.shape)}."
            )
        if score.device != selected.device:
            raise ValueError("score and selected must be on the same device.")
        if score.dtype != selected.dtype:
            raise ValueError("score and selected must use the same dtype.")
        _require_finite("score", score)

        normalized_mask = _normalize_padding_mask(
            padding_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=selected.device,
        )

        if normalized_mask is not None:
            pad = normalized_mask.unsqueeze(-1)
            original = selected.masked_fill(pad, 0.0)
            replacement = replacement.masked_fill(pad, 0.0)
        else:
            original = selected

        cue_mask = self._topk_mask(score, normalized_mask)

        soft_mask = torch.sigmoid(score)
        if normalized_mask is not None:
            soft_mask = soft_mask.masked_fill(normalized_mask, 0.0)

        hard_mask = cue_mask.to(dtype=selected.dtype)
        selection_mask = (
            hard_mask
            + soft_mask
            - soft_mask.detach()
        ).unsqueeze(-1)

        neutral = self.neutral.to(
            device=selected.device,
            dtype=selected.dtype,
        ).view(1, 1, -1)


        masked = (
            (1.0 - selection_mask) * original
            + selection_mask * neutral
        )


        replaced = (
            (1.0 - selection_mask) * original
            + selection_mask * replacement
        )


        kept = (
            selection_mask * original
            + (1.0 - selection_mask) * neutral
        )

        if normalized_mask is not None:
            pad = normalized_mask.unsqueeze(-1)
            masked = masked.masked_fill(pad, 0.0)
            replaced = replaced.masked_fill(pad, 0.0)
            kept = kept.masked_fill(pad, 0.0)

        for name, value in {
            "original": original,
            "masked": masked,
            "replaced": replaced,
            "kept": kept,
        }.items():
            _require_finite(name, value)

        return InterventionOutput(
            original=original,
            masked=masked,
            replaced=replaced,
            kept=kept,
            cue_mask=cue_mask,
        )


class AuxiliaryPredictor(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        num_outputs: int,
    ) -> None:
        super().__init__()

        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        if num_outputs != 1:
            raise ValueError(
                "The formal CLCFI experiments require one regression output."
            )

        self.feature_dim = int(feature_dim)
        self.num_outputs = int(num_outputs)
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.num_outputs),
        )

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor],
    ) -> Tensor:
        _validate_sequence(
            x,
            "auxiliary_input",
            self.feature_dim,
        )
        pooled = masked_mean(x, padding_mask)
        prediction = self.network(pooled)
        _require_finite("auxiliary_prediction", prediction)
        return prediction


@dataclass
class SNEOutput:
    corrected_ta: Tensor
    corrected_tv: Tensor
    weight_ta: Tensor
    weight_tv: Tensor
    sufficiency_ta: Tensor
    sufficiency_tv: Tensor
    necessity_ta: Tensor
    necessity_tv: Tensor
    predictions: Dict[str, Tensor]


class SufficiencyNecessityEvaluator(nn.Module):


    def __init__(
        self,
        feature_dim: int,
        num_outputs: int,
        task_type: str,
        dropout: float,
    ) -> None:
        super().__init__()

        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        if num_outputs != 1:
            raise ValueError("num_outputs must equal 1 for sentiment regression.")
        if task_type != "regression":
            raise ValueError(
                "SufficiencyNecessityEvaluator supports regression only."
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.feature_dim = int(feature_dim)
        self.num_outputs = int(num_outputs)
        self.task_type = task_type

        self.predictor = AuxiliaryPredictor(
            feature_dim=self.feature_dim,
            num_outputs=self.num_outputs,
        )

    @staticmethod
    def _validate_branch(
        branch: InterventionOutput,
        name: str,
        feature_dim: int,
        padding_mask: Optional[Tensor],
    ) -> Tuple[int, int]:
        if not isinstance(branch, InterventionOutput):
            raise TypeError(f"{name} must be an InterventionOutput.")

        original_shape = _validate_sequence(
            branch.original,
            f"{name}.original",
            feature_dim,
        )
        batch_size, sequence_length, _ = original_shape

        for variant_name in ("masked", "replaced", "kept"):
            variant = getattr(branch, variant_name)
            variant_shape = _validate_sequence(
                variant,
                f"{name}.{variant_name}",
                feature_dim,
            )
            if variant_shape != original_shape:
                raise ValueError(
                    f"{name}.{variant_name} must match {name}.original."
                )
            if variant.device != branch.original.device:
                raise ValueError(
                    f"{name}.{variant_name} and original must share a device."
                )
            if variant.dtype != branch.original.dtype:
                raise ValueError(
                    f"{name}.{variant_name} and original must share a dtype."
                )

        if not isinstance(branch.cue_mask, Tensor):
            raise TypeError(f"{name}.cue_mask must be a torch.Tensor.")
        if branch.cue_mask.dtype != torch.bool:
            raise TypeError(f"{name}.cue_mask must use torch.bool.")
        if tuple(branch.cue_mask.shape) != (batch_size, sequence_length):
            raise ValueError(
                f"{name}.cue_mask must have shape "
                f"{(batch_size, sequence_length)}."
            )
        if branch.cue_mask.device != branch.original.device:
            raise ValueError(
                f"{name}.cue_mask and branch tensors must share a device."
            )

        normalized_mask = _normalize_padding_mask(
            padding_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=branch.original.device,
        )
        if normalized_mask is not None:
            if bool(branch.cue_mask[normalized_mask].any()):
                raise ValueError(
                    f"{name}.cue_mask selects an excluded position."
                )
            pad = normalized_mask.unsqueeze(-1)
            for variant_name in (
                "original",
                "masked",
                "replaced",
                "kept",
            ):
                variant = getattr(branch, variant_name)
                if not bool((variant.masked_select(pad.expand_as(variant)) == 0).all()):
                    raise ValueError(
                        f"{name}.{variant_name} must be zero at padded positions."
                    )

        return batch_size, sequence_length

    def _predict_all(
        self,
        branch: InterventionOutput,
        padding_mask: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        predictions = {
            "original": self.predictor(branch.original, padding_mask),
            "mask": self.predictor(branch.masked, padding_mask),
            "replace": self.predictor(branch.replaced, padding_mask),
            "keep": self.predictor(branch.kept, padding_mask),
        }
        for name, value in predictions.items():
            _require_finite(f"prediction.{name}", value)
        return predictions

    @staticmethod
    def _reference_target(
        ta_original: Tensor,
        tv_original: Tensor,
    ) -> Tensor:
        if ta_original.shape != tv_original.shape:
            raise ValueError(
                "TA and TV original predictions must have identical shapes."
            )
        reference = (
            (ta_original + tv_original) / 2.0
        ).detach()
        _require_finite("reference_target", reference)
        return reference

    @staticmethod
    def _errors(
        predictions: Dict[str, Tensor],
        reference_target: Tensor,
    ) -> Dict[str, Tensor]:
        required = {"original", "mask", "replace", "keep"}
        if set(predictions) != required:
            raise ValueError(
                "predictions must contain original, mask, replace and keep."
            )

        errors: Dict[str, Tensor] = {}
        for name, value in predictions.items():
            if value.shape != reference_target.shape:
                raise ValueError(
                    f"Prediction '{name}' and reference target must match."
                )
            error = (value - reference_target).abs().mean(dim=-1)
            _require_finite(f"error.{name}", error)
            errors[name] = error
        return errors

    @staticmethod
    def _scores(
        errors: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:

        sufficiency = torch.sigmoid(
            errors["mask"] - errors["keep"]
        )

        necessity = torch.sigmoid(
            (
                errors["mask"]
                + errors["replace"]
            ) / 2.0
            - errors["original"]
        )

        _require_finite("sufficiency", sufficiency)
        _require_finite("necessity", necessity)

        if bool(((sufficiency < 0) | (sufficiency > 1)).any()):
            raise RuntimeError("Sufficiency scores must be in [0, 1].")
        if bool(((necessity < 0) | (necessity > 1)).any()):
            raise RuntimeError("Necessity scores must be in [0, 1].")
        return sufficiency, necessity

    def forward(
        self,
        ta: InterventionOutput,
        tv: InterventionOutput,
        padding_mask: Optional[Tensor],
        target: Optional[Tensor],
    ) -> SNEOutput:
        ta_batch, ta_length = self._validate_branch(
            ta,
            "ta",
            self.feature_dim,
            padding_mask,
        )
        tv_batch, tv_length = self._validate_branch(
            tv,
            "tv",
            self.feature_dim,
            padding_mask,
        )
        if (ta_batch, ta_length) != (tv_batch, tv_length):
            raise ValueError(
                "TA and TV branches must share batch and sequence dimensions."
            )
        if ta.original.device != tv.original.device:
            raise ValueError("TA and TV branches must be on the same device.")
        if ta.original.dtype != tv.original.dtype:
            raise ValueError("TA and TV branches must use the same dtype.")

        if target is not None:
            _normalize_regression_target(
                target,
                batch_size=ta_batch,
                device=ta.original.device,
                name="target",
            )

        ta_predictions = self._predict_all(ta, padding_mask)
        tv_predictions = self._predict_all(tv, padding_mask)

        reference_target = self._reference_target(
            ta_predictions["original"],
            tv_predictions["original"],
        )

        ta_errors = self._errors(
            ta_predictions,
            reference_target,
        )
        tv_errors = self._errors(
            tv_predictions,
            reference_target,
        )
        ta_sf, ta_nc = self._scores(ta_errors)
        tv_sf, tv_nc = self._scores(tv_errors)

        contribution = torch.stack(
            [
                ta_sf + ta_nc,
                tv_sf + tv_nc,
            ],
            dim=-1,
        )
        weights = torch.softmax(contribution, dim=-1)
        _require_finite("branch_weights", weights)

        if not torch.allclose(
            weights.sum(dim=-1),
            torch.ones_like(weights[:, 0]),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError("TA and TV branch weights do not sum to one.")
        if bool(((weights < 0) | (weights > 1)).any()):
            raise RuntimeError("Branch weights must be in [0, 1].")

        weight_ta = weights[:, 0].view(-1, 1, 1)
        weight_tv = weights[:, 1].view(-1, 1, 1)

        # Eq. (24): apply each contribution weight exactly once.
        corrected_ta = weight_ta * ta.original
        corrected_tv = weight_tv * tv.original
        _require_finite("corrected_ta", corrected_ta)
        _require_finite("corrected_tv", corrected_tv)

        predictions: Dict[str, Tensor] = {
            f"ta_{name}": value
            for name, value in ta_predictions.items()
        }
        predictions.update(
            {
                f"tv_{name}": value
                for name, value in tv_predictions.items()
            }
        )
        predictions["reference"] = reference_target

        return SNEOutput(
            corrected_ta=corrected_ta,
            corrected_tv=corrected_tv,
            weight_ta=weight_ta,
            weight_tv=weight_tv,
            sufficiency_ta=ta_sf,
            sufficiency_tv=tv_sf,
            necessity_ta=ta_nc,
            necessity_tv=tv_nc,
            predictions=predictions,
        )
