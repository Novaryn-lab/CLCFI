from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


_MIN_STD = 1e-5


def _require_finite(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf values.")


def _validate_sequence(
    x: Tensor,
    name: str,
    feature_dim: Optional[int] = None,
) -> Tuple[int, int, int]:
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if x.ndim != 3:
        raise ValueError(
            f"{name} must have shape [batch, sequence, feature], "
            f"but found {tuple(x.shape)}."
        )

    batch_size, sequence_length, actual_dim = x.shape
    if batch_size < 1 or sequence_length < 1 or actual_dim < 1:
        raise ValueError(f"{name} must have positive dimensions.")
    if feature_dim is not None and actual_dim != feature_dim:
        raise ValueError(
            f"{name} feature dimension is {actual_dim}, "
            f"but expected {feature_dim}."
        )
    if not x.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype.")

    _require_finite(name, x)
    return batch_size, sequence_length, actual_dim


def _normalize_padding_mask(
    padding_mask: Optional[Tensor],
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> Optional[Tensor]:

    if padding_mask is None:
        return None
    if not isinstance(padding_mask, Tensor):
        raise TypeError("padding_mask must be a torch.Tensor or None.")
    if padding_mask.ndim != 2:
        raise ValueError(
            "padding_mask must have shape [batch, sequence], "
            f"but found {tuple(padding_mask.shape)}."
        )
    if tuple(padding_mask.shape) != (batch_size, sequence_length):
        raise ValueError(
            f"padding_mask must have shape {(batch_size, sequence_length)}, "
            f"but found {tuple(padding_mask.shape)}."
        )
    if padding_mask.device != device:
        raise ValueError("padding_mask and cue features must be on the same device.")

    if padding_mask.dtype != torch.bool:
        binary = torch.logical_or(padding_mask == 0, padding_mask == 1)
        if not bool(binary.all()):
            raise ValueError("padding_mask must contain only bool or 0/1 values.")
        padding_mask = padding_mask.to(dtype=torch.bool)

    if bool(padding_mask.all(dim=1).any()):
        raise ValueError(
            "Each sample must contain at least one non-padding cue position."
        )
    return padding_mask


def _masked_average(
    x: Tensor,
    padding_mask: Optional[Tensor],
) -> Tensor:
    batch_size, sequence_length, _ = _validate_sequence(x, "x")
    mask = _normalize_padding_mask(
        padding_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=x.device,
    )

    if mask is None:
        return x.mean(dim=1)

    valid = (~mask).unsqueeze(-1).to(dtype=x.dtype)
    counts = valid.sum(dim=1)
    if bool((counts <= 0).any()):
        raise ValueError("Cannot average a sample with no valid cue positions.")
    return (x * valid).sum(dim=1) / counts


def _masked_sample_mean(
    values: Tensor,
    padding_mask: Optional[Tensor],
) -> Tensor:

    if not isinstance(values, Tensor) or values.ndim != 2:
        raise ValueError("values must have shape [batch, sequence].")
    _require_finite("values", values)

    batch_size, sequence_length = values.shape
    mask = _normalize_padding_mask(
        padding_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=values.device,
    )

    if mask is None:
        return values.mean(dim=1)

    valid = (~mask).to(dtype=values.dtype)
    counts = valid.sum(dim=1)
    if bool((counts <= 0).any()):
        raise ValueError("Cannot reduce a sample with no valid cue positions.")
    return (values * valid).sum(dim=1) / counts


def _joint_distribution_embedding(mean: Tensor, std: Tensor) -> Tensor:
    """
    Represent one cue distribution using both its mean and standard deviation.

    Mean and standard-deviation components are normalized separately, so one
    component cannot dominate prototype matching only because of scale.
    """
    if mean.shape != std.shape:
        raise ValueError("mean and std must have identical shapes.")
    if mean.ndim != 2:
        raise ValueError("mean and std must have shape [batch, feature].")
    _require_finite("mean", mean)
    _require_finite("std", std)
    if bool((std <= 0).any()):
        raise ValueError("std must be strictly positive.")

    normalized_mean = F.normalize(mean, p=2, dim=-1, eps=1e-12)
    normalized_std = F.normalize(std, p=2, dim=-1, eps=1e-12)
    return torch.cat([normalized_mean, normalized_std], dim=-1) / (2.0 ** 0.5)


class CuePrototypeBank(nn.Module):
    """
    Cue prototype bank for Eqs. (9)-(12).

    Each prototype stores a paired mean and standard deviation. Prototype
    assignment and negative selection therefore use both statistics rather
    than matching only the mean.
    """

    def __init__(
        self,
        size: int,
        feature_dim: int,
        momentum: float,
    ) -> None:
        super().__init__()

        if size < 1:
            raise ValueError("Prototype-bank size must be positive.")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1].")

        self.size = int(size)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)

        self.register_buffer(
            "mean_bank",
            torch.zeros(self.size, self.feature_dim),
        )
        self.register_buffer(
            "std_bank",
            torch.ones(self.size, self.feature_dim),
        )
        self.register_buffer(
            "valid_count",
            torch.zeros((), dtype=torch.long),
        )

    def _validate_distribution(
        self,
        mean: Tensor,
        std: Tensor,
        name: str,
    ) -> None:
        if not isinstance(mean, Tensor) or not isinstance(std, Tensor):
            raise TypeError(f"{name} mean and std must be torch.Tensor objects.")
        if mean.ndim != 2 or std.ndim != 2:
            raise ValueError(f"{name} mean and std must have shape [batch, feature].")
        if mean.shape != std.shape:
            raise ValueError(f"{name} mean and std must have identical shapes.")
        if mean.size(0) < 1 or mean.size(1) != self.feature_dim:
            raise ValueError(
                f"{name} distributions must have shape [batch, {self.feature_dim}]."
            )
        if mean.device != self.mean_bank.device or std.device != self.std_bank.device:
            raise ValueError(f"{name} distributions and prototype bank must share a device.")
        if mean.dtype != self.mean_bank.dtype or std.dtype != self.std_bank.dtype:
            raise ValueError(f"{name} distributions and prototype bank must share a dtype.")

        _require_finite(f"{name}.mean", mean)
        _require_finite(f"{name}.std", std)
        if bool((std <= 0).any()):
            raise ValueError(f"{name}.std must be strictly positive.")

    def _prototype_similarity(
        self,
        current_mean: Tensor,
        current_std: Tensor,
        count: int,
    ) -> Tensor:
        current = _joint_distribution_embedding(current_mean, current_std)
        prototypes = _joint_distribution_embedding(
            self.mean_bank[:count],
            self.std_bank[:count],
        )
        return current @ prototypes.transpose(0, 1)

    @torch.no_grad()
    def update(self, mean: Tensor, std: Tensor) -> None:

        self._validate_distribution(mean, std, "update")
        detached_mean = mean.detach()
        detached_std = std.detach()

        for sample_mean, sample_std in zip(detached_mean, detached_std):
            count = int(self.valid_count.item())

            if count < self.size:
                slot = count
                self.mean_bank[slot].copy_(sample_mean)
                self.std_bank[slot].copy_(sample_std.clamp_min(_MIN_STD))
                self.valid_count.add_(1)
                continue

            similarity = self._prototype_similarity(
                sample_mean.unsqueeze(0),
                sample_std.unsqueeze(0),
                count=count,
            )
            slot = int(similarity.argmax(dim=1).item())
            gamma = self.momentum

            self.mean_bank[slot].mul_(1.0 - gamma).add_(
                sample_mean,
                alpha=gamma,
            )
            self.std_bank[slot].mul_(1.0 - gamma).add_(
                sample_std,
                alpha=gamma,
            )
            self.std_bank[slot].clamp_min_(_MIN_STD)

    def negative_distribution(
        self,
        current_mean: Tensor,
        current_std: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        self._validate_distribution(current_mean, current_std, "current")
        batch_size = current_mean.size(0)
        count = int(self.valid_count.item())

        if count > 0:
            similarity = self._prototype_similarity(
                current_mean,
                current_std,
                count=count,
            )
            indices = similarity.argmin(dim=1)
            valid_negative = torch.ones(
                batch_size,
                dtype=torch.bool,
                device=current_mean.device,
            )
            return (
                self.mean_bank[indices].detach(),
                self.std_bank[indices].detach(),
                valid_negative,
            )

        if batch_size > 1:
            current = _joint_distribution_embedding(current_mean, current_std)
            pairwise_similarity = current @ current.transpose(0, 1)
            pairwise_similarity.fill_diagonal_(float("inf"))
            indices = pairwise_similarity.argmin(dim=1)
            valid_negative = torch.ones(
                batch_size,
                dtype=torch.bool,
                device=current_mean.device,
            )
            return (
                current_mean[indices].detach(),
                current_std[indices].detach(),
                valid_negative,
            )

        valid_negative = torch.zeros(
            1,
            dtype=torch.bool,
            device=current_mean.device,
        )
        return (
            current_mean.detach(),
            current_std.detach(),
            valid_negative,
        )


class DistributionHead(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.feature_dim = int(feature_dim)
        self.backbone = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mean = nn.Linear(self.feature_dim, self.feature_dim)
        self.std = nn.Linear(self.feature_dim, self.feature_dim)
        self.score = nn.Linear(self.feature_dim, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        _validate_sequence(x, "distribution_input", self.feature_dim)

        hidden = self.backbone(x)
        mean = self.mean(hidden)
        std = F.softplus(self.std(hidden)) + _MIN_STD
        score = self.score(hidden).squeeze(-1)

        _require_finite("distribution_mean", mean)
        _require_finite("distribution_std", std)
        _require_finite("cue_score", score)
        return mean, std, score


@dataclass
class BranchCueOutput:
    selected: Tensor
    negative: Tensor
    score: Tensor
    mean: Tensor
    std: Tensor
    ccl_loss: Tensor


@dataclass
class CueSelectionOutput:
    ta: BranchCueOutput
    tv: BranchCueOutput
    ccl_loss: Tensor
    consistency_loss: Tensor


class CueSelectionModule(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        prototype_size: int,
        temperature: float,
        prototype_momentum: float,
        positive_weight: float,
        dropout: float,
    ) -> None:
        super().__init__()

        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        if prototype_size < 1:
            raise ValueError("prototype_size must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0.0 <= prototype_momentum <= 1.0:
            raise ValueError("prototype_momentum must be in [0, 1].")
        if positive_weight < 0:
            raise ValueError("positive_weight must be non-negative.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.feature_dim = int(feature_dim)
        self.temperature = float(temperature)
        self.positive_weight = float(positive_weight)

        self.ta_distribution = DistributionHead(feature_dim, dropout)
        self.tv_distribution = DistributionHead(feature_dim, dropout)
        self.ta_bank = CuePrototypeBank(
            prototype_size,
            feature_dim,
            prototype_momentum,
        )
        self.tv_bank = CuePrototypeBank(
            prototype_size,
            feature_dim,
            prototype_momentum,
        )

    @staticmethod
    def _sample(
        mean: Tensor,
        std: Tensor,
        stochastic: bool,
    ) -> Tensor:
        if mean.shape != std.shape:
            raise ValueError("mean and std must have identical shapes.")
        if bool((std <= 0).any()):
            raise ValueError("std must be strictly positive.")

        if not stochastic:
            return mean
        return mean + std * torch.randn_like(std)

    def _contrastive_loss(
        self,
        query: Tensor,
        positive: Tensor,
        negative: Tensor,
        padding_mask: Optional[Tensor],
        valid_negative: Tensor,
    ) -> Tensor:
        positive_similarity = F.cosine_similarity(
            query,
            positive,
            dim=-1,
            eps=1e-8,
        )
        negative_similarity = F.cosine_similarity(
            query,
            negative,
            dim=-1,
            eps=1e-8,
        )
        logits = torch.stack(
            [positive_similarity, negative_similarity],
            dim=-1,
        ) / self.temperature

        targets = torch.zeros(
            logits.size(0) * logits.size(1),
            dtype=torch.long,
            device=logits.device,
        )
        token_loss = F.cross_entropy(
            logits.reshape(-1, 2),
            targets,
            reduction="none",
        ).reshape(logits.size(0), logits.size(1))

        sample_loss = _masked_sample_mean(token_loss, padding_mask)
        if valid_negative.ndim != 1 or valid_negative.size(0) != sample_loss.size(0):
            raise ValueError(
                "valid_negative must have shape [batch]."
            )

        if bool(valid_negative.any()):
            loss = sample_loss[valid_negative].mean()
        else:

            loss = token_loss.sum() * 0.0

        _require_finite("ccl_loss", loss)
        return loss

    def _branch(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor],
        head: DistributionHead,
        bank: CuePrototypeBank,
    ) -> BranchCueOutput:
        batch_size, sequence_length, _ = _validate_sequence(
            x,
            "branch_input",
            self.feature_dim,
        )
        normalized_mask = _normalize_padding_mask(
            padding_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=x.device,
        )

        mean, std, score = head(x)
        summary_mean = _masked_average(mean, normalized_mask)
        summary_std = _masked_average(std, normalized_mask)

        negative_mean, negative_std, valid_negative = (
            bank.negative_distribution(summary_mean, summary_std)
        )

        stochastic = self.training
        query = self._sample(mean, std, stochastic=stochastic)
        positive = self._sample(mean, std, stochastic=stochastic)

        expanded_negative_mean = negative_mean.unsqueeze(1).expand_as(mean)
        expanded_negative_std = negative_std.unsqueeze(1).expand_as(std)
        negative = self._sample(
            expanded_negative_mean,
            expanded_negative_std,
            stochastic=stochastic,
        )

        ccl_loss = self._contrastive_loss(
            query=query,
            positive=positive,
            negative=negative,
            padding_mask=normalized_mask,
            valid_negative=valid_negative,
        )

        # Eq. (16): F_tm^s = F_tm^q + alpha_tm F_tm^k.
        selected = query + self.positive_weight * positive

        if normalized_mask is not None:
            pad = normalized_mask.unsqueeze(-1)
            selected = selected.masked_fill(pad, 0.0)
            negative = negative.masked_fill(pad, 0.0)

            score = score.masked_fill(
                normalized_mask,
                torch.finfo(score.dtype).min,
            )

        if self.training:

            bank.update(summary_mean, summary_std)

        _require_finite("selected_cues", selected)
        _require_finite("negative_cues", negative)
        _require_finite("cue_scores", score)

        return BranchCueOutput(
            selected=selected,
            negative=negative,
            score=score,
            mean=mean,
            std=std,
            ccl_loss=ccl_loss,
        )

    @staticmethod
    def _consistency(
        ta_value: Tensor,
        tv_value: Tensor,
        padding_mask: Optional[Tensor],
    ) -> Tensor:
        if ta_value.shape != tv_value.shape:
            raise ValueError(
                "TA and TV distribution tensors must have identical shapes."
            )
        batch_size, sequence_length, _ = _validate_sequence(
            ta_value,
            "ta_distribution",
        )
        _validate_sequence(
            tv_value,
            "tv_distribution",
            ta_value.size(-1),
        )
        normalized_mask = _normalize_padding_mask(
            padding_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=ta_value.device,
        )

        token_distance = (ta_value - tv_value).pow(2).sum(dim=-1)
        sample_distance = _masked_sample_mean(
            token_distance,
            normalized_mask,
        )
        loss = sample_distance.mean()
        _require_finite("consistency_component", loss)
        return loss

    def forward(
        self,
        f_ta: Tensor,
        f_tv: Tensor,
        text_padding_mask: Optional[Tensor],
    ) -> CueSelectionOutput:
        ta_shape = _validate_sequence(
            f_ta,
            "f_ta",
            self.feature_dim,
        )
        tv_shape = _validate_sequence(
            f_tv,
            "f_tv",
            self.feature_dim,
        )
        if ta_shape != tv_shape:
            raise ValueError(
                "f_ta and f_tv must have identical [batch, sequence, feature] "
                f"shapes, but found {tuple(f_ta.shape)} and {tuple(f_tv.shape)}."
            )
        if f_ta.device != f_tv.device:
            raise ValueError("f_ta and f_tv must be on the same device.")
        if f_ta.dtype != f_tv.dtype:
            raise ValueError("f_ta and f_tv must use the same dtype.")

        normalized_mask = _normalize_padding_mask(
            text_padding_mask,
            batch_size=f_ta.size(0),
            sequence_length=f_ta.size(1),
            device=f_ta.device,
        )

        ta = self._branch(
            f_ta,
            normalized_mask,
            self.ta_distribution,
            self.ta_bank,
        )
        tv = self._branch(
            f_tv,
            normalized_mask,
            self.tv_distribution,
            self.tv_bank,
        )

        consistency_loss = self._consistency(
            ta.mean,
            tv.mean,
            normalized_mask,
        )
        consistency_loss = consistency_loss + self._consistency(
            ta.std,
            tv.std,
            normalized_mask,
        )
        ccl_loss = ta.ccl_loss + tv.ccl_loss

        _require_finite("total_ccl_loss", ccl_loss)
        _require_finite("total_consistency_loss", consistency_loss)

        return CueSelectionOutput(
            ta=ta,
            tv=tv,
            ccl_loss=ccl_loss,
            consistency_loss=consistency_loss,
        )
