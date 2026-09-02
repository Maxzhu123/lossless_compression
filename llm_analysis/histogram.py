from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from matplotlib import pyplot as plt
import torch
from torch import Tensor


MAX_CHUNK_ELEMENTS = 16 * 1024 * 1024
DEFAULT_BULK_MASS = 0.95


@torch.no_grad()
def tensor_histogram(
    tensors: Sequence[Tensor],
    bin_width: float = 0.1,
    limit: float = 10.0,
):
    """
    Histogram with:
        bucket 0:          x < -limit
        buckets 1..N:     width `bin_width` from -limit to +limit
        bucket N+1:        x > +limit

    Returns:
        counts: [N + 2] int64 tensor
        edges:  [N + 1] bin edges for the normal buckets
        minimum, maximum: scalar tensors containing the overall extrema

    Assumes all tensors are on the same device.
    """
    if not tensors:
        raise ValueError("tensors must not be empty")

    device = tensors[0].device
    n_bins = round(2 * limit / bin_width)

    # +2 for negative and positive overflow buckets
    counts = torch.zeros(n_bins + 2, dtype=torch.int64, device=device)
    minimum = None
    maximum = None

    for tensor in tensors:
        tensor = tensor.detach().reshape(-1)
        if tensor.numel() == 0:
            continue

        tensor_minimum = tensor.amin()
        tensor_maximum = tensor.amax()
        minimum = (
            tensor_minimum
            if minimum is None
            else torch.minimum(minimum, tensor_minimum)
        )
        maximum = (
            tensor_maximum
            if maximum is None
            else torch.maximum(maximum, tensor_maximum)
        )

        # Keep the temporary int64 bucket indices bounded for very large model
        # tensors (for example Nemotron's embedding and output matrices).
        for start in range(0, tensor.numel(), MAX_CHUNK_ELEMENTS):
            x = tensor[start : start + MAX_CHUNK_ELEMENTS]
            # Histogram arithmetic needs more precision than model activations.
            # In BF16, adding a large limit (for example 150) can quantize away
            # bin widths smaller than one before the division is performed.
            bucket_values = x.to(torch.float32)

            # Normal buckets:
            # [-10.0, -9.9) -> 1
            # [-9.9,  -9.8) -> 2
            # ...
            idx = (
                torch.floor((bucket_values + limit) / bin_width).to(torch.int64) + 1
            )

            # Overflow buckets
            idx = torch.where(bucket_values < -limit, 0, idx)
            idx = torch.where(bucket_values > limit, n_bins + 1, idx)

            # x == +limit goes into the final regular bucket
            idx.clamp_(0, n_bins + 1)

            counts += torch.bincount(idx, minlength=n_bins + 2)

    edges = torch.linspace(
        -limit, limit, n_bins + 1,
        device=device,
    )

    if minimum is None or maximum is None:
        raise ValueError("tensors must contain at least one value")
    return counts, edges, minimum, maximum


def plot_histogram(counts, edges, log_y=False, title="Tensor value histograms"):
    edges = edges.detach().cpu()
    if torch.is_tensor(counts):
        counts = {"Values": counts}
    counts = {
        label: values.detach().cpu()
        for label, values in counts.items()
    }

    lower_limit = edges[0].item()
    upper_limit = edges[-1].item()
    figure, axes = plt.subplots(
        len(counts),
        1,
        figsize=(12, 4 * len(counts)),
        sharex=True,
        squeeze=False,
    )
    for axis, (label, values) in zip(axes[:, 0], counts.items()):
        axis.stairs(values[1:-1].numpy(), edges.numpy())
        axis.set_title(label)
        axis.set_ylabel("Count")
        if log_y:
            axis.set_yscale("log")
        axis.text(
            0.01,
            0.95,
            f"x < {lower_limit:g}: {values[0]:,}\n"
            f"x > {upper_limit:g}: {values[-1]:,}",
            transform=axis.transAxes,
            va="top",
        )

    axes[-1, 0].set_xlabel("Value")
    figure.suptitle(title)

    plt.tight_layout()
    plt.show()


@dataclass(frozen=True)
class DistributionFit:
    family: str
    region: str
    center: float
    intercept: float
    slope: float
    radial_scale: float
    boundary: float
    log_rmse: float
    r_squared: float
    log_likelihood: float
    num_bins: int

    @property
    def parameters(self) -> str:
        if self.family == "gaussian":
            sigma = (-0.5 / self.slope) ** 0.5
            return f"mu={self.center:.4g}, sigma={sigma:.4g}"
        if self.family == "exponential":
            scale = -1.0 / self.slope
            return f"mu={self.center:.4g}, scale={scale:.4g}"
        exponent = -self.slope
        return (
            f"mu={self.center:.4g}, scale={self.radial_scale:.4g}, "
            f"exponent={exponent:.4g}"
        )

    def log_density(self, values: Tensor) -> Tensor:
        distance = (values - self.center).abs()
        if self.family == "gaussian":
            feature = distance.square()
        elif self.family == "exponential":
            feature = distance
        else:
            feature = torch.log1p(distance / self.radial_scale)
        return self.intercept + self.slope * feature

    @property
    def equation(self) -> str:
        center_term = (
            f"(x - {self.center:.4g})"
            if self.center >= 0
            else f"(x + {abs(self.center):.4g})"
        )
        if self.family == "gaussian":
            feature = f"{center_term}²"
        elif self.family == "exponential":
            feature = f"|{center_term}|"
        else:
            feature = f"(1 + |{center_term}| / {self.radial_scale:.4g})"
        amplitude = math.exp(self.intercept)
        if self.family == "polynomial":
            return f"p(x) = {amplitude:.4g} · {feature}^{self.slope:+.4g}"
        return f"p(x) = {amplitude:.4g} · exp({self.slope:+.4g}·{feature})"


def _weighted_quantile(values: Tensor, weights: Tensor, quantile: float) -> float:
    order = torch.argsort(values)
    sorted_values = values[order]
    cumulative = weights[order].cumsum(0)
    target = quantile * cumulative[-1]
    index = int(torch.searchsorted(cumulative, target).clamp_max(len(values) - 1))
    return float(sorted_values[index])


def _log_region_probability(
    family: str,
    scale: Tensor,
    boundary: float,
    region: str,
    exponent: Tensor | None = None,
) -> Tensor:
    """Return log P(X is in a radial fit region) for a candidate family."""
    boundary_tensor = torch.as_tensor(boundary, dtype=scale.dtype)
    if family == "gaussian":
        z = boundary_tensor / (scale * torch.tensor(2.0, dtype=scale.dtype).sqrt())
        probability = torch.special.erf(z) if region == "bulk" else torch.special.erfc(z)
    elif family == "exponential":
        z = boundary_tensor / scale
        probability = -torch.expm1(-z) if region == "bulk" else torch.exp(-z)
    else:
        if exponent is None:
            raise ValueError("polynomial fits require an exponent")
        log_tail_probability = (1.0 - exponent) * torch.log1p(boundary_tensor / scale)
        probability = (
            -torch.expm1(log_tail_probability)
            if region == "bulk"
            else torch.exp(log_tail_probability)
        )
    return probability.clamp_min(torch.finfo(scale.dtype).tiny).log()


def _fit_mle(
    family: str,
    distance: Tensor,
    counts: Tensor,
    center: float,
    initial_scale: float,
    observed_log_density: Tensor,
    region: str,
    boundary: float,
    observed_region_mass: float,
) -> DistributionFit | None:
    """Fit one family by conditional MLE on weighted histogram bin centers."""
    weight_sum = counts.sum()
    initial_variance = (counts * distance.square()).sum() / weight_sum
    if family == "gaussian":
        initial_scale = max(float(initial_variance.sqrt()), initial_scale)
    elif family == "exponential":
        initial_scale = max(float((counts * distance).sum() / weight_sum), initial_scale)

    log_scale = torch.nn.Parameter(
        torch.tensor(initial_scale, dtype=distance.dtype).log()
    )
    parameters = [log_scale]
    log_excess = None
    if family == "polynomial":
        log_excess = torch.nn.Parameter(torch.zeros((), dtype=distance.dtype))
        parameters.append(log_excess)
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=0.5,
        max_iter=100,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def model_terms() -> tuple[Tensor, Tensor, Tensor, Tensor]:
        scale = log_scale.exp().clamp_min(torch.finfo(distance.dtype).tiny)
        if family == "gaussian":
            exponent = None
            normalizer = torch.tensor(2 * torch.pi, dtype=distance.dtype).sqrt()
            intercept = -torch.log(scale * normalizer)
            slope = -0.5 / scale.square()
            feature = distance.square()
        elif family == "exponential":
            exponent = None
            intercept = -torch.log(2 * scale)
            slope = -1.0 / scale
            feature = distance
        else:
            exponent = 1.0 + log_excess.exp().clamp_min(torch.finfo(distance.dtype).tiny)
            intercept = torch.log(exponent - 1.0) - torch.log(2 * scale)
            slope = -exponent
            feature = torch.log1p(distance / scale)
        log_density = intercept + slope * feature
        log_probability = _log_region_probability(
            family,
            scale,
            boundary,
            region,
            exponent,
        )
        return scale, log_density, log_probability, slope

    def closure() -> Tensor:
        optimizer.zero_grad()
        _, log_density, log_probability, _ = model_terms()
        loss = -(counts * (log_density - log_probability)).sum() / weight_sum
        if not torch.isfinite(loss):
            return torch.tensor(float("inf"), dtype=distance.dtype)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        return None

    scale, raw_log_density, log_probability, slope = model_terms()
    if not torch.isfinite(scale) or not torch.isfinite(raw_log_density).all():
        return None

    # The MLE is conditional on the selected region. Scale it by the observed
    # region mass so the plotted curve is comparable to the unconditional
    # histogram density.
    raw_feature = (
        distance.square()
        if family == "gaussian"
        else distance
        if family == "exponential"
        else torch.log1p(distance / scale)
    )
    raw_intercept = raw_log_density[0] - slope * raw_feature[0]
    adjusted_intercept = float(
        (
            raw_intercept
            - log_probability
            + torch.tensor(observed_region_mass, dtype=distance.dtype).log()
        ).detach()
    )
    model_log_density = adjusted_intercept + slope * raw_feature
    residual = observed_log_density - model_log_density
    squared_error = (counts * residual.square()).sum()
    total_error = (counts * (observed_log_density - observed_log_density.mean()).square()).sum()
    log_rmse = torch.sqrt(squared_error / weight_sum)
    r_squared = 1.0 - squared_error / total_error if total_error > 0 else torch.nan
    conditional_log_likelihood = (counts * (raw_log_density - log_probability)).sum()
    radial_scale = float(scale.detach())
    return DistributionFit(
        family=family,
        region=region,
        center=center,
        intercept=adjusted_intercept,
        slope=float(slope.detach()),
        radial_scale=radial_scale,
        boundary=boundary,
        log_rmse=float(log_rmse.detach()),
        r_squared=float(r_squared.detach()),
        log_likelihood=float(conditional_log_likelihood.detach()),
        num_bins=int(len(distance)),
    )


def fit_distribution_regions(
    counts: Tensor,
    edges: Tensor,
    *,
    bulk_mass: float = DEFAULT_BULK_MASS,
) -> dict[str, list[DistributionFit]]:
    """Fit normalized candidate models by MLE on weighted histogram centers."""
    if not 0.0 < bulk_mass < 1.0:
        raise ValueError("bulk_mass must be between zero and one")

    counts = counts[1:-1].detach().cpu().to(torch.float64)
    edges = edges.detach().cpu().to(torch.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    positive = counts > 0
    if positive.sum() < 6:
        raise ValueError("At least six populated histogram bins are required")

    centers = centers[positive]
    counts = counts[positive]
    bin_width = float(edges[1] - edges[0])
    density = counts / (counts.sum() * bin_width)
    log_density = density.log()
    center = _weighted_quantile(centers, counts, 0.5)
    distance = (centers - center).abs()
    boundary = _weighted_quantile(distance, counts, bulk_mass)
    radial_scale = max(
        _weighted_quantile(distance, counts, 0.5),
        bin_width,
    )

    region_masks = {
        "bulk": distance <= boundary,
        "tail": distance > boundary,
    }
    results: dict[str, list[DistributionFit]] = {}
    for region, mask in region_masks.items():
        if mask.sum() < 3:
            continue
        region_distance = distance[mask]
        observed_region_mass = float((counts[mask].sum() / counts.sum()).item())
        fits = []
        for family in ("gaussian", "exponential", "polynomial"):
            fit = _fit_mle(
                family,
                region_distance,
                counts[mask],
                center,
                radial_scale,
                log_density[mask],
                region,
                boundary,
                observed_region_mass,
            )
            if fit is not None:
                fits.append(fit)
        if fits:
            results[region] = sorted(
                fits,
                key=lambda fit: fit.log_likelihood,
                reverse=True,
            )
    if not results:
        raise RuntimeError("No region contained enough bins for a decaying fit")
    return results


def fit_histogram_distributions(
    histograms: Mapping[str, Tensor],
    edges: Tensor,
    *,
    bulk_mass: float = DEFAULT_BULK_MASS,
) -> dict[str, dict[str, list[DistributionFit]]]:
    """Fit every named histogram by maximum likelihood."""
    return {
        label: fit_distribution_regions(counts, edges, bulk_mass=bulk_mass)
        for label, counts in histograms.items()
    }


def print_fit_report(
    fits: Mapping[str, Mapping[str, list[DistributionFit]]],
) -> None:
    for label, regions in fits.items():
        print(f"\n{label}")
        for region, candidates in regions.items():
            best = candidates[0]
            print(
                f"  {region} (|x - mu| "
                f"{'<=' if region == 'bulk' else '>'} {best.boundary:.4g}):"
            )
            for fit in candidates:
                marker = "best" if fit is best else "    "
                print(
                    f"    {marker:>4} {fit.family:<11} "
                    f"log-likelihood={fit.log_likelihood:.4g}, "
                    f"log-RMSE={fit.log_rmse:.4f}, R^2={fit.r_squared:.4f}, "
                    f"{fit.parameters}"
                )


def plot_histogram_fits(
    histograms: Mapping[str, Tensor],
    edges: Tensor,
    fits: Mapping[str, Mapping[str, list[DistributionFit]]],
    *,
    bulk_mass: float = DEFAULT_BULK_MASS,
    title: str = "Histogram distribution fits",
    x_label: str = "Value",
    x_limits: tuple[float, float] | None = None,
) -> None:
    edges = edges.detach().cpu().to(torch.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_width = float(edges[1] - edges[0])
    figure, axes = plt.subplots(
        len(histograms),
        1,
        figsize=(12, 4 * len(histograms)),
        sharex=True,
        squeeze=False,
    )
    for axis, (label, raw_counts) in zip(axes[:, 0], histograms.items()):
        counts = raw_counts[1:-1].detach().cpu().to(torch.float64)
        density = counts / (counts.sum() * bin_width)
        positive = counts > 0
        observed_density = density[positive]
        observed_min = float(observed_density.quantile(0.01))
        observed_max = float(observed_density.max())
        y_floor = max(observed_min * 0.5, observed_max * 1e-8)
        y_ceiling = observed_max * 2.0

        fit_segments = []
        annotations = []

        for region, color in (("bulk", "tab:orange"), ("tail", "tab:red")):
            candidates = fits[label].get(region)
            if not candidates:
                annotations.append((f"{region}: not fitted", color))
                continue
            fit = candidates[0]
            annotations.append(
                (f"{region}: R²={fit.r_squared:.3f}\n{fit.equation}", color)
            )
            distance = (centers - fit.center).abs()
            masks = (
                [distance <= fit.boundary]
                if region == "bulk"
                else [
                    centers < fit.center - fit.boundary,
                    centers > fit.center + fit.boundary,
                ]
            )
            for index, mask in enumerate(masks):
                prediction = fit.log_density(centers[mask]).exp()
                fit_segments.append(
                    (
                        centers[mask],
                        prediction,
                        color,
                        f"{region}: {fit.family}" if index == 0 else None,
                    )
                )

        fit_maxima = [
            float(prediction[torch.isfinite(prediction)].max())
            for _, prediction, _, _ in fit_segments
            if torch.isfinite(prediction).any()
        ]
        if fit_maxima:
            y_ceiling = max(
                y_ceiling,
                min(max(fit_maxima), observed_max * 10) * 1.2,
            )

        axis.semilogy(centers[positive], observed_density, label="observed")
        axis.set_ylim(y_floor, y_ceiling)
        for x_values, prediction, color, fit_label in fit_segments:
            visible = (
                torch.isfinite(prediction)
                & (prediction >= y_floor)
                & (prediction <= y_ceiling)
            )
            if visible.any():
                axis.semilogy(
                    x_values[visible],
                    prediction[visible],
                    color=color,
                    linewidth=2,
                    label=fit_label,
                )

        axis.set_title(label)
        axis.set_ylabel("Density")
        for index, (annotation, color) in enumerate(annotations):
            axis.text(
                0.02,
                0.03 + index * 0.16,
                annotation,
                transform=axis.transAxes,
                color=color,
                fontsize=8,
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "black",
                    "edgecolor": color,
                    "alpha": 0.55,
                },
            )
        axis.legend()

    axes[-1, 0].set_xlabel(x_label)
    if x_limits is not None:
        axes[-1, 0].set_xlim(*x_limits)
    figure.suptitle(f"{title} (central mass={bulk_mass:.0%})")
    figure.tight_layout()
    plt.show()
