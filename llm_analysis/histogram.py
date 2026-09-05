from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math

from matplotlib import pyplot as plt
import torch
from torch import Tensor


MAX_CHUNK_ELEMENTS = 16 * 1024 * 1024
DEFAULT_BULK_MASS = 0.95
ZERO_MASS_DISPLAY_THRESHOLD = 5e-6


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
    zero_mass: float = 0.0

    @property
    def has_material_zero_atom(self) -> bool:
        """Whether the zero atom is large enough to show in fit reports."""
        return self.zero_mass >= ZERO_MASS_DISPLAY_THRESHOLD

    @property
    def parameters(self) -> str:
        if self.family == "gaussian":
            sigma = (-0.5 / self.slope) ** 0.5
            parameters = f"mu={self.center:.4g}, sigma={sigma:.4g}"
        elif self.family == "exponential":
            scale = -1.0 / self.slope
            parameters = f"mu={self.center:.4g}, scale={scale:.4g}"
        else:
            exponent = -self.slope
            parameters = (
                f"mu={self.center:.4g}, scale={self.radial_scale:.4g}, "
                f"exponent={exponent:.4g}"
            )
        if self.has_material_zero_atom:
            parameters += f", zero_mass={self.zero_mass:.3%}"
        return parameters

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


def _fit_feature(fit: DistributionFit, distance: Tensor) -> Tensor:
    if fit.family == "gaussian":
        return distance.square()
    if fit.family == "exponential":
        return distance
    return torch.log1p(distance / fit.radial_scale)


def _update_fit_diagnostics(
    fit: DistributionFit,
    distance: Tensor,
    counts: Tensor,
    observed_log_density: Tensor,
) -> DistributionFit:
    """Recalculate plot-facing diagnostics after rescaling a fitted curve."""
    model_log_density = fit.intercept + fit.slope * _fit_feature(fit, distance)
    residual = observed_log_density - model_log_density
    weight_sum = counts.sum()
    squared_error = (counts * residual.square()).sum()
    total_error = (
        counts * (observed_log_density - observed_log_density.mean()).square()
    ).sum()
    log_rmse = torch.sqrt(squared_error / weight_sum)
    r_squared = 1.0 - squared_error / total_error if total_error > 0 else torch.nan
    return replace(
        fit,
        log_rmse=float(log_rmse),
        r_squared=float(r_squared),
    )


def _join_tail_to_bulk(
    bulk: DistributionFit,
    tail_fits: Sequence[DistributionFit],
    distance: Tensor,
    counts: Tensor,
    observed_log_density: Tensor,
) -> list[DistributionFit]:
    """Rescale tails so they meet the primary bulk fit at the shared boundary."""
    boundary_value = torch.tensor(
        [bulk.center + bulk.boundary],
        dtype=distance.dtype,
    )
    bulk_log_density = bulk.log_density(boundary_value).item()
    joined_fits = []
    for tail in tail_fits:
        tail_log_density = tail.log_density(boundary_value).item()
        joined_tail = replace(
            tail,
            intercept=tail.intercept + bulk_log_density - tail_log_density,
        )
        joined_fits.append(
            _update_fit_diagnostics(
                joined_tail,
                distance,
                counts,
                observed_log_density,
            )
        )
    return joined_fits


def _weighted_quantile(values: Tensor, weights: Tensor, quantile: float) -> float:
    order = torch.argsort(values)
    sorted_values = values[order]
    cumulative = weights[order].cumsum(0)
    target = quantile * cumulative[-1]
    index = int(torch.searchsorted(cumulative, target).clamp_max(len(values) - 1))
    return float(sorted_values[index])


def _weighted_binned_quantile(
    centers: Tensor,
    counts: Tensor,
    quantile: float,
    bin_width: float,
) -> float:
    """Estimate a weighted quantile within its histogram bin."""
    order = torch.argsort(centers)
    sorted_centers = centers[order]
    sorted_counts = counts[order]
    cumulative = sorted_counts.cumsum(0)
    target = quantile * cumulative[-1]
    index = int(torch.searchsorted(cumulative, target).clamp_max(len(centers) - 1))
    previous = (
        cumulative[index - 1]
        if index
        else torch.zeros((), dtype=counts.dtype)
    )
    fraction = (target - previous) / sorted_counts[index]
    return float(sorted_centers[index] - bin_width / 2 + fraction * bin_width)


def _remove_zero_atom(
    counts: Tensor,
    edges: Tensor,
    zero_count: int | Tensor,
) -> tuple[Tensor, float]:
    """Separate an exact-zero atom from the regular histogram bins."""
    regular_counts = counts[1:-1].detach().cpu().to(torch.float64)
    total_count = regular_counts.sum()
    if total_count <= 0:
        raise ValueError("Histogram must contain observations in its regular bins")

    zero_count = int(torch.as_tensor(zero_count).item())
    if zero_count < 0:
        raise ValueError("zero_count must not be negative")

    zero_index = int(
        torch.searchsorted(
            edges,
            torch.zeros((), dtype=edges.dtype),
            right=True,
        ).item()
        - 1
    )
    if not 0 <= zero_index < regular_counts.numel():
        raise ValueError("Histogram edges must contain zero in a regular bin")
    if zero_count > regular_counts[zero_index]:
        raise ValueError("zero_count exceeds the count in zero's histogram bin")

    continuous_counts = regular_counts.clone()
    continuous_counts[zero_index] -= zero_count
    return continuous_counts, float(zero_count / total_count)


def _distribution_cdf(
    family: str,
    values: Tensor,
    center: float,
    scale: Tensor,
    exponent: Tensor | None = None,
) -> Tensor:
    """Evaluate a symmetric candidate distribution's CDF."""
    shifted = values - center
    if family == "gaussian":
        return 0.5 * (
            1.0
            + torch.erf(
                shifted
                / (scale * torch.tensor(2.0, dtype=scale.dtype).sqrt())
            )
        )
    if family == "exponential":
        tail = 0.5 * torch.exp(-shifted.abs() / scale)
        return torch.where(shifted < 0, tail, 1.0 - tail)
    if exponent is None:
        raise ValueError("polynomial fits require an exponent")
    tail = 0.5 * (1.0 + shifted.abs() / scale).pow(1.0 - exponent)
    return torch.where(shifted < 0, tail, 1.0 - tail)


def _fit_mle(
    family: str,
    lower_edges: Tensor,
    upper_edges: Tensor,
    counts: Tensor,
    center: float,
    initial_scale: float,
    region: str,
    boundary: float,
    observed_region_mass: float,
) -> DistributionFit | None:
    """Fit one family by conditional MLE on integrated bin probabilities."""
    weight_sum = counts.sum()
    bin_centers = (lower_edges + upper_edges) / 2
    distance = (bin_centers - center).abs()
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

    def model_terms() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        scale = log_scale.exp().clamp_min(torch.finfo(counts.dtype).tiny)
        if family == "gaussian":
            exponent = None
            normalizer = torch.tensor(2 * torch.pi, dtype=counts.dtype).sqrt()
            intercept = -torch.log(scale * normalizer)
            slope = -0.5 / scale.square()
        elif family == "exponential":
            exponent = None
            intercept = -torch.log(2 * scale)
            slope = -1.0 / scale
        else:
            exponent = 1.0 + torch.nn.functional.softplus(log_excess)
            intercept = torch.log(exponent - 1.0) - torch.log(2 * scale)
            slope = -exponent
        bin_probability = (
            _distribution_cdf(
                family,
                upper_edges,
                center,
                scale,
                exponent,
            )
            - _distribution_cdf(
                family,
                lower_edges,
                center,
                scale,
                exponent,
            )
        ).clamp_min(torch.finfo(counts.dtype).tiny)
        log_region_probability = bin_probability.sum().log()
        return scale, intercept, slope, log_region_probability, bin_probability

    def closure() -> Tensor:
        optimizer.zero_grad()
        _, _, _, log_region_probability, bin_probability = model_terms()
        loss = -(
            counts * (bin_probability.log() - log_region_probability)
        ).sum() / weight_sum
        if not torch.isfinite(loss):
            return torch.tensor(float("inf"), dtype=counts.dtype)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        return None

    scale, raw_intercept, slope, log_region_probability, bin_probability = model_terms()
    if not torch.isfinite(scale) or not torch.isfinite(bin_probability).all():
        return None

    # The MLE is conditional on the selected histogram bins. Scale it by the
    # observed region mass so the plotted curve is an unconditional density.
    adjusted_intercept = float(
        (
            raw_intercept
            - log_region_probability
            + torch.tensor(observed_region_mass, dtype=counts.dtype).log()
        ).detach()
    )
    conditional_log_likelihood = (
        counts * (bin_probability.log() - log_region_probability)
    ).sum()
    radial_scale = float(scale.detach())
    return DistributionFit(
        family=family,
        region=region,
        center=center,
        intercept=adjusted_intercept,
        slope=float(slope.detach()),
        radial_scale=radial_scale,
        boundary=boundary,
        log_rmse=float("nan"),
        r_squared=float("nan"),
        log_likelihood=float(conditional_log_likelihood.detach()),
        num_bins=int((counts > 0).sum()),
    )


def fit_distribution_regions(
    counts: Tensor,
    edges: Tensor,
    *,
    zero_count: int | Tensor = 0,
    bulk_mass: float = DEFAULT_BULK_MASS,
) -> dict[str, list[DistributionFit]]:
    """Fit a zero-inflated, piecewise continuous density to a histogram."""
    if not 0.0 < bulk_mass < 1.0:
        raise ValueError("bulk_mass must be between zero and one")

    raw_counts = counts.detach().cpu().to(torch.float64)
    edges = edges.detach().cpu().to(torch.float64)
    counts, zero_mass = _remove_zero_atom(raw_counts, edges, zero_count)
    continuous_mass = 1.0 - zero_mass
    if continuous_mass <= 0.0:
        raise ValueError("A continuous fit requires at least one nonzero observation")

    centers = (edges[:-1] + edges[1:]) / 2
    positive = counts > 0
    if positive.sum() < 6:
        raise ValueError("At least six populated histogram bins are required")

    bin_width = float(edges[1] - edges[0])
    total_continuous_count = counts.sum()
    positive_centers = centers[positive]
    positive_counts = counts[positive]
    conditional_density = positive_counts / (total_continuous_count * bin_width)
    conditional_log_density = conditional_density.log()
    log_density = conditional_log_density + math.log(continuous_mass)
    center = _weighted_binned_quantile(
        positive_centers,
        positive_counts,
        0.5,
        bin_width,
    )
    distance = (positive_centers - center).abs()
    boundary = _weighted_quantile(distance, positive_counts, bulk_mass)
    # A sharply quantized central mode can place the requested central mass in
    # fewer than three populated bins. Expand only enough to make that region
    # identifiable by the multi-parameter candidate families.
    minimum_bulk_boundary = float(torch.sort(distance).values[2])
    boundary = max(boundary, minimum_bulk_boundary)
    radial_scale = max(
        _weighted_quantile(distance, positive_counts, 0.5),
        bin_width,
    )

    region_masks = {
        "bulk": (centers - center).abs() <= boundary,
        "tail": (centers - center).abs() > boundary,
    }
    results: dict[str, list[DistributionFit]] = {}
    region_data: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
    for region, mask in region_masks.items():
        positive_mask = mask & positive
        if positive_mask.sum() < 3:
            continue
        region_positive_mask = positive_mask[positive]
        region_distance = (centers[positive_mask] - center).abs()
        observed_region_mass = float(
            (counts[mask].sum() / total_continuous_count).item()
        )
        fits = []
        for family in ("gaussian", "exponential", "polynomial"):
            fit = _fit_mle(
                family,
                edges[:-1][mask],
                edges[1:][mask],
                counts[mask],
                center,
                radial_scale,
                region,
                boundary,
                observed_region_mass,
            )
            if fit is not None:
                zero_inflated_fit = replace(
                    fit,
                    intercept=fit.intercept + math.log(continuous_mass),
                    zero_mass=zero_mass,
                )
                fits.append(
                    _update_fit_diagnostics(
                        zero_inflated_fit,
                        region_distance,
                        counts[positive_mask],
                        log_density[region_positive_mask],
                    )
                )
        if fits:
            results[region] = sorted(
                fits,
                key=lambda fit: fit.log_likelihood,
                reverse=True,
            )
            region_data[region] = (
                region_distance,
                counts[positive_mask],
                log_density[region_positive_mask],
            )

    # Each region is initially normalized to its empirical mass. Those separate
    # normalizations can leave a visible jump where the piecewise curves meet.
    # Keep the bulk fit fixed and vertically anchor every tail candidate to the
    # primary bulk candidate at the common radial boundary.
    if "bulk" in results and "tail" in results:
        results["tail"] = _join_tail_to_bulk(
            results["bulk"][0],
            results["tail"],
            *region_data["tail"],
        )
    if not results:
        raise RuntimeError("No region contained enough bins for a decaying fit")
    return results


def fit_histogram_distributions(
    histograms: Mapping[str, Tensor],
    edges: Tensor,
    *,
    zero_counts: Mapping[str, int | Tensor] | None = None,
    bulk_mass: float = DEFAULT_BULK_MASS,
) -> dict[str, dict[str, list[DistributionFit]]]:
    """Fit every named histogram, optionally separating exact-zero atoms."""
    if zero_counts is not None and set(zero_counts) != set(histograms):
        raise ValueError("zero_counts and histograms must have matching labels")
    return {
        label: fit_distribution_regions(
            counts,
            edges,
            zero_count=0 if zero_counts is None else zero_counts[label],
            bulk_mass=bulk_mass,
        )
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
    zero_counts: Mapping[str, int | Tensor] | None = None,
    bulk_mass: float = DEFAULT_BULK_MASS,
    title: str = "Histogram distribution fits",
    x_label: str = "Value",
    x_limits: tuple[float, float] | None = None,
) -> None:
    if zero_counts is not None and set(zero_counts) != set(histograms):
        raise ValueError("zero_counts and histograms must have matching labels")

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
        raw_counts = raw_counts.detach().cpu().to(torch.float64)
        zero_count = 0 if zero_counts is None else zero_counts[label]
        counts, zero_mass = _remove_zero_atom(raw_counts, edges, zero_count)
        density = counts / (raw_counts[1:-1].sum() * bin_width)
        positive = counts > 0
        if positive.any():
            observed_density = density[positive]
            observed_min = float(observed_density.quantile(0.01))
            observed_max = float(observed_density.max())
            y_floor = max(observed_min * 0.5, observed_max * 1e-8)
            y_ceiling = observed_max * 2.0
        else:
            observed_density = density[positive]
            observed_max = 1.0
            y_floor = 1e-12
            y_ceiling = 1.0

        show_zero_atom = zero_mass >= ZERO_MASS_DISPLAY_THRESHOLD
        zero_display_density = zero_mass / bin_width
        if show_zero_atom:
            y_ceiling = max(y_ceiling, zero_display_density * 1.2)

        fit_segments = []
        annotations = (
            [(f"zero atom: {zero_mass:.3%}", "tab:purple")]
            if show_zero_atom
            else []
        )

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
                x_values = centers[mask]
                if region == "bulk":
                    join_points = torch.tensor(
                        [
                            fit.center - fit.boundary,
                            fit.center + fit.boundary,
                        ],
                        dtype=centers.dtype,
                    )
                else:
                    join_points = torch.tensor(
                        [
                            fit.center - fit.boundary
                            if index == 0
                            else fit.center + fit.boundary
                        ],
                        dtype=centers.dtype,
                    )
                x_values = torch.cat((x_values, join_points)).unique(sorted=True)
                prediction = fit.log_density(x_values).exp()
                fit_segments.append(
                    (
                        x_values,
                        prediction,
                        color,
                        (
                            f"{region}: "
                            f"{'zero-inflated ' if fit.has_material_zero_atom else ''}"
                            f"{fit.family}"
                            if index == 0
                            else None
                        ),
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
        if show_zero_atom:
            axis.vlines(
                0.0,
                y_floor,
                min(zero_display_density, y_ceiling),
                color="tab:purple",
                linewidth=2,
                linestyle=":",
                label="exact-zero atom",
            )
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
