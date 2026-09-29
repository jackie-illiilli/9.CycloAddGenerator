"""Shared reference loading and distribution plots for discovery notebooks."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline


REFERENCE_RELATIVE_PATH = Path(
    "Data/Reported_BO/Reported Bioorthogonal Reactions for Selection.csv"
)
POPULATION_STYLES = (
    ("IEDDA candidates (predicted)", "#b55475"),
    ("Reported reactions (DFT)", "#7d9263"),
)


def load_reference(repository_root):
    """Use the complete curated reference CSV without additional exclusions."""
    return pd.read_csv(Path(repository_root) / REFERENCE_RELATIVE_PATH)


def orbital_gaps(frame, fmo):
    if not fmo.index.is_unique:
        raise ValueError("FMO indices must be unique.")
    result = pd.DataFrame({
        "inverse_gap": frame["Diene_Index"].map(fmo["LUMO"])
        - frame["Ene_Index"].map(fmo["HOMO"]),
        "normal_gap": frame["Ene_Index"].map(fmo["LUMO"])
        - frame["Diene_Index"].map(fmo["HOMO"]),
    }, index=frame.index)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite frontier orbital energies.")
    return result


def energy_values(frame, predicted=False):
    suffix = "_p" if predicted else ""
    values = pd.DataFrame({
        "4pi distortion": frame[f"Diene_Distort{suffix}"],
        "2pi distortion": frame[f"Ene_Distort{suffix}"],
        "Interaction": frame[f"Interaction{suffix}"],
    })
    values["Total distortion"] = values["4pi distortion"] + values["2pi distortion"]
    if (values["Total distortion"] == 0).any():
        raise ValueError("Zero total distortion makes the distortion fraction undefined.")
    values["4pi distortion fraction"] = values["4pi distortion"] / values["Total distortion"]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite comparison energies or distortion fractions.")
    return values


def describe_populations(populations):
    rows = []
    for label, values, _ in populations:
        for metric in values.columns:
            series = values[metric]
            rows.append({
                "metric": metric, "population": label, "n": len(series),
                "excluded_nonfinite": 0, "mean": series.mean(),
                "std": series.std(ddof=1), "median": series.median(),
                "min": series.min(), "max": series.max(),
            })
    return pd.DataFrame(rows)


def plot_frequency_curves(ax, curves, bins=None, vertical=False):
    """Match the original ratio plot: cubic splines of normalized bin counts.

    Each curve is (label, values, color). Thirty edges define 29 shared bins,
    matching the original ratio implementation. The ordinate is relative
    frequency per bin, not probability density. No observations are discarded.
    """
    arrays = [(label, np.asarray(values, dtype=float), color)
              for label, values, color in curves]
    if any(len(values) == 0 or not np.isfinite(values).all()
           for _, values, _ in arrays):
        raise ValueError("Plot inputs must be nonempty and finite.")
    pooled = np.concatenate([values for _, values, _ in arrays])
    if bins is None:
        lower, upper = pooled.min(), pooled.max()
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        bins = np.linspace(lower, upper, 30)
    if pooled.min() < bins[0] or pooled.max() > bins[-1]:
        raise ValueError("Histogram edges must include every observation.")
    centers = (bins[1:] + bins[:-1]) / 2
    grid = np.linspace(centers.min(), centers.max(), 300)
    peak = 0.0
    for label, values, color in arrays:
        counts, _ = np.histogram(values, bins=bins)
        frequencies = counts / len(values)
        smooth = make_interp_spline(centers, frequencies, k=3)(grid)
        peak = max(peak, float(smooth.max()))
        if vertical:
            ax.plot(smooth, grid, color=color, linewidth=1.2, label=label)
            ax.fill_betweenx(grid, 0, smooth, where=smooth > 0,
                             interpolate=True, color=color, alpha=0.2)
        else:
            ax.plot(grid, smooth, color=color, linewidth=1.2, label=label)
            ax.fill_between(grid, 0, smooth, where=smooth > 0,
                            interpolate=True, color=color, alpha=0.2)
    if vertical:
        ax.set_xlim(0, peak * 1.15)
        ax.set_ylim(bins[0], bins[-1])
    else:
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, peak * 1.15)
    return bins
