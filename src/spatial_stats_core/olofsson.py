from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class OlofssonResult:
    """Container for stratified area-adjusted accuracy and area estimates."""

    overall_accuracy: float
    overall_accuracy_se: float
    overall_accuracy_ci: tuple[float, float]
    proportion_matrix: np.ndarray
    class_table: pd.DataFrame


def olofsson_area_estimate(
    confusion_matrix: np.ndarray | Sequence[Sequence[float]],
    mapped_areas: np.ndarray | Sequence[float],
    class_names: Sequence[str] | None = None,
    confidence: float = 0.95,
) -> OlofssonResult:
    """Compute unbiased area-adjusted accuracies and class areas (Olofsson et al., 2014).

    Parameters
    ----------
    confusion_matrix : array-like of shape (q, q)
        Sample error matrix of raw counts n_ij where rows (i) are map classes
        (strata) and columns (j) are reference classes.
    mapped_areas : array-like of shape (q,)
        Total mapped area (or pixel count) for each map class i.
    class_names : sequence of str, optional
        Names of the q classes. Defaults to class_0, class_1, ...
    confidence : float, default 0.95
        Confidence level for two-sided normal confidence intervals.
    """
    counts = np.asarray(confusion_matrix, dtype=np.float64)
    areas = np.asarray(mapped_areas, dtype=np.float64)

    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("confusion_matrix must be a square 2D array of shape (q, q).")
    q = counts.shape[0]
    if areas.ndim != 1 or areas.shape[0] != q:
        raise ValueError("mapped_areas must be a 1D array matching confusion_matrix dimension.")
    if not np.all(np.isfinite(counts)) or not np.all(np.isfinite(areas)):
        raise ValueError("confusion_matrix and mapped_areas must contain finite numbers.")
    if np.any(counts < 0) or np.any(areas <= 0):
        raise ValueError("Counts must be non-negative and mapped_areas must be strictly positive.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1.")

    row_totals = counts.sum(axis=1)
    if np.any(row_totals < 2):
        raise ValueError("Each map stratum row must contain at least 2 sample points.")

    if class_names is None:
        labels = [f"class_{i}" for i in range(q)]
    else:
        if len(class_names) != q:
            raise ValueError("Length of class_names must match number of classes q.")
        labels = list(class_names)

    total_area = float(areas.sum())
    weights = areas / total_area  # W_i

    # Eq. 1: Estimated area proportions p_ij = W_i * (n_ij / n_i.)
    row_fractions = counts / row_totals[:, None]
    p_ij = weights[:, None] * row_fractions

    # Cell-level variance terms: v_ij = p_ij * (W_i - p_ij) / (n_i. - 1)
    denom = (row_totals - 1.0)[:, None]
    v_ij = p_ij * (weights[:, None] - p_ij) / denom

    # Eq. 2 & Eq. 6: User's accuracy U_i and its standard error
    users_acc = np.diag(row_fractions)
    users_var = users_acc * (1.0 - users_acc) / (row_totals - 1.0)
    users_se = np.sqrt(users_var)

    # Eq. 9 & Eq. 10: Stratified area proportions p_.k and standard errors
    area_prop = p_ij.sum(axis=0)
    area_prop_var = v_ij.sum(axis=0)
    area_prop_se = np.sqrt(area_prop_var)

    # Eq. 3 & Eq. 7: Producer's accuracy P_j and its standard error
    p_diag = np.diag(p_ij)
    v_diag = np.diag(v_ij)
    producers_acc = np.where(area_prop > 0, p_diag / area_prop, 0.0)

    v_off_diag_sum = area_prop_var - v_diag
    producers_var = np.where(
        area_prop > 0,
        ((1.0 - producers_acc) ** 2 * v_diag + (producers_acc**2) * v_off_diag_sum)
        / (area_prop**2),
        0.0,
    )
    producers_se = np.sqrt(producers_var)

    # Eq. 4 & Eq. 5: Overall accuracy O and its standard error
    overall_acc = float(p_diag.sum())
    overall_se = float(np.sqrt(v_diag.sum()))

    z = float(norm.ppf(0.5 + confidence / 2.0))
    oa_ci = (overall_acc - z * overall_se, overall_acc + z * overall_se)

    est_area = area_prop * total_area
    est_area_se = area_prop_se * total_area
    est_area_ci = z * est_area_se

    table = pd.DataFrame(
        {
            "class_name": labels,
            "mapped_area": areas,
            "mapped_proportion": weights,
            "estimated_proportion": area_prop,
            "proportion_se": area_prop_se,
            "estimated_area": est_area,
            "area_se": est_area_se,
            "area_ci_halfwidth": est_area_ci,
            "area_ci_lower": est_area - est_area_ci,
            "area_ci_upper": est_area + est_area_ci,
            "users_accuracy": users_acc,
            "users_se": users_se,
            "users_ci_halfwidth": z * users_se,
            "producers_accuracy": producers_acc,
            "producers_se": producers_se,
            "producers_ci_halfwidth": z * producers_se,
        }
    )

    return OlofssonResult(
        overall_accuracy=overall_acc,
        overall_accuracy_se=overall_se,
        overall_accuracy_ci=oa_ci,
        proportion_matrix=p_ij,
        class_table=table,
    )


def allocate_stratified_sample(
    mapped_areas: np.ndarray | Sequence[float],
    expected_users_acc: np.ndarray | Sequence[float],
    target_overall_se: float = 0.01,
    min_per_class: int = 50,
) -> dict[str, np.ndarray | int]:
    """Compute total sample size and proportional/minimum allocation (Olofsson Eq. 13)."""
    areas = np.asarray(mapped_areas, dtype=np.float64)
    u_i = np.asarray(expected_users_acc, dtype=np.float64)

    if areas.shape != u_i.shape or areas.ndim != 1 or len(areas) == 0:
        raise ValueError("mapped_areas and expected_users_acc must be non-empty 1D arrays of equal length.")
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError("mapped_areas must contain finite, strictly positive numbers.")
    if not np.all(np.isfinite(u_i)) or np.any((u_i <= 0.0) | (u_i >= 1.0)):
        raise ValueError("expected_users_acc values must be strictly between 0 and 1.")
    if target_overall_se <= 0:
        raise ValueError("target_overall_se must be strictly positive.")
    if min_per_class < 2:
        raise ValueError("min_per_class must be at least 2.")

    weights = areas / areas.sum()
    s_i = np.sqrt(u_i * (1.0 - u_i))
    raw_n = int(np.ceil(((weights * s_i).sum() / target_overall_se) ** 2))

    q = len(areas)
    target_n = max(raw_n, q * int(min_per_class))

    prop_alloc = np.round(weights * target_n).astype(int)
    enforced = np.maximum(prop_alloc, int(min_per_class))
    return {
        "formula_sample_size": raw_n,
        "total_sample_size": int(enforced.sum()),
        "proportional_allocation": prop_alloc,
        "recommended_allocation": enforced,
    }
