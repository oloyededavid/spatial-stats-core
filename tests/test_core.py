from __future__ import annotations

import numpy as np
import pytest

from spatial_stats_core import (
    SpatialBlockKFold,
    allocate_stratified_sample,
    assign_spatial_blocks,
    kling_gupta_efficiency,
    nash_sutcliffe_efficiency,
    olofsson_area_estimate,
    percent_bias,
    rmse,
    summarize_fold_distances,
)


def test_olofsson_benchmark():
    # Table 8 benchmark from Olofsson et al. (2014)
    counts = np.array([
        [66,  0,   5,   4],
        [ 0, 55,   8,  12],
        [ 1,  0, 153,  11],
        [ 2,  1,   9, 313],
    ])
    mapped_ha = np.array([200_000, 150_000, 3_200_000, 6_450_000]) * 0.09
    res = olofsson_area_estimate(counts, mapped_ha)
    assert np.isclose(res.overall_accuracy, 0.9465, atol=1e-4)
    assert np.isclose(res.class_table.loc[0, "estimated_area"], 21157.7622, atol=1e-2)


def test_spatial_cv_leakage():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 50_000, size=(200, 2))
    X = rng.normal(size=(200, 2))
    y = rng.integers(0, 2, size=200)

    blocks = assign_spatial_blocks(coords, block_size=10_000.0)
    cv = SpatialBlockKFold(n_splits=3, block_size=10_000.0, stratify=True, random_state=42)
    splits = list(cv.split(X, y, coords=coords))

    assert len(splits) == 3
    for tr, te in splits:
        assert len(set(blocks[tr]).intersection(set(blocks[te]))) == 0

    dist_summary = summarize_fold_distances(coords, splits)
    assert dist_summary["min_train_test_distance"] > 0.0


def test_hydro_metrics_analytical():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    kge_09 = kling_gupta_efficiency(obs, sim, variant="2009")
    kge_12 = kling_gupta_efficiency(obs, sim, variant="2012")
    nse = nash_sutcliffe_efficiency(obs, sim)
    pbias = percent_bias(obs, sim)
    err = rmse(obs, sim)

    assert np.isclose(kge_09.kge, 1.0 - np.sqrt(2.0))
    assert np.isclose(kge_12.kge, 0.0)
    assert np.isclose(nse, -4.5)
    assert np.isclose(pbias, 100.0)
    assert np.isclose(err, np.sqrt(11.0))
