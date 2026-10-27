from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def assign_spatial_blocks(
    coords: np.ndarray | Sequence[Sequence[float]],
    block_size: float | tuple[float, float],
    origin: tuple[float, float] | None = None,
) -> np.ndarray:
    """Assign 2D coordinates to rectangular spatial grid blocks.

    Parameters
    ----------
    coords : array-like of shape (n_samples, 2)
        Planar (x, y) or geographic (lon, lat) coordinates for each sample.
    block_size : float or (float, float)
        Grid cell width (dx) and height (dy) in the same units as coords.
    origin : (float, float), optional
        Grid origin (x0, y0). Defaults to minimum (x, y) in coords.

    Returns
    -------
    block_ids : np.ndarray of shape (n_samples,)
        Contiguous integer block labels in [0, n_blocks - 1].
    """
    xy = np.asarray(coords, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0:
        raise ValueError("coords must be a non-empty 2D array of shape (n_samples, 2).")
    if not np.all(np.isfinite(xy)):
        raise ValueError("coords must contain only finite numbers (no NaN or Inf).")

    if isinstance(block_size, (int, float, np.number)) and not isinstance(block_size, bool):
        dx = dy = float(block_size)
    elif isinstance(block_size, Sequence) and len(block_size) == 2:
        dx, dy = float(block_size[0]), float(block_size[1])
    else:
        raise ValueError("block_size must be a positive scalar or a 2-tuple (dx, dy).")

    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0 or dy <= 0:
        raise ValueError("block_size dimensions must be finite and strictly positive.")

    if origin is None:
        x0, y0 = xy.min(axis=0)
    else:
        if len(origin) != 2:
            raise ValueError("origin must be a 2-tuple (x0, y0).")
        x0, y0 = float(origin[0]), float(origin[1])

    gx = np.floor((xy[:, 0] - x0) / dx).astype(np.int64)
    gy = np.floor((xy[:, 1] - y0) / dy).astype(np.int64)

    grid_pairs = np.column_stack((gx, gy))
    _, inverse_ids = np.unique(grid_pairs, axis=0, return_inverse=True)
    return inverse_ids


class SpatialBlockKFold:
    """Spatial block K-Fold cross-validator (Roberts et al., 2017).

    Compatible with scikit-learn functions (`cross_val_score`, `GridSearchCV`)
    by passing 2D coordinates via either `groups=coords` or `coords=coords`.
    """

    def __init__(
        self,
        n_splits: int = 5,
        block_size: float | tuple[float, float] = 10_000.0,
        stratify: bool = False,
        shuffle: bool = True,
        random_state: int | None = 42,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = int(n_splits)
        self.block_size = block_size
        self.stratify = bool(stratify)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def get_n_splits(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        coords: np.ndarray | None = None,
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        coords: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate (train_idx, test_idx) splits partitioned by spatial blocks."""
        xy = coords if coords is not None else groups
        if xy is None:
            raise ValueError("2D sample coordinates must be provided via `coords` or `groups`.")

        n_samples = len(X)
        if len(xy) != n_samples:
            raise ValueError("X and coordinates must have the same number of samples.")

        block_ids = assign_spatial_blocks(xy, self.block_size)
        unique_blocks = np.unique(block_ids)

        if len(unique_blocks) < self.n_splits:
            raise ValueError(
                f"block_size produced only {len(unique_blocks)} non-empty blocks, "
                f"which is fewer than n_splits={self.n_splits}. Decrease block_size."
            )

        if self.stratify:
            if y is None:
                raise ValueError("y must be provided when stratify=True.")
            eff_seed = self.random_state if self.shuffle else None
            splitter = StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=eff_seed,
            )
            yield from splitter.split(X, y, groups=block_ids)
        else:
            if self.shuffle:
                rng = np.random.default_rng(self.random_state)
                perm = rng.permutation(len(unique_blocks))
                block_ids = perm[block_ids]
            splitter = GroupKFold(n_splits=self.n_splits)
            yield from splitter.split(X, y, groups=block_ids)


def summarize_fold_distances(
    coords: np.ndarray | Sequence[Sequence[float]],
    splits: Sequence[tuple[np.ndarray, np.ndarray]] | Iterator[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """Compute nearest-neighbor distances from test samples to training samples across folds."""
    xy = np.asarray(coords, dtype=np.float64)
    min_dists: list[float] = []
    median_dists: list[float] = []

    for train_idx, test_idx in splits:
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError("Each fold must contain non-empty train and test indices.")
        tree = cKDTree(xy[train_idx])
        dists, _ = tree.query(xy[test_idx], k=1)
        min_dists.append(float(np.min(dists)))
        median_dists.append(float(np.median(dists)))

    if not min_dists:
        raise ValueError("splits cannot be empty.")

    return {
        "min_train_test_distance": float(np.min(min_dists)),
        "mean_fold_median_distance": float(np.mean(median_dists)),
    }
