"""Core validation and uncertainty routines for geospatial and hydrological models."""

from .olofsson import OlofssonResult, allocate_stratified_sample, olofsson_area_estimate
from .spatial_cv import SpatialBlockKFold, assign_spatial_blocks, summarize_fold_distances

__all__ = [
    "OlofssonResult",
    "SpatialBlockKFold",
    "allocate_stratified_sample",
    "assign_spatial_blocks",
    "olofsson_area_estimate",
    "summarize_fold_distances",
]
__version__ = "0.1.0"
