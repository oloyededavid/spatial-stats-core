"""Core validation and uncertainty routines for geospatial and hydrological models."""

from .olofsson import OlofssonResult, allocate_stratified_sample, olofsson_area_estimate

__all__ = [
    "OlofssonResult",
    "allocate_stratified_sample",
    "olofsson_area_estimate",
]
__version__ = "0.1.0"
