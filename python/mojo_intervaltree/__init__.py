"""Mutable interval trees with Mojo-accelerated queries."""

from ._lib import build
from .interval import Interval
from .tree import IntervalTree

__all__ = ["Interval", "IntervalTree", "build"]
__version__ = "0.1.0"
