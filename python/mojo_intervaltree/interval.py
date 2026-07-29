"""The public immutable interval value type."""

from __future__ import annotations

from collections import namedtuple


class Interval(namedtuple("_IntervalBase", "begin end data", defaults=(None,))):
    __slots__ = ()

    def __hash__(self):
        return hash((self.begin, self.end))

    def __repr__(self):
        if self.data is None:
            return f"Interval({self.begin!r}, {self.end!r})"
        return f"Interval({self.begin!r}, {self.end!r}, {self.data!r})"

    def copy(self):
        return Interval(self.begin, self.end, self.data)

    def is_null(self):
        return self.begin >= self.end

    def length(self):
        if self.is_null():
            return 0
        return self.end - self.begin

    def contains_point(self, p):
        return self.begin <= p < self.end

    def range_matches(self, other):
        return self.begin == other.begin and self.end == other.end

    def contains_interval(self, other):
        return self.begin <= other.begin and self.end >= other.end

    def overlaps(self, begin, end=None):
        if end is None:
            try:
                return self.begin < begin.end and self.end > begin.begin
            except AttributeError:
                return self.contains_point(begin)
        return self.begin < end and self.end > begin

    def overlap_size(self, begin, end=None):
        if end is None:
            begin, end = begin.begin, begin.end
        overlap = min(self.end, end) - max(self.begin, begin)
        return overlap if overlap > 0 else 0

    def distance_to(self, other):
        if self.overlaps(other):
            return 0
        try:
            if self.begin < other.begin:
                return other.begin - self.end
            return self.begin - other.end
        except AttributeError:
            if self.end <= other:
                return other - self.end
            return self.begin - other
