"""A mutable interval set with Mojo-accelerated point and range queries."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral, Real
from operator import attrgetter

import numpy as np

from ._lib import addr, lib
from .interval import Interval


class IntervalTree:
    __slots__ = (
        "_intervals",
        "_dirty",
        "_records",
        "_begins",
        "_ends",
        "_max_ends",
        "_left",
        "_right",
        "_order",
        "_result",
        "_stack",
    )

    def __init__(self, intervals=None):
        self._intervals = set()
        self._dirty = True
        self._records = []
        self._begins = self._empty_float()
        self._ends = self._empty_float()
        self._max_ends = self._empty_float()
        self._left = self._empty_int()
        self._right = self._empty_int()
        self._order = self._empty_int()
        self._result = self._empty_int()
        self._stack = self._empty_int()
        if intervals is not None:
            self.update(intervals)

    @staticmethod
    def _empty_float():
        return np.empty(0, dtype=np.float64)

    @staticmethod
    def _empty_int():
        return np.empty(0, dtype=np.int64)

    @staticmethod
    def _coordinate(value):
        if not isinstance(value, Real):
            raise TypeError("Mojo interval queries require real-number coordinates")
        try:
            converted = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError("coordinate is not representable as float64") from exc
        if not np.isfinite(converted):
            raise ValueError("coordinates must be finite")
        if isinstance(value, Integral):
            exact = int(value) == int(converted)
        else:
            exact = value == converted
        if not exact:
            raise ValueError("coordinate is not exactly representable as float64")
        return converted

    @classmethod
    def _coordinate_array(cls, values):
        raw = np.asarray(values)
        if raw.ndim != 1:
            raise ValueError("query coordinates must be 1D arrays")
        if raw.dtype == np.float64 and raw.flags.c_contiguous:
            converted = raw
        else:
            converted = np.ascontiguousarray(
                [cls._coordinate(value) for value in raw], dtype=np.float64
            )
        if not np.all(np.isfinite(converted)):
            raise ValueError("coordinates must be finite")
        return converted

    @classmethod
    def from_tuples(cls, tuples):
        tree = cls()
        intervals = {Interval(*args) for args in tuples}
        for interval in intervals:
            begin = interval.begin
            end = interval.end
            if begin >= end:
                raise ValueError(
                    "IntervalTree: Null Interval objects not allowed in "
                    f"IntervalTree: {interval!r}"
                )
            tree._coordinate(begin)
            tree._coordinate(end)
        tree._intervals = intervals
        return tree

    def copy(self):
        return type(self)(iv.copy() for iv in self)

    def _coerce_interval(self, interval):
        if isinstance(interval, Interval):
            iv = interval
        else:
            try:
                iv = Interval(interval.begin, interval.end, interval.data)
            except AttributeError as exc:
                raise TypeError("interval must have begin, end, and data attributes") from exc
        if iv.is_null():
            raise ValueError(
                f"IntervalTree: Null Interval objects not allowed in IntervalTree: {iv!r}"
            )
        self._coordinate(iv.begin)
        self._coordinate(iv.end)
        return iv

    def _invalidate(self):
        self._dirty = True

    def _ensure_index(self):
        if not self._dirty:
            return
        self._records = list(self._intervals)
        n = len(self._records)
        if n == 0:
            self._begins = self._empty_float()
            self._ends = self._empty_float()
            self._max_ends = self._empty_float()
            self._left = self._empty_int()
            self._right = self._empty_int()
            self._order = self._empty_int()
            self._result = self._empty_int()
            self._stack = self._empty_int()
            self._dirty = False
            return

        source_begins = np.fromiter(
            map(attrgetter("begin"), self._records),
            dtype=np.float64,
            count=n,
        )
        source_ends = np.fromiter(
            map(attrgetter("end"), self._records),
            dtype=np.float64,
            count=n,
        )
        sorted_order = np.lexsort((source_ends, source_begins))
        self._begins = np.empty(n, dtype=np.float64)
        self._ends = np.empty(n, dtype=np.float64)
        self._max_ends = np.empty(n, dtype=np.float64)
        self._left = np.empty(n, dtype=np.int64)
        self._right = np.empty(n, dtype=np.int64)
        self._order = np.empty(n, dtype=np.int64)
        self._result = np.empty(n, dtype=np.int64)
        self._stack = np.empty(4 * (n.bit_length() + 1), dtype=np.int64)
        status = lib().mit_build_index(
            addr(source_begins, np.float64),
            addr(source_ends, np.float64),
            addr(sorted_order, np.int64),
            n,
            addr(self._begins, np.float64, writable=True),
            addr(self._ends, np.float64, writable=True),
            addr(self._max_ends, np.float64, writable=True),
            addr(self._left, np.int64, writable=True),
            addr(self._right, np.int64, writable=True),
            addr(self._order, np.int64, writable=True),
            addr(self._stack, np.int64, writable=True),
        )
        if status != 0:
            raise RuntimeError(f"native index construction failed with status {status}")
        self._dirty = False

    def build_index(self):
        self._ensure_index()
        return self

    def _query(self, kind, begin, end):
        begin = self._coordinate(begin)
        end = self._coordinate(end)
        self._ensure_index()
        n = len(self._records)
        if n == 0:
            return set()
        count = lib().mit_query_one(
            addr(self._begins, np.float64),
            addr(self._ends, np.float64),
            addr(self._max_ends, np.float64),
            addr(self._left, np.int64),
            addr(self._right, np.int64),
            addr(self._order, np.int64),
            n,
            kind,
            float(begin),
            float(end),
            addr(self._result, np.int64, writable=True),
            addr(self._stack, np.int64, writable=True),
        )
        if count < 0 or count > n:
            raise RuntimeError(f"native query failed with status {count}")
        return {self._records[int(i)] for i in self._result[:count]}

    def at(self, p):
        return self._query(0, p, p)

    def overlap(self, begin, end=None):
        if end is None:
            try:
                begin, end = begin.begin, begin.end
            except AttributeError as exc:
                raise TypeError("overlap() needs begin and end") from exc
        if begin >= end:
            return set()
        return self._query(1, begin, end)

    def envelop(self, begin, end=None):
        if end is None:
            try:
                begin, end = begin.begin, begin.end
            except AttributeError as exc:
                raise TypeError("envelop() needs begin and end") from exc
        if begin >= end:
            return set()
        return self._query(2, begin, end)

    def _query_many(self, kind, lowers, uppers):
        lower = self._coordinate_array(lowers)
        upper = self._coordinate_array(uppers)
        if lower.ndim != 1 or upper.ndim != 1 or lower.shape != upper.shape:
            raise ValueError("query coordinates must be equal-length 1D arrays")
        q = lower.size
        if q == 0:
            return []
        if kind:
            valid = lower < upper
            if not np.all(valid):
                answers = [set() for _ in range(q)]
                valid_answers = self._query_many(kind, lower[valid], upper[valid])
                for index, answer in zip(np.flatnonzero(valid), valid_answers):
                    answers[int(index)] = answer
                return answers
        self._ensure_index()
        n = len(self._records)
        if n == 0:
            return [set() for _ in range(q)]
        counts = np.empty(q, dtype=np.int64)
        native = lib()
        common = (
            addr(self._begins, np.float64),
            addr(self._ends, np.float64),
            addr(self._max_ends, np.float64),
            addr(self._left, np.int64),
            addr(self._right, np.int64),
            addr(self._order, np.int64),
            n,
            kind,
            addr(lower, np.float64),
            addr(upper, np.float64),
            q,
        )
        status = native.mit_count_many(
            *common,
            addr(counts, np.int64, writable=True),
            addr(self._result, np.int64, writable=True),
            addr(self._stack, np.int64, writable=True),
        )
        if status != 0:
            raise RuntimeError(f"native batch count failed with status {status}")
        if np.any(counts < 0) or np.any(counts > n):
            raise RuntimeError("native batch count returned an invalid result size")
        offsets = np.empty(q + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        result = np.empty(int(offsets[-1]), dtype=np.int64)
        if result.size:
            status = native.mit_fill_many(
                *common,
                addr(offsets, np.int64),
                addr(result, np.int64, writable=True),
                addr(self._stack, np.int64, writable=True),
            )
            if status != 0:
                raise RuntimeError(f"native batch fill failed with status {status}")
        return [
            {self._records[int(i)] for i in result[offsets[j] : offsets[j + 1]]}
            for j in range(q)
        ]

    def at_many(self, points):
        points = self._coordinate_array(points)
        return self._query_many(0, points, points)

    def overlap_many(self, begins, ends):
        return self._query_many(1, begins, ends)

    def envelop_many(self, begins, ends):
        return self._query_many(2, begins, ends)

    def overlaps(self, begin, end=None):
        return bool(self.at(begin) if end is None else self.overlap(begin, end))

    def overlaps_point(self, p):
        return bool(self.at(p))

    def overlaps_range(self, begin, end):
        return bool(self.overlap(begin, end))

    def add(self, interval):
        iv = self._coerce_interval(interval)
        before = len(self._intervals)
        self._intervals.add(iv)
        if len(self._intervals) != before:
            self._invalidate()

    def addi(self, begin, end, data=None):
        self.add(Interval(begin, end, data))

    append = add
    appendi = addi

    def update(self, intervals):
        additions = {self._coerce_interval(interval) for interval in intervals}
        before = len(self._intervals)
        self._intervals.update(additions)
        if len(self._intervals) != before:
            self._invalidate()

    def remove(self, interval):
        iv = self._coerce_interval(interval)
        try:
            self._intervals.remove(iv)
        except KeyError as exc:
            raise ValueError(iv) from exc
        self._invalidate()

    def discard(self, interval):
        try:
            iv = self._coerce_interval(interval)
        except ValueError:
            return
        before = len(self._intervals)
        self._intervals.discard(iv)
        if len(self._intervals) != before:
            self._invalidate()

    def removei(self, begin, end, data=None):
        self.remove(Interval(begin, end, data))

    def discardi(self, begin, end, data=None):
        self.discard(Interval(begin, end, data))

    def remove_overlap(self, begin, end=None):
        doomed = self.at(begin) if end is None else self.overlap(begin, end)
        if doomed:
            self._intervals.difference_update(doomed)
            self._invalidate()

    def remove_envelop(self, begin, end):
        doomed = self.envelop(begin, end)
        if doomed:
            self._intervals.difference_update(doomed)
            self._invalidate()

    def clear(self):
        if self._intervals:
            self._intervals.clear()
            self._invalidate()

    def pop(self):
        interval = self._intervals.pop()
        self._invalidate()
        return interval

    def containsi(self, begin, end, data=None):
        return Interval(begin, end, data) in self._intervals

    @property
    def all_intervals(self):
        return set(self._intervals)

    def items(self):
        return set(self._intervals)

    def is_empty(self):
        return not self._intervals

    def begin(self):
        if not self._intervals:
            return 0
        return min(iv.begin for iv in self._intervals)

    def end(self):
        if not self._intervals:
            return 0
        return max(iv.end for iv in self._intervals)

    def range(self):
        return Interval(self.begin(), self.end())

    def span(self):
        if not self._intervals:
            return 0
        return self.end() - self.begin()

    def chop(self, begin, end, datafunc=None):
        affected = self.overlap(begin, end)
        replacements = []
        for iv in affected:
            if iv.begin < begin:
                data = datafunc(iv, True) if datafunc else iv.data
                replacements.append(Interval(iv.begin, begin, data))
            if iv.end > end:
                data = datafunc(iv, False) if datafunc else iv.data
                replacements.append(Interval(end, iv.end, data))
        self._intervals.difference_update(affected)
        self._intervals.update(replacements)
        if affected:
            self._invalidate()

    def slice(self, point, datafunc=None):
        affected = {iv for iv in self.at(point) if iv.begin < point < iv.end}
        replacements = []
        for iv in affected:
            lower_data = datafunc(iv, True) if datafunc else iv.data
            upper_data = datafunc(iv, False) if datafunc else iv.data
            replacements.extend(
                (
                    Interval(iv.begin, point, lower_data),
                    Interval(point, iv.end, upper_data),
                )
            )
        self._intervals.difference_update(affected)
        self._intervals.update(replacements)
        if affected:
            self._invalidate()

    def union(self, other):
        return type(self)(self._intervals.union(other))

    def difference(self, other):
        return type(self)(self._intervals.difference(other))

    def intersection(self, other):
        return type(self)(self._intervals.intersection(other))

    def symmetric_difference(self, other):
        return type(self)(self._intervals.symmetric_difference(other))

    def isdisjoint(self, other):
        return self._intervals.isdisjoint(other)

    def difference_update(self, other):
        before = len(self._intervals)
        self._intervals.difference_update(other)
        if len(self._intervals) != before:
            self._invalidate()

    def intersection_update(self, other):
        old = self._intervals.copy()
        self._intervals.intersection_update(other)
        if self._intervals != old:
            self._invalidate()

    def symmetric_difference_update(self, other):
        self._intervals.symmetric_difference_update(
            self._coerce_interval(iv) for iv in other
        )
        self._invalidate()

    def issubset(self, other):
        return self._intervals.issubset(other)

    def issuperset(self, other):
        return self._intervals.issuperset(other)

    def find_nested(self):
        nested = {}
        by_length = sorted(self._intervals, key=Interval.length, reverse=True)
        for index, parent in enumerate(by_length):
            for child in by_length[index + 1 :]:
                if parent.contains_interval(child):
                    nested.setdefault(parent, set()).add(child)
        return nested

    def verify(self):
        for iv in self._intervals:
            assert isinstance(iv, Interval)
            assert not iv.is_null()
        self._ensure_index()
        assert len(self._records) == len(self._intervals)

    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.step is not None:
                raise ValueError("slice steps are not supported")
            if index.start is None or index.stop is None:
                raise ValueError("both slice bounds are required")
            return self.overlap(index.start, index.stop)
        return self.at(index)

    def __setitem__(self, index, data):
        if not isinstance(index, slice):
            raise TypeError("interval assignment requires tree[begin:end] = data")
        if index.step is not None:
            raise ValueError("slice steps are not supported")
        self.addi(index.start, index.stop, data)

    def __delitem__(self, index):
        if isinstance(index, slice):
            self.remove_overlap(index.start, index.stop)
        else:
            self.remove_overlap(index)

    def __contains__(self, interval):
        return interval in self._intervals

    def __iter__(self):
        return iter(self._intervals)

    def iter(self):
        return iter(self._intervals)

    def __len__(self):
        return len(self._intervals)

    def __bool__(self):
        return bool(self._intervals)

    def __repr__(self):
        if not self:
            return f"{type(self).__name__}()"
        body = ", ".join(map(repr, sorted(self, key=attrgetter("begin", "end"))))
        return f"{type(self).__name__}([{body}])"

    def __eq__(self, other):
        if isinstance(other, IntervalTree):
            return self._intervals == other._intervals
        return False

    def __le__(self, other):
        return self.issubset(other)

    def __lt__(self, other):
        return self._intervals < set(other)

    def __ge__(self, other):
        return self.issuperset(other)

    def __gt__(self, other):
        return self._intervals > set(other)

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __sub__(self, other):
        return self.difference(other)

    def __xor__(self, other):
        return self.symmetric_difference(other)

    def __ior__(self, other):
        self.update(other)
        return self

    def __iand__(self, other):
        self.intersection_update(other)
        return self

    def __isub__(self, other):
        self.difference_update(other)
        return self

    def __ixor__(self, other):
        self.symmetric_difference_update(other)
        return self
