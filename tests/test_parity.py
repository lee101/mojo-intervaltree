"""Behavioural parity with the real intervaltree package."""

from __future__ import annotations

import random
from fractions import Fraction

import numpy as np
import pytest
from intervaltree import Interval as ReferenceInterval
from intervaltree import IntervalTree as ReferenceTree

from mojo_intervaltree import Interval, IntervalTree
from mojo_intervaltree._lib import addr, lib


def normalized(intervals):
    return {(iv.begin, iv.end, iv.data) for iv in intervals}


def paired(tuples):
    return IntervalTree.from_tuples(tuples), ReferenceTree.from_tuples(tuples)


def assert_same(ours, reference):
    assert normalized(ours) == normalized(reference)


def test_interval_value_api_parity():
    ours = Interval(1, 5, "x")
    ref = ReferenceInterval(1, 5, "x")
    assert tuple(ours) == tuple(ref)
    assert ours.length() == ref.length()
    assert ours.contains_point(1) == ref.contains_point(1)
    assert ours.contains_point(5) == ref.contains_point(5)
    assert ours.overlaps(4, 8) == ref.overlaps(4, 8)
    assert ours.overlap_size(4, 8) == ref.overlap_size(4, 8)
    other, ref_other = Interval(8, 10), ReferenceInterval(8, 10)
    assert ours.distance_to(other) == ref.distance_to(ref_other)


def test_construction_iteration_and_bounds_parity():
    values = [(1, 2, "a"), (4, 7, "b"), (5, 9, "c")]
    ours, ref = paired(values)
    assert_same(ours, ref)
    assert len(ours) == len(ref)
    assert ours.begin() == ref.begin()
    assert ours.end() == ref.end()
    assert tuple(ours.range()) == tuple(ref.range())
    assert ours.is_empty() == ref.is_empty()
    assert normalized(ours.items()) == normalized(ref.items())


@pytest.mark.parametrize(
    "point, expected",
    [
        (0, set()),
        (1, {(1, 2, "a")}),
        (2, set()),
        (4, {(4, 7, "b")}),
        (6, {(4, 7, "b"), (5, 9, "c")}),
        (9, set()),
    ],
)
def test_point_boundary_parity(point, expected):
    ours, ref = paired([(1, 2, "a"), (4, 7, "b"), (5, 9, "c")])
    assert normalized(ours.at(point)) == expected
    assert_same(ours.at(point), ref.at(point))
    assert_same(ours[point], ref[point])


@pytest.mark.parametrize("begin,end", [(0, 1), (1, 2), (2, 4), (1, 5), (6, 8)])
def test_overlap_boundary_parity(begin, end):
    ours, ref = paired([(1, 2, "a"), (4, 7, "b"), (5, 9, "c")])
    assert_same(ours.overlap(begin, end), ref.overlap(begin, end))
    assert_same(ours[begin:end], ref[begin:end])
    assert ours.overlaps(begin, end) == ref.overlaps(begin, end)


def test_envelop_parity():
    ours, ref = paired([(0, 10, "wide"), (1, 2, "a"), (4, 7, "b"), (5, 9, "c")])
    for begin, end in [(0, 10), (1, 5), (4, 9), (6, 8)]:
        assert_same(ours.envelop(begin, end), ref.envelop(begin, end))


def test_random_query_parity():
    rng = random.Random(42)
    values = []
    for i in range(1500):
        begin = rng.randint(-2000, 2000) / 10
        end = begin + rng.randint(1, 200) / 10
        values.append((begin, end, i))
    ours, ref = paired(values)
    for _ in range(300):
        point = rng.randint(-2200, 2200) / 10
        assert_same(ours.at(point), ref.at(point))
        begin = rng.randint(-2200, 2100) / 10
        end = begin + rng.randint(1, 300) / 10
        assert_same(ours.overlap(begin, end), ref.overlap(begin, end))
        assert_same(ours.envelop(begin, end), ref.envelop(begin, end))


def test_batch_queries_match_upstream_loop():
    rng = np.random.default_rng(7)
    starts = rng.integers(-1000, 1000, size=2000)
    lengths = rng.integers(1, 50, size=2000)
    values = [(int(a), int(a + b), int(i)) for i, (a, b) in enumerate(zip(starts, lengths))]
    ours, ref = paired(values)
    points = rng.integers(-1100, 1100, size=200)
    for actual, point in zip(ours.at_many(points), points):
        assert_same(actual, ref.at(int(point)))
    qbegin = rng.integers(-1100, 1000, size=100)
    qend = qbegin + rng.integers(1, 100, size=100)
    for actual, begin, end in zip(ours.overlap_many(qbegin, qend), qbegin, qend):
        assert_same(actual, ref.overlap(int(begin), int(end)))
    for actual, begin, end in zip(ours.envelop_many(qbegin, qend), qbegin, qend):
        assert_same(actual, ref.envelop(int(begin), int(end)))


def test_native_index_simd_tail_and_duplicate_boundaries():
    # Exercise every possible tail length around likely SIMD widths.
    for size in range(1, 34):
        values = [
            (index % 7, index % 7 + 1 + index % 5, f"value-{index}")
            for index in range(size)
        ]
        ours, ref = paired(values)
        ours.build_index()
        for point in np.linspace(-1, 12, 29):
            assert_same(ours.at(point), ref.at(point))
        for begin in range(-1, 10):
            assert_same(ours.overlap(begin, begin + 3), ref.overlap(begin, begin + 3))
            assert_same(ours.envelop(begin, begin + 6), ref.envelop(begin, begin + 6))


def test_batch_queries_cross_parallel_threshold():
    rng = np.random.default_rng(11)
    values = [(index * 3, index * 3 + 7, index) for index in range(3000)]
    ours, ref = paired(values)
    points = rng.integers(-10, 9010, size=4097, dtype=np.int64)
    actual = ours.at_many(points)
    for answer, point in zip(actual, points):
        assert_same(answer, ref.at(int(point)))


def test_coordinates_reject_unsafe_float64_narrowing_and_nonfinite_values():
    tree = IntervalTree.from_tuples([(0, 10)])
    unsafe_integer = 2**53 + 1
    for value in [unsafe_integer, Fraction(1, 3), np.longdouble("1.0000000000000000001")]:
        with pytest.raises((TypeError, ValueError)):
            tree.at(value)
    for value in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            tree.at(value)
    with pytest.raises(ValueError):
        IntervalTree.from_tuples([(unsafe_integer, unsafe_integer + 2)])
    with pytest.raises(ValueError):
        tree.at_many(np.array([0, unsafe_integer], dtype=np.int64))


def test_batch_inputs_accept_strided_arrays_by_making_owned_contiguous_buffers():
    tree = IntervalTree.from_tuples([(0, 2), (2, 4)])
    points = np.arange(8, dtype=np.int32)[::2]
    assert [len(answer) for answer in tree.at_many(points)] == [1, 1, 0, 0]


def test_ffi_rejects_invalid_buffers_before_dereferencing():
    native = lib()
    assert native.mit_query_one(0, 0, 0, 0, 0, 0, 1, 0, 0.0, 0.0, 0, 0) < 0
    with pytest.raises(TypeError):
        addr(np.ones(1, dtype=np.float32), np.float64)
    with pytest.raises(ValueError):
        addr(np.ones(4, dtype=np.float64)[::2], np.float64)
    readonly = np.ones(1, dtype=np.int64)
    readonly.flags.writeable = False
    with pytest.raises(ValueError):
        addr(readonly, np.int64, writable=True)


def test_mutation_parity():
    ours, ref = paired([(0, 5, "a")])
    ours.addi(5, 10, "b")
    ref.addi(5, 10, "b")
    ours[3:7] = "c"
    ref[3:7] = "c"
    ours.discardi(100, 101, "missing")
    ref.discardi(100, 101, "missing")
    ours.removei(5, 10, "b")
    ref.removei(5, 10, "b")
    assert_same(ours, ref)
    ours.remove_overlap(4)
    ref.remove_overlap(4)
    assert_same(ours, ref)
    ours.clear()
    ref.clear()
    assert_same(ours, ref)


def test_remove_range_and_envelop_parity():
    values = [(0, 10, "a"), (10, 20, "b"), (20, 30, "c"), (30, 40, "d")]
    ours, ref = paired(values)
    ours.remove_overlap(25, 35)
    ref.remove_overlap(25, 35)
    assert_same(ours, ref)
    ours.remove_envelop(5, 20)
    ref.remove_envelop(5, 20)
    assert_same(ours, ref)


def test_chop_and_slice_parity():
    ours, ref = paired([(0, 10, "a"), (5, 15, "b")])
    ours.slice(3)
    ref.slice(3)
    assert_same(ours, ref)
    ours.chop(7, 12)
    ref.chop(7, 12)
    assert_same(ours, ref)


def test_set_operations_parity():
    left_values = [(0, 2, "a"), (2, 4, "b")]
    right_values = [(2, 4, "b"), (4, 6, "c")]
    ours_left, ref_left = paired(left_values)
    ours_right, ref_right = paired(right_values)
    for ours, ref in [
        (ours_left | ours_right, ref_left | ref_right),
        (ours_left & ours_right, ref_left & ref_right),
        (ours_left - ours_right, ref_left - ref_right),
        (ours_left ^ ours_right, ref_left ^ ref_right),
    ]:
        assert_same(ours, ref)
    assert (ours_left <= ours_left.copy()) == (ref_left <= ref_left.copy())
    assert ours_left.isdisjoint(ours_right) == ref_left.isdisjoint(ref_right)


def test_aliases_pop_span_and_iter_parity():
    ours, ref = paired([(0, 2, "a"), (5, 8, "b")])
    ours.append(Interval(9, 10, "c"))
    ref.append(ReferenceInterval(9, 10, "c"))
    ours.appendi(10, 11, "d")
    ref.appendi(10, 11, "d")
    assert ours.span() == ref.span()
    assert normalized(ours.iter()) == normalized(ref.iter())
    ours.verify()
    pop_ours, pop_ref = paired([(20, 30, "only")])
    assert tuple(pop_ours.pop()) == tuple(pop_ref.pop())
    assert not pop_ours and not pop_ref


def test_find_nested_parity():
    values = [(0, 20, "outer"), (2, 5, "a"), (4, 12, "middle"), (7, 9, "b")]
    ours, ref = paired(values)
    ours_result = {
        (key.begin, key.end, key.data): normalized(value)
        for key, value in ours.find_nested().items()
    }
    ref_result = {
        (key.begin, key.end, key.data): normalized(value)
        for key, value in ref.find_nested().items()
    }
    assert ours_result == ref_result


def test_duplicate_and_unhashable_data_parity():
    ours = IntervalTree()
    ours.add(Interval(0, 1, {"value": 1}))
    result = ours.at(0)
    assert len(result) == 1
    iv = next(iter(result))
    assert (iv.begin, iv.end, iv.data) == (0, 1, {"value": 1})


def test_null_intervals_rejected_and_null_queries_are_empty():
    with pytest.raises(ValueError):
        IntervalTree([Interval(1, 1)])
    tree = IntervalTree.from_tuples([(0, 1)])
    assert tree.overlap(1, 1) == set()
    assert tree.envelop(2, 1) == set()
    assert tree.overlap_many([0, 2], [0, 1]) == [set(), set()]


def test_empty_tree_parity():
    ours, ref = paired([])
    assert_same(ours.at(10), ref.at(10))
    assert ours.begin() == ref.begin() == 0
    assert ours.end() == ref.end() == 0
    assert tuple(ours.range()) == tuple(ref.range()) == (0, 0, None)
