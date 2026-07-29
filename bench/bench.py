"""Benchmark Mojo interval queries against upstream intervaltree."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from intervaltree import IntervalTree as ReferenceTree  # noqa: E402
from mojo_intervaltree import IntervalTree  # noqa: E402


def timeit(fn, repeat=5):
    best = math.inf
    value = None
    for _ in range(repeat):
        start = time.perf_counter()
        value = fn()
        best = min(best, time.perf_counter() - start)
    return best, value


def machine():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def normalized(intervals):
    return {(iv.begin, iv.end, iv.data) for iv in intervals}


def assert_same_batches(actual, expected):
    assert len(actual) == len(expected)
    assert all(normalized(a) == normalized(e) for a, e in zip(actual, expected))


def main():
    rng = np.random.default_rng(2026)
    n = 200_000
    starts = rng.integers(0, 10_000_000, size=n, dtype=np.int64)
    lengths = rng.integers(1, 1000, size=n, dtype=np.int64)
    tuples = [
        (int(begin), int(begin + length), int(i))
        for i, (begin, length) in enumerate(zip(starts, lengths))
    ]

    build_mojo, ours = timeit(lambda: IntervalTree.from_tuples(tuples).build_index(), repeat=3)
    build_ref, theirs = timeit(lambda: ReferenceTree.from_tuples(tuples), repeat=3)

    point_values = rng.integers(0, 10_000_000, size=20_000, dtype=np.int64)
    range_begins = rng.integers(0, 9_999_000, size=5_000, dtype=np.int64)
    range_ends = range_begins + rng.integers(1, 2000, size=5_000, dtype=np.int64)

    ours.at_many(point_values[:10])
    point_mojo, point_result = timeit(lambda: ours.at_many(point_values))
    point_ref, ref_point_result = timeit(
        lambda: [theirs.at(int(point)) for point in point_values]
    )
    assert_same_batches(point_result, ref_point_result)

    range_mojo, range_result = timeit(
        lambda: ours.overlap_many(range_begins, range_ends)
    )
    range_ref, ref_range_result = timeit(
        lambda: [
            theirs.overlap(int(begin), int(end))
            for begin, end in zip(range_begins, range_ends)
        ]
    )
    assert_same_batches(range_result, ref_range_result)

    envelop_mojo, envelop_result = timeit(
        lambda: ours.envelop_many(range_begins, range_ends)
    )
    envelop_ref, ref_envelop_result = timeit(
        lambda: [
            theirs.envelop(int(begin), int(end))
            for begin, end in zip(range_begins, range_ends)
        ]
    )
    assert_same_batches(envelop_result, ref_envelop_result)

    cases = [
        ("build/index 200k intervals", build_mojo, build_ref),
        ("20k point queries", point_mojo, point_ref),
        ("5k overlap queries", range_mojo, range_ref),
        ("5k envelop queries", envelop_mojo, envelop_ref),
    ]
    print(f"Machine: {machine()}")
    print()
    print("| case | mojo-intervaltree | intervaltree 3.1.0 | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_time, ref_time in cases:
        ratio = ref_time / mojo_time
        word = "faster" if ratio >= 1 else "slower"
        shown = ratio if ratio >= 1 else 1 / ratio
        print(
            f"| {name} | {mojo_time * 1000:.2f} ms | "
            f"{ref_time * 1000:.2f} ms | {shown:.2f}x {word} |"
        )


if __name__ == "__main__":
    main()
