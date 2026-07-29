# mojo-intervaltree

A Mojo-accelerated mutable interval tree for Python. It implements a tested
subset of the upstream `intervaltree` API, plus batch-query extensions. Large
batches of queries run in one FFI call.

## Status

Covered and tested against upstream:

- `Interval` value, range, overlap, containment, and distance behavior
- `IntervalTree` construction, iteration, bounds, copying, and slice syntax
- `at`, `overlap`, `envelop`, `overlaps`, and `find_nested`
- add, update, remove, discard, range removal, `chop`, `slice`, `clear`, and `pop`
- union, intersection, difference, symmetric difference, and set comparisons
- batch extensions: `at_many`, `overlap_many`, and `envelop_many`

Not covered: `split_overlaps`, `merge_equals`, `merge_overlaps`,
`print_structure`, `score`, the `Interval.lt`/`le`/`gt`/`ge` helpers, and
internal `Node` APIs. Coordinates must be finite real numbers exactly
representable as `float64`; datetime and arbitrary comparable coordinate types
supported by upstream are outside this port.

## Install

From a source checkout:

```console
pixi install
pixi run build
pixi run test
```

## Usage

```python
from mojo_intervaltree import IntervalTree

tree = IntervalTree.from_tuples([
    (1, 2, "one"),
    (4, 7, "four-seven"),
    (5, 9, "five-nine"),
])

assert {iv.data for iv in tree[6]} == {"four-seven", "five-nine"}
assert {iv.data for iv in tree.overlap(1, 5)} == {"one", "four-seven"}

answers = tree.at_many([1, 6, 9])
assert [len(answer) for answer in answers] == [1, 2, 0]
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz. The
query rows use a tree of 200,000 intervals and include creation of the returned
Python sets. `intervaltree` is the conda-forge 3.1.0 package used by the parity
suite.

| case | mojo-intervaltree | intervaltree 3.1.0 | ratio |
| --- | ---: | ---: | ---: |
| build/index 200k intervals | 1370.43 ms | 6279.67 ms | 4.58x faster |
| 20k point queries | 585.08 ms | 1170.96 ms | 2.00x faster |
| 5k overlap queries | 293.37 ms | 6426.42 ms | 21.91x faster |
| 5k envelop queries | 177.16 ms | 6975.61 ms | 39.38x faster |

These are best-of-five timings except construction, which is best-of-three.
They are measurements from this machine, not portable performance guarantees.
There is no GPU path.

## How it works

The Python layer owns immutable `Interval` objects and a mutable set. On the
first query after a mutation it lays the intervals out as a balanced binary
search tree in contiguous NumPy arrays. Every node stores its subtree's maximum
end coordinate, allowing the Mojo traversal to prune subtrees that cannot
overlap a query.

Index construction sorts coordinates in NumPy, then builds the balanced topology
and subtree maxima in Mojo. The contiguous maximum-end initialization uses
native-width SIMD with a scalar tail. The topology has serial parent-child
dependencies, so it does not use a parallel path.

Arrays cross the C ABI as integer addresses and are reconstructed as
`UnsafePointer[..., AnyOrigin[mut=True]]` inside the exported Mojo functions.
The Mojo library never allocates or owns Python memory. Batch queries use one
counting pass, one exactly-sized result allocation, and one filling pass; the
returned integer IDs are mapped back to the original Python `Interval` objects,
so arbitrary data payloads remain in Python.

Licensed under the MIT License. See `LICENSE`.
