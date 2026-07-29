"""Augmented balanced-tree queries over caller-owned arrays."""

from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


@export("mit_build_index")
def mit_build_index(
    source_begins_addr: Int,
    source_ends_addr: Int,
    sorted_order_addr: Int,
    n: Int,
    begins_addr: Int,
    ends_addr: Int,
    max_ends_addr: Int,
    left_addr: Int,
    right_addr: Int,
    order_addr: Int,
    stack_addr: Int,
) abi("C") -> Int:
    if n <= 0:
        return -1
    if (
        source_begins_addr == 0 or source_ends_addr == 0
        or sorted_order_addr == 0 or begins_addr == 0 or ends_addr == 0
        or max_ends_addr == 0 or left_addr == 0 or right_addr == 0
        or order_addr == 0 or stack_addr == 0
    ):
        return -2
    var source_begins = FPtr(unsafe_from_address=source_begins_addr)
    var source_ends = FPtr(unsafe_from_address=source_ends_addr)
    var sorted_order = IPtr(unsafe_from_address=sorted_order_addr)
    var begins = FPtr(unsafe_from_address=begins_addr)
    var ends = FPtr(unsafe_from_address=ends_addr)
    var max_ends = FPtr(unsafe_from_address=max_ends_addr)
    var left = IPtr(unsafe_from_address=left_addr)
    var right = IPtr(unsafe_from_address=right_addr)
    var order = IPtr(unsafe_from_address=order_addr)
    var stack = IPtr(unsafe_from_address=stack_addr)

    var top = 1
    var next_node = 0
    stack[0] = 0
    stack[1] = Int64(n)
    stack[2] = -1
    stack[3] = 0
    while top > 0:
        top -= 1
        var base = top * 4
        var lo = Int(stack[base])
        var hi = Int(stack[base + 1])
        var parent = Int(stack[base + 2])
        var is_right = Int(stack[base + 3])
        var mid = (lo + hi) // 2
        var node = next_node
        next_node += 1
        var source = Int(sorted_order[mid])
        begins[node] = source_begins[source]
        ends[node] = source_ends[source]
        order[node] = Int64(source)
        left[node] = -1
        right[node] = -1
        if parent >= 0:
            if is_right:
                right[parent] = Int64(node)
            else:
                left[parent] = Int64(node)

        if mid + 1 < hi:
            base = top * 4
            stack[base] = Int64(mid + 1)
            stack[base + 1] = Int64(hi)
            stack[base + 2] = Int64(node)
            stack[base + 3] = 1
            top += 1
        if lo < mid:
            base = top * 4
            stack[base] = Int64(lo)
            stack[base + 1] = Int64(mid)
            stack[base + 2] = Int64(node)
            stack[base + 3] = 0
            top += 1

    comptime W = simdwidthof[DType.float64]()
    var i = 0
    while i + W <= n:
        max_ends.store(i, ends.load[width=W](i))
        i += W
    while i < n:
        max_ends[i] = ends[i]
        i += 1

    i = n
    while i > 0:
        i -= 1
        var maximum = max_ends[i]
        var left_node = Int(left[i])
        if left_node >= 0:
            maximum = max(maximum, max_ends[left_node])
        var right_node = Int(right[i])
        if right_node >= 0:
            maximum = max(maximum, max_ends[right_node])
        max_ends[i] = maximum
    return 0


def query(
    begins: FPtr,
    ends: FPtr,
    max_ends: FPtr,
    left: IPtr,
    right: IPtr,
    order: IPtr,
    n: Int,
    kind: Int,
    lower: Float64,
    upper: Float64,
    result: IPtr,
    result_base: Int,
    stack: IPtr,
) -> Int:
    if n == 0:
        return 0
    var top = 1
    var count = 0
    stack[0] = 0
    while top > 0:
        top -= 1
        var node = Int(stack[top])

        if kind == 0:
            if max_ends[node] <= lower:
                continue
            if begins[node] <= lower:
                if lower < ends[node]:
                    result[result_base + count] = order[node]
                    count += 1
                var l = Int(left[node])
                if l >= 0:
                    stack[top] = Int64(l)
                    top += 1
                var r = Int(right[node])
                if r >= 0:
                    stack[top] = Int64(r)
                    top += 1
            else:
                var l = Int(left[node])
                if l >= 0:
                    stack[top] = Int64(l)
                    top += 1
            continue

        if kind == 1:
            if max_ends[node] <= lower:
                continue
            if begins[node] < upper:
                if ends[node] > lower:
                    result[result_base + count] = order[node]
                    count += 1
                var l = Int(left[node])
                if l >= 0:
                    stack[top] = Int64(l)
                    top += 1
                var r = Int(right[node])
                if r >= 0:
                    stack[top] = Int64(r)
                    top += 1
            else:
                var l = Int(left[node])
                if l >= 0:
                    stack[top] = Int64(l)
                    top += 1
            continue

        if begins[node] < lower:
            var r = Int(right[node])
            if r >= 0:
                stack[top] = Int64(r)
                top += 1
        elif begins[node] >= upper:
            var l = Int(left[node])
            if l >= 0:
                stack[top] = Int64(l)
                top += 1
        else:
            if ends[node] <= upper:
                result[result_base + count] = order[node]
                count += 1
            var l = Int(left[node])
            if l >= 0:
                stack[top] = Int64(l)
                top += 1
            var r = Int(right[node])
            if r >= 0:
                stack[top] = Int64(r)
                top += 1
    return count


@export("mit_query_one")
def mit_query_one(
    begins_addr: Int,
    ends_addr: Int,
    max_ends_addr: Int,
    left_addr: Int,
    right_addr: Int,
    order_addr: Int,
    n: Int,
    kind: Int,
    lower: Float64,
    upper: Float64,
    result_addr: Int,
    stack_addr: Int,
) abi("C") -> Int:
    if n <= 0 or kind < 0 or kind > 2:
        return -1
    if (
        begins_addr == 0 or ends_addr == 0 or max_ends_addr == 0
        or left_addr == 0 or right_addr == 0 or order_addr == 0
        or result_addr == 0 or stack_addr == 0
    ):
        return -1
    return query(
        FPtr(unsafe_from_address=begins_addr),
        FPtr(unsafe_from_address=ends_addr),
        FPtr(unsafe_from_address=max_ends_addr),
        IPtr(unsafe_from_address=left_addr),
        IPtr(unsafe_from_address=right_addr),
        IPtr(unsafe_from_address=order_addr),
        n,
        kind,
        lower,
        upper,
        IPtr(unsafe_from_address=result_addr),
        0,
        IPtr(unsafe_from_address=stack_addr),
    )


@export("mit_count_many")
def mit_count_many(
    begins_addr: Int,
    ends_addr: Int,
    max_ends_addr: Int,
    left_addr: Int,
    right_addr: Int,
    order_addr: Int,
    n: Int,
    kind: Int,
    lowers_addr: Int,
    uppers_addr: Int,
    q: Int,
    counts_addr: Int,
    result_addr: Int,
    stack_addr: Int,
) abi("C") -> Int:
    if n <= 0 or q <= 0 or kind < 0 or kind > 2:
        return -1
    if (
        begins_addr == 0 or ends_addr == 0 or max_ends_addr == 0
        or left_addr == 0 or right_addr == 0 or order_addr == 0
        or lowers_addr == 0 or uppers_addr == 0 or counts_addr == 0
        or result_addr == 0 or stack_addr == 0
    ):
        return -2
    var begins = FPtr(unsafe_from_address=begins_addr)
    var ends = FPtr(unsafe_from_address=ends_addr)
    var max_ends = FPtr(unsafe_from_address=max_ends_addr)
    var left = IPtr(unsafe_from_address=left_addr)
    var right = IPtr(unsafe_from_address=right_addr)
    var order = IPtr(unsafe_from_address=order_addr)
    var lowers = FPtr(unsafe_from_address=lowers_addr)
    var uppers = FPtr(unsafe_from_address=uppers_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)
    var result = IPtr(unsafe_from_address=result_addr)
    var stack = IPtr(unsafe_from_address=stack_addr)
    for i in range(q):
        counts[i] = Int64(query(
            begins,
            ends,
            max_ends,
            left,
            right,
            order,
            n,
            kind,
            lowers[i],
            uppers[i],
            result,
            0,
            stack,
        ))
    return 0


@export("mit_fill_many")
def mit_fill_many(
    begins_addr: Int,
    ends_addr: Int,
    max_ends_addr: Int,
    left_addr: Int,
    right_addr: Int,
    order_addr: Int,
    n: Int,
    kind: Int,
    lowers_addr: Int,
    uppers_addr: Int,
    q: Int,
    offsets_addr: Int,
    result_addr: Int,
    stack_addr: Int,
) abi("C") -> Int:
    if n <= 0 or q <= 0 or kind < 0 or kind > 2:
        return -1
    if (
        begins_addr == 0 or ends_addr == 0 or max_ends_addr == 0
        or left_addr == 0 or right_addr == 0 or order_addr == 0
        or lowers_addr == 0 or uppers_addr == 0 or offsets_addr == 0
        or result_addr == 0 or stack_addr == 0
    ):
        return -2
    var begins = FPtr(unsafe_from_address=begins_addr)
    var ends = FPtr(unsafe_from_address=ends_addr)
    var max_ends = FPtr(unsafe_from_address=max_ends_addr)
    var left = IPtr(unsafe_from_address=left_addr)
    var right = IPtr(unsafe_from_address=right_addr)
    var order = IPtr(unsafe_from_address=order_addr)
    var lowers = FPtr(unsafe_from_address=lowers_addr)
    var uppers = FPtr(unsafe_from_address=uppers_addr)
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var result = IPtr(unsafe_from_address=result_addr)
    var stack = IPtr(unsafe_from_address=stack_addr)
    for i in range(q):
        _ = query(
            begins,
            ends,
            max_ends,
            left,
            right,
            order,
            n,
            kind,
            lowers[i],
            uppers[i],
            result,
            Int(offsets[i]),
            stack,
        )
    return 0
