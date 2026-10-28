"""
Nearest-neighbour searches with a parallel k-d tree.

Two searches, both exact (the same as ``scipy.spatial.cKDTree.query``):

+ :func:`~swiftsimio.visualisation.kdtree.k_nearest_distances` finds the distance
  from every particle to its k-th nearest neighbour (for smoothing lengths);
+ :func:`~swiftsimio.visualisation.kdtree.nearest_particles` finds the particle
  nearest to each of a set of points (for nearest-neighbour images).

Notes
-----
The tree is balanced (each node splits its particles in half at the median of
its widest axis), so it is stored implicitly: node ``i`` has children
``2i + 1`` and ``2i + 2``, and every node is a contiguous range of the permuted
particles. It is built level by level, the nodes of a level in parallel, and
the coordinates are copied in tree order (one array per axis) so that the
loops over the particles of a leaf read memory in order, and the compiler can
process several particles at once.

Queries go a leaf at a time, in groups of ``LEAF_GROUP`` neighbouring leaves
interleaved between threads (so that each thread gets its share of the dense
regions). Each particle is bounded by those solved before it in its leaf (or,
for the first, in the previous leaf): the k-th distance changes by at most the
distance moved, ``|d_k(q) - d_k(a)| <= |q - a|``, so
``max(d_k(a) - |q - a|) <= d_k(q) <= min(d_k(a) + |q - a|)``. The k-th
distance of the nearest solved particle is also a guess at ``d_k(q)``. The
first particle of a group is found from scratch.

For :func:`~swiftsimio.visualisation.kdtree.nearest_particles` the query points
go in chunks of consecutive points (as the pixels along a row of an image), and
each search starts from the particle nearest to the point before, whose distance
bounds the nearest.

Up to ``k = HEAP_K`` the k-th distance is found with a max-heap of the ``k``
nearest squared distances, searching only below the upper bound. Above it the
search is heap-free: given a shell ``r_lo < d_k <= r_hi`` (first within
``EXACT_GUESS`` of the guess), the particles inside ``r_lo`` are counted, and
the k-th distance is selected among the few in the shell. A shell that misses
the k-th neighbour says on which side it is, and the next shell is there.

Periodic boxes are handled in the metric rather than by copying particles: for
coordinates wrapped into ``[0, L)``, the separation along a periodic axis is
``min(|dx|, L - |dx|)``, and the distance from a query to a node's bounding box
also tries the box's image through the wrap. A free (non-periodic) axis has
period ``FREE_AXIS``, for which both are the non-periodic expressions. Before
building the tree, the boundary of each periodic axis is moved into the widest
gap between particles (``_periodic_cut``), and an axis the particles span less
than half of becomes free, as no distance along it can be shorter through the
wrap.
"""

import numpy as np
from numpy.typing import NDArray
from numba import get_num_threads, njit, prange

LEAF_SIZE = 64
# The heap of the k nearest is used up to this k; above it, a shell is selected from
HEAP_K = 96
# Relative width of the bracket on the k-th distance found by counting before
# selecting from a shell, when there is no bound from solved particles
EXACT_SHELL = 0.1
# The first shell, given bounds from solved particles: within this much of their guess
EXACT_GUESS = 0.12
# Leaves queried one after another by a thread (a power of 2): the first
# particle of each is bounded from the previous leaf's
LEAF_GROUP = 8
# Consecutive query points searched one after another by a thread (each from
# the nearest particle to the point before it) for nearest_particles
QUERY_CHUNK = 256
# The period of an axis that is not periodic: huge but finite, so that the
# periodic separation min(|dx|, period - |dx|) is |dx| (see _separation)
FREE_AXIS = 1e300
# Floating-point optimisations for numba, except those that assume there are no
# NaNs or infinities ("nnan" and "ninf"): the searches use inf for "no bound yet"
FASTMATH = {"nsz", "arcp", "contract", "afn", "reassoc"}


def _tree_depth(n: int, leaf_size: int) -> int:
    """
    Find the number of levels below the root so that every leaf is small enough.

    Parameters
    ----------
    n : int
        Number of particles.
    leaf_size : int
        Largest number of particles in a leaf.

    Returns
    -------
    int
        The depth of the tree.
    """
    depth = 0
    while (n + (1 << depth) - 1) >> depth > leaf_size:
        depth += 1
    return depth


@njit(cache=True)
def _node_ranges(n: int, depth: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Compute the particle range ``[lo, hi)`` of every node of the implicit tree.

    Parameters
    ----------
    n : int
        Number of particles.
    depth : int
        Depth of the tree.

    Returns
    -------
    tuple[NDArray[np.int64], NDArray[np.int64]]
        The first particle of each node, and one past its last.
    """
    n_nodes = (1 << (depth + 1)) - 1
    lo = np.empty(n_nodes, dtype=np.int64)
    hi = np.empty(n_nodes, dtype=np.int64)
    lo[0] = 0
    hi[0] = n
    for i in range((1 << depth) - 1):
        mid = (lo[i] + hi[i]) // 2
        lo[2 * i + 1] = lo[i]
        hi[2 * i + 1] = mid
        lo[2 * i + 2] = mid
        hi[2 * i + 2] = hi[i]
    return lo, hi


@njit(cache=True)
def _bounding_box(
    pos: NDArray[np.float64],
    perm: NDArray[np.int64],
    lo: int,
    hi: int,
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    node: int,
) -> None:
    """
    Compute the bounding box of the particles of a node.

    Parameters
    ----------
    pos : NDArray[np.float64]
        Particle positions, shape ``(N, 3)``.
    perm : NDArray[np.int64]
        Particle indices in tree order.
    lo : int
        First particle (in tree order) of the node.
    hi : int
        One past the last particle of the node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes, written for ``node``.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes, written for ``node``.
    node : int
        The node.
    """
    for d in range(3):
        bmin[node, d] = np.inf
        bmax[node, d] = -np.inf
    for j in range(lo, hi):
        p = perm[j]
        for d in range(3):
            x = pos[p, d]
            if x < bmin[node, d]:
                bmin[node, d] = x
            if x > bmax[node, d]:
                bmax[node, d] = x


@njit(cache=True)
def _select(
    pos: NDArray[np.float64],
    perm: NDArray[np.int64],
    lo: int,
    hi: int,
    nth: int,
    dim: int,
) -> None:
    """
    Partition ``perm[lo:hi]`` about the particle of rank ``nth`` along ``dim``.

    Afterwards ``perm[nth]`` is the particle of that rank, with none smaller
    after it and none larger before it (quickselect, median-of-three pivots).

    Parameters
    ----------
    pos : NDArray[np.float64]
        Particle positions, shape ``(N, 3)``.
    perm : NDArray[np.int64]
        Particle indices, reordered in place.
    lo : int
        First index of the range.
    hi : int
        One past the last index of the range.
    nth : int
        The rank to select.
    dim : int
        The axis to select along.
    """
    left = lo
    right = hi - 1
    while right > left:
        mid = (left + right) // 2
        a = pos[perm[left], dim]
        b = pos[perm[mid], dim]
        c = pos[perm[right], dim]
        if (a <= b <= c) or (c <= b <= a):
            pivot = b
        elif (b <= a <= c) or (c <= a <= b):
            pivot = a
        else:
            pivot = c
        i = left
        j = right
        while i <= j:
            while pos[perm[i], dim] < pivot:
                i += 1
            while pos[perm[j], dim] > pivot:
                j -= 1
            if i <= j:
                t = perm[i]
                perm[i] = perm[j]
                perm[j] = t
                i += 1
                j -= 1
        if nth <= j:
            right = j
        elif nth >= i:
            left = i
        else:
            return


@njit(cache=True, parallel=True)
def _build(pos: NDArray[np.float64], depth: int) -> tuple:
    """
    Build the k-d tree.

    Parameters
    ----------
    pos : NDArray[np.float64]
        Particle positions, shape ``(N, 3)``.
    depth : int
        Depth of the tree.

    Returns
    -------
    tuple
        The particle indices in tree order, the ranges ``lo`` and ``hi`` and
        the bounding boxes ``bmin`` and ``bmax`` of the nodes, and the x, y and
        z coordinates in tree order.
    """
    n = pos.shape[0]
    lo, hi = _node_ranges(n, depth)
    n_nodes = lo.size
    perm = np.arange(n)
    bmin = np.empty((n_nodes, 3), dtype=pos.dtype)
    bmax = np.empty((n_nodes, 3), dtype=pos.dtype)
    for level in range(depth):
        first = (1 << level) - 1
        for k in prange(1 << level):
            node = first + k
            _bounding_box(pos, perm, lo[node], hi[node], bmin, bmax, node)
            dim = 0
            for d in range(1, 3):
                if bmax[node, d] - bmin[node, d] > bmax[node, dim] - bmin[node, dim]:
                    dim = d
            mid = (lo[node] + hi[node]) // 2
            _select(pos, perm, lo[node], hi[node], mid, dim)
    first_leaf = (1 << depth) - 1
    for k in prange(1 << depth):
        node = first_leaf + k
        _bounding_box(pos, perm, lo[node], hi[node], bmin, bmax, node)
    # Coordinates in tree order, one array per axis
    xs = np.empty(n, dtype=pos.dtype)
    ys = np.empty(n, dtype=pos.dtype)
    zs = np.empty(n, dtype=pos.dtype)
    for j in prange(n):
        xs[j] = pos[perm[j], 0]
        ys[j] = pos[perm[j], 1]
        zs[j] = pos[perm[j], 2]
    return perm, lo, hi, bmin, bmax, xs, ys, zs


@njit(cache=True, inline="always")
def _separation(dx: float, period: NDArray[np.float64], d: int) -> float:
    """
    Compute the minimum-image separation along an axis.

    For coordinates wrapped into ``[0, period[d])`` this is
    ``min(|dx|, period[d] - |dx|)``, written without an ``if`` so that the loops
    over a leaf's particles can still process several at once. On a free axis
    (period ``FREE_AXIS``) it is ``|dx|``.

    Parameters
    ----------
    dx : float
        Difference of the coordinates.
    period : NDArray[np.float64]
        Period of each axis.
    d : int
        The axis.

    Returns
    -------
    float
        The separation.
    """
    a = abs(dx)
    return min(a, period[d] - a)


@njit(cache=True, inline="always")
def _axis_gap(
    low: float, high: float, x: float, period: NDArray[np.float64], d: int
) -> float:
    """
    Compute the distance from ``x`` to ``[low, high]`` along an axis.

    The distance through the periodic wrap is used if that is shorter.

    Parameters
    ----------
    low : float
        Lower end of the interval.
    high : float
        Upper end of the interval.
    x : float
        The coordinate.
    period : NDArray[np.float64]
        Period of each axis.
    d : int
        The axis.

    Returns
    -------
    float
        The distance (0 inside the interval).
    """
    if x < low:
        return min(low - x, period[d] - (high - x))
    if x > high:
        return min(x - high, period[d] - (x - low))
    return 0.0


@njit(cache=True, inline="always")
def _farthest(
    low: float, high: float, x: float, period: NDArray[np.float64], d: int
) -> float:
    """
    Bound the distance from ``x`` to the farthest point of ``[low, high]``.

    No particle is more than half a period away along a periodic axis.

    Parameters
    ----------
    low : float
        Lower end of the interval.
    high : float
        Upper end of the interval.
    x : float
        The coordinate.
    period : NDArray[np.float64]
        Period of each axis.
    d : int
        The axis.

    Returns
    -------
    float
        The upper bound.
    """
    return min(max(x - low, high - x), 0.5 * period[d])


@njit(cache=True, inline="always")
def _box_min_distance2(
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    node: int,
    qx: float,
    qy: float,
    qz: float,
    period: NDArray[np.float64],
) -> float:
    """
    Compute the squared distance from a point to the nearest point of a node.

    Parameters
    ----------
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    node : int
        The node.
    qx : float
        The point's x coordinate.
    qy : float
        The point's y coordinate.
    qz : float
        The point's z coordinate.
    period : NDArray[np.float64]
        Period of each axis.

    Returns
    -------
    float
        The squared distance.
    """
    m = _axis_gap(bmin[node, 0], bmax[node, 0], qx, period, 0)
    d2 = m * m
    m = _axis_gap(bmin[node, 1], bmax[node, 1], qy, period, 1)
    d2 += m * m
    m = _axis_gap(bmin[node, 2], bmax[node, 2], qz, period, 2)
    d2 += m * m
    return d2


@njit(cache=True, inline="always")
def _box_max_distance2(
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    node: int,
    qx: float,
    qy: float,
    qz: float,
    period: NDArray[np.float64],
) -> float:
    """
    Bound the squared distance from a point to the farthest point of a node.

    Only used to find nodes wholly inside a radius.

    Parameters
    ----------
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    node : int
        The node.
    qx : float
        The point's x coordinate.
    qy : float
        The point's y coordinate.
    qz : float
        The point's z coordinate.
    period : NDArray[np.float64]
        Period of each axis.

    Returns
    -------
    float
        The upper bound on the squared distance.
    """
    m = _farthest(bmin[node, 0], bmax[node, 0], qx, period, 0)
    d2 = m * m
    m = _farthest(bmin[node, 1], bmax[node, 1], qy, period, 1)
    d2 += m * m
    m = _farthest(bmin[node, 2], bmax[node, 2], qz, period, 2)
    d2 += m * m
    return d2


@njit(cache=True, inline="always")
def _sift_down(heap: NDArray[np.float64], k: int) -> None:
    """
    Restore the max-heap after ``heap[0]`` was replaced.

    Parameters
    ----------
    heap : NDArray[np.float64]
        The heap, in its first ``k`` elements.
    k : int
        The size of the heap.
    """
    j = 0
    value = heap[0]
    while True:
        child = 2 * j + 1
        if child >= k:
            break
        if child + 1 < k and heap[child + 1] > heap[child]:
            child += 1
        if heap[child] <= value:
            break
        heap[j] = heap[child]
        j = child
    heap[j] = value


@njit(cache=True, fastmath=FASTMATH)
def _heap_query(
    qx: float,
    qy: float,
    qz: float,
    k: int,
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    period: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    first_leaf: int,
    bound2: float,
    heap: NDArray[np.float64],
    stack: NDArray[np.int64],
    stack_d2: NDArray[np.float64],
    d2: NDArray[np.float64],
) -> float:
    """
    Find the squared distance from a point to its k-th nearest particle.

    A max-heap keeps the ``k`` nearest squared distances below ``bound2``
    (``bound2`` is returned if fewer than ``k`` are): the nearer child of a node
    is searched first, and nodes whose bounding box is no nearer than the k-th
    distance so far are skipped.

    Parameters
    ----------
    qx : float
        The point's x coordinate.
    qy : float
        The point's y coordinate.
    qz : float
        The point's z coordinate.
    k : int
        The neighbour to find.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    period : NDArray[np.float64]
        Period of each axis.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order.
    first_leaf : int
        Index of the first leaf node.
    bound2 : float
        Upper bound on the squared distance.
    heap : NDArray[np.float64]
        Workspace of at least ``k`` elements.
    stack : NDArray[np.int64]
        Workspace for the nodes to visit.
    stack_d2 : NDArray[np.float64]
        Workspace for the squared distances to the nodes to visit.
    d2 : NDArray[np.float64]
        Workspace of ``LEAF_SIZE`` elements.

    Returns
    -------
    float
        The squared distance to the k-th nearest particle (or ``bound2``).
    """
    for i in range(k):
        heap[i] = bound2
    stack[0] = 0
    stack_d2[0] = 0.0
    top = 1
    while top > 0:
        top -= 1
        node = stack[top]
        if stack_d2[top] >= heap[0]:
            continue
        if node >= first_leaf:
            start = lo[node]
            n = hi[node] - start
            # All the leaf's squared distances first, then their minimum: two
            # simple loops, which the compiler can run on several particles at once
            nearest = np.inf
            for j in range(n):
                dx = _separation(xs[start + j] - qx, period, 0)
                dy = _separation(ys[start + j] - qy, period, 1)
                dz = _separation(zs[start + j] - qz, period, 2)
                d2[j] = dx * dx + dy * dy + dz * dz
                nearest = min(nearest, d2[j])
            if nearest >= heap[0]:
                continue
            for j in range(n):
                if d2[j] < heap[0]:
                    heap[0] = d2[j]
                    _sift_down(heap, k)
            continue
        left = 2 * node + 1
        right = left + 1
        d_left = _box_min_distance2(bmin, bmax, left, qx, qy, qz, period)
        d_right = _box_min_distance2(bmin, bmax, right, qx, qy, qz, period)
        # Nearer child on top, so it is searched first
        if d_left <= d_right:
            near, d_near, far, d_far = left, d_left, right, d_right
        else:
            near, d_near, far, d_far = right, d_right, left, d_left
        if d_far < heap[0]:
            stack[top] = far
            stack_d2[top] = d_far
            top += 1
        if d_near < heap[0]:
            stack[top] = near
            stack_d2[top] = d_near
            top += 1
    return heap[0]


@njit(cache=True)
def _select_value(values: NDArray[np.float64], n: int, nth: int) -> float:
    """
    Find the ``nth`` smallest of ``values[:n]`` (quickselect).

    Parameters
    ----------
    values : NDArray[np.float64]
        The values, reordered in place.
    n : int
        Number of values.
    nth : int
        The rank to find (0 for the smallest).

    Returns
    -------
    float
        The value of rank ``nth``.
    """
    left = 0
    right = n - 1
    while right > left:
        mid = (left + right) // 2
        a = values[left]
        b = values[mid]
        c = values[right]
        if (a <= b <= c) or (c <= b <= a):
            pivot = b
        elif (b <= a <= c) or (c <= a <= b):
            pivot = a
        else:
            pivot = c
        i = left
        j = right
        while i <= j:
            while values[i] < pivot:
                i += 1
            while values[j] > pivot:
                j -= 1
            if i <= j:
                t = values[i]
                values[i] = values[j]
                values[j] = t
                i += 1
                j -= 1
        if nth <= j:
            right = j
        elif nth >= i:
            left = i
        else:
            break
    return values[nth]


@njit(cache=True, fastmath=FASTMATH)
def _count_within(
    qx: float,
    qy: float,
    qz: float,
    r2: float,
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    period: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    first_leaf: int,
    stack: NDArray[np.int64],
) -> int:
    """
    Count the particles within a radius of a point.

    Parameters
    ----------
    qx : float
        The point's x coordinate.
    qy : float
        The point's y coordinate.
    qz : float
        The point's z coordinate.
    r2 : float
        The squared radius.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    period : NDArray[np.float64]
        Period of each axis.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order.
    first_leaf : int
        Index of the first leaf node.
    stack : NDArray[np.int64]
        Workspace for the nodes to visit.

    Returns
    -------
    int
        The number of particles within the radius.
    """
    count = 0
    stack[0] = 0
    top = 1
    while top > 0:
        top -= 1
        node = stack[top]
        if _box_min_distance2(bmin, bmax, node, qx, qy, qz, period) > r2:
            continue
        if _box_max_distance2(bmin, bmax, node, qx, qy, qz, period) <= r2:
            count += hi[node] - lo[node]
            continue
        if node >= first_leaf:
            # Without an if, so the compiler can test several particles at once
            inside = 0
            for j in range(lo[node], hi[node]):
                dx = _separation(xs[j] - qx, period, 0)
                dy = _separation(ys[j] - qy, period, 1)
                dz = _separation(zs[j] - qz, period, 2)
                inside += dx * dx + dy * dy + dz * dz <= r2
            count += inside
            continue
        stack[top] = 2 * node + 1
        stack[top + 1] = 2 * node + 2
        top += 2
    return count


@njit(cache=True)
def _refine(
    qx: float,
    qy: float,
    qz: float,
    k: int,
    tolerance: float,
    r_lo: float,
    r_hi: float,
    r_last: float,
    c_last: int,
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    period: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    first_leaf: int,
    stack: NDArray[np.int64],
) -> tuple[float, float]:
    """
    Narrow a bracket ``count(r_lo) < k <= count(r_hi)`` to a relative tolerance.

    Newton steps on ``count ~ r^3`` from the last count, guarded by bisection.
    A step that does not halve the bracket (as when the count jumps over ``k``
    between near-duplicate neighbours) is followed by a bisection, so the
    bracket at least halves every two counts.

    Parameters
    ----------
    qx : float
        The point's x coordinate.
    qy : float
        The point's y coordinate.
    qz : float
        The point's z coordinate.
    k : int
        The neighbour to bracket.
    tolerance : float
        Stop once ``r_hi - r_lo <= tolerance * r_hi``.
    r_lo : float
        Lower end of the bracket.
    r_hi : float
        Upper end of the bracket.
    r_last : float
        Radius of the last count.
    c_last : int
        The last count (0 if none).
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    period : NDArray[np.float64]
        Period of each axis.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order.
    first_leaf : int
        Index of the first leaf node.
    stack : NDArray[np.int64]
        Workspace for the nodes to visit.

    Returns
    -------
    tuple[float, float]
        The narrowed bracket.
    """
    bisect = False
    for _ in range(200):
        width = r_hi - r_lo
        if r_lo > 0.0 and width <= tolerance * r_hi:
            break
        r = r_last * (k / c_last) ** (1.0 / 3.0) if c_last > 0 else 0.0
        if bisect or not (r_lo < r < r_hi):
            r = 0.5 * (r_lo + r_hi) if r_lo > 0.0 else 0.5 * r_hi
        c = _count_within(
            qx, qy, qz, r * r, lo, hi, bmin, bmax, period, xs, ys, zs, first_leaf, stack
        )
        if c >= k:
            r_hi = r
        else:
            r_lo = r
        bisect = r_hi - r_lo > 0.5 * width
        c_last = c
        r_last = r
    return r_lo, r_hi


@njit(cache=True)
def _bracket(
    qx: float,
    qy: float,
    qz: float,
    k: int,
    leaf: int,
    tolerance: float,
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    period: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    first_leaf: int,
    stack: NDArray[np.int64],
) -> tuple[float, float]:
    """
    Bracket the k-th distance of a point by counting, from scratch.

    Finds ``r_lo < r_hi`` with ``count(r_lo) < k <= count(r_hi)`` and
    ``r_hi - r_lo <= tolerance * r_hi``, where ``count(r)`` is the number of
    particles within ``r`` of the point (``2 <= k <= N``).

    Parameters
    ----------
    qx : float
        The point's x coordinate.
    qy : float
        The point's y coordinate.
    qz : float
        The point's z coordinate.
    k : int
        The neighbour to bracket.
    leaf : int
        The leaf holding the point, for a first guess from its density.
    tolerance : float
        Relative width of the bracket.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    period : NDArray[np.float64]
        Period of each axis.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order.
    first_leaf : int
        Index of the first leaf node.
    stack : NDArray[np.int64]
        Workspace for the nodes to visit.

    Returns
    -------
    tuple[float, float]
        The bracket (both 0 if ``k`` particles coincide with the point).
    """
    # First guess from the density of the query's own leaf
    volume = 1.0
    for d in range(3):
        volume *= bmax[leaf, d] - bmin[leaf, d] + 1e-9
    r = 1.0
    in_leaf = hi[leaf] - lo[leaf]
    if volume > 0.0 and in_leaf > 0:
        guess = (3.0 * k / (4.0 * np.pi * (in_leaf / volume))) ** (1.0 / 3.0)
        if np.isfinite(guess) and guess > 0.0:
            r = guess
    args = (lo, hi, bmin, bmax, period, xs, ys, zs, first_leaf, stack)

    # Bracket: count(r_lo) < k <= count(r_hi)
    r_lo = 0.0
    r_hi = 0.0
    c = _count_within(qx, qy, qz, r * r, *args)
    c_last = c
    r_last = r
    if c >= k:
        # k coincident particles: d_k is 0 (a count that reaches only the
        # query's own leaf)
        if _count_within(qx, qy, qz, 0.0, *args) >= k:
            return 0.0, 0.0
        r_hi = r
        for _ in range(64):
            r *= 0.6
            c = _count_within(qx, qy, qz, r * r, *args)
            if c < k:
                r_lo = r
                break
            r_hi = r
            c_last = c
            r_last = r
    else:
        r_lo = r
        for _ in range(64):
            r *= 1.6
            c = _count_within(qx, qy, qz, r * r, *args)
            if c >= k:
                r_hi = r
                c_last = c
                r_last = r
                break
            r_lo = r
        if r_hi == 0.0:
            r_hi = r
    return _refine(qx, qy, qz, k, tolerance, r_lo, r_hi, r_last, c_last, *args)


@njit(cache=True, fastmath=FASTMATH)
def _shell(
    qx: float,
    qy: float,
    qz: float,
    r2_lo: float,
    r2_hi: float,
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    period: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    first_leaf: int,
    stack: NDArray[np.int64],
    buffer: NDArray[np.float64],
) -> tuple[int, int, NDArray[np.float64]]:
    """
    Count the particles inside a shell's inner radius, and collect those in it.

    Parameters
    ----------
    qx : float
        The point's x coordinate.
    qy : float
        The point's y coordinate.
    qz : float
        The point's z coordinate.
    r2_lo : float
        Squared inner radius of the shell.
    r2_hi : float
        Squared outer radius of the shell.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    period : NDArray[np.float64]
        Period of each axis.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order.
    first_leaf : int
        Index of the first leaf node.
    stack : NDArray[np.int64]
        Workspace for the nodes to visit.
    buffer : NDArray[np.float64]
        Workspace for the squared distances in the shell (grown if needed).

    Returns
    -------
    tuple[int, int, NDArray[np.float64]]
        The number of particles inside the inner radius, the number in the
        shell, and the buffer holding their squared distances.
    """
    inside = 0
    m = 0
    stack[0] = 0
    top = 1
    while top > 0:
        top -= 1
        node = stack[top]
        if _box_min_distance2(bmin, bmax, node, qx, qy, qz, period) > r2_hi:
            continue
        if _box_max_distance2(bmin, bmax, node, qx, qy, qz, period) <= r2_lo:
            inside += hi[node] - lo[node]
            continue
        if node < first_leaf:
            stack[top] = 2 * node + 1
            stack[top + 1] = 2 * node + 2
            top += 2
            continue
        if m + hi[node] - lo[node] > buffer.size:
            grown = np.empty(2 * buffer.size + hi[node] - lo[node], dtype=buffer.dtype)
            grown[:m] = buffer[:m]
            buffer = grown
        for i in range(lo[node], hi[node]):
            dx = _separation(xs[i] - qx, period, 0)
            dy = _separation(ys[i] - qy, period, 1)
            dz = _separation(zs[i] - qz, period, 2)
            d2 = dx * dx + dy * dy + dz * dz
            # Without an if (as in _count_within): d2 is always written, but only
            # kept (by moving on to the next slot) if it is in the shell
            inside += d2 <= r2_lo
            buffer[m] = d2
            m += (d2 > r2_lo) & (d2 <= r2_hi)
    return inside, m, buffer


@njit(cache=True)
def _shell_query(
    qx: float,
    qy: float,
    qz: float,
    k: int,
    leaf: int,
    r_lo: float,
    r_hi: float,
    guess: float,
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    period: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    first_leaf: int,
    stack: NDArray[np.int64],
    buffer: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """
    Find the squared distance from a point to its k-th nearest particle by shells.

    No heap of the k nearest: the particles inside an inner radius are counted
    (``c`` of them), and the ``(k - c)``-th smallest squared distance in the
    shell between it and an outer radius is the answer, if the shell holds it.
    Given bounds ``r_lo < d_k <= r_hi`` (``r_hi > 0``), the shell is first the
    part of them within ``EXACT_GUESS`` of the guess, then if that misses, the
    rest on the side it missed; without, the bounds are found by counting.

    Parameters
    ----------
    qx : float
        The point's x coordinate.
    qy : float
        The point's y coordinate.
    qz : float
        The point's z coordinate.
    k : int
        The neighbour to find (``2 <= k <= N``).
    leaf : int
        The leaf holding the point.
    r_lo : float
        Lower bound on the k-th distance.
    r_hi : float
        Upper bound on the k-th distance (0 if there is none).
    guess : float
        A guess at the k-th distance.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    period : NDArray[np.float64]
        Period of each axis.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order.
    first_leaf : int
        Index of the first leaf node.
    stack : NDArray[np.int64]
        Workspace for the nodes to visit.
    buffer : NDArray[np.float64]
        Workspace for the squared distances in a shell (grown if needed).

    Returns
    -------
    tuple[float, NDArray[np.float64]]
        The squared distance, and the (possibly grown) buffer.
    """
    if r_hi <= 0.0:
        r_lo, r_hi = _bracket(
            qx,
            qy,
            qz,
            k,
            leaf,
            EXACT_SHELL,
            lo,
            hi,
            bmin,
            bmax,
            period,
            xs,
            ys,
            zs,
            first_leaf,
            stack,
        )
        s_lo = r_lo
        s_hi = r_hi
    else:
        s_lo = max(r_lo, guess * (1.0 - EXACT_GUESS))
        s_hi = min(r_hi, guess * (1.0 + EXACT_GUESS))
        if not s_lo < s_hi:
            s_lo = r_lo
            s_hi = r_hi
    aimed = False
    while True:
        # Nothing inside a shell from 0 (so coincident particles are in it)
        inside, m, buffer = _shell(
            qx,
            qy,
            qz,
            s_lo * s_lo if s_lo > 0.0 else -1.0,
            s_hi * s_hi,
            lo,
            hi,
            bmin,
            bmax,
            period,
            xs,
            ys,
            zs,
            first_leaf,
            stack,
            buffer,
        )
        if inside < k <= inside + m:
            return _select_value(buffer, m, k - inside - 1), buffer
        # A miss: the next shell is on the side it missed, once from where
        # count ~ r^3 puts d_k, then out to the bound
        if inside >= k:
            if s_lo > r_lo:
                r_hi = s_lo
                s_hi = s_lo
                s_lo = r_lo
                if not aimed:
                    aimed = True
                    s_lo = max(
                        r_lo, s_hi * (k / inside) ** (1.0 / 3.0) * (1.0 - EXACT_GUESS)
                    )
            else:
                # Rounding at the bound: from 0 (nothing inside)
                s_lo = 0.0
                r_lo = 0.0
        elif s_hi < r_hi:
            r_lo = s_hi
            s_lo = s_hi
            s_hi = r_hi
            c = inside + m
            if not aimed and c > 0:
                aimed = True
                s_hi = min(r_hi, s_lo * (k / c) ** (1.0 / 3.0) * (1.0 + EXACT_GUESS))
        else:
            # Rounding at the bound: widen it (ends beyond every particle)
            s_hi = 2.0 * s_hi if s_hi > 0.0 else np.inf
            r_hi = s_hi


@njit(cache=True, fastmath=FASTMATH)
def _leaf_bounds(
    qx: float,
    qy: float,
    qz: float,
    done_x: NDArray[np.float64],
    done_y: NDArray[np.float64],
    done_z: NDArray[np.float64],
    done_d: NDArray[np.float64],
    n_done: int,
    period: NDArray[np.float64],
) -> tuple[float, float, float]:
    """
    Bound the k-th distance of a point from those of particles already solved.

    By the triangle inequality ``|d_k(q) - d_k(a)| <= |q - a|``, so
    ``max(d_k(a) - |q - a|) <= d_k(q) <= min(d_k(a) + |q - a|)``. The k-th
    distance of the nearest solved particle is also a guess at that of ``q``.

    Parameters
    ----------
    qx : float
        The point's x coordinate.
    qy : float
        The point's y coordinate.
    qz : float
        The point's z coordinate.
    done_x : NDArray[np.float64]
        The x coordinates of the solved particles.
    done_y : NDArray[np.float64]
        The y coordinates of the solved particles.
    done_z : NDArray[np.float64]
        The z coordinates of the solved particles.
    done_d : NDArray[np.float64]
        The k-th distances of the solved particles.
    n_done : int
        Number of solved particles.
    period : NDArray[np.float64]
        Period of each axis.

    Returns
    -------
    tuple[float, float, float]
        The lower bound, the upper bound (``inf`` if none is solved) and the
        guess.
    """
    r_lo = 0.0
    r_hi = np.inf
    guess = 0.0
    nearest = np.inf
    for a in range(n_done):
        dx = _separation(done_x[a] - qx, period, 0)
        dy = _separation(done_y[a] - qy, period, 1)
        dz = _separation(done_z[a] - qz, period, 2)
        sep = np.sqrt(dx * dx + dy * dy + dz * dz)
        r_lo = max(r_lo, done_d[a] - sep)
        r_hi = min(r_hi, done_d[a] + sep)
        if sep < nearest:
            nearest = sep
            guess = done_d[a]
    return r_lo, r_hi, guess


@njit(cache=True)
def _query_worker(
    w: int,
    n_workers: int,
    group: int,
    k: int,
    use_heap: bool,
    perm: NDArray[np.int64],
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    period: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    depth: int,
    out: NDArray[np.float64],
) -> None:
    """
    Find the k-th distances of the particles of one worker.

    Worker ``w`` takes the leaves in groups of ``group`` neighbouring leaves (a
    subtree): groups ``w``, ``w + n_workers``, ...

    Parameters
    ----------
    w : int
        The worker.
    n_workers : int
        Number of workers.
    group : int
        Number of neighbouring leaves queried one after another.
    k : int
        The neighbour to find (``2 <= k <= N``).
    use_heap : bool
        Whether to search with a heap (else by shells).
    perm : NDArray[np.int64]
        Particle indices in tree order.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    period : NDArray[np.float64]
        Period of each axis.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order.
    depth : int
        Depth of the tree.
    out : NDArray[np.float64]
        The k-th distances, written for this worker's particles.
    """
    first_leaf = (1 << depth) - 1
    n_leaves = 1 << depth
    stack = np.empty(2 * depth + 4, dtype=np.int64)
    buffer = np.empty(max(k, 1024), dtype=xs.dtype)
    stack_d2 = np.empty(2 * depth + 4, dtype=xs.dtype)
    d2_leaf = np.empty(LEAF_SIZE, dtype=xs.dtype)
    done_x = np.empty(LEAF_SIZE, dtype=xs.dtype)
    done_y = np.empty(LEAF_SIZE, dtype=xs.dtype)
    done_z = np.empty(LEAF_SIZE, dtype=xs.dtype)
    done_d = np.empty(LEAF_SIZE, dtype=xs.dtype)
    n_done = 0
    for leaf in range(first_leaf, first_leaf + n_leaves):
        if (leaf - first_leaf) // group % n_workers != w:
            continue
        if (leaf - first_leaf) % group == 0:
            # A new group: nothing solved nearby yet
            n_done = 0
        first = True
        for j in range(lo[leaf], hi[leaf]):
            qx = xs[j]
            qy = ys[j]
            qz = zs[j]
            # Bounds from the leaf's particles solved so far, or for its
            # first from the previous leaf's (found by counting if none)
            r_lo, r_hi, guess = _leaf_bounds(
                qx, qy, qz, done_x, done_y, done_z, done_d, n_done, period
            )
            if first:
                first = False
                n_done = 0
            if r_hi == 0.0:
                # A solved particle at the same place, with k there: so has q
                d2 = 0.0
            else:
                if r_hi < np.inf:
                    # Margins for the rounding of the distances
                    r_lo *= 1.0 - 1e-9
                    r_hi *= 1.0 + 1e-9
                else:
                    r_hi = 0.0
                if use_heap:
                    bound2 = r_hi * r_hi if r_hi > 0.0 else np.inf
                    args = (lo, hi, bmin, bmax, period, xs, ys, zs, first_leaf)
                    d2 = _heap_query(
                        qx, qy, qz, k, *args, bound2, buffer, stack, stack_d2, d2_leaf
                    )
                    if d2 >= bound2:
                        d2 = _heap_query(
                            qx,
                            qy,
                            qz,
                            k,
                            *args,
                            np.inf,
                            buffer,
                            stack,
                            stack_d2,
                            d2_leaf,
                        )
                else:
                    d2, buffer = _shell_query(
                        qx,
                        qy,
                        qz,
                        k,
                        leaf,
                        r_lo,
                        r_hi,
                        guess,
                        lo,
                        hi,
                        bmin,
                        bmax,
                        period,
                        xs,
                        ys,
                        zs,
                        first_leaf,
                        stack,
                        buffer,
                    )
            d = np.sqrt(d2)
            out[perm[j]] = d
            done_x[n_done] = qx
            done_y[n_done] = qy
            done_z[n_done] = qz
            done_d[n_done] = d
            n_done += 1


@njit(cache=True, parallel=True)
def _query_all(
    k: int,
    heap_k: int,
    n_workers: int,
    perm: NDArray[np.int64],
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    period: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    depth: int,
) -> NDArray[np.float64]:
    """
    Find the k-th distances of all the particles, in parallel.

    Parameters
    ----------
    k : int
        The neighbour to find.
    heap_k : int
        Largest ``k`` searched for with a heap.
    n_workers : int
        Number of workers.
    perm : NDArray[np.int64]
        Particle indices in tree order.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    period : NDArray[np.float64]
        Period of each axis.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order.
    depth : int
        Depth of the tree.

    Returns
    -------
    NDArray[np.float64]
        The k-th distance of each particle (``inf`` if there are fewer than
        ``k`` particles).
    """
    n = perm.size
    out = np.zeros(n)
    if k > n:
        out[:] = np.inf
        return out
    if k <= 1:
        return out
    use_heap = k <= heap_k
    # A leaf at a time, in groups of LEAF_GROUP neighbouring leaves (a subtree),
    # the groups interleaved between the threads (prange gives each worker one
    # contiguous chunk, which would leave the threads that get the densest
    # region the last to finish)
    group = min(LEAF_GROUP, 1 << depth)
    for w in prange(n_workers):
        _query_worker(
            w,
            n_workers,
            group,
            k,
            use_heap,
            perm,
            lo,
            hi,
            bmin,
            bmax,
            period,
            xs,
            ys,
            zs,
            depth,
            out,
        )
    return out


@njit(cache=True, fastmath=FASTMATH)
def _nearest_query(
    qx: float,
    qy: float,
    qz: float,
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    period: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    first_leaf: int,
    best2: float,
    best: int,
    stack: NDArray[np.int64],
    stack_d2: NDArray[np.float64],
    d2: NDArray[np.float64],
) -> int:
    """
    Find the particle nearest to a point.

    The nearer child of a node is searched first, and nodes no nearer than the
    nearest particle so far are skipped. The search starts from a particle
    already known (the nearest to the previous point), whose distance bounds
    that of the nearest.

    Parameters
    ----------
    qx : float
        The point's x coordinate.
    qy : float
        The point's y coordinate.
    qz : float
        The point's z coordinate.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    period : NDArray[np.float64]
        Period of each axis.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order.
    first_leaf : int
        Index of the first leaf node.
    best2 : float
        Squared distance to the particle ``best`` (``inf`` if none).
    best : int
        A particle (in tree order) to start from (-1 if none).
    stack : NDArray[np.int64]
        Workspace for the nodes to visit.
    stack_d2 : NDArray[np.float64]
        Workspace for the squared distances to the nodes to visit.
    d2 : NDArray[np.float64]
        Workspace of ``LEAF_SIZE`` elements.

    Returns
    -------
    int
        The nearest particle, in tree order.
    """
    stack[0] = 0
    stack_d2[0] = 0.0
    top = 1
    while top > 0:
        top -= 1
        node = stack[top]
        if stack_d2[top] >= best2:
            continue
        if node >= first_leaf:
            start = lo[node]
            n = hi[node] - start
            # All the leaf's squared distances first, then their minimum: two
            # simple loops, which the compiler can run on several particles at once
            nearest = np.inf
            for j in range(n):
                dx = _separation(xs[start + j] - qx, period, 0)
                dy = _separation(ys[start + j] - qy, period, 1)
                dz = _separation(zs[start + j] - qz, period, 2)
                d2[j] = dx * dx + dy * dy + dz * dz
                nearest = min(nearest, d2[j])
            if nearest < best2:
                for j in range(n):
                    if d2[j] == nearest:
                        best = start + j
                        break
                best2 = nearest
            continue
        left = 2 * node + 1
        right = left + 1
        d_left = _box_min_distance2(bmin, bmax, left, qx, qy, qz, period)
        d_right = _box_min_distance2(bmin, bmax, right, qx, qy, qz, period)
        # Nearer child on top, so it is searched first
        if d_left <= d_right:
            near, d_near, far, d_far = left, d_left, right, d_right
        else:
            near, d_near, far, d_far = right, d_right, left, d_left
        if d_far < best2:
            stack[top] = far
            stack_d2[top] = d_far
            top += 1
        if d_near < best2:
            stack[top] = near
            stack_d2[top] = d_near
            top += 1
    return best


@njit(cache=True, parallel=True)
def _nearest_all(
    query_x: NDArray[np.float64],
    query_y: NDArray[np.float64],
    query_z: NDArray[np.float64],
    n_workers: int,
    perm: NDArray[np.int64],
    lo: NDArray[np.int64],
    hi: NDArray[np.int64],
    bmin: NDArray[np.float64],
    bmax: NDArray[np.float64],
    period: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    depth: int,
) -> NDArray[np.int64]:
    """
    Find the particle nearest to each query point, in parallel.

    The points go in chunks of ``QUERY_CHUNK`` consecutive points interleaved
    between the threads; each point starts its search from the nearest particle
    to the one before it in its chunk.

    Parameters
    ----------
    query_x : NDArray[np.float64]
        The query points' x coordinates.
    query_y : NDArray[np.float64]
        The query points' y coordinates.
    query_z : NDArray[np.float64]
        The query points' z coordinates.
    n_workers : int
        Number of workers.
    perm : NDArray[np.int64]
        Particle indices in tree order.
    lo : NDArray[np.int64]
        First particle of each node.
    hi : NDArray[np.int64]
        One past the last particle of each node.
    bmin : NDArray[np.float64]
        Lower corners of the nodes' boxes.
    bmax : NDArray[np.float64]
        Upper corners of the nodes' boxes.
    period : NDArray[np.float64]
        Period of each axis.
    xs : NDArray[np.float64]
        Particle x coordinates in tree order.
    ys : NDArray[np.float64]
        Particle y coordinates in tree order.
    zs : NDArray[np.float64]
        Particle z coordinates in tree order.
    depth : int
        Depth of the tree.

    Returns
    -------
    NDArray[np.int64]
        The index of the nearest particle to each query point.
    """
    m = query_x.size
    out = np.empty(m, dtype=np.int64)
    first_leaf = (1 << depth) - 1
    n_chunks = (m + QUERY_CHUNK - 1) // QUERY_CHUNK
    for w in prange(n_workers):
        stack = np.empty(2 * depth + 4, dtype=np.int64)
        stack_d2 = np.empty(2 * depth + 4, dtype=xs.dtype)
        d2 = np.empty(LEAF_SIZE, dtype=xs.dtype)
        for chunk in range(w, n_chunks, n_workers):
            best = -1
            for i in range(chunk * QUERY_CHUNK, min(m, (chunk + 1) * QUERY_CHUNK)):
                qx = query_x[i]
                qy = query_y[i]
                qz = query_z[i]
                best2 = np.inf
                if best >= 0:
                    dx = _separation(xs[best] - qx, period, 0)
                    dy = _separation(ys[best] - qy, period, 1)
                    dz = _separation(zs[best] - qz, period, 2)
                    best2 = dx * dx + dy * dy + dz * dz
                best = _nearest_query(
                    qx,
                    qy,
                    qz,
                    lo,
                    hi,
                    bmin,
                    bmax,
                    period,
                    xs,
                    ys,
                    zs,
                    first_leaf,
                    best2,
                    best,
                    stack,
                    stack_d2,
                    d2,
                )
                out[i] = perm[best]
    return out


def _periodic_cut(x: NDArray[np.float64], period: float, bins: int = 4096) -> float:
    """
    Choose where to put the boundary of a periodic axis.

    Periodic distances do not depend on where the boundary is, but the tree
    does: a clump of particles across the boundary would be cut into pieces at
    opposite faces, and nodes holding both would have bounding boxes as wide as
    the box, which prune nothing. The boundary is put in the middle of the
    widest run of empty bins of a histogram of ``x`` (0 if none is empty).

    Parameters
    ----------
    x : NDArray[np.float64]
        Coordinates along the axis.
    period : float
        Period of the axis.
    bins : int, optional
        Number of bins of the histogram. Default is ``4096``.

    Returns
    -------
    float
        The coordinate of the boundary.
    """
    which = np.minimum((x % period) * (bins / period), bins - 1).astype(np.int64)
    empty = np.bincount(which, minlength=bins) == 0
    if not empty.any():
        return 0.0
    # Runs of empty bins, from a filled bin, so none wraps around the end
    start = int(np.argmin(empty))
    rolled = np.roll(empty, -start)
    edges = np.flatnonzero(np.diff(np.r_[0, rolled.astype(np.int8), 0]))
    run_starts, run_ends = edges[::2], edges[1::2]
    widest = int(np.argmax(run_ends - run_starts))
    middle = 0.5 * (run_starts[widest] + run_ends[widest]) + start
    return (middle % bins) * (period / bins)


@njit(cache=True, parallel=True)
def _shift_axis(
    x: NDArray[np.float64], offset: float, period: float, out: NDArray[np.float64]
) -> None:
    """
    Shift coordinates along an axis, and wrap them into ``[0, period)``.

    Parameters
    ----------
    x : NDArray[np.float64]
        The coordinates.
    offset : float
        Subtracted from the coordinates.
    period : float
        Period of the axis (0 for no wrapping).
    out : NDArray[np.float64]
        The shifted coordinates.
    """
    for i in prange(x.size):
        v = x[i] - offset
        if period > 0.0:
            v %= period
            # v % L rounds up to L for tiny negative v
            if v >= period:
                v = 0.0
        out[i] = v


def _shift_into_frame(
    positions: NDArray[np.float64],
    boxsize: float | NDArray[np.floating] | None,
    queries: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """
    Shift the particles (and query points) into the frame the tree is built in.

    On periodic axes the coordinates are wrapped into ``[0, period)``, which the
    periodic separation ``min(|dx|, period - |dx|)`` needs, with the boundary moved into the widest gap
    between particles (distances do not change); on free axes they are centred,
    for the precision of the distances. A periodic axis along which all the
    points are within half a period of each other becomes free, as no distance
    along it is shorter through the wrap (the case for a small region of a large
    box).

    Parameters
    ----------
    positions : NDArray[np.float64]
        Particle positions, shape ``(N, 3)`` with ``N > 0``.
    boxsize : float or NDArray[np.floating] or None
        Period of each axis (``None``, or a period ``<= 0``, for a free axis).
    queries : NDArray[np.float64], optional
        Query points, shape ``(M, 3)``, shifted with the particles. Default is
        ``None``.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]
        The shifted positions, shape ``(N, 3)``; the period of each axis
        (``FREE_AXIS`` if free); and the shifted query points, one row per axis,
        shape ``(3, M)`` (``None`` if none were given).
    """
    period = np.full(3, FREE_AXIS)
    if boxsize is not None:
        box = np.broadcast_to(np.asarray(boxsize, dtype=np.float64), (3,))
        period[box > 0] = box[box > 0]
    shifted = np.empty_like(positions)
    shifted_queries = None if queries is None else np.empty((3, queries.shape[0]))
    for d in range(3):
        x = np.ascontiguousarray(positions[:, d])
        column = np.empty_like(x)
        if period[d] < FREE_AXIS:
            offset = _periodic_cut(x, period[d])
            _shift_axis(x, offset, period[d], column)
            low, high = column.min(), column.max()
        else:
            offset = 0.5 * (x.min() + x.max())
            _shift_axis(x, offset, 0.0, column)
        shifted[:, d] = column
        if queries is not None:
            q = np.ascontiguousarray(queries[:, d])
            wrap = period[d] if period[d] < FREE_AXIS else 0.0
            _shift_axis(q, offset, wrap, shifted_queries[d])
            if period[d] < FREE_AXIS:
                # A query outside the particles' range can be nearer through
                # the wrap, so all the points must be within half a period
                low = min(low, shifted_queries[d].min())
                high = max(high, shifted_queries[d].max())
        if period[d] < FREE_AXIS and high - low <= 0.5 * period[d]:
            period[d] = FREE_AXIS
    return shifted, period, shifted_queries


def k_nearest_distances(
    positions: NDArray[np.floating],
    k: int,
    boxsize: float | NDArray[np.floating] | None = None,
    heap_k: int = HEAP_K,
) -> NDArray[np.float64]:
    """
    Compute the distance from every particle to its k-th nearest neighbour.

    The particle itself is included (so ``k=1`` gives 0), as in
    ``scipy.spatial.cKDTree(positions, boxsize=boxsize).query(positions, k=[k])``.
    Runs on ``numba.get_num_threads()`` threads.

    Parameters
    ----------
    positions : NDArray[np.floating]
        Particle positions, shape ``(N, 3)``.
    k : int
        The neighbour to find.
    boxsize : float or NDArray[np.floating] or None, optional
        Periodic box: distances are minimum-image distances along every axis
        with a positive period (the coordinates are wrapped into it). ``None``,
        or a period ``<= 0``, leaves that axis non-periodic. Default is ``None``.
    heap_k : int, optional
        Largest ``k`` searched for with a heap (larger ``k`` are selected from
        shells); for testing. Default is ``HEAP_K`` (96).

    Returns
    -------
    NDArray[np.float64]
        The distances, shape ``(N,)`` (``inf`` where there are fewer than ``k``
        particles).
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = positions.shape[0]
    if n == 0:
        return np.zeros(0)
    shifted, period, _ = _shift_into_frame(positions, boxsize)
    depth = _tree_depth(n, LEAF_SIZE)
    tree = _build(shifted, depth)
    return _query_all(
        int(k), int(heap_k), get_num_threads(), *tree[:5], period, *tree[5:], depth
    )


def nearest_particles(
    positions: NDArray[np.floating],
    queries: NDArray[np.floating],
    boxsize: float | NDArray[np.floating] | None = None,
) -> NDArray[np.int64]:
    """
    Find the particle nearest to each of a set of points.

    The same as ``scipy.spatial.cKDTree(positions, boxsize=boxsize).query(queries)``
    (up to the choice between particles at equal distances). Consecutive query
    points should be near each other (as the pixels of an image), as each search
    starts from the particle nearest to the point before. Runs on
    ``numba.get_num_threads()`` threads.

    Parameters
    ----------
    positions : NDArray[np.floating]
        Particle positions, shape ``(N, 3)`` with ``N > 0``.
    queries : NDArray[np.floating]
        Query points, shape ``(M, 3)``.
    boxsize : float or NDArray[np.floating] or None, optional
        Periodic box: distances are minimum-image distances along every axis
        with a positive period (the coordinates are wrapped into it). ``None``,
        or a period ``<= 0``, leaves that axis non-periodic. Default is ``None``.

    Returns
    -------
    NDArray[np.int64]
        The index of the nearest particle to each query point, shape ``(M,)``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64).reshape(-1, 3)
    if positions.shape[0] == 0:
        raise ValueError("Cannot find the nearest of no particles.")
    shifted, period, shifted_queries = _shift_into_frame(positions, boxsize, queries)
    depth = _tree_depth(positions.shape[0], LEAF_SIZE)
    tree = _build(shifted, depth)
    return _nearest_all(
        shifted_queries[0],
        shifted_queries[1],
        shifted_queries[2],
        get_num_threads(),
        *tree[:5],
        period,
        *tree[5:],
        depth,
    )
