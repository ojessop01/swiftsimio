"""
Nested multi-resolution volume render backend for SPH data.

Particles whose smoothing kernels span many voxels at the target resolution
are scattered onto successively coarser sub-grids and then trilinearly
upsampled back to the target resolution.  This bounds the per-particle work
regardless of smoothing length, giving large speedups for simulations with
wide kernel-size distributions (e.g. zoom-in runs or IGM particles).

Notes
-----
Algorithm:

1. For each particle compute how many finest-grid cells its compact-support
   kernel spans::

       support_cells = kernel_gamma * h * res

2. Assign particle to level ``L`` where::

       L = clip(ceil(log2(support_cells / ntarget)), 0, nlevels)

   Level 0 is the finest grid (``res``); level ``L`` uses a grid of size
   ``res // 2**L``.  At their assigned level every kernel spans roughly
   ``ntarget`` cells, bounding the number of voxels visited per particle.

3. Scatter each level's particles onto their coarse grid.

4. Collapse the hierarchy coarse-to-fine: trilinearly upsample each coarse
   grid and accumulate it into the next finer grid.

5. Return the finest grid.

The algorithm follows a novel Sparse Multi-Scale Grid algorithm
(as described in Benitez-Llambay 2025) to place particles on a grid using
an Adaptive Mesh Refinement (AMR) approach.

Resolution constraints: ``res`` must be even and divisible by ``2**nlevels``
so that each level has an integer grid size.  With the default ``nlevels=4``
this requires ``res`` to be a multiple of 16.
"""

from math import ceil, floor, sqrt

import numpy as np
from numpy import float32, float64, int32, zeros, int64

from numba import get_num_threads, njit, prange

from swiftsimio.visualisation.slice_backends.sph import kernel, kernel_gamma, kernel_constant

@njit(fastmath=True, cache=True, nogil=True, boundscheck=False, error_model="numpy")
def _assign_serial_levels(
    h: np.ndarray,
    m: np.ndarray,
    res: int,
    ntarget: int,
    nlevels: int,
) -> tuple:
    """
    Assign a hierarchy level to every particle in a single pass.

    Replaces ``ceil(log2(support_cells / ntarget))`` with a cheap doubling
    loop, avoiding transcendental calls in the hot path.  Particles with
    zero mass or negative smoothing length are marked -1 and skipped.

    Parameters
    ----------
    h : np.ndarray[float32]
        Smoothing lengths, in [0, 1] units.
    m : np.ndarray[float32]
        Particle masses or weights.
    res : int
        Finest-grid resolution along one axis.
    ntarget : int
        Target cells per kernel side at the assigned level.
    nlevels : int
        Maximum hierarchy depth.

    Returns
    -------
    level_index : np.ndarray[int8]
        Per-particle level assignment; -1 for inactive particles.
    deepest : int32
        Highest level actually occupied (bounds hierarchy allocation).
    """
    level_index = np.empty(h.size, dtype=np.int8)
    support_scale = float32(kernel_gamma) * float32(res)
    target = float32(ntarget)
    deepest = int32(0)

    for particle in range(h.size):
        if m[particle] != float32(0.0) and h[particle] >= float32(0.0):
            # Double the threshold at each step rather than computing log2.
            support_cells = h[particle] * support_scale
            threshold = target
            level = int32(0)
            while support_cells > threshold and level < nlevels:
                level += int32(1)
                threshold *= float32(2.0)
            level_index[particle] = level
            if level > deepest:
                deepest = level
        else:
            # Skip zero-mass and unphysical particles entirely.
            level_index[particle] = np.int8(-1)

    return level_index, deepest


@njit(
    fastmath=True,
    cache=True,
    nogil=True,
    boundscheck=False,
    error_model="numpy",
    inline="always",
)
def _deposit_particle_flat(
    destination: np.ndarray,
    level_offset: int,
    level_res: int,
    level_plane: int,
    x_pos: np.float64,
    y_pos: np.float64,
    z_pos: np.float64,
    mass_icv: np.float32,
    weighted_prefactor: np.float32,
    radius_cells_64: np.float64,
    level: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> None:
    """
    Deposit one particle onto the flat destination array at the given level.

    The destination is a flat (1-D) view shared across all levels; ``level_offset``
    locates the start of this level's sub-array.  Bounds arrays track the
    bounding box of occupied coarse cells so that the collapse step only
    visits cells that were actually written.

    All particle-invariant quantities (``weighted_prefactor``, ``radius_cells_64``,
    ``mass_icv``) are pre-computed by the caller before the periodic-image loop.
    Kernel distances are evaluated in cell-space coordinates so that the inner-loop
    increment is exactly ``1.0`` and the z-loop runs without a branch.

    Parameters
    ----------
    destination : np.ndarray[float32]
        Flat output array (finest or coarse, depending on level).
    level_offset : int64
        Index of the first cell belonging to this level in ``destination``.
    level_res : int32
        Number of cells along one axis at this level.
    level_plane : int32
        ``level_res**2`` — x-plane stride in the flat array.
    x_pos : float64
        Particle x coordinate in [0, 1].
    y_pos : float64
        Particle y coordinate in [0, 1].
    z_pos : float64
        Particle z coordinate in [0, 1].
    mass_icv : float32
        ``mass * level_res**3`` — used by the sub-pixel deposit path.
    weighted_prefactor : float32
        ``mass * (21/2π) * H**-3`` — kernel amplitude folded into one scalar.
    radius_cells_64 : float64
        ``kernel_gamma * hsml * level_res`` — kernel radius in cell units, kept
        in float64 for accurate bounding-box derivation.
    level : int32
        Hierarchy level of this particle.
    bounds_min : np.ndarray[int32, shape=(nlevels+1, 3)]
        Per-level bounding-box minimum cell indices (updated in-place).
    bounds_max : np.ndarray[int32, shape=(nlevels+1, 3)]
        Per-level bounding-box maximum cell indices (updated in-place).
    """
    # Positions scaled to cell-index space; float64 for accurate index derivation.
    scaled_x_64 = float64(level_res) * x_pos
    scaled_y_64 = float64(level_res) * y_pos
    scaled_z_64 = float64(level_res) * z_pos

    radius_cells_f32 = float32(radius_cells_64)
    maximal_index = level_res - 1

    # Sub-pixel fast path: kernel spans less than one cell half-width.
    # Deposit the full particle weight into the containing cell to preserve mass.
    if radius_cells_f32 < float32(0.5):
        px = int32(scaled_x_64)
        py = int32(scaled_y_64)
        pz = int32(scaled_z_64)
        if 0 <= px <= maximal_index and 0 <= py <= maximal_index and 0 <= pz <= maximal_index:
            flat_cell = level_offset + (px * level_res + py) * level_res + pz
            destination[flat_cell] += mass_icv
            if level != 0:
                if px < bounds_min[level, 0]:
                    bounds_min[level, 0] = px
                if py < bounds_min[level, 1]:
                    bounds_min[level, 1] = py
                if pz < bounds_min[level, 2]:
                    bounds_min[level, 2] = pz
                if px > bounds_max[level, 0]:
                    bounds_max[level, 0] = px
                if py > bounds_max[level, 1]:
                    bounds_max[level, 1] = py
                if pz > bounds_max[level, 2]:
                    bounds_max[level, 2] = pz
        return

    # Exact kernel bounding box in cell space.  Cell i has its centre at
    # i + 0.5; we include cells where that centre falls strictly inside the
    # kernel compact support.  Float64 avoids the asymmetric rounding produced
    # by the previous (particle_cell ± cells_spanned) approach.
    x_start = int32(floor(scaled_x_64 - radius_cells_64 - 0.5)) + 1
    x_stop  = int32(ceil(scaled_x_64  + radius_cells_64 - 0.5))
    if x_start < 0:
        x_start = 0
    if x_stop > level_res:
        x_stop = level_res

    y_start = int32(floor(scaled_y_64 - radius_cells_64 - 0.5)) + 1
    y_stop  = int32(ceil(scaled_y_64  + radius_cells_64 - 0.5))
    if y_start < 0:
        y_start = 0
    if y_stop > level_res:
        y_stop = level_res

    z_start = int32(floor(scaled_z_64 - radius_cells_64 - 0.5)) + 1
    z_stop  = int32(ceil(scaled_z_64  + radius_cells_64 - 0.5))
    if z_start < 0:
        z_start = 0
    if z_stop > level_res:
        z_stop = level_res

    if x_start >= x_stop or y_start >= y_stop or z_start >= z_stop:
        return

    if level != 0:
        if x_start < bounds_min[level, 0]:
            bounds_min[level, 0] = x_start
        if y_start < bounds_min[level, 1]:
            bounds_min[level, 1] = y_start
        if z_start < bounds_min[level, 2]:
            bounds_min[level, 2] = z_start
        if x_stop - 1 > bounds_max[level, 0]:
            bounds_max[level, 0] = x_stop - 1
        if y_stop - 1 > bounds_max[level, 1]:
            bounds_max[level, 1] = y_stop - 1
        if z_stop - 1 > bounds_max[level, 2]:
            bounds_max[level, 2] = z_stop - 1

    # float32 particle positions for kernel arithmetic; inner-loop distances
    # accumulate by exactly 1.0 rather than pixel_width, removing that multiply.
    scaled_x_f32 = float32(scaled_x_64)
    scaled_y_f32 = float32(scaled_y_64)
    scaled_z_f32 = float32(scaled_z_64)
    radius_cells_2_64 = radius_cells_64 * radius_cells_64
    inverse_radius_cells = float32(1.0) / radius_cells_f32

    # Single-cell fast path: evaluate the kernel once, no loop overhead.
    if x_stop - x_start == 1 and y_stop - y_start == 1 and z_stop - z_start == 1:
        dx = float32(x_start) + float32(0.5) - scaled_x_f32
        dy = float32(y_start) + float32(0.5) - scaled_y_f32
        dz = float32(z_start) + float32(0.5) - scaled_z_f32
        radius_2 = dx * dx + dy * dy + dz * dz
        ratio = sqrt(radius_2) * inverse_radius_cells
        one_minus_ratio = float32(1.0) - ratio
        one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
        flat_cell = level_offset + (x_start * level_res + y_start) * level_res + z_start
        destination[flat_cell] += (
            weighted_prefactor
            * one_minus_ratio_2
            * one_minus_ratio_2
            * (float32(1.0) + float32(4.0) * ratio)
        )
        return

    # General kernel loop in cell-space coordinates.
    # Float64 is used only for per-strip analytic bound derivation; the kernel
    # arithmetic stays in float32.  The innermost z-loop is branch-free because
    # tight_z bounds already exclude cells outside the kernel support.
    # Wendland-C2: W(r,H) = (21/2π) H⁻³ (1 - r/H)⁴ (1 + 4r/H)
    dx_64 = float64(x_start) + 0.5 - scaled_x_64
    dx = float32(dx_64)
    for cell_x in range(x_start, x_stop):
        dx_2 = dx * dx

        # Tight y-strip: analytic bound given the current x-distance.
        rem_x_sq = radius_cells_2_64 - dx_64 * dx_64
        if rem_x_sq < 0.0:
            rem_x_sq = 0.0
        rem_x_64 = sqrt(rem_x_sq)
        tight_y_start = int32(floor(scaled_y_64 - rem_x_64 - 0.5)) + 1
        tight_y_stop  = int32(ceil(scaled_y_64  + rem_x_64 - 0.5))
        if tight_y_start < y_start:
            tight_y_start = y_start
        if tight_y_stop > y_stop:
            tight_y_stop = y_stop

        if tight_y_start < tight_y_stop:
            x_base = level_offset + cell_x * level_plane
            dy_64 = float64(tight_y_start) + 0.5 - scaled_y_64
            dy = float32(dy_64)

            for cell_y in range(tight_y_start, tight_y_stop):
                dx_dy_2 = dx_2 + dy * dy

                # Tight z-pencil: analytic bound given current x- and y-distances.
                rem_xy_sq = radius_cells_2_64 - dx_64 * dx_64 - dy_64 * dy_64
                if rem_xy_sq < 0.0:
                    rem_xy_sq = 0.0
                rem_xy_64 = sqrt(rem_xy_sq)
                tight_z_start = int32(floor(scaled_z_64 - rem_xy_64 - 0.5)) + 1
                tight_z_stop  = int32(ceil(scaled_z_64  + rem_xy_64 - 0.5))
                if tight_z_start < z_start:
                    tight_z_start = z_start
                if tight_z_stop > z_stop:
                    tight_z_stop = z_stop

                if tight_z_start < tight_z_stop:
                    flat_cell = x_base + cell_y * level_res + tight_z_start
                    dz = float32(tight_z_start) + float32(0.5) - scaled_z_f32

                    for cell_z in range(tight_z_start, tight_z_stop):
                        radius_2 = dx_dy_2 + dz * dz
                        ratio = sqrt(radius_2) * inverse_radius_cells
                        one_minus_ratio = float32(1.0) - ratio
                        one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
                        destination[flat_cell] += (
                            weighted_prefactor
                            * one_minus_ratio_2
                            * one_minus_ratio_2
                            * (float32(1.0) + float32(4.0) * ratio)
                        )
                        flat_cell += 1
                        dz += float32(1.0)
                dy_64 += 1.0
                dy += float32(1.0)
        dx_64 += 1.0
        dx += float32(1.0)

@njit(fastmath=True, cache=True, nogil=True, boundscheck=False, error_model="numpy")
def _scatter_particles(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    m: np.ndarray,
    h: np.ndarray,
    level_index: np.ndarray,
    level_offsets: np.ndarray,
    level_resolutions: np.ndarray,
    finest: np.ndarray,
    coarse: np.ndarray,
    box_x: float64,
    box_y: float64,
    box_z: float64,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> None:
    """
    Scatter all particles onto their assigned hierarchy levels.

    Iterates once over all particles, dispatching each to either the finest
    grid (level 0) or the shared coarse flat-array (levels 1+).

    Periodic wrapping is handled by depositing each particle once per image
    whose kernel support overlaps [0, 1].  When a box dimension is 0.0 the
    shift range collapses to a single image, so both periodic and
    non-periodic cases are handled by the same loop.  Per-particle image
    ranges are computed analytically so interior particles (whose kernels do
    not reach any periodic boundary) make only one call to
    ``_deposit_particle_flat``, saving the overhead of 26 no-op calls.

    Parameters
    ----------
    x, y, z : np.ndarray[float64]
        Particle positions in [0, 1].
    m : np.ndarray[float32]
        Particle masses or weights.
    h : np.ndarray[float32]
        Particle smoothing lengths.
    level_index : np.ndarray[int8]
        Pre-assigned hierarchy level per particle; -1 = skip.
    level_offsets : np.ndarray[int64]
        Flat-array start index of each level.
    level_resolutions : np.ndarray[int32]
        Cell count per axis at each level.
    finest : np.ndarray[float32]
        Flat level-0 accumulator of size ``res**3``.
    coarse : np.ndarray[float32]
        Flat accumulator for levels 1+ (total coarse cells).
    box_x : float64
        Box size in x; 0.0 means no wrapping in x.
    box_y : float64
        Box size in y; 0.0 means no wrapping in y.
    box_z : float64
        Box size in z; 0.0 means no wrapping in z.
    bounds_min : np.ndarray[int32, shape=(nlevels+1, 3)]
        Per-level bounding-box minimum (updated in-place).
    bounds_max : np.ndarray[int32, shape=(nlevels+1, 3)]
        Per-level bounding-box maximum (updated in-place).
    """
    finest_cells = finest.size

    # Global shift bounds: the widest range any particle could need.
    # Per-particle ranges are derived analytically below and clipped to these.
    xshift_min = int32(0) if box_x == 0.0 else int32(-1)
    yshift_min = int32(0) if box_y == 0.0 else int32(-1)
    zshift_min = int32(0) if box_z == 0.0 else int32(-1)
    xshift_max = int32(1) if box_x == 0.0 else int32(ceil(1.0 / box_x) + 1)
    yshift_max = int32(1) if box_y == 0.0 else int32(ceil(1.0 / box_y) + 1)
    zshift_max = int32(1) if box_z == 0.0 else int32(ceil(1.0 / box_z) + 1)

    for particle in range(h.size):
        level = int32(level_index[particle])
        if level < 0:
            continue

        mass = m[particle]
        hsml = h[particle]
        if level == 0:
            destination = finest
            level_offset = int64(0)
        else:
            destination = coarse
            level_offset = level_offsets[level] - finest_cells

        level_res = level_resolutions[level]
        level_plane_p = level_res * level_res
        original_x = x[particle]
        original_y = y[particle]
        original_z = z[particle]

        # Hoist all particle-invariant constants before the periodic image loop.
        kernel_width = float32(kernel_gamma) * hsml
        inverse_kernel_width = float32(1.0) / kernel_width
        weighted_prefactor = (
            mass
            * float32(kernel_constant)
            * inverse_kernel_width
            * inverse_kernel_width
            * inverse_kernel_width
        )
        float_res = float32(level_res)
        mass_icv = mass * float_res * float_res * float_res
        kernel_width_64 = float64(kernel_width)
        radius_cells_64 = kernel_width_64 * float64(level_res)

        # Per-particle image range: only shifts whose kernel overlaps [0, 1].
        # For an interior particle this collapses to a single image (shift = 0).
        if box_x != 0.0:
            local_x_min = int32(ceil((-kernel_width_64 - original_x) / box_x))
            local_x_max = int32(floor((1.0 + kernel_width_64 - original_x) / box_x)) + 1
            if local_x_min < xshift_min:
                local_x_min = xshift_min
            if local_x_max > xshift_max:
                local_x_max = xshift_max
        else:
            local_x_min = int32(0)
            local_x_max = int32(1)

        if box_y != 0.0:
            local_y_min = int32(ceil((-kernel_width_64 - original_y) / box_y))
            local_y_max = int32(floor((1.0 + kernel_width_64 - original_y) / box_y)) + 1
            if local_y_min < yshift_min:
                local_y_min = yshift_min
            if local_y_max > yshift_max:
                local_y_max = yshift_max
        else:
            local_y_min = int32(0)
            local_y_max = int32(1)

        if box_z != 0.0:
            local_z_min = int32(ceil((-kernel_width_64 - original_z) / box_z))
            local_z_max = int32(floor((1.0 + kernel_width_64 - original_z) / box_z)) + 1
            if local_z_min < zshift_min:
                local_z_min = zshift_min
            if local_z_max > zshift_max:
                local_z_max = zshift_max
        else:
            local_z_min = int32(0)
            local_z_max = int32(1)

        for xshift in range(local_x_min, local_x_max):
            x_pos = original_x + xshift * box_x
            for yshift in range(local_y_min, local_y_max):
                y_pos = original_y + yshift * box_y
                for zshift in range(local_z_min, local_z_max):
                    _deposit_particle_flat(
                        destination,
                        level_offset,
                        level_res,
                        level_plane_p,
                        x_pos,
                        y_pos,
                        original_z + zshift * box_z,
                        mass_icv,
                        weighted_prefactor,
                        radius_cells_64,
                        level,
                        bounds_min,
                        bounds_max,
                    )


@njit(fastmath=True, cache=True, nogil=True, boundscheck=False, error_model="numpy")
def _collapse_serial_flat(
    finest: np.ndarray,
    coarse: np.ndarray,
    level_offsets: np.ndarray,
    level_resolutions: np.ndarray,
    nlevels: int32,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> None:
    """
    Trilinearly upsample each coarse level and accumulate into the next finer one.

    Works coarse-to-fine (highest level first) so each upsampled contribution
    is ready before it is needed by the next collapse step.  Only the
    bounding box of occupied coarse cells is visited, keeping the work
    proportional to the number of particles rather than the grid volume.

    The trilinear weights follow the stencil used by scatter.py:
    an even fine cell at index ``2i`` draws 3/4 from coarse cell ``i`` and
    1/4 from coarse cell ``i-1``; an odd fine cell at ``2i+1`` draws 3/4
    from ``i`` and 1/4 from ``i+1``.

    Parameters
    ----------
    finest : np.ndarray[float32]
        Flat level-0 accumulator (modified in-place).
    coarse : np.ndarray[float32]
        Flat accumulator for levels 1+ (read and modified in-place).
    level_offsets : np.ndarray[int64]
        Start index of each level in the flat arrays.
    level_resolutions : np.ndarray[int32]
        Cell count per axis at each level.
    nlevels : int
        Deepest occupied level (collapse stops here).
    bounds_min : np.ndarray[int32, shape=(nlevels+1, 3)]
        Per-level bounding-box minimum populated during scatter.
    bounds_max : np.ndarray[int32, shape=(nlevels+1, 3)]
        Per-level bounding-box maximum populated during scatter.
    """
    finest_cells = finest.size

    for level in range(nlevels, 0, -1):
        # Skip levels that received no particles (bounds_max stays at -1).
        if bounds_max[level, 0] < 0:
            continue

        coarse_res = level_resolutions[level]
        fine_res = level_resolutions[level - 1]
        coarse_max = coarse_res - 1
        coarse_plane = coarse_res * coarse_res
        fine_plane = fine_res * fine_res
        source_offset = level_offsets[level] - finest_cells

        if level == 1:
            destination = finest
            destination_offset = 0
        else:
            destination = coarse
            destination_offset = level_offsets[level - 1] - finest_cells

        # One occupied coarse cell can influence fine indices 2*i-1 through
        # 2*i+2. Restricting to this box is exact because all cells outside
        # the tracked source box are known to be zero.
        fine_x_start = (bounds_min[level, 0] << 1) - 1
        fine_y_start = (bounds_min[level, 1] << 1) - 1
        fine_z_start = (bounds_min[level, 2] << 1) - 1
        fine_x_stop = (bounds_max[level, 0] << 1) + 3
        fine_y_stop = (bounds_max[level, 1] << 1) + 3
        fine_z_stop = (bounds_max[level, 2] << 1) + 3
        if fine_x_start < 0:
            fine_x_start = 0
        if fine_y_start < 0:
            fine_y_start = 0
        if fine_z_start < 0:
            fine_z_start = 0
        if fine_x_stop > fine_res:
            fine_x_stop = fine_res
        if fine_y_stop > fine_res:
            fine_y_stop = fine_res
        if fine_z_stop > fine_res:
            fine_z_stop = fine_res

        pair_z_start = fine_z_start >> 1
        pair_z_stop = (fine_z_stop + 1) >> 1

        for fine_x in range(fine_x_start, fine_x_stop):
            coarse_x0 = (fine_x - 1) >> 1
            coarse_x1 = coarse_x0 + 1
            weight_x1 = float32(0.75) if (fine_x & 1) == 0 else float32(0.25)
            weight_x0 = float32(1.0) - weight_x1
            if coarse_x0 < 0:
                coarse_x0 = 0
            if coarse_x1 > coarse_max:
                coarse_x1 = coarse_max

            x0_base = source_offset + coarse_x0 * coarse_plane
            x1_base = source_offset + coarse_x1 * coarse_plane
            fine_x_base = destination_offset + fine_x * fine_plane

            for fine_y in range(fine_y_start, fine_y_stop):
                coarse_y0 = (fine_y - 1) >> 1
                coarse_y1 = coarse_y0 + 1
                weight_y1 = float32(0.75) if (fine_y & 1) == 0 else float32(0.25)
                weight_y0 = float32(1.0) - weight_y1
                if coarse_y0 < 0:
                    coarse_y0 = 0
                if coarse_y1 > coarse_max:
                    coarse_y1 = coarse_max

                base_00 = x0_base + coarse_y0 * coarse_res
                base_01 = x0_base + coarse_y1 * coarse_res
                base_10 = x1_base + coarse_y0 * coarse_res
                base_11 = x1_base + coarse_y1 * coarse_res
                output_base = fine_x_base + fine_y * fine_res

                # Two adjacent fine z cells share the middle coarse sample.
                # Bilinear xy interpolation is performed once per z-plane (left,
                # centre, right), then the two fine z outputs are derived with two
                # FMA-friendly operations.  This reduces multiplications vs the
                # previous corner-by-corner approach.
                for coarse_z in range(pair_z_start, pair_z_stop):
                    left_z = coarse_z - 1
                    if left_z < 0:
                        left_z = 0
                    right_z = coarse_z + 1
                    if right_z > coarse_max:
                        right_z = coarse_max

                    xy_l_x0 = weight_y0 * coarse[base_00 + left_z]  + weight_y1 * coarse[base_01 + left_z]
                    xy_l_x1 = weight_y0 * coarse[base_10 + left_z]  + weight_y1 * coarse[base_11 + left_z]
                    xy_l    = weight_x0 * xy_l_x0 + weight_x1 * xy_l_x1

                    xy_c_x0 = weight_y0 * coarse[base_00 + coarse_z] + weight_y1 * coarse[base_01 + coarse_z]
                    xy_c_x1 = weight_y0 * coarse[base_10 + coarse_z] + weight_y1 * coarse[base_11 + coarse_z]
                    xy_c    = weight_x0 * xy_c_x0 + weight_x1 * xy_c_x1

                    xy_r_x0 = weight_y0 * coarse[base_00 + right_z] + weight_y1 * coarse[base_01 + right_z]
                    xy_r_x1 = weight_y0 * coarse[base_10 + right_z] + weight_y1 * coarse[base_11 + right_z]
                    xy_r    = weight_x0 * xy_r_x0 + weight_x1 * xy_r_x1

                    fine_z = coarse_z << 1
                    destination[output_base + fine_z]     += xy_c + float32(0.25) * (xy_l - xy_c)
                    destination[output_base + fine_z + 1] += xy_c + float32(0.25) * (xy_r - xy_c)

        # Bounds are only consumed by the next collapse. Level zero is final.
        if level > 1:
            if fine_x_start < bounds_min[level - 1, 0]:
                bounds_min[level - 1, 0] = fine_x_start
            if fine_y_start < bounds_min[level - 1, 1]:
                bounds_min[level - 1, 1] = fine_y_start
            if fine_z_start < bounds_min[level - 1, 2]:
                bounds_min[level - 1, 2] = fine_z_start
            if fine_x_stop - 1 > bounds_max[level - 1, 0]:
                bounds_max[level - 1, 0] = fine_x_stop - 1
            if fine_y_stop - 1 > bounds_max[level - 1, 1]:
                bounds_max[level - 1, 1] = fine_y_stop - 1
            if fine_z_stop - 1 > bounds_max[level - 1, 2]:
                bounds_max[level - 1, 2] = fine_z_stop - 1


@njit(cache=True, fastmath=True, nogil=True)
def _build_hierarchy_layout(res: int, nlevels: int) -> tuple:
    """
    Build flattened hierarchy metadata and per-level hot-loop constants.

    Computes the per-level resolution, flat-array offsets, cell widths, and
    inverse cell volumes used by the scatter and collapse kernels.  Each level
    ``L`` has resolution ``res >> L``; level 0 is the finest grid.

    Parameters
    ----------
    res : int
        Finest-grid resolution (cells per axis).
    nlevels : int
        Number of coarsening levels (deepest level has resolution ``res >> nlevels``).

    Returns
    -------
    level_offsets : np.ndarray[int64, shape=(nlevels+1,)]
        Cumulative flat-array start index for each level.
    level_resolutions : np.ndarray[int32, shape=(nlevels+1,)]
        Cell count per axis at each level.
    level_pixel_widths : np.ndarray[float32, shape=(nlevels+1,)]
        Physical cell width (``1 / level_res``) at each level.
    level_inverse_cell_volumes : np.ndarray[float32, shape=(nlevels+1,)]
        ``level_res**3``; used to convert mass to density per voxel.
    total_cells : int
        Total number of cells across all levels.
    """
    level_resolutions = np.empty(nlevels + 1, dtype=np.int32)
    level_offsets = np.empty(nlevels + 1, dtype=int64)
    level_pixel_widths = np.empty(nlevels + 1, dtype=float32)
    level_inverse_cell_volumes = np.empty(nlevels + 1, dtype=float32)
    total_cells = 0

    for level in range(nlevels + 1):
        level_res = res >> level
        float_res = float32(level_res)
        level_resolutions[level] = level_res
        level_offsets[level] = total_cells
        level_pixel_widths[level] = float32(1.0) / float_res
        level_inverse_cell_volumes[level] = float_res * float_res * float_res
        total_cells += level_res * level_res * level_res

    return (
        level_offsets,
        level_resolutions,
        level_pixel_widths,
        level_inverse_cell_volumes,
        total_cells,
    )


# ---------------------------------------------------------------------------
# Parallel scatter: split particles across the active Numba threads, run the
# full serial nested pipeline per chunk via prange, accumulate into a shared
# output grid.
# ---------------------------------------------------------------------------


@njit(fastmath=True, cache=True, nogil=True, boundscheck=False, error_model="numpy")
def _scatter_serial_chunk(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    m: np.ndarray,
    h: np.ndarray,
    res: int,
    box_x: float64,
    box_y: float64,
    box_z: float64,
    ntarget: int,
    nlevels: int,
) -> np.ndarray:
    """
    Run the complete serial nested scatter pipeline on one particle chunk.

    JIT-callable wrapper around the full assign → scatter → collapse pipeline,
    designed to be invoked from ``prange`` inside the parallel kernel and as
    the single-thread entry point from the public API.  Each call builds its
    own private hierarchy and accumulates into a private finest grid.

    Parameters
    ----------
    x, y, z : np.ndarray[float64]
        Particle positions for this chunk, in [0, 1].
    m : np.ndarray[float32]
        Particle masses or weights for this chunk.
    h : np.ndarray[float32]
        Smoothing lengths for this chunk.
    res : int
        Finest-grid resolution (cells per axis).
    box_x, box_y, box_z : float64
        Periodic box sizes; 0.0 disables wrapping on that axis.
    ntarget : int
        Target cells per kernel side at the assigned level.
    nlevels : int
        Maximum hierarchy depth.

    Returns
    -------
    np.ndarray[float32, shape=(res, res, res)]
        Partial voxel grid for this chunk.
    """
    level_index, active_nlevels_val = _assign_serial_levels(h, m, res, ntarget, nlevels)
    active_nlevels = int32(active_nlevels_val)

    (
        level_offsets,
        level_resolutions,
        level_pixel_widths,
        level_inverse_cell_volumes,
        total_cells,
    ) = _build_hierarchy_layout(res, active_nlevels)

    finest_cells = int64(res) * int64(res) * int64(res)
    finest = zeros(finest_cells, dtype=float32)
    n_coarse = total_cells - finest_cells
    coarse = zeros(n_coarse if n_coarse > int64(0) else int64(1), dtype=float32)

    bounds_min = np.empty((active_nlevels + 1, 3), dtype=np.int32)
    bounds_max = np.full((active_nlevels + 1, 3), np.int32(-1), dtype=np.int32)
    for level in range(active_nlevels + 1):
        lr = level_resolutions[level]
        bounds_min[level, 0] = lr
        bounds_min[level, 1] = lr
        bounds_min[level, 2] = lr

    _scatter_particles(
        x,
        y,
        z,
        m,
        h,
        level_index,
        level_offsets,
        level_resolutions,
        finest,
        coarse,
        box_x,
        box_y,
        box_z,
        bounds_min,
        bounds_max,
    )

    _collapse_serial_flat(
        finest,
        coarse,
        level_offsets,
        level_resolutions,
        active_nlevels,
        bounds_min,
        bounds_max,
    )
    return finest.reshape((res, res, res))


@njit(fastmath=True, parallel=True)
def _scatter_parallel_impl(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    m: np.ndarray,
    h: np.ndarray,
    res: int,
    box_x: float64 = float64(0.0),
    box_y: float64 = float64(0.0),
    box_z: float64 = float64(0.0),
    ntarget: int = 6,
    nlevels: int = 4,
) -> np.ndarray:
    """
    Parallel nested multi-resolution scatter.

    Splits particles into one contiguous chunk per active Numba thread and
    calls the full serial nested pipeline on each chunk via ``prange``.  Each
    thread independently builds its own hierarchy and finest grid, avoiding
    any write-after-write races.  The per-thread grids are summed into the
    shared output after all chunks complete.

    Particle-level parallelism (rather than voxel-level) is used because
    different particles visit different subsets of voxels; voxel-level
    splitting would require synchronisation on every write.

    Parameters
    ----------
    x, y, z : np.ndarray[float64]
        Contiguous particle positions in [0, 1].
    m : np.ndarray[float32]
        Contiguous particle masses or weights.
    h : np.ndarray[float32]
        Contiguous particle smoothing lengths.
    res : int
        Finest-grid resolution (cells per axis).
    box_x, box_y, box_z : float64
        Periodic box sizes; 0.0 disables wrapping on that axis.
    ntarget : int
        Target cells per kernel side.
    nlevels : int
        Maximum hierarchy depth.

    Returns
    -------
    np.ndarray[float32, shape=(res, res, res)]
        Voxel grid of the projected quantity.

    See Also
    --------
    scatter : Serial implementation.
    """
    number_of_particles = x.size
    number_of_chunks = min(get_num_threads(), number_of_particles)

    output = zeros((res, res, res), dtype=float32)

    for chunk in prange(number_of_chunks):
        left_edge = chunk * number_of_particles // number_of_chunks
        right_edge = (chunk + 1) * number_of_particles // number_of_chunks

        output += _scatter_serial_chunk(
            x[left_edge:right_edge],
            y[left_edge:right_edge],
            z[left_edge:right_edge],
            m[left_edge:right_edge],
            h[left_edge:right_edge],
            res,
            box_x,
            box_y,
            box_z,
            ntarget,
            nlevels,
        )
    return output

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _prepare(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    m: np.ndarray,
    h: np.ndarray,
    res: int,
    ntarget: int,
    nlevels: int,
) -> tuple:
    """
    Validate all inputs and return typed, contiguous arrays with scalar config.

    Positions (x, y, z) are promoted to float64 to avoid rounding errors in
    cell-index arithmetic at high resolution.  Masses and smoothing lengths
    (m, h) are cast to float32 to match the Numba kernel signatures.

    Parameters
    ----------
    x, y, z : array-like
        Particle positions; must be 1-D with equal length.
    m : array-like
        Particle masses or weights; must be 1-D with equal length.
    h : array-like
        Particle smoothing lengths; must be 1-D with equal length.
    res : int-like
        Finest-grid resolution (cells per axis).
    ntarget : int-like
        Target cells per kernel side.
    nlevels : int-like
        Maximum hierarchy depth.

    Returns
    -------
    x, y, z : np.ndarray[float64]
        Contiguous position arrays.
    m, h : np.ndarray[float32]
        Contiguous mass and smoothing-length arrays.
    res, ntarget, nlevels : int
        Validated integer scalars.
    """
    res = int(res)
    ntarget = int(ntarget)
    nlevels = int(nlevels)

    # Validate hierarchy geometry.
    if res <= 0:
        raise ValueError(f"Pixel size must be a positive integer. Got res={res}.")
    if nlevels < 0:
        raise ValueError(
            f"The number of hierarchy levels cannot be negative. Got nlevels={nlevels}."
        )
    if res % 2 != 0:
        raise ValueError(
            f"The nested backend requires an even pixel size because each "
            f"hierarchy step coarsens the grid by exactly a factor of two. "
            f"Got res={res}. Choose an even pixel size."
        )
    required_divisor = 1 << nlevels
    if res % required_divisor != 0:
        max_levels = 0
        remaining_res = res
        while remaining_res % 2 == 0:
            max_levels += 1
            remaining_res //= 2
        lower_valid_res = (res // required_divisor) * required_divisor
        upper_valid_res = lower_valid_res + required_divisor
        alternatives = f"{upper_valid_res}"
        if lower_valid_res > 0:
            alternatives = f"{lower_valid_res} or {upper_valid_res}"
        raise ValueError(
            f"Pixel size res={res} is incompatible with nlevels={nlevels}. "
            f"A hierarchy with {nlevels} coarsening levels requires the pixel "
            f"size to be divisible by 2**{nlevels}={required_divisor}, so that "
            f"every grid has an integer size. With res={res}, use "
            f"nlevels<={max_levels}, or choose a pixel size divisible by "
            f"{required_divisor}, such as {alternatives}."
        )
    if ntarget <= 0:
        raise ValueError(f"ntarget must be greater than zero. Got ntarget={ntarget}.")

    # Validate and convert particle arrays.
    names = ("x", "y", "z", "m", "h")
    arrays = tuple(np.asarray(value) for value in (x, y, z, m, h))

    invalid_shapes = [
        f"{name}.shape={array.shape}"
        for name, array in zip(names, arrays)
        if array.ndim != 1
    ]
    if invalid_shapes:
        raise ValueError(
            "Particle inputs x, y, z, m, and h must all be one-dimensional. "
            f"Invalid inputs: {', '.join(invalid_shapes)}."
        )

    lengths = tuple(array.size for array in arrays)
    if len(set(lengths)) != 1:
        length_details = ", ".join(
            f"{name}={length}" for name, length in zip(names, lengths)
        )
        raise ValueError(
            "Particle inputs x, y, z, m, and h must have identical lengths. "
            f"Got {length_details}."
        )

    x_arr, y_arr, z_arr, m_arr, h_arr = arrays
    return (
        np.ascontiguousarray(x_arr, dtype=float64),
        np.ascontiguousarray(y_arr, dtype=float64),
        np.ascontiguousarray(z_arr, dtype=float64),
        np.ascontiguousarray(m_arr, dtype=float32),
        np.ascontiguousarray(h_arr, dtype=float32),
        res,
        ntarget,
        nlevels,
    )

def scatter(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    m: np.ndarray,
    h: np.ndarray,
    res: int,
    box_x: float64 = float64(0.0),
    box_y: float64 = float64(0.0),
    box_z: float64 = float64(0.0),
    ntarget: int = 6,
    nlevels: int = 4,
) -> np.ndarray:
    """
    Create a weighted voxel grid using nested multi-resolution scatter.

    Particle inputs must be one-dimensional arrays with identical lengths.
    They are converted to contiguous float32/float64 arrays before entering Numba.

    Parameters
    ----------
    x : np.ndarray[np.float64]
        Array of x-positions of the particles. Must be bounded by [0, 1].

    y : np.ndarray[np.float64]
        Array of y-positions of the particles. Must be bounded by [0, 1].

    z : np.ndarray[np.float64]
        Array of z-positions of the particles. Must be bounded by [0, 1].

    m : np.ndarray[np.float32]
        Array of masses (or otherwise weights) of the particles.

    h : np.ndarray[np.float32]
        Array of smoothing lengths of the particles.

    res : int
        The number of voxels along one axis, i.e. this returns a cube of
        ``res * res * res``. Must be divisible by ``2**nlevels``.

    box_x : np.float64
        Box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping. Set to 0.0 to disable wrapping in x.

    box_y : np.float64
        Box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping. Set to 0.0 to disable wrapping in y.

    box_z : np.float64
        Box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping. Set to 0.0 to disable wrapping in z.

    ntarget : int
        Target number of voxels spanned by each kernel at its assigned level.
        Controls the accuracy/speed trade-off. Default is 6.

    nlevels : int
        Maximum number of coarsening levels in the hierarchy. Default is 4.

    Returns
    -------
    np.ndarray[np.float32, np.float32, np.float32]
        Voxel grid of the projected quantity, shape ``(res, res, res)``.

    See Also
    --------
    scatter_parallel : Parallel implementation of this function.
    """
    x, y, z, m, h, res, ntarget, nlevels = _prepare(
        x, y, z, m, h, res, ntarget, nlevels
    )
    return _scatter_serial_chunk(
        x,
        y,
        z,
        m,
        h,
        res,
        float64(box_x),
        float64(box_y),
        float64(box_z),
        ntarget,
        nlevels,
    )


def scatter_parallel(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    m: np.ndarray,
    h: np.ndarray,
    res: int,
    box_x: float64 = float64(0.0),
    box_y: float64 = float64(0.0),
    box_z: float64 = float64(0.0),
    ntarget: int = 6,
    nlevels: int = 4,
) -> np.ndarray:
    """
    Create a weighted voxel grid using nested multi-resolution scatter in parallel.

    Uses the serial implementation when only one Numba thread is active, otherwise
    splits particles across threads and accumulates per-thread grids.

    Parameters
    ----------
    x : np.ndarray[np.float64]
        Array of x-positions of the particles. Must be bounded by [0, 1].

    y : np.ndarray[np.float64]
        Array of y-positions of the particles. Must be bounded by [0, 1].

    z : np.ndarray[np.float64]
        Array of z-positions of the particles. Must be bounded by [0, 1].

    m : np.ndarray[np.float32]
        Array of masses (or otherwise weights) of the particles.

    h : np.ndarray[np.float32]
        Array of smoothing lengths of the particles.

    res : int
        The number of voxels along one axis, i.e. this returns a cube of
        ``res * res * res``. Must be divisible by ``2**nlevels``.

    box_x : np.float64
        Box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping. Set to 0.0 to disable wrapping in x.

    box_y : np.float64
        Box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping. Set to 0.0 to disable wrapping in y.

    box_z : np.float64
        Box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping. Set to 0.0 to disable wrapping in z.

    ntarget : int
        Target number of voxels spanned by each kernel at its assigned level.
        Controls the accuracy/speed trade-off. Default is 6.

    nlevels : int
        Maximum number of coarsening levels in the hierarchy. Default is 4.

    Returns
    -------
    np.ndarray[np.float32, np.float32, np.float32]
        Voxel grid of the projected quantity, shape ``(res, res, res)``.

    See Also
    --------
    scatter : Serial implementation of this function.
    """
    x, y, z, m, h, res, ntarget, nlevels = _prepare(
        x, y, z, m, h, res, ntarget, nlevels
    )
    box_x = float64(box_x)
    box_y = float64(box_y)
    box_z = float64(box_z)

    if get_num_threads() == 1:
        return _scatter_serial_chunk(
            x, y, z, m, h, res, box_x, box_y, box_z, ntarget, nlevels
        )

    return _scatter_parallel_impl(
        x, y, z, m, h, res, box_x, box_y, box_z, ntarget, nlevels
    )
