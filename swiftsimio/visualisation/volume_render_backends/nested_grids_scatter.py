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

from math import ceil, sqrt

import numpy as np
from numpy import float32, float64, int32, zeros, int64

from numba import get_num_threads, njit, prange

# SWIFT uses the normalised 3-D Wendland-C2 kernel. The serial hot loop
# expands it algebraically so that 1/H and 1/H**3 are computed once per
# particle instead of once per visited voxel.
_KERNEL_NORMALISATION_3D = float32(21.0 / (2.0 * np.pi))
kernel_gamma = float32(1.936492)


@njit(fastmath=True, cache=True, nogil=True, boundscheck=False, error_model="numpy")
def _assign_serial_levels(h, m, res, ntarget, nlevels):
    """Assign a hierarchy level to every particle in a single pass.

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
    destination,
    level_offset,
    level_res,
    pixel_width,
    inverse_cell_volume,
    x_pos,
    y_pos,
    z_pos,
    mass,
    hsml,
    level,
    bounds_min,
    bounds_max,
):
    """Deposit one particle onto the flat destination array at the given level.

    The destination is a flat (1-D) view shared across all levels; ``level_offset``
    locates the start of this level's sub-array.  Bounds arrays track the
    bounding box of occupied coarse cells so that the collapse step only
    visits cells that were actually written.

    Parameters
    ----------
    destination : np.ndarray[float32]
        Flat output array (finest or coarse, depending on level).
    level_offset : int64
        Index of the first cell belonging to this level in ``destination``.
    level_res : int32
        Number of cells along one axis at this level.
    pixel_width : float32
        Physical width of one cell at this level (``1 / level_res``).
    inverse_cell_volume : float32
        ``level_res**3`` — converts mass-per-particle to density-per-voxel.
    x_pos : float64
        Particle x coordinate in [0, 1].
    y_pos : float64
        Particle y coordinate in [0, 1].
    z_pos : float64
        Particle z coordinate in [0, 1].
    mass : float32
        Particle mass or weight.
    hsml : float32
        Particle smoothing length in [0, 1] units.
    level : int32
        Hierarchy level of this particle.
    bounds_min : np.ndarray[int32, shape=(nlevels+1, 3)]
        Per-level bounding-box minimum cell indices (updated in-place).
    bounds_max : np.ndarray[int32, shape=(nlevels+1, 3)]
        Per-level bounding-box maximum cell indices (updated in-place).
    """
    maximal_index = level_res - 1
    float_res = float32(level_res)
    # float64 resolution for cell-index computation avoids float32 rounding
    # near cell boundaries at high resolution.
    float_res_64 = float64(level_res)
    kernel_width = float32(kernel_gamma) * hsml

    particle_cell_x = int32(float_res_64 * x_pos)
    particle_cell_y = int32(float_res_64 * y_pos)
    particle_cell_z = int32(float_res_64 * z_pos)
    cells_spanned = int32(float32(1.0) + kernel_width * float_res)

    # Early-exit when the kernel bounding box is entirely outside the grid.
    if (
        particle_cell_x + cells_spanned < 0
        or particle_cell_x - cells_spanned > maximal_index
        or particle_cell_y + cells_spanned < 0
        or particle_cell_y - cells_spanned > maximal_index
        or particle_cell_z + cells_spanned < 0
        or particle_cell_z - cells_spanned > maximal_index
    ):
        return

    # Sub-pixel kernel: deposit into the single nearest cell without
    # evaluating the full kernel loop.
    if kernel_width < pixel_width * float32(0.5):
        if (
            0 <= particle_cell_x <= maximal_index
            and 0 <= particle_cell_y <= maximal_index
            and 0 <= particle_cell_z <= maximal_index
        ):
            flat_cell = level_offset + (
                (particle_cell_x * level_res + particle_cell_y) * level_res
                + particle_cell_z
            )
            destination[flat_cell] += mass * inverse_cell_volume
            # Level zero is returned directly and is never a collapse source.
            if level != 0:
                if particle_cell_x < bounds_min[level, 0]:
                    bounds_min[level, 0] = particle_cell_x
                if particle_cell_y < bounds_min[level, 1]:
                    bounds_min[level, 1] = particle_cell_y
                if particle_cell_z < bounds_min[level, 2]:
                    bounds_min[level, 2] = particle_cell_z
                if particle_cell_x > bounds_max[level, 0]:
                    bounds_max[level, 0] = particle_cell_x
                if particle_cell_y > bounds_max[level, 1]:
                    bounds_max[level, 1] = particle_cell_y
                if particle_cell_z > bounds_max[level, 2]:
                    bounds_max[level, 2] = particle_cell_z
        return

    x_start = particle_cell_x - cells_spanned
    if x_start < 0:
        x_start = 0
    x_stop = particle_cell_x + cells_spanned
    if x_stop > level_res:
        x_stop = level_res

    y_start = particle_cell_y - cells_spanned
    if y_start < 0:
        y_start = 0
    y_stop = particle_cell_y + cells_spanned
    if y_stop > level_res:
        y_stop = level_res

    z_start = particle_cell_z - cells_spanned
    if z_start < 0:
        z_start = 0
    z_stop = particle_cell_z + cells_spanned
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

    # Precompute reciprocals so the inner loop avoids per-voxel divisions.
    kernel_width_2 = kernel_width * kernel_width
    inverse_kernel_width = float32(1.0) / kernel_width
    # Absorb mass and kernel normalisation into a single prefactor evaluated
    # once per particle rather than once per visited voxel.
    weighted_prefactor = (
        mass
        * _KERNEL_NORMALISATION_3D
        * inverse_kernel_width
        * inverse_kernel_width
        * inverse_kernel_width
    )
    level_plane = level_res * level_res

    # Walk cells in x-y-z order (z innermost) to maximise sequential memory
    # access in the flat destination array (z is the contiguous axis).
    distance_x = (float32(x_start) + float32(0.5)) * pixel_width - x_pos
    for cell_x in range(x_start, x_stop):
        distance_x_2 = distance_x * distance_x
        if distance_x_2 < kernel_width_2:
            distance_y = (float32(y_start) + float32(0.5)) * pixel_width - y_pos
            x_base = level_offset + cell_x * level_plane

            for cell_y in range(y_start, y_stop):
                distance_xy_2 = distance_x_2 + distance_y * distance_y
                if distance_xy_2 < kernel_width_2:
                    distance_z = (float32(z_start) + float32(0.5)) * pixel_width - z_pos
                    flat_cell = x_base + cell_y * level_res + z_start

                    for cell_z in range(z_start, z_stop):
                        radius_2 = distance_xy_2 + distance_z * distance_z
                        if radius_2 < kernel_width_2:
                            # Wendland-C2 kernel: W(r,H) = (21/2π) H⁻³ (1-r/H)⁴ (1+4r/H)
                            ratio = sqrt(radius_2) * inverse_kernel_width
                            one_minus_ratio = float32(1.0) - ratio
                            one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
                            kernel_shape = (
                                one_minus_ratio_2
                                * one_minus_ratio_2
                                * (float32(1.0) + float32(4.0) * ratio)
                            )
                            destination[flat_cell] += weighted_prefactor * kernel_shape
                        flat_cell += 1
                        distance_z += pixel_width
                distance_y += pixel_width
        distance_x += pixel_width


@njit(fastmath=True, cache=True, nogil=True, boundscheck=False, error_model="numpy")
def _scatter_particles(
    x,
    y,
    z,
    m,
    h,
    level_index,
    level_offsets,
    level_resolutions,
    level_pixel_widths,
    level_inverse_cell_volumes,
    finest,
    coarse,
    box_x,
    box_y,
    box_z,
    bounds_min,
    bounds_max,
):
    """Scatter all particles onto their assigned hierarchy levels.

    Iterates once over all particles, dispatching each to either the finest
    grid (level 0) or the shared coarse flat-array (levels 1+).

    Periodic wrapping is handled by depositing each particle once per image
    whose kernel support overlaps [0, 1].  When a box dimension is 0.0 the
    shift range collapses to a single image, so both periodic and
    non-periodic cases are handled by the same loop.

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
    level_pixel_widths : np.ndarray[float32]
        Cell width (``1/res``) at each level.
    level_inverse_cell_volumes : np.ndarray[float32]
        ``res**3`` at each level; converts mass to density.
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

    # Number of periodic images needed per axis. A box size of 0 means the
    # axis is non-periodic, so range(0, 1) deposits only the original image.
    xshift_min = 0 if box_x == 0.0 else -1
    yshift_min = 0 if box_y == 0.0 else -1
    zshift_min = 0 if box_z == 0.0 else -1
    xshift_max = 1 if box_x == 0.0 else int32(ceil(1.0 / box_x) + 1)
    yshift_max = 1 if box_y == 0.0 else int32(ceil(1.0 / box_y) + 1)
    zshift_max = 1 if box_z == 0.0 else int32(ceil(1.0 / box_z) + 1)

    for particle in range(h.size):
        level = int32(level_index[particle])
        if level < 0:
            continue

        mass = m[particle]
        hsml = h[particle]
        # Level-0 particles go directly into the finest grid; coarser levels
        # share a single flat array offset by level_offsets.
        if level == 0:
            destination = finest
            level_offset = 0
        else:
            destination = coarse
            level_offset = level_offsets[level] - finest_cells

        level_res = level_resolutions[level]
        pixel_width = level_pixel_widths[level]
        inverse_cell_volume = level_inverse_cell_volumes[level]
        original_x = x[particle]
        original_y = y[particle]
        original_z = z[particle]

        for xshift in range(xshift_min, xshift_max):
            x_pos = original_x + xshift * box_x
            for yshift in range(yshift_min, yshift_max):
                y_pos = original_y + yshift * box_y
                for zshift in range(zshift_min, zshift_max):
                    _deposit_particle_flat(
                        destination,
                        level_offset,
                        level_res,
                        pixel_width,
                        inverse_cell_volume,
                        x_pos,
                        y_pos,
                        original_z + zshift * box_z,
                        mass,
                        hsml,
                        level,
                        bounds_min,
                        bounds_max,
                    )


@njit(fastmath=True, cache=True, nogil=True, boundscheck=False, error_model="numpy")
def _collapse_serial_flat(
    finest, coarse, level_offsets, level_resolutions, nlevels, bounds_min, bounds_max
):
    """Trilinearly upsample each coarse level and accumulate into the next finer one.

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
                for coarse_z in range(pair_z_start, pair_z_stop):
                    left_z = coarse_z - 1
                    if left_z < 0:
                        left_z = 0
                    right_z = coarse_z + 1
                    if right_z > coarse_max:
                        right_z = coarse_max

                    even_00 = (
                        float32(0.25) * coarse[base_00 + left_z]
                        + float32(0.75) * coarse[base_00 + coarse_z]
                    )
                    even_01 = (
                        float32(0.25) * coarse[base_01 + left_z]
                        + float32(0.75) * coarse[base_01 + coarse_z]
                    )
                    even_10 = (
                        float32(0.25) * coarse[base_10 + left_z]
                        + float32(0.75) * coarse[base_10 + coarse_z]
                    )
                    even_11 = (
                        float32(0.25) * coarse[base_11 + left_z]
                        + float32(0.75) * coarse[base_11 + coarse_z]
                    )
                    odd_00 = (
                        float32(0.75) * coarse[base_00 + coarse_z]
                        + float32(0.25) * coarse[base_00 + right_z]
                    )
                    odd_01 = (
                        float32(0.75) * coarse[base_01 + coarse_z]
                        + float32(0.25) * coarse[base_01 + right_z]
                    )
                    odd_10 = (
                        float32(0.75) * coarse[base_10 + coarse_z]
                        + float32(0.25) * coarse[base_10 + right_z]
                    )
                    odd_11 = (
                        float32(0.75) * coarse[base_11 + coarse_z]
                        + float32(0.25) * coarse[base_11 + right_z]
                    )

                    even_x0 = weight_y0 * even_00 + weight_y1 * even_01
                    even_x1 = weight_y0 * even_10 + weight_y1 * even_11
                    odd_x0 = weight_y0 * odd_00 + weight_y1 * odd_01
                    odd_x1 = weight_y0 * odd_10 + weight_y1 * odd_11
                    fine_z = coarse_z << 1
                    destination[output_base + fine_z] += (
                        weight_x0 * even_x0 + weight_x1 * even_x1
                    )
                    destination[output_base + fine_z + 1] += (
                        weight_x0 * odd_x0 + weight_x1 * odd_x1
                    )

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
def _build_hierarchy_layout(res: int, nlevels: int):
    """Build flattened hierarchy metadata and per-level hot-loop constants.

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
    x, y, z, m, h, res, box_x: float64, box_y: float64, box_z: float64, ntarget, nlevels
):
    """Run the complete serial nested scatter pipeline on one particle chunk.

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
        level_pixel_widths,
        level_inverse_cell_volumes,
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


def _prepare(x, y, z, m, h, res, ntarget, nlevels):
    """Validate all inputs and return typed, contiguous arrays with scalar config.

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
