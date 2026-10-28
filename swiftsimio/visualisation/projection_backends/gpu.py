"""Kernel evaluation on the GPU."""

import math

import numpy as np
from swiftsimio.optional_packages import (
    CUDA_AVAILABLE,
    cuda_jit,
    CudaSupportError,
    cuda,
)
from swiftsimio.visualisation.kernels import (
    get_kernel_id,
    kernel_shape,
    kernel_gamma_2d,
    kernel_constant_2d,
)

# Taken from Dehnen & Aly 2012; this is the Wendland-C2 (default) value.
kernel_gamma = np.float32(1.897367)

# Support radii and 2D normalisations of each kernel, indexed by kernel_id.
kernel_gammas = kernel_gamma_2d.astype(np.float32)
kernel_constants = kernel_constant_2d.astype(np.float32)


@cuda_jit("float32(float32, float32, int64)", device=True)
def kernel(r: np.float32, H: np.float32, kernel_id: int) -> np.float32:
    """
    Single precision kernel implementation for swiftsimio.

    This is the 2D form of the kernel selected by ``kernel_id`` (see
    :mod:`swiftsimio.visualisation.kernels`); the default is the Wendland-C2
    kernel as shown in Denhen & Aly (2012) [1]_.

    Parameters
    ----------
    r : np.float32
        Radius used in kernel computation.

    H : np.float32
        Kernel width (i.e. radius of compact support for the kernel).

    kernel_id : int
        Index of the kernel in :data:`swiftsimio.visualisation.kernels.kernel_names`.

    Returns
    -------
    np.float32
        Contribution to the density by the particle.

    Notes
    -----
    This is the cuda-compiled version of the kernel, designed for use within the gpu
    backend. It has no double precision cousin.

    References
    ----------
    .. [1] Dehnen W., Aly H., 2012, MNRAS, 425, 1068
    """
    kernel_constant = kernel_constants[kernel_id]

    inverse_H = 1.0 / H
    ratio = r * inverse_H

    kernel = 0.0

    if ratio < 1.0:
        kernel = kernel_shape(ratio, kernel_id)

        kernel *= kernel_constant * inverse_H * inverse_H

    return kernel


@cuda_jit(
    "void(float64[:], float64[:], float32[:], float32[:], float64, float64, "
    "float32[:,:], int64)"
)
def scatter_gpu(
    x: np.float64,
    y: np.float64,
    m: np.float32,
    h: np.float32,
    box_x: np.float64,
    box_y: np.float64,
    img: np.float32,
    kernel_id: int,
) -> None:
    """
    Create a weighted scatter plot.

    Computes contributions to from particles with positions (`x`,`y`) with smoothing
    lengths `h` weighted by quantities `m`. This includes periodic boundary effects.

    Parameters
    ----------
    x : np.ndarray[np.float64]
        Array of x-positions of the particles. Must be bounded by [0, 1].

    y : np.ndarray[np.float64]
        Array of y-positions of the particles. Must be bounded by [0, 1].

    m : np.ndarray[np.float32]
        Array of masses (or otherwise weights) of the particles.

    h : np.ndarray[np.float32]
        Array of smoothing lengths of the particles.

    box_x : np.float64
        Box size in x, in the same rescaled length units as x and y. Used
        for periodic wrapping.

    box_y : np.float64
        Box size in y, in the same rescaled length units as x and y. Used
        for periodic wrapping.

    img : np.ndarray[np.float32]
        The output image.

    kernel_id : int
        Index of the SPH kernel to use, see
        :data:`swiftsimio.visualisation.kernels.kernel_names`.

    Notes
    -----
    Explicitly defining the types in this function allows for a performance improvement.
    This is the cuda version, and as such can only be run on systems with a supported
    GPU. Do not call this where cuda is not available (checks can be performed using
    ``swiftsimio.optional_packages.CUDA_AVAILABLE``).
    """
    # Output array for our image
    res = img.shape[0]
    maximal_array_index = np.int32(res) - 1

    # Change that integer to a float, we know that our x, y are bounded
    # by [0, 1].
    float_res = np.float32(res)
    pixel_width = 1.0 / float_res

    # We need this for combining with the x_pos and y_pos variables.
    float_res_64 = np.float64(res)

    # Pre-calculate this constant for use with the above
    inverse_cell_area = res * res

    # get the particle index and the x and y index of its periodic copy
    i, dx, dy = cuda.grid(3)
    if i < len(x):
        # Get the correct particle
        mass = m[i]
        hsml = h[i]
        x_pos = x[i] + (dx - 1.0) * box_x
        y_pos = y[i] + (dy - 1.0) * box_y

        # Calculate the cell that this particle; use the 64 bit version of the
        # resolution as this is the same type as the positions
        particle_cell_x = np.int32(math.floor(float_res_64 * x_pos))
        particle_cell_y = np.int32(math.floor(float_res_64 * y_pos))

        # SWIFT stores hsml as the FWHM.
        kernel_width = kernel_gammas[kernel_id] * hsml

        # The number of cells that this kernel spans
        cells_spanned = np.int32(1.0 + kernel_width * float_res)

        if (
            particle_cell_x + cells_spanned < 0
            or particle_cell_x - cells_spanned > maximal_array_index
            or particle_cell_y + cells_spanned < 0
            or particle_cell_y - cells_spanned > maximal_array_index
        ):
            # Can happily skip this particle
            return

        if cells_spanned <= 1:
            # Easygame, gg
            if (
                particle_cell_x >= 0
                and particle_cell_x <= maximal_array_index
                and particle_cell_y >= 0
                and particle_cell_y <= maximal_array_index
            ):
                cuda.atomic.add(
                    img, (particle_cell_x, particle_cell_y), mass * inverse_cell_area
                )
        else:
            # Now we loop over the square of cells that the kernel lives in
            for cell_x in range(
                # Ensure that the lowest x value is 0, otherwise we'll segfault
                max(0, particle_cell_x - cells_spanned),
                # Ensure that the highest x value lies within the array bounds,
                # otherwise we'll segfault (oops).
                min(particle_cell_x + cells_spanned + 1, maximal_array_index + 1),
            ):
                # The distance in x to our new favourite cell
                # remember that our x, y are all in a box of [0, 1]
                # calculate the distance to the cell center
                distance_x = (np.float32(cell_x) + 0.5) * pixel_width
                distance_x -= np.float32(x_pos)
                distance_x_2 = distance_x * distance_x
                for cell_y in range(
                    max(0, particle_cell_y - cells_spanned),
                    min(particle_cell_y + cells_spanned + 1, maximal_array_index + 1),
                ):
                    distance_y = (np.float32(cell_y) + 0.5) * pixel_width
                    distance_y -= np.float32(y_pos)
                    distance_y_2 = distance_y * distance_y

                    r = math.sqrt(distance_x_2 + distance_y_2)

                    kernel_eval = kernel(r, kernel_width, kernel_id)

                    cuda.atomic.add(img, (cell_x, cell_y), mass * kernel_eval)


def scatter(
    x: np.float64,
    y: np.float64,
    m: np.float32,
    h: np.float32,
    res: int,
    box_x: np.float64 = 0.0,
    box_y: np.float64 = 0.0,
    kernel_id: int = 0,
) -> np.ndarray:
    """
    Create a weighted scatter plot in parallel.

    Creates a weighted scatter plot. Computes contributions from particles with positions
    (`x`,`y`) with smoothing lengths `h` weighted by quantities `m`. This includes
    periodic boundary effects.

    Parameters
    ----------
    x : np.ndarray[np.float64]
        Array of x-positions of the particles. Must be bounded by [0, 1].

    y : np.ndarray[np.float64]
        Array of y-positions of the particles. Must be bounded by [0, 1].

    m : np.ndarray[np.float32]
        Array of masses (or otherwise weights) of the particles.

    h : np.ndarray[np.float32]
        Array of smoothing lengths of the particles.

    res : int
        The number of pixels along one axis, i.e. this returns a square
        of res * res.

    box_x : np.float64
        Box size in x, in the same rescaled length units as x and y. Used
        for periodic wrapping.

    box_y : np.float64
        Box size in y, in the same rescaled length units as x and y. Used
        for periodic wrapping.

    kernel_id : int
        Index of the SPH kernel to use, see
        :data:`swiftsimio.visualisation.kernels.kernel_names`. Defaults to
        ``0`` (Wendland-C2).

    Returns
    -------
    np.ndarray[np.float32, np.float32, np.float32]
        Pixel grid of quantity.

    See Also
    --------
    scatter
        Creates 2D scatter plot from SWIFT data.

    Notes
    -----
    Explicitly defining the types in this function allows a performance improvement.
    """
    if not CUDA_AVAILABLE or cuda is None:
        raise CudaSupportError(
            "Unable to load the CUDA extension to numba. This function "
            "is only available on systems with supported GPUs."
        )

    # The GPU kernel is compiled for these exact types and needs contiguous arrays.
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    m = np.ascontiguousarray(m, dtype=np.float32)
    h = np.ascontiguousarray(h, dtype=np.float32)
    box_x = float(box_x)
    box_y = float(box_y)

    output = cuda.device_array((res, res), dtype=np.float32)
    output[:] = 0

    n_part = len(x)
    if box_x == 0.0:
        n_xshift = 1
    else:
        n_xshift = int(np.ceil(1 / box_x) + 2)
    if box_y == 0.0:
        n_yshift = 1
    else:
        n_yshift = int(np.ceil(1 / box_y) + 2)
    # set up a 3D grid:
    # the first dimension are the particles
    # the second and third dimension are the periodic
    # copies for each particle
    threads_per_block = (16, 1, 1)
    blocks_per_grid = (
        int(np.ceil(n_part / threads_per_block[0])),
        n_xshift // threads_per_block[1],
        n_yshift // threads_per_block[2],
    )
    scatter_gpu[blocks_per_grid, threads_per_block](
        x, y, m, h, box_x, box_y, output, get_kernel_id(kernel_id)
    )

    return output.copy_to_host()


scatter_parallel = scatter
