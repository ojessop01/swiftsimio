"""Backend tools for image slices with nearest neighbour interpolation."""

import numpy as np

from numba import config, get_num_threads, set_num_threads

from swiftsimio.visualisation._kdtree import nearest_particles


def slice_scatter(
    x: np.float64,
    y: np.float64,
    z: np.float64,
    m: np.float32,
    h: np.float32,
    z_slice: float,
    xres: int,
    yres: int,
    box_x: float = 0.0,
    box_y: float = 0.0,
    box_z: float = 0.0,
    workers: int = 1,
) -> np.ndarray:
    """
    Create a 2D image slice through a volume.

    Creates a 2D numpy array (image) of the given quantities of all particles in
    a data slice including periodic boundary effects.

    Parameters
    ----------
    x : array of float64
        The x-positions of the particles. Must be bounded by [0, 1].

    y : array of float64
        The y-positions of the particles. Must be bounded by [0, 1].

    z : array of float64
        The z-positions of the particles. Must be bounded by [0, 1].

    m : array of float32
        Masses (or otherwise weights) of the particles.

    h : array of float32
        Smoothing lengths of the particles.

    z_slice : float64
        The position at which we wish to create the slice.

    xres : int
        The number of pixels in x direction.

    yres : int
        The number of pixels in the y direction.

    box_x : float
        Box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).

    box_y : float
        Box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).

    box_z : float
        Box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).

    workers : int
        The number of threads to use for the nearest-neighbour calculations.
        Set to -1 to use all of ``numba``'s threads.

    Returns
    -------
    ndarray of float32
        Output array for the slice image.

    See Also
    --------
    scatter
        Create 3D scatter plot of SWIFT data.

    scatter_parallel
        Create 3D scatter plot of SWIFT data in parallel.

    slice_scatter_parallel
        Create scatter plot of a slice of data in parallel.
    """
    res = max(xres, yres)
    pixel_coordinates = np.stack(
        [
            arr.ravel()
            for arr in np.meshgrid(
                (np.arange(xres) + 0.5) / res,
                (np.arange(yres) + 0.5) / res,
                np.array([z_slice]),
                indexing="ij",
            )
        ],
        axis=1,
    )

    values = m / h**3
    if values.size == 0:
        return np.zeros((xres, yres), dtype=values.dtype)

    # In the periodic case the particles and pixels are wrapped into the box (a
    # box size of 0 leaves that axis non-periodic)
    threads = get_num_threads()
    try:
        if workers > 0:
            set_num_threads(min(workers, config.NUMBA_NUM_THREADS))
        i = nearest_particles(
            np.stack((x, y, z), axis=1), pixel_coordinates, [box_x, box_y, box_z]
        )
    finally:
        set_num_threads(threads)

    return values[i].reshape(xres, yres)


def slice_scatter_parallel(
    x: np.float64,
    y: np.float64,
    z: np.float64,
    m: np.float32,
    h: np.float32,
    z_slice: float,
    xres: int,
    yres: int,
    box_x: float = 0.0,
    box_y: float = 0.0,
    box_z: float = 0.0,
) -> np.ndarray:
    """
    Parallel implementation of slice_scatter.

    Creates a 2D numpy array (image) of the given quantities of all particles in
    a data slice including periodic boundary effects.

    Parameters
    ----------
    x : array of float64
        The x-positions of the particles. Must be bounded by [0, 1].

    y : array of float64
        The y-positions of the particles. Must be bounded by [0, 1].

    z : array of float64
        The z-positions of the particles. Must be bounded by [0, 1].

    m : array of float32
        Masses (or otherwise weights) of the particles.

    h : array of float32
        Smoothing lengths of the particles.

    z_slice : float64
        The position at which we wish to create the slice.

    xres : int
        The number of pixels in x direction.

    yres : int
        The number of pixels in the y direction.

    box_x : float
        Box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).

    box_y : float
        Box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).

    box_z : float
        Box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).

    Returns
    -------
    ndarray of float32
        Output array for the slice image.

    See Also
    --------
    scatter
        Create 3D scatter plot of SWIFT data.

    scatter_parallel
        Create 3D scatter plot of SWIFT data in parallel.

    slice_scatter_parallel
        Create scatter plot of a slice of data in parallel.
    """
    return slice_scatter(
        x=x,
        y=y,
        z=z,
        m=m,
        h=h,
        z_slice=z_slice,
        xres=xres,
        yres=yres,
        box_x=box_x,
        box_y=box_y,
        box_z=box_z,
        workers=-1,
    )
