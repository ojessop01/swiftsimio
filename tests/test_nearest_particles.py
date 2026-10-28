"""Test the nearest-particle search and the nearest-neighbour slice backend."""

import numba
import numpy as np
import pytest
from swiftsimio.visualisation._kdtree import nearest_particles
from swiftsimio.visualisation.slice_backends import backends, backends_parallel

# Coordinates are shifted before building the tree (moving the periodic boundary
# into a gap, centring free axes), which rounds them by about one ulp of the box
DISTANCE_ATOL = 1e-14


def _clumpy_particles(n: int, seed: int = 1) -> np.ndarray:
    """
    Make particles in a unit box with a dense clump and a clump across a corner.

    Parameters
    ----------
    n : int
        Number of particles.
    seed : int
        Seed of the random numbers.

    Returns
    -------
    np.ndarray
        Positions wrapped into the unit box, shape ``(n, 3)``.
    """
    rng = np.random.default_rng(seed)
    clump = rng.normal(0.5, 0.02, (n // 2, 3))
    corner = rng.normal(0.0, 0.05, (n // 4, 3))
    field = rng.random((n - n // 2 - n // 4, 3))
    return np.concatenate([clump, corner, field]) % 1.0


def _pixels(res: int, z: float, low: float = 0.0, span: float = 1.0) -> np.ndarray:
    """
    Make the pixel centres of an image, in the order of the slice backend.

    Parameters
    ----------
    res : int
        Number of pixels along each side.
    z : float
        The z coordinate of the image.
    low : float
        Lower edge of the image along x and y.
    span : float
        Side of the image.

    Returns
    -------
    np.ndarray
        The pixel centres, shape ``(res * res, 3)``.
    """
    centres = low + span * (np.arange(res) + 0.5) / res
    grids = np.meshgrid(centres, centres, [z], indexing="ij")
    return np.stack([grid.ravel() for grid in grids], axis=1)


def _distances(
    positions: np.ndarray, queries: np.ndarray, boxsize: list[float] | None
) -> np.ndarray:
    """
    Compute minimum-image distances between pairs of points.

    Parameters
    ----------
    positions : np.ndarray
        The first point of each pair.
    queries : np.ndarray
        The second point of each pair.
    boxsize : list[float] or None
        Period of each axis (0 for none), or ``None`` for no periodicity.

    Returns
    -------
    np.ndarray
        The distances.
    """
    separation = np.abs(positions - queries)
    if boxsize is not None:
        box = np.asarray(boxsize, dtype=np.float64)
        periodic = box > 0
        wrapped = separation % np.where(periodic, box, np.inf)
        separation = np.where(periodic, np.minimum(wrapped, box - wrapped), separation)
    return np.sqrt((separation**2).sum(axis=1))


@pytest.mark.parametrize("boxsize", [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], None])
@pytest.mark.parametrize(
    "image",
    [
        dict(res=128, z=0.5),
        dict(res=96, z=0.01),
        # Pixels beyond the box, which are wrapped into it
        dict(res=64, z=1.7, low=-1.0, span=3.0),
        # A zoom onto the dense clump
        dict(res=64, z=0.5, low=0.45, span=0.1),
    ],
)
def test_nearest_particles_match_scipy(boxsize, image):
    """Check the nearest particle to each pixel against scipy."""
    spatial = pytest.importorskip("scipy.spatial")
    positions = _clumpy_particles(20000)
    queries = _pixels(**image)
    nearest = nearest_particles(positions, queries, boxsize)
    expected, _ = spatial.cKDTree(positions, boxsize=boxsize).query(queries)
    # Compare distances (particles at equal distances may be chosen differently)
    np.testing.assert_allclose(
        _distances(positions[nearest], queries, boxsize),
        expected,
        rtol=1e-12,
        atol=DISTANCE_ATOL,
    )


def test_nearest_particles_small_region_of_periodic_box():
    """Check pixels far from a small region of a periodic box (nearer by the wrap)."""
    spatial = pytest.importorskip("scipy.spatial")
    positions = np.random.default_rng(2).normal(0.2, 0.01, (5000, 3))
    queries = _pixels(64, 0.9)
    nearest = nearest_particles(positions, queries, 1.0)
    expected, _ = spatial.cKDTree(positions % 1.0, boxsize=1.0).query(queries)
    np.testing.assert_allclose(
        _distances(positions[nearest], queries, [1.0] * 3),
        expected,
        rtol=1e-12,
        atol=DISTANCE_ATOL,
    )


def test_nearest_particles_threads():
    """Check that the nearest particles do not depend on the number of threads."""
    positions = _clumpy_particles(20000, seed=3)
    queries = _pixels(128, 0.5)
    parallel = nearest_particles(positions, queries, 1.0)
    threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        serial = nearest_particles(positions, queries, 1.0)
    finally:
        numba.set_num_threads(threads)
    np.testing.assert_array_equal(parallel, serial)


def test_nearest_particles_no_particles():
    """Check that searching no particles is an error."""
    with pytest.raises(ValueError):
        nearest_particles(np.zeros((0, 3)), _pixels(4, 0.5))


@pytest.mark.parametrize("box", [1.0, 0.0])
def test_nearest_neighbours_slice_matches_scipy(box):
    """Check the slice backend against the scipy query it replaced."""
    spatial = pytest.importorskip("scipy.spatial")
    rng = np.random.default_rng(4)
    positions = _clumpy_particles(5000, seed=4)
    masses = rng.random(5000).astype(np.float32)
    hsml = (0.01 + 0.01 * rng.random(5000)).astype(np.float32)
    kwargs = dict(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        m=masses,
        h=hsml,
        z_slice=0.3,
        xres=80,
        yres=60,
        box_x=box,
        box_y=box,
        box_z=box,
    )
    image = backends["nearest_neighbours"](**kwargs)
    image_parallel = backends_parallel["nearest_neighbours"](**kwargs)

    res = max(kwargs["xres"], kwargs["yres"])
    grids = np.meshgrid(
        (np.arange(kwargs["xres"]) + 0.5) / res,
        (np.arange(kwargs["yres"]) + 0.5) / res,
        [kwargs["z_slice"]],
        indexing="ij",
    )
    pixels = np.stack([grid.ravel() for grid in grids], axis=1)
    tree = spatial.cKDTree(positions, boxsize=[box] * 3 if box else None)
    _, nearest = tree.query(pixels)
    expected = (masses / hsml**3)[nearest].reshape(kwargs["xres"], kwargs["yres"])

    assert image.dtype == expected.dtype
    np.testing.assert_array_equal(image, expected)
    np.testing.assert_array_equal(image_parallel, expected)


def test_nearest_neighbours_slice_no_particles():
    """Check that a slice of no particles is empty."""
    empty = np.zeros(0)
    image = backends["nearest_neighbours"](
        x=empty,
        y=empty,
        z=empty,
        m=empty.astype(np.float32),
        h=empty.astype(np.float32),
        z_slice=0.5,
        xres=8,
        yres=4,
        box_x=1.0,
        box_y=1.0,
        box_z=1.0,
    )
    assert image.shape == (8, 4)
    assert not image.any()
