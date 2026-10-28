"""Test the smoothing length generation code."""

import numba
import numpy as np
import pytest
import unyt as u
from swiftsimio import load, cosmo_array
from swiftsimio.visualisation.smoothing_length import generate_smoothing_lengths
from swiftsimio.visualisation._kdtree import (
    HEAP_K,
    k_nearest_distances,
)

from numpy import isclose


def test_generate_smoothing_length(cosmological_volume):
    """Check that generated smoothign lengths match the ones stored with the data."""
    data = load(cosmological_volume)

    smoothing_lengths = data.gas.smoothing_lengths

    # Parameters required to generate smoothing lengths
    number_of_neighbours = int(
        round(data.metadata.hydro_scheme["Kernel target N_ngb"][0])
    )
    kernel_eta = data.metadata.hydro_scheme["Kernel eta"][0]

    kernel_gamma = ((3.0 * number_of_neighbours) / (4.0 * 3.14159)) ** (
        1 / 3
    ) / kernel_eta

    generated_smoothing_lengths = generate_smoothing_lengths(
        data.gas.coordinates,
        boxsize=data.metadata.boxsize,
        kernel_gamma=kernel_gamma,
        neighbours=number_of_neighbours,
        speedup_fac=1,
        dimension=3,
    ).to(smoothing_lengths.units)

    assert (
        isclose(generated_smoothing_lengths.value, smoothing_lengths.value, 0.2).sum()
        / generated_smoothing_lengths.size
        > 0.5
    )

    return


def test_generate_smoothing_length_faster(cosmological_volume):
    """Check that turning up the speed of the generator gives acceptable results."""
    data = load(cosmological_volume)

    smoothing_lengths = data.gas.smoothing_lengths

    # Parameters required to generate smoothing lengths
    number_of_neighbours = int(
        round(data.metadata.hydro_scheme["Kernel target N_ngb"][0])
    )
    kernel_eta = data.metadata.hydro_scheme["Kernel eta"][0]

    kernel_gamma = ((3.0 * number_of_neighbours) / (4.0 * 3.14159)) ** (
        1 / 3
    ) / kernel_eta

    generated_smoothing_lengths = generate_smoothing_lengths(
        data.gas.coordinates,
        boxsize=data.metadata.boxsize,
        kernel_gamma=kernel_gamma,
        neighbours=number_of_neighbours,
        speedup_fac=2,
        dimension=3,
    ).to(smoothing_lengths.units)

    assert (
        isclose(generated_smoothing_lengths.value, smoothing_lengths.value, 0.2).sum()
        / generated_smoothing_lengths.size
        > 0.5
    )

    return


def test_generate_smoothing_length_return_type():
    """Check that return types agree with input types (unyt vs cosmo arrays)."""
    x = cosmo_array(
        np.arange(20), u.Mpc, comoving=False, scale_factor=0.5, scale_exponent=1
    )
    xgrid, ygrid, zgrid = np.meshgrid(x, x, x)
    coords = np.vstack((xgrid.flatten(), ygrid.flatten(), zgrid.flatten())).T
    lbox = cosmo_array(
        [20, 20, 20], u.Mpc, comoving=False, scale_factor=0.5, scale_exponent=1
    )
    from_ca_input = generate_smoothing_lengths(coords, lbox, 1)
    assert from_ca_input.units == coords.units
    assert from_ca_input.comoving == coords.comoving
    assert from_ca_input.cosmo_factor == coords.cosmo_factor
    from_ua_input = generate_smoothing_lengths(u.unyt_array(coords), lbox, 1)
    assert isinstance(from_ua_input, u.unyt_array) and not isinstance(
        from_ua_input, cosmo_array
    )


def _clumpy_particles(n: int, boxsize: float, seed: int = 1) -> np.ndarray:
    """
    Make particles with densities spanning several decades, some across the edge.

    Parameters
    ----------
    n : int
        Number of particles.
    boxsize : float
        Side of the periodic box.
    seed : int
        Seed of the random numbers.

    Returns
    -------
    np.ndarray
        Positions wrapped into the box, shape ``(n, 3)``.
    """
    rng = np.random.default_rng(seed)
    n_core, n_edge = n // 3, n // 3
    # A dense core, a clump centred on a corner of the box, and a uniform field
    radius = 0.01 * boxsize * rng.random(n_core) ** 3
    direction = rng.normal(size=(n_core, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    core = 0.4 * boxsize + direction * radius[:, None]
    edge = rng.normal(0.0, 0.03 * boxsize, (n_edge, 3))
    field = rng.random((n - n_core - n_edge, 3)) * boxsize
    return np.concatenate([core, edge, field]) % boxsize


def _scipy_distances(
    positions: np.ndarray, k: int, boxsize: float | None
) -> np.ndarray:
    """
    Compute the k-th neighbour distances with scipy.

    Parameters
    ----------
    positions : np.ndarray
        Particle positions.
    k : int
        The neighbour to find.
    boxsize : float or None
        Side of the periodic box, or ``None`` for no periodicity.

    Returns
    -------
    np.ndarray
        The distances.
    """
    spatial = pytest.importorskip("scipy.spatial")
    tree = spatial.cKDTree(positions, boxsize=boxsize)
    return tree.query(positions, k=[k])[0][:, 0]


# Coordinates are shifted before building the tree (moving the periodic boundary
# into a gap, centring free axes), which rounds them by about one ulp of the box
DISTANCE_ATOL = 1e-14


@pytest.mark.parametrize("boxsize", [10.0, None])
@pytest.mark.parametrize("k", [1, 2, 7, 40, 150])
@pytest.mark.parametrize("heap_k", [0, HEAP_K])
def test_k_nearest_distances_match_scipy(boxsize, k, heap_k):
    """Check the k-d tree distances against scipy, by heap and by shells."""
    positions = _clumpy_particles(6000, 10.0)
    distances = k_nearest_distances(positions, k, boxsize, heap_k=heap_k)
    np.testing.assert_allclose(
        distances,
        _scipy_distances(positions, k, boxsize),
        rtol=1e-12,
        atol=DISTANCE_ATOL * 10.0,
    )


@pytest.mark.parametrize("boxsize", [10.0, None])
def test_k_nearest_distances_duplicates(boxsize):
    """Check coincident particles (k-th distance 0 for clumps of k or more)."""
    rng = np.random.default_rng(2)
    clumps = np.repeat(rng.random((50, 3)) * 10.0, rng.integers(1, 40, 50), axis=0)
    positions = np.concatenate([clumps, rng.random((2000, 3)) * 10.0])
    for k in (5, 20, 120):
        for heap_k in (0, HEAP_K):
            np.testing.assert_allclose(
                k_nearest_distances(positions, k, boxsize, heap_k=heap_k),
                _scipy_distances(positions, k, boxsize),
                rtol=1e-12,
                atol=DISTANCE_ATOL * 10.0,
            )


def test_k_nearest_distances_mixed_periodicity():
    """Check a box periodic along some axes only (a period of 0 is free)."""
    positions = _clumpy_particles(4000, 10.0, seed=3)
    distances = k_nearest_distances(positions, 32, np.array([10.0, 0.0, 10.0]))
    # scipy treats a boxsize of 0 along an axis as non-periodic too
    np.testing.assert_allclose(
        distances,
        _scipy_distances(positions, 32, np.array([10.0, 0.0, 10.0])),
        rtol=1e-12,
        atol=DISTANCE_ATOL * 10.0,
    )


def test_k_nearest_distances_small_inputs():
    """Check empty inputs, k = 1, and fewer particles than k."""
    assert k_nearest_distances(np.zeros((0, 3)), 4).shape == (0,)
    positions = np.random.default_rng(4).random((10, 3))
    np.testing.assert_array_equal(k_nearest_distances(positions, 1, 1.0), 0.0)
    assert np.all(np.isinf(k_nearest_distances(positions, 11, 1.0)))
    np.testing.assert_allclose(
        k_nearest_distances(positions, 10, 1.0),
        _scipy_distances(positions, 10, 1.0),
        rtol=1e-12,
        atol=DISTANCE_ATOL,
    )


def test_k_nearest_distances_threads():
    """Check that the distances do not depend on the number of threads."""
    positions = _clumpy_particles(20000, 10.0, seed=5)
    parallel = k_nearest_distances(positions, 32, 10.0)
    threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        serial = k_nearest_distances(positions, 32, 10.0)
    finally:
        numba.set_num_threads(threads)
    np.testing.assert_array_equal(parallel, serial)


@pytest.mark.parametrize("speedup_fac", [1, 2, 4])
@pytest.mark.parametrize("dimension", [2, 3])
def test_generate_smoothing_lengths_match_scipy(speedup_fac, dimension):
    """Check generate_smoothing_lengths against the scipy search it replaced."""
    positions = _clumpy_particles(5000, 20.0, seed=6)
    coordinates = cosmo_array(
        positions, u.Mpc, comoving=True, scale_factor=0.5, scale_exponent=1
    )
    boxsize = cosmo_array(
        [20.0] * 3, u.Mpc, comoving=True, scale_factor=0.5, scale_exponent=1
    )
    kernel_gamma = 1.936492
    neighbours = 48
    smoothing_lengths = generate_smoothing_lengths(
        coordinates,
        boxsize,
        kernel_gamma,
        neighbours=neighbours,
        speedup_fac=speedup_fac,
        dimension=dimension,
    )
    expected = _scipy_distances(positions, neighbours // speedup_fac, 20.0)
    expected *= speedup_fac ** (1 / dimension) / kernel_gamma
    assert smoothing_lengths.dtype == np.float32
    assert smoothing_lengths.units == u.Mpc
    np.testing.assert_allclose(smoothing_lengths.value, expected, rtol=1e-6)
