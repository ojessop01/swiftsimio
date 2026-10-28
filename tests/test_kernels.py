"""Tests of the selectable SPH kernels used by the visualisation tools."""

import numpy as np
import pytest

from swiftsimio import load
from swiftsimio.visualisation import project_gas, render_gas, slice_gas
from swiftsimio.visualisation.kernels import (
    get_kernel_id,
    get_kernel_name,
    kernel_constant_2d,
    kernel_constant_3d,
    kernel_gamma_2d,
    kernel_gamma_3d,
    kernel_names,
    kernel_shape,
    specialise_for_kernel,
)
from swiftsimio.visualisation.projection_backends import (
    backends as projection_backends,
)
from swiftsimio.visualisation.slice_backends import (
    backends as slice_backends,
)
from swiftsimio.visualisation.volume_render_backends import (
    backends as volume_render_backends,
)

kernel_ids = list(range(len(kernel_names)))

# Piecewise polynomial coefficients (highest power first) from SWIFT's
# src/kernel_hydro.h, as (number of intervals, coefficients per interval).
swift_coefficients = {
    "wendland_C2": (1, [[4.0, -15.0, 20.0, -10.0, 0.0, 1.0]]),
    "cubic_spline": (2, [[3.0, -3.0, 0.0, 0.5], [-1.0, 3.0, -3.0, 1.0]]),
    "quartic_spline": (
        5,
        [
            [6.0, 0.0, -2.4, 0.0, 0.368],
            [-4.0, 8.0, -4.8, 0.32, 0.352],
            [-4.0, 8.0, -4.8, 0.32, 0.352],
            [1.0, -4.0, 6.0, -4.0, 1.0],
            [1.0, -4.0, 6.0, -4.0, 1.0],
        ],
    ),
    "quintic_spline": (
        3,
        [
            [-10.0, 10.0, 0.0, -20.0 / 9.0, 0.0, 22.0 / 81.0],
            [5.0, -15.0, 50.0 / 3.0, -70.0 / 9.0, 25.0 / 27.0, 17.0 / 81.0],
            [-1.0, 5.0, -10.0, 10.0, -5.0, 1.0],
        ],
    ),
    "wendland_C4": (
        1,
        [[35.0 / 3.0, -64.0, 140.0, -448.0 / 3.0, 70.0, 0.0, -28.0 / 3.0, 0.0, 1.0]],
    ),
    "wendland_C6": (
        1,
        [
            [
                32.0,
                -231.0,
                704.0,
                -1155.0,
                1056.0,
                -462.0,
                0.0,
                66.0,
                0.0,
                -11.0,
                0.0,
                1.0,
            ]
        ],
    ),
}


def _swift_shape(name: str, ratio: np.ndarray) -> np.ndarray:
    """
    Evaluate SWIFT's piecewise-polynomial form of a kernel shape.

    Parameters
    ----------
    name : str
        Canonical kernel name.

    ratio : np.ndarray
        Values of ``r / H`` in ``[0, 1)``.

    Returns
    -------
    np.ndarray
        The kernel shape ``w(r / H)``.
    """
    n_intervals, coefficients = swift_coefficients[name]
    interval = np.minimum((ratio * n_intervals).astype(int), n_intervals - 1)
    out = np.empty_like(ratio)
    for i, c in enumerate(coefficients):
        sel = interval == i
        out[sel] = np.polyval(c, ratio[sel])
    return out


def _shape(kernel_id: int, ratio: np.ndarray) -> np.ndarray:
    """
    Evaluate :func:`kernel_shape` on an array.

    Parameters
    ----------
    kernel_id : int
        Kernel index.

    ratio : np.ndarray
        Values of ``r / H``.

    Returns
    -------
    np.ndarray
        The kernel shape ``w(r / H)``.
    """
    return np.array([kernel_shape(q, kernel_id) for q in ratio])


@pytest.mark.parametrize("kernel_id", kernel_ids)
def test_shape_matches_swift(kernel_id):
    """The kernel shapes agree with the polynomials in SWIFT's kernel_hydro.h."""
    ratio = np.linspace(0.0, 0.999, 2000)
    np.testing.assert_allclose(
        _shape(kernel_id, ratio),
        _swift_shape(kernel_names[kernel_id], ratio),
        rtol=1e-6,
        atol=1e-9,
    )
    assert kernel_shape(1.0, kernel_id) == 0.0
    assert kernel_shape(1.3, kernel_id) == 0.0


@pytest.mark.parametrize("kernel_id", kernel_ids)
def test_normalisation_and_gamma(kernel_id):
    """Kernels integrate to one, and h = H / gamma is twice the kernel's sigma."""
    ratio = np.linspace(0.0, 1.0, 200001)
    w = _shape(kernel_id, ratio)
    c3 = kernel_constant_3d[kernel_id]
    c2 = kernel_constant_2d[kernel_id]

    assert np.trapezoid(4 * np.pi * ratio**2 * c3 * w, ratio) == pytest.approx(
        1.0, rel=1e-6
    )
    assert np.trapezoid(2 * np.pi * ratio * c2 * w, ratio) == pytest.approx(
        1.0, rel=2e-6
    )

    # Dehnen & Aly (2012) / SWIFT definition of the smoothing length: h = 2 sigma.
    sigma_3d = np.sqrt(np.trapezoid(4 * np.pi * ratio**4 * c3 * w, ratio) / 3)
    sigma_2d = np.sqrt(np.trapezoid(2 * np.pi * ratio**3 * c2 * w, ratio) / 2)
    assert 1.0 / kernel_gamma_3d[kernel_id] == pytest.approx(2 * sigma_3d, rel=1e-5)
    assert 1.0 / kernel_gamma_2d[kernel_id] == pytest.approx(2 * sigma_2d, rel=1e-5)


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("wendland_C2", 0),
        (b"Wendland C2", 0),
        ("Cubic spline (M4)", 1),
        ("cubic-spline", 1),
        ("Quartic spline (M5)", 2),
        ("Quintic spline (M6)", 3),
        ("Wendland C4", 4),
        ("WENDLAND_C6", 5),
        (3, 3),
    ],
)
def test_get_kernel_id(spelling, expected):
    """Kernel names, including SWIFT's metadata strings, map to the right id."""
    assert get_kernel_id(spelling) == expected
    assert get_kernel_name(spelling) == kernel_names[expected]


@pytest.mark.parametrize("bad", ["gaussian", 6, -1])
def test_get_kernel_id_invalid(bad):
    """Unknown kernels are rejected."""
    with pytest.raises(ValueError):
        get_kernel_id(bad)


@pytest.mark.parametrize("kernel_id", kernel_ids)
def test_projection_kernel_profile(kernel_id):
    """A single resolved particle projects to the 2D kernel profile."""
    res = 256
    h = np.float32(0.1)
    centre = (128 + 0.5) / res
    image = projection_backends["reference"](
        np.array([centre]),
        np.array([centre]),
        np.array([1.0], dtype=np.float32),
        np.array([h]),
        res,
        kernel_id=kernel_id,
    )
    H = float(kernel_gamma_2d[kernel_id]) * h
    offsets = np.arange(0, 40)
    expected = (
        kernel_constant_2d[kernel_id] / H**2 * _shape(kernel_id, (offsets / res) / H)
    )
    np.testing.assert_allclose(image[128 + offsets, 128], expected, rtol=1e-6)


@pytest.mark.parametrize("kernel_id", kernel_ids)
def test_slice_kernel_profile(kernel_id):
    """A slice through a single particle follows the 3D kernel profile."""
    res = 128
    h = np.float32(0.1)
    centre = (64 + 0.5) / res
    image = slice_backends["sph"](
        np.array([centre]),
        np.array([centre]),
        np.array([0.5]),
        np.array([1.0], dtype=np.float32),
        np.array([h]),
        0.52,
        res,
        res,
        kernel_id=kernel_id,
    )
    H = kernel_gamma_3d[kernel_id] * h
    offsets = np.arange(0, 20)
    r = np.sqrt((offsets / res) ** 2 + 0.02**2)
    expected = kernel_constant_3d[kernel_id] / H**3 * _shape(kernel_id, r / H)
    np.testing.assert_allclose(image[64 + offsets, 64], expected, rtol=1e-5)


@pytest.mark.parametrize("kernel_id", kernel_ids[1:])
def test_specialised_backends_match_direct_calls(kernel_id):
    """Backends specialised for a kernel agree with passing ``kernel_id`` directly."""
    rng = np.random.default_rng(7)
    x, y, z = rng.random((3, 300))
    m = rng.random(300).astype(np.float32)
    h = rng.uniform(0.01, 0.08, 300).astype(np.float32)
    cases = [
        (projection_backends["fast"], (x, y, m, h, 64, 1.0, 1.0)),
        (projection_backends["nested"], (x, y, m, h, 64, 1.0, 1.0)),
        (slice_backends["sph"], (x, y, z, m, h, 0.5, 64, 64, 1.0, 1.0, 1.0)),
        (volume_render_backends["scatter"], (x, y, z, m, h, 16, 1.0, 1.0, 1.0)),
        (volume_render_backends["nested"], (x, y, z, m, h, 16, 1.0, 1.0, 1.0)),
    ]
    for backend, args in cases:
        specialised = specialise_for_kernel(backend, kernel_names[kernel_id])
        assert specialise_for_kernel(backend, kernel_id) is specialised
        np.testing.assert_allclose(
            specialised(*args), backend(*args, kernel_id=kernel_id), rtol=1e-5
        )
    assert (
        specialise_for_kernel(projection_backends["fast"], 0)
        is (projection_backends["fast"])
    )


def test_frontends_accept_kernel(cosmological_volume_only_single_local):
    """The kernel argument reaches the backends of all three frontends."""
    data = load(cosmological_volume_only_single_local)
    snapshot_kernel = data.metadata.hydro_scheme["Kernel function"]

    for function, resolution in [(project_gas, 64), (slice_gas, 64), (render_gas, 16)]:
        default = function(data, resolution)
        np.testing.assert_array_equal(
            default, function(data, resolution, kernel="wendland_C2")
        )
        assert not np.allclose(
            default, function(data, resolution, kernel="wendland_C6")
        )
        # SWIFT's metadata string selects the kernel the simulation used
        np.testing.assert_array_equal(
            function(data, resolution, kernel=snapshot_kernel),
            function(data, resolution, kernel=get_kernel_name(snapshot_kernel)),
        )

    # Backends without a kernel ignore the argument
    np.testing.assert_array_equal(
        project_gas(data, 64, backend="histogram", kernel="wendland_C6"),
        project_gas(data, 64, backend="histogram"),
    )


def test_backend_without_kernel_choice():
    """A backend without a kernel_id argument rejects other kernels."""
    histogram = projection_backends["histogram"]
    assert specialise_for_kernel(histogram, "wendland_C2") is histogram
    with pytest.raises(ValueError, match="only supports"):
        specialise_for_kernel(histogram, "cubic_spline")
