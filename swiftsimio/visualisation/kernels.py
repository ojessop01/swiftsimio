r"""
SPH smoothing kernels available to the visualisation backends.

These are the six kernels that SWIFT can be configured with (see
``src/kernel_hydro.h`` in SWIFT). All of them are written as

.. math::

    W(r, H) = C_d \, H^{-d} \, w(r / H),

where ``H = kernel_gamma * h`` is the radius of compact support, ``h`` is the
smoothing length stored in the snapshot, ``w`` is the dimensionless shape
function (zero for ``r >= H``) and ``C_d`` normalises the kernel in ``d``
dimensions. Both ``kernel_gamma`` and ``C_d`` depend on the dimension: slices and
volume renders use the 3D values, projections use the 2D values (the kernel is
projected by using its 2D form, as swiftsimio has always done for Wendland-C2).

Backends take the kernel as an integer ``kernel_id`` (an index into the tables
below); use :func:`get_kernel_id` to turn a kernel name into its id, and
:func:`specialise_for_kernel` to get a version of a backend that runs the chosen
kernel as fast as the default one.
"""

from types import FunctionType
from typing import Callable

import numpy as np

try:
    from numba import jit, types
    from numba.extending import overload, register_jitable

    NUMBA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NUMBA_AVAILABLE = False

# The kernel shapes are plain Python functions that numba can also compile, both
# inside the CPU backends and in the GPU backend.
_jitable = register_jitable(fastmath=True) if NUMBA_AVAILABLE else (lambda f: f)

__all__ = [
    "kernel_names",
    "kernel_gamma_3d",
    "kernel_gamma_2d",
    "kernel_constant_3d",
    "kernel_constant_2d",
    "get_kernel_id",
    "get_kernel_name",
    "kernel_shape",
    "specialise_for_kernel",
]

#: Canonical kernel names, in ``kernel_id`` order.
kernel_names = (
    "wendland_C2",
    "cubic_spline",
    "quartic_spline",
    "quintic_spline",
    "wendland_C4",
    "wendland_C6",
)

# Ratio H / h of the compact support radius to the smoothing length, taken from
# SWIFT (Dehnen & Aly 2012 definition of h).
kernel_gamma_3d = np.array(
    [1.936492, 1.825742, 2.018932, 2.195775, 2.207940, 2.449490], dtype=np.float64
)
kernel_gamma_2d = np.array(
    [1.897367, 1.778002, 1.977173, 2.158131, 2.171239, 2.415230], dtype=np.float32
)

# Normalisation constants C_d such that the kernel integrates to unity.
kernel_constant_3d = np.array(
    [
        21.0 * 0.31830988618379067154 / 2.0,
        16.0 / np.pi,
        15625.0 / (512.0 * np.pi),
        2187.0 / (40.0 * np.pi),
        495.0 / (32.0 * np.pi),
        1365.0 / (64.0 * np.pi),
    ],
    dtype=np.float64,
)
kernel_constant_2d = np.array(
    [
        2.22817109,
        80.0 / (7.0 * np.pi),
        46875.0 / (2398.0 * np.pi),
        15309.0 / (478.0 * np.pi),
        9.0 / np.pi,
        78.0 / (7.0 * np.pi),
    ],
    dtype=np.float64,
)

# Accepted spellings, including the "Kernel function" strings SWIFT writes to the
# HydroScheme group of its snapshots.
_aliases = {
    "wendlandc2": 0,
    "c2": 0,
    "cubicspline": 1,
    "cubicsplinem4": 1,
    "cubic": 1,
    "m4": 1,
    "quarticspline": 2,
    "quarticsplinem5": 2,
    "quartic": 2,
    "m5": 2,
    "quinticspline": 3,
    "quinticsplinem6": 3,
    "quintic": 3,
    "m6": 3,
    "wendlandc4": 4,
    "c4": 4,
    "wendlandc6": 5,
    "c6": 5,
}


def get_kernel_id(kernel: str | int | bytes) -> int:
    """
    Convert a kernel name to the integer id used by the visualisation backends.

    Parameters
    ----------
    kernel : str, bytes or int
        Kernel name, e.g. ``"wendland_C2"``, ``"cubic_spline"`` or the
        ``"Kernel function"`` string stored in SWIFT snapshot metadata (e.g.
        ``"Quartic spline (M5)"``). Case, spaces, hyphens, underscores and
        brackets are ignored. An integer is checked and returned unchanged.

    Returns
    -------
    int
        Index of the kernel in :data:`kernel_names`.

    Raises
    ------
    ValueError
        If the kernel is not recognised.
    """
    if isinstance(kernel, (int, np.integer)):
        if 0 <= kernel < len(kernel_names):
            return int(kernel)
        raise ValueError(f"Kernel id {kernel} out of range [0, {len(kernel_names)}).")
    if isinstance(kernel, bytes):
        kernel = kernel.decode("utf-8")
    key = "".join(c for c in str(kernel).lower() if c.isalnum())
    try:
        return _aliases[key]
    except KeyError:
        raise ValueError(
            f"Unknown SPH kernel '{kernel}'. Available kernels: {kernel_names}."
        ) from None


def get_kernel_name(kernel: str | int | bytes) -> str:
    """
    Return the canonical name of a kernel.

    Parameters
    ----------
    kernel : str, bytes or int
        Any spelling accepted by :func:`get_kernel_id`.

    Returns
    -------
    str
        The canonical name, an entry of :data:`kernel_names`.
    """
    return kernel_names[get_kernel_id(kernel)]


@_jitable
def _shape_wendland_C2(ratio: float) -> float:
    """
    Wendland-C2 shape, ``(1 - q)^4 (1 + 4q)``.

    Parameters
    ----------
    ratio : float
        ``q = r / H``, in ``[0, 1)``.

    Returns
    -------
    float
        The kernel shape at ``ratio``.
    """
    one_minus_ratio = 1.0 - ratio
    one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
    one_minus_ratio_4 = one_minus_ratio_2 * one_minus_ratio_2
    return max(one_minus_ratio_4 * (1.0 + 4.0 * ratio), 0.0)


@_jitable
def _shape_cubic_spline(ratio: float) -> float:
    """
    Cubic spline (M4) shape, ``(1 - q)^3 - 4 (1/2 - q)_+^3``.

    Parameters
    ----------
    ratio : float
        ``q = r / H``, in ``[0, 1)``.

    Returns
    -------
    float
        The kernel shape at ``ratio``.
    """
    one_minus_ratio = 1.0 - ratio
    kernel = one_minus_ratio * one_minus_ratio * one_minus_ratio
    if ratio < 0.5:
        a = 0.5 - ratio
        kernel -= 4.0 * a * a * a
    return kernel


@_jitable
def _shape_quartic_spline(ratio: float) -> float:
    """
    Quartic spline (M5) shape, ``(1 - q)^4 - 5 (3/5 - q)_+^4 + 10 (1/5 - q)_+^4``.

    Parameters
    ----------
    ratio : float
        ``q = r / H``, in ``[0, 1)``.

    Returns
    -------
    float
        The kernel shape at ``ratio``.
    """
    one_minus_ratio = 1.0 - ratio
    one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
    kernel = one_minus_ratio_2 * one_minus_ratio_2
    if ratio < 0.6:
        a = 0.6 - ratio
        a_2 = a * a
        kernel -= 5.0 * a_2 * a_2
        if ratio < 0.2:
            b = 0.2 - ratio
            b_2 = b * b
            kernel += 10.0 * b_2 * b_2
    return kernel


@_jitable
def _shape_quintic_spline(ratio: float) -> float:
    """
    Quintic spline (M6) shape, ``(1 - q)^5 - 6 (2/3 - q)_+^5 + 15 (1/3 - q)_+^5``.

    Parameters
    ----------
    ratio : float
        ``q = r / H``, in ``[0, 1)``.

    Returns
    -------
    float
        The kernel shape at ``ratio``.
    """
    one_minus_ratio = 1.0 - ratio
    one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
    kernel = one_minus_ratio_2 * one_minus_ratio_2 * one_minus_ratio
    if ratio < 2.0 / 3.0:
        a = 2.0 / 3.0 - ratio
        a_2 = a * a
        kernel -= 6.0 * a_2 * a_2 * a
        if ratio < 1.0 / 3.0:
            b = 1.0 / 3.0 - ratio
            b_2 = b * b
            kernel += 15.0 * b_2 * b_2 * b
    return kernel


@_jitable
def _shape_wendland_C4(ratio: float) -> float:
    """
    Wendland-C4 shape, ``(1 - q)^6 (1 + 6q + 35/3 q^2)``.

    Parameters
    ----------
    ratio : float
        ``q = r / H``, in ``[0, 1)``.

    Returns
    -------
    float
        The kernel shape at ``ratio``.
    """
    one_minus_ratio = 1.0 - ratio
    one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
    one_minus_ratio_6 = one_minus_ratio_2 * one_minus_ratio_2 * one_minus_ratio_2
    return one_minus_ratio_6 * (1.0 + ratio * (6.0 + ratio * (35.0 / 3.0)))


@_jitable
def _shape_wendland_C6(ratio: float) -> float:
    """
    Wendland-C6 shape, ``(1 - q)^8 (1 + 8q + 25q^2 + 32q^3)``.

    Parameters
    ----------
    ratio : float
        ``q = r / H``, in ``[0, 1)``.

    Returns
    -------
    float
        The kernel shape at ``ratio``.
    """
    one_minus_ratio = 1.0 - ratio
    one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
    one_minus_ratio_4 = one_minus_ratio_2 * one_minus_ratio_2
    return (
        one_minus_ratio_4
        * one_minus_ratio_4
        * (1.0 + ratio * (8.0 + ratio * (25.0 + ratio * 32.0)))
    )


_shapes = (
    _shape_wendland_C2,
    _shape_cubic_spline,
    _shape_quartic_spline,
    _shape_quintic_spline,
    _shape_wendland_C4,
    _shape_wendland_C6,
)


def kernel_shape(ratio: float, kernel_id: int) -> float:
    """
    Evaluate the dimensionless kernel shape ``w(r / H)``.

    The shape is the same in 2D and 3D; only the normalisation and
    ``kernel_gamma`` differ. In a backend returned by
    :func:`specialise_for_kernel`, numba compiles only the chosen kernel's
    formula.

    Parameters
    ----------
    ratio : float
        ``r / H``, the distance in units of the compact support radius.

    kernel_id : int
        Index of the kernel in :data:`kernel_names`.

    Returns
    -------
    float
        ``w(ratio)``, which is zero for ``ratio >= 1``.

    Notes
    -----
    The splines are written as sums of terms ``(a - ratio)**n`` (each used
    only where ``ratio < a``) and the Wendland kernels as products. Both are
    equal to the polynomials in SWIFT, but lose less precision near
    ``ratio = 1`` than the expanded polynomials do.
    """
    if ratio >= 1.0:
        return 0.0 * ratio
    if kernel_id == 0:
        return _shape_wendland_C2(ratio)
    elif kernel_id == 1:
        return _shape_cubic_spline(ratio)
    elif kernel_id == 2:
        return _shape_quartic_spline(ratio)
    elif kernel_id == 3:
        return _shape_quintic_spline(ratio)
    elif kernel_id == 4:
        return _shape_wendland_C4(ratio)
    return _shape_wendland_C6(ratio)


if NUMBA_AVAILABLE:

    @overload(kernel_shape, jit_options={"fastmath": True})
    def _kernel_shape_overload(ratio: float, kernel_id: int) -> Callable:
        """
        Choose how numba compiles :func:`kernel_shape`.

        When numba already knows which kernel will be used (because
        ``kernel_id`` was left at its default, see
        :func:`specialise_for_kernel`), only that kernel's formula is compiled.
        Otherwise the kernel is chosen each time the function is called.

        Numba passes the types of the arguments rather than their values; the
        parameter names and annotations have to match :func:`kernel_shape`.

        Parameters
        ----------
        ratio : float
            Numba type of ``ratio``.

        kernel_id : int
            Numba type of ``kernel_id``.

        Returns
        -------
        Callable
            The function numba should compile for these argument types.
        """
        if not isinstance(kernel_id, types.Omitted):
            return kernel_shape
        shape = _shapes[kernel_id.value]

        def specialised(ratio: float, kernel_id: int) -> float:
            if ratio >= 1.0:
                return 0.0 * ratio
            return shape(ratio)

        return specialised


_specialisations = {}


def specialise_for_kernel(backend: Callable, kernel_id: int) -> Callable:
    """
    Return a version of a backend that uses a given kernel by default.

    The backends accept ``kernel_id`` as an argument, but numba produces much
    faster code when the kernel is fixed before the backend is compiled. Numba
    treats arguments that are left at their default value as fixed, so this
    returns a copy of ``backend`` whose default ``kernel_id`` is the requested
    kernel. Call the result *without* passing ``kernel_id``. Each copy is made
    (and compiled by numba) once, the first time it is needed. For the default
    kernel, ``backend`` itself is returned.

    Parameters
    ----------
    backend : Callable
        A visualisation backend with a ``kernel_id`` argument that defaults to
        ``0``. It may be a numba-compiled function or a plain Python function.

    kernel_id : int
        The kernel to use, as an id or any name accepted by
        :func:`get_kernel_id`.

    Returns
    -------
    Callable
        ``backend``, or a copy of it that uses ``kernel_id`` by default.

    Raises
    ------
    ValueError
        If ``backend`` has no ``kernel_id`` argument and a kernel other than
        Wendland-C2 is requested.
    """
    kernel_id = get_kernel_id(kernel_id)
    if kernel_id == 0:
        return backend
    key = (backend, kernel_id)
    if key not in _specialisations:
        python_function = getattr(backend, "py_func", backend)
        names = python_function.__code__.co_varnames[
            : python_function.__code__.co_argcount
        ]
        if "kernel_id" not in names:
            raise ValueError(
                f"The {python_function.__module__}.{python_function.__name__} "
                f"backend only supports the {kernel_names[0]} kernel, not "
                f"{kernel_names[kernel_id]}."
            )
        defaults = python_function.__defaults__
        first_default = len(names) - len(defaults)
        copy = FunctionType(
            python_function.__code__,
            python_function.__globals__,
            python_function.__name__,
            tuple(
                kernel_id if name == "kernel_id" else value
                for name, value in zip(names[first_default:], defaults)
            ),
            python_function.__closure__,
        )
        copy.__doc__ = python_function.__doc__
        copy.__qualname__ = python_function.__qualname__
        copy.__annotations__ = dict(python_function.__annotations__)
        if hasattr(backend, "targetoptions"):
            copy = jit(**backend.targetoptions)(copy)
        _specialisations[key] = copy
    return _specialisations[key]
