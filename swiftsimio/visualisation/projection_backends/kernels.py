"""Projection kernels."""

import numpy as np
from swiftsimio.accelerated import jit
from swiftsimio.visualisation.kernels import (
    kernel_shape,
    kernel_gamma_2d as kernel_gammas,
    kernel_constant_2d,
)

__all__ = [
    "kernel_gamma",
    "kernel_constant",
    "kernel_gammas",
    "kernel_constants_single",
    "kernel_constants_double",
    "kernel_single_precision",
    "kernel_double_precision",
]

# Taken from Dehnen & Aly 2012; these are the Wendland-C2 (default) values.
kernel_gamma = np.float32(1.897367)
kernel_constant = np.float32(7.0 / 3.14159)

# 2D normalisations of each kernel, indexed by kernel_id.
kernel_constants_single = kernel_constant_2d.astype(np.float32)
kernel_constants_double = kernel_constant_2d.astype(np.float64)


@jit(nopython=True, fastmath=True)
def kernel_single_precision(
    r: np.float32, H: np.float32, kernel_id: int = 0
) -> np.float32:
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

    See Also
    --------
    kernel_double_precision

    References
    ----------
    .. [1] Dehnen W., Aly H., 2012, MNRAS, 425, 1068
    """
    # Single precision in and out, whatever the caller passes.
    r = np.float32(r)
    H = np.float32(H)
    kernel_constant = kernel_constants_single[kernel_id]

    inverse_H = 1.0 / H
    ratio = r * inverse_H

    kernel = 0.0

    if ratio < 1.0:
        if kernel_id == 0:
            # Wendland-C2, written out as before other kernels were added: going
            # through kernel_shape makes the fast backend about 2% slower.
            one_minus_ratio = 1.0 - ratio
            one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
            one_minus_ratio_4 = one_minus_ratio_2 * one_minus_ratio_2
            kernel = max(one_minus_ratio_4 * (1.0 + 4.0 * ratio), 0.0)
        else:
            kernel = kernel_shape(ratio, kernel_id)

        kernel *= kernel_constant * inverse_H * inverse_H

    return np.float32(kernel)


@jit(nopython=True, fastmath=True)
def kernel_double_precision(
    r: np.float64, H: np.float64, kernel_id: int = 0
) -> np.float64:
    """
    Double precision kernel implementation for swiftsimio.

    This is the 2D form of the kernel selected by ``kernel_id`` (see
    :mod:`swiftsimio.visualisation.kernels`); the default is the Wendland-C2
    kernel as shown in Denhen & Aly (2012) [2]_.

    Parameters
    ----------
    r : np.float64
        Radius used in kernel computation.

    H : np.float64
        Kernel width (i.e. radius of compact support for the kernel).

    kernel_id : int
        Index of the kernel in :data:`swiftsimio.visualisation.kernels.kernel_names`.

    Returns
    -------
    np.float64
        Contribution to the density by the particle.

    See Also
    --------
    kernel_single_precision

    References
    ----------
    .. [2] Dehnen W., Aly H., 2012, MNRAS, 425, 1068
    """
    # Double precision in and out, whatever the caller passes.
    r = np.float64(r)
    H = np.float64(H)
    kernel_constant = kernel_constants_double[kernel_id]

    inverse_H = 1.0 / H
    ratio = r * inverse_H

    kernel = 0.0

    if ratio < 1.0:
        if kernel_id == 0:
            # Wendland-C2, written out as before other kernels were added: going
            # through kernel_shape makes the fast backend about 2% slower.
            one_minus_ratio = 1.0 - ratio
            one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
            one_minus_ratio_4 = one_minus_ratio_2 * one_minus_ratio_2
            kernel = max(one_minus_ratio_4 * (1.0 + 4.0 * ratio), 0.0)
        else:
            kernel = kernel_shape(ratio, kernel_id)

        kernel *= kernel_constant * inverse_H * inverse_H

    return np.float64(kernel)
