SPH kernels
===========

The smoothed visualisation routines (projections, slices and volume renders)
deposit each particle with an SPH kernel. :mod:`swiftsimio` supports the same
six kernels that SWIFT can be configured with:

=====================  ==============================  ====================  ====================
``kernel=``            SWIFT ``Kernel function``       :math:`\gamma` 3D     :math:`\gamma` 2D
=====================  ==============================  ====================  ====================
``"wendland_C2"``      ``Wendland C2``                 1.936492              1.897367
``"cubic_spline"``     ``Cubic spline (M4)``           1.825742              1.778002
``"quartic_spline"``   ``Quartic spline (M5)``         2.018932              1.977173
``"quintic_spline"``   ``Quintic spline (M6)``         2.195775              2.158131
``"wendland_C4"``      ``Wendland C4``                 2.207940              2.171239
``"wendland_C6"``      ``Wendland C6``                 2.449490              2.415230
=====================  ==============================  ====================  ====================

Each kernel has compact support radius :math:`H = \gamma h`, where :math:`h` is
the smoothing length stored in the snapshot, and is normalised to integrate to
one (Dehnen & Aly 2012). Slices and volume renders use the 3D kernel;
projections use the 2D form of the kernel with the 2D :math:`\gamma`, as
:mod:`swiftsimio` has always done for the Wendland-C2 kernel.

Choosing a kernel
-----------------

All of the high-level functions use the Wendland-C2 kernel unless told
otherwise. To choose another kernel, pass ``kernel``:

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation import project_gas, slice_gas, render_gas

   data = load("cosmo_volume_example.hdf5")

   image = project_gas(data, resolution=1024, kernel="quintic_spline")
   slice_image = slice_gas(data, resolution=1024, kernel="cubic_spline")
   voxels = render_gas(data, resolution=256, kernel="wendland_C4")

The SWIFT names in the table are also accepted (ignoring case, spaces, hyphens
and the bracketed ``M4``/``M5``/``M6``), so the kernel the simulation was run
with can be used directly:

.. code-block:: python

   image = project_gas(
       data, resolution=1024, kernel=data.metadata.hydro_scheme["Kernel function"]
   )

The ``kernel`` argument works with every backend that uses a kernel, including
``nested`` and ``gpu``. The ``histogram`` projection backend and the
``nearest_neighbours`` slice backend do not use a kernel and ignore it.

Low-level API
-------------

The backend functions take the kernel as an integer ``kernel_id``, its index in
:data:`swiftsimio.visualisation.kernels.kernel_names`. They run fastest when the
kernel is chosen before they are first called, using
:func:`~swiftsimio.visualisation.kernels.specialise_for_kernel` (this is what the
high-level functions do):

.. code-block:: python

   from swiftsimio.visualisation.kernels import specialise_for_kernel
   from swiftsimio.visualisation.projection_backends import backends

   scatter = specialise_for_kernel(backends["fast"], "cubic_spline")
   image = scatter(x, y, m, h, res=512)

Passing ``kernel_id`` directly, e.g. ``backends["fast"](x, y, m, h, res=512,
kernel_id=1)``, gives the same image but runs several times slower.

The kernel shapes, :math:`\gamma` values and normalisations are available from
:mod:`swiftsimio.visualisation.kernels`.
