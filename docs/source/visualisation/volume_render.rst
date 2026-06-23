Volume Rendering
================

The :mod:`swiftsimio.visualisation.volume_render` sub-module provides an
interface to render SWIFT data onto a fixed grid. This takes your 3D data and
finds the 3D density at fixed positions, allowing it to be used in codes that
require fixed grids such as radiative transfer programs.

This effectively solves the equation:

:math:`\tilde{A}_i = \sum_j A_j W_{ij, 3D}`

with :math:`\tilde{A}_i` the smoothed quantity in pixel :math:`i`, and
:math:`j` all particles in the simulation, with :math:`W` the 3D kernel.
Here we use the Wendland-C2 kernel.

The primary function here is
:func:`swiftsimio.visualisation.volume_render.render_gas`, which allows you
to create a gas density grid of any field, see the example below.

Example
-------

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.volume_render import render_gas

   data = load("cosmo_volume_example.hdf5")

   # This creates a grid that has units msun / Mpc^3, and can be transformed like
   # any other unyt quantity.
   mass_grid = render_gas(
       data,
       resolution=256,
       project="masses",
       parallel=True,
       periodic=True,
   )

This basic demonstration creates a mass density cube.

To create, for example, a projected temperature cube, we need to remove the
density dependence (i.e. :func:`~swiftsimio.visualisation.volume_render.render_gas` returns a volumetric
temperature in units of K / kpc^3 and we just want K) by dividing out by
this:

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.volume_render import render_gas

   data = load("cosmo_volume_example.hdf5")

   # First create a mass-weighted temperature dataset
   data.gas.mass_weighted_temps = data.gas.masses * data.gas.temperatures

   # Map in msun / mpc^3
   mass_cube = render_gas(
       data,
       resolution=256,
       project="masses",
       parallel=True,
       periodic=True,
   )

   # Map in msun * K / mpc^3
   mass_weighted_temp_cube = render_gas(
       data,
       resolution=256,
       project="mass_weighted_temps",
       parallel=True,
       periodic=True,
   )

   # A 256 x 256 x 256 cube with dimensions of temperature
   temp_cube = mass_weighted_temp_cube / mass_cube

Periodic boundaries
-------------------

Cosmological simulations and many other simulations use periodic boundary
conditions. This has implications for the particles at the edge of the
simulation box: they can contribute to voxels on multiple sides of the image.
If this effect is not taken into account, then the voxels close to the edge
will have values that are too low because of missing contributions.

All visualisation functions by default assume a periodic box. Rather than
simply summing each individual particle once, eight additional periodic copies
of each particle are also taken into account. Most copies will contribute
outside the valid voxel range, but the copies that do not ensure that voxels
close to the edge receive all necessary contributions. Thanks to :mod:`numba`
optimisations, the overhead of these additional copies is relatively small.

There are some caveats with this approach. If you try to visualise a subset of
the particles in the box (e.g. using a mask), then only periodic copies of
particles in this subset will be used. If the subset does not include particles
on the other side of the periodic boundary, then these will still be missing
from the voxel cube. The same is true if you visualise a region of the box.
The periodic boundary wrapping is also not compatible with rotations (see below)
and should therefore not be used together with a rotation.

Rotations
---------

Rotations of the box prior to volume rendering are provided in a similar fashion
to the :mod:`swiftsimio.visualisation.projection` sub-module, by using the
:mod:`swiftsimio.visualisation.rotation` sub-module. To rotate the perspective
prior to slicing a ``rotation_center`` argument in :func:`~swiftsimio.visualisation.volume_render.render_gas` needs
to be provided, specifying the point around which the rotation takes place.
The angle of rotation is specified with a matrix, supplied by ``rotation_matrix``
in :func:`~swiftsimio.visualisation.volume_render.render_gas`. The rotation matrix may be computed with
:func:`~swiftsimio.visualisation.rotation.rotation_matrix_from_vector`. This will result in the perspective being
rotated to be along the provided vector. This approach to rotations applied to
the above example is shown below.

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.volume_render import render_gas
   from swiftsimio.visualisation.rotation import rotation_matrix_from_vector

   data = load("cosmo_volume_example.hdf5")

   # First create a mass-weighted temperature dataset
   data.gas.mass_weighted_temps = data.gas.masses * data.gas.temperatures

   # Specify the rotation parameters
   center = 0.5 * data.metadata.boxsize
   rotate_vec = [0.5,0.5,1]
   matrix = rotation_matrix_from_vector(rotate_vec, axis='z')

   # Map in msun / mpc^3
   mass_cube = render_gas(
       data,
       resolution=256,
       project="masses",
       rotation_matrix=matrix,
       rotation_center=center,
       parallel=True,
       periodic=False,  # disable periodic boundaries for rotations
   )

   # Map in msun * K / mpc^3
   mass_weighted_temp_cube = render_gas(
       data,
       resolution=256,
       project="mass_weighted_temps",
       rotation_matrix=matrix,
       rotation_center=center,
       parallel=True,
       periodic=False,
   )

   # A 256 x 256 x 256 cube with dimensions of temperature
   temp_cube = mass_weighted_temp_cube / mass_cube


Rendering
---------

We provide a volume rendering function that can be used to make images highlighting
specific density contours. The key function here is
:func:`swiftsimio.visualisation.volume_render.visualise_render`. This takes
in your volume rendering, along with a colour map and centers, to create
these highlights. The example below shows how to use this.

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.colors import LogNorm

   from swiftsimio import load
   from swiftsimio.visualisation import volume_render

   # Load the data
   data = load("eagle_6.hdf5")

   # Rough location of an interesting galaxy in the volume.
   region = [
       0.225 * data.metadata.boxsize[0],
       0.275 * data.metadata.boxsize[0],
       0.12 * data.metadata.boxsize[1],
       0.17 * data.metadata.boxsize[1],
       0.45 * data.metadata.boxsize[2],
       0.5 * data.metadata.boxsize[2],
   ]

   # Render the volume (note 1024 is reasonably high resolution so this won't complete
   # immediately; you should consider using 256, etc. for testing).
   rendered = volume_render.render_gas(data, resolution=1024, region=region, parallel=True)

   # Quick view! By projecting along the final axis you can get
   # the projected density from the rendered image.
   plt.imsave("volume_render_quick_view.png", LogNorm()(rendered.sum(-1)))

Here we can see the quick view of this image. It's just a regular density projection:

.. image:: volume_render_quick_view.png

.. code-block:: python

   # Now we will move onto the real volume rendering. Let's use the log of the density;
   # using the real density leads to low contrast images.
   log_rendered = np.log10(rendered)

   # The volume rendering function expects centers of 'bins' and widths. These
   # bins actually represent gaussian functions around a specific density (or other
   # visualization quantity). The brightest pixel value is at center. We will
   # visualise this later!
   width = 0.1
   std = np.std(log_rendered)
   mean = np.mean(log_rendered)

   # It's helpful to choose the centers relative to the data you have. When making
   # a movie, you will obviously want to choose the centers to be the same for each
   # frame.
   centers = [mean + x * std for x in [1.0, 3.0, 5.0, 7.0]]

   # This will visualize your render options. The centers are shown as gaussians and
   # vertical lines.
   fig, ax = volume_render.visualise_render_options(
       centers=centers, widths=width, cmap="viridis"
   )

   histogram, edges = np.histogram(
       log_rendered.flat,
       bins=128,
       range=(min(centers) - 5.0 * width, max(centers) + 5.0 * width),
   )
   bc = (edges[:-1] + edges[1:]) / 2.0

   # The normalization here is the height of a gaussian!
   ax.plot(bc, histogram / (np.max(histogram) * np.sqrt(2.0 * np.pi) * width))
   ax.semilogy()
   ax.set_xlabel("$\\log_{10}(\\rho)$")

   plt.savefig("volume_render_options.png")

This function :func:`swiftsimio.visualisation.volume_render.visualise_render_options` allows
you to see what densities your rendering is picking out:

.. image:: volume_render_options.png

.. code-block:: python

   # Now we can really visualize the rendering.
   img, norms = volume_render.visualise_render(
       log_rendered,
       centers,
       widths=width,
       cmap="viridis",
   )

   # Sometimes, these images can be a bit dark. You can increase the brightness using
   # tools like PIL or in your favourite image editor.
   from PIL import Image, ImageEnhance

   pilimg = Image.fromarray((img * 255.0).astype(np.uint8))
   enhanced = ImageEnhance.Contrast(ImageEnhance.Brightness(pilimg).enhance(2.0)).enhance(
       1.2
   )

   enhanced.save("volume_render_example.png")

Which produces the image:

.. image:: volume_render_example.png

Once you have this base image, you can always use your photo editor to tweak it further.
In particular, open the 'levels' panel and play around with the sliders!


Scatter backends
----------------

Two scatter backends are available via the ``backend`` argument to
:func:`~swiftsimio.visualisation.volume_render.render_gas`:

* ``"scatter"`` — the standard single-resolution backend (default).
* ``"nested"`` — the nested multi-resolution backend, described in detail
  below. This is faster for simulations with a wide range of smoothing
  lengths, such as zoom-in runs or boxes containing both high-density gas
  and diffuse IGM.

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.volume_render import render_gas

   data = load("cosmo_volume_example.hdf5")

   # Standard backend — identical results, familiar behaviour.
   mass_grid_standard = render_gas(
       data,
       resolution=256,
       project="masses",
       parallel=True,
       backend="scatter",
   )

   # Nested backend — faster for wide smoothing-length distributions.
   mass_grid_nested = render_gas(
       data,
       resolution=256,
       project="masses",
       parallel=True,
       backend="nested",
       ntarget=6,    # target voxels per kernel diameter at each level
       nlevels=4,    # number of coarsening levels (requires res % 2**nlevels == 0)
   )


Nested multi-resolution backend
--------------------------------

Motivation
~~~~~~~~~~

In the standard scatter backend every particle visits every voxel within its
kernel compact-support radius. For a particle whose smoothing length spans
:math:`N` voxels in each dimension, the cost is :math:`\mathcal{O}(N^3)`.
Simulations with large dynamic ranges in smoothing length — zoom-in
simulations, or cosmological volumes that include both dense filaments and
low-density IGM — contain particles whose kernels span tens or hundreds of
voxels. These few large particles can dominate the total rendering time even
though they carry little structural information at the target resolution.

The nested backend bounds the per-particle cost at :math:`\mathcal{O}(n_\mathrm{target}^3)`
regardless of smoothing length by scattering large particles onto a coarser
grid and upsampling afterwards. This follows the Sparse Multi-Scale Grid
approach described in Benitez-Llambay (2025).

Algorithm
~~~~~~~~~

**Step 1 — level assignment.**
For each particle, compute how many finest-grid voxels its compact-support
radius spans:

.. math::

   N_\mathrm{cells} = \gamma_k \, h \, N_\mathrm{res}

where :math:`\gamma_k = 1.936492` is the kernel gamma for the Wendland-C2
kernel and :math:`N_\mathrm{res}` is the target resolution. The particle is
assigned to level :math:`L`:

.. math::

   L = \mathrm{clip}\!\left(\left\lceil \log_2\!\left(\frac{N_\mathrm{cells}}{n_\mathrm{target}}\right) \right\rceil,\ 0,\ N_\mathrm{levels}\right)

Level 0 is the finest grid (resolution :math:`N_\mathrm{res}`); level
:math:`L` uses a grid of resolution :math:`N_\mathrm{res} / 2^L`. At
their assigned level, every particle's kernel spans approximately
:math:`n_\mathrm{target}` voxels, bounding the work per particle.

**Step 2 — scatter.**
Each particle is scattered onto its assigned coarse grid using the standard
3D Wendland-C2 kernel evaluated at the coarse voxel centres.

**Step 3 — collapse.**
Working from the coarsest occupied level back to the finest, each coarse
grid is trilinearly upsampled by a factor of two and accumulated into the
next finer grid. Trilinear upsampling weights are:

.. math::

   w_\mathrm{even} = 0.75, \quad w_\mathrm{odd} = 0.25

for each spatial dimension, applied as a separable 3D interpolation.
Only the occupied bounding box of each coarse grid is traversed during
collapse, minimising unnecessary work.

**Step 4 — return.**
The finest grid, now containing contributions from all levels, is returned.

Why is it faster?
~~~~~~~~~~~~~~~~~

The standard backend scales as :math:`\mathcal{O}(\sum_j h_j^3 N_\mathrm{res}^3)` — a
particle with a kernel spanning 32 voxels costs :math:`32^3 = 32{,}768`
kernel evaluations. The nested backend caps this at
:math:`\mathcal{O}(n_\mathrm{target}^3)` per particle: with ``ntarget=6`` the
same particle costs only :math:`6^3 = 216` evaluations on a grid
:math:`4\times` coarser, and the trilinear collapse adds a fixed overhead
that is typically much smaller. The speedup grows with the width of the
smoothing-length distribution and with resolution, making it most effective
for high-resolution renders of multi-scale simulations.

For example, on a cosmological zoom-in simulation with 50 million gas
particles rendered at :math:`512^3`, the nested backend is typically
**5–10× faster** than the standard backend in serial, with similar
parallel scaling.

Resolution constraints
~~~~~~~~~~~~~~~~~~~~~~

Because each level coarsens by exactly a factor of two, the resolution must
be divisible by :math:`2^{N_\mathrm{levels}}`. With the default
``nlevels=4`` this requires ``res`` to be a multiple of 16. Common choices
that satisfy this are 128, 256, 512, 1024, and 2048.

Tuning ``ntarget`` and ``nlevels``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ntarget`` controls how many finest-grid voxels a particle's kernel spans
at its assigned level. Smaller values push more particles to coarser levels,
increasing speed but also increasing the approximation error from the
trilinear collapse. Values in the range 4–10 provide a good trade-off
between speed and accuracy; the default of 6 is a reasonable starting point.

``nlevels`` sets the depth of the hierarchy. Deeper hierarchies accommodate
larger dynamic ranges in smoothing length, but each additional level adds a
factor-of-two drop in coarse-grid resolution. Four levels (the default) is
appropriate for most cosmological simulations; two or three levels may be
sufficient for simulations with a narrower smoothing-length distribution.

Example — choosing backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from swiftsimio import load
   from swiftsimio.visualisation.volume_render import render_gas

   data = load("zoom_simulation.hdf5")

   # Nested backend with default parameters.
   grid = render_gas(
       data,
       resolution=512,
       project="masses",
       parallel=True,
       periodic=True,
       backend="nested",
   )

   # Tighter accuracy: more kernel evaluations per particle.
   grid_accurate = render_gas(
       data,
       resolution=512,
       project="masses",
       parallel=True,
       periodic=True,
       backend="nested",
       ntarget=10,
       nlevels=4,
   )

   # Faster render at the cost of slightly coarser upsampling.
   grid_fast = render_gas(
       data,
       resolution=512,
       project="masses",
       parallel=True,
       periodic=True,
       backend="nested",
       ntarget=4,
       nlevels=5,   # requires res % 32 == 0
   )


Lower-level API
---------------

The lower-level API for volume rendering allows for any general positions,
smoothing lengths, and smoothed quantities, to generate a pixel grid that
represents the smoothed, volume rendered, version of the data.

This API is available through
:obj:`swiftsimio.visualisation.volume_render_backends.backends` and
:obj:`swiftsimio.visualisation.volume_render_backends.backends_parallel` for parallel
implementations. The parallel versions use significantly more memory as they allocate
a thread-local image array for each thread, summing them in the end.

To use this function, you will need:

+ x-positions of all of your particles, ``x``.
+ y-positions of all of your particles, ``y``.
+ z-positions of all of your particles, ``z``.
+ A quantity which you wish to smooth for all particles, such as their
  mass, ``m``.
+ Smoothing lengths for all particles, ``h``.
+ The resolution you wish to make your cube at, ``res``.

Optionally, you will also need:

+ the size of the simulation box in x, y and z, ``box_x``, ``box_y`` and ``box_z``.

The key here is that only particles in the domain [0, 1] in x, [0, 1] in y,
and [0, 1] in z will be visible in the cube. You may have particles outside
of this range; they will not crash the code, and may even contribute to the
image if their smoothing lengths overlap with [0, 1]. You will need to
re-scale your data such that it lives within this range. You should pass in
raw numpy array (not :class:`~swiftsimio.objects.cosmo_array` or :class:`~unyt.array.unyt_array`).
Then you may use the functions as follows:

.. code-block:: python

   from swiftsimio.visualisation.volume_render_backends import backends, backends_parallel

   # Standard single-resolution scatter.
   out = backends["scatter"](x=x, y=y, z=z, h=h, m=m, res=res)

   # Nested multi-resolution scatter (serial).
   out = backends["nested"](
       x=x, y=y, z=z, h=h, m=m, res=res,
       ntarget=6,
       nlevels=4,
   )

   # Parallel variants — use the same keyword arguments.
   out = backends_parallel["scatter"](x=x, y=y, z=z, h=h, m=m, res=res)
   out = backends_parallel["nested"](
       x=x, y=y, z=z, h=h, m=m, res=res,
       ntarget=6,
       nlevels=4,
   )

``out`` will be a 3D :class:`~numpy.ndarray` grid of shape ``[res, res, res]``. You will
need to re-scale this back to your original dimensions to get it in the
correct units, and do not forget that it now represents the smoothed quantity
per volume.

If the optional arguments ``box_x``, ``box_y`` and ``box_z`` are provided, they
should contain the simulation box size in the same re-scaled coordinates as
``x``, ``y`` and ``z``. The rendering function will then correctly apply
periodic boundary wrapping. If ``box_x``, ``box_y`` and ``box_z`` are not
provided or set to 0, no periodic boundaries are applied.
