"""Project SPH kernel splats using WebGPU."""

from functools import lru_cache

import numpy as np

from swiftsimio.optional_packages import WGPU_AVAILABLE, wgpu
from swiftsimio.visualisation.kernels import get_kernel_id, kernel_constant_2d
from swiftsimio.visualisation.projection_backends.kernels import kernel_gammas


_SPLAT_WGSL = """
struct Params {
    box: vec4<f32>, // box sizes and first periodic shifts
    counts: vec4<u32>, // particle count, number of x copies, resolution, kernel
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> particles: array<vec4<f32>>;
struct Vertex {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) particle: vec4<f32>,
};
@vertex
fn vs_main(@builtin(vertex_index) vertex: u32,
           @builtin(instance_index) index: u32) -> Vertex {
    let particle = particles[index % p.counts.x];
    let tile = index / p.counts.x;
    let shift = vec2<f32>(f32(tile % p.counts.y), f32(tile / p.counts.y));
    let centre = particle.xy + (shift + p.box.zw) * p.box.xy;
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0,-1.0), vec2<f32>(1.0,-1.0), vec2<f32>(-1.0,1.0),
        vec2<f32>(-1.0,1.0), vec2<f32>(1.0,-1.0), vec2<f32>(1.0,1.0));
    let xy = centre + corners[vertex] * particle.z;
    var out: Vertex;
    // Framebuffer rows increase downwards; return rows in increasing physical y.
    out.position = vec4<f32>(2.0*xy.x-1.0, 1.0-2.0*xy.y, 0.0, 1.0);
    out.particle = vec4<f32>(centre, particle.zw);
    return out;
}
// Shapes of the kernels other than Wendland-C2, for 0 <= r < 1 (see
// swiftsimio.visualisation.kernels); their normalisation is in the particle weight.
fn kernel_shape(r: f32, kernel: u32) -> f32 {
    let t = 1.0-r;
    let t2 = t*t;
    switch kernel {
        case 1u: { // cubic spline
            var w = t2*t;
            if (r < 0.5) { let a = 0.5-r; w -= 4.0*a*a*a; }
            return w;
        }
        case 2u: { // quartic spline
            var w = t2*t2;
            if (r < 0.6) { let a = 0.6-r; w -= 5.0*a*a*a*a; }
            if (r < 0.2) { let b = 0.2-r; w += 10.0*b*b*b*b; }
            return w;
        }
        case 3u: { // quintic spline
            var w = t2*t2*t;
            if (r < 2.0/3.0) { let a = 2.0/3.0-r; w -= 6.0*a*a*a*a*a; }
            if (r < 1.0/3.0) { let b = 1.0/3.0-r; w += 15.0*b*b*b*b*b; }
            return w;
        }
        case 4u: { // Wendland-C4
            return t2*t2*t2 * (1.0+r*(6.0+r*(35.0/3.0)));
        }
        default: { // Wendland-C6
            return t2*t2*t2*t2 * (1.0+r*(8.0+r*(25.0+r*32.0)));
        }
    }
}
@fragment
fn fs_main(in: Vertex) -> @location(0) f32 {
    // Evaluate from pixel centres to avoid rasterizer vertex/interpolation rounding.
    let coord = (in.position.xy / f32(p.counts.z) - in.particle.xy) / in.particle.z;
    let r = length(coord);
    if (r >= 1.0) { discard; }
    if (p.counts.w != 0u) {
        return in.particle.w * kernel_shape(r, p.counts.w);
    }
    let t = 1.0-r;
    let kernel = 2.22817109 * t*t*t*t * (1.0+4.0*r);
    return in.particle.w * kernel;
}
"""


@lru_cache(maxsize=1)
def _get_pipeline() -> tuple:
    """
    Create the device and additive splat pipeline on first use.

    Returns
    -------
    tuple[wgpu.GPUDevice, wgpu.GPURenderPipeline]
        WebGPU device and the pipeline for additive kernel deposition.
    """
    if not WGPU_AVAILABLE:
        raise ImportError("The wgpu projection backend requires swiftsimio[wgpu].")

    adapter = wgpu.gpu.request_adapter_sync()
    if "float32-blendable" not in adapter.features:
        raise RuntimeError(
            "The wgpu projection backend requires a GPU supporting float32-blendable."
        )
    device = adapter.request_device_sync(required_features=["float32-blendable"])
    module = device.create_shader_module(code=_SPLAT_WGSL)
    blend = dict(operation="add", src_factor="one", dst_factor="one")
    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex=dict(module=module, entry_point="vs_main"),
        primitive=dict(topology="triangle-list"),
        fragment=dict(
            module=module,
            entry_point="fs_main",
            targets=[dict(format="r32float", blend=dict(color=blend, alpha=blend))],
        ),
    )
    return device, pipeline


def scatter(
    x: np.ndarray,
    y: np.ndarray,
    m: np.ndarray,
    h: np.ndarray,
    res: int,
    box_x: float = 0.0,
    box_y: float = 0.0,
    kernel_id: int = 0,
) -> np.ndarray:
    """
    Project particles with additive GPU splats.

    Parameters
    ----------
    x, y : np.ndarray
        One-dimensional positions in coordinates scaled to the image domain [0, 1].
        Particles outside the domain contribute if their kernels overlap it.
    m : np.ndarray
        Particle weights (e.g. masses), including signed values.
    h : np.ndarray
        Nonnegative smoothing lengths in the same coordinates as x and y.
    res : int
        Number of pixels along each image axis.
    box_x, box_y : float, optional
        Periodic box sizes in the same coordinates. Zero disables wrapping.
    kernel_id : int, optional
        SPH kernel to use, see :data:`swiftsimio.visualisation.kernels.kernel_names`.
        Defaults to ``0`` (Wendland-C2).

    Returns
    -------
    np.ndarray
        Float32 grid of shape (res, res), indexed by x then y.

    Notes
    -----
    Requires the optional ``swiftsimio[wgpu]`` dependency and a WebGPU device
    supporting ``float32-blendable``. Availability is checked in
    ``optional_packages``; the device is created only on first use. No canvas,
    window or event loop is needed. Deposition uses a single-channel float32
    texture, which is read back to the CPU;
    units and rotations are handled by the projection API.

    Uses the 2D form of the chosen kernel, with support radius ``gamma*h`` (for
    the default Wendland-C2 kernel, ``1.897367*h``). Support
    radii smaller than two pixels are broadened to two pixels, with an amplitude
    correction preserving the analytic integral. Pixel-centre
    sampling does not guarantee discrete mass conservation. There is no colour
    mapping or weighted-mean division. ``parallel=True`` uses the same GPU
    implementation.
    """
    if not isinstance(res, (int, np.integer)) or res <= 0:
        raise ValueError("res must be a positive integer")
    arrays = [np.asarray(a, dtype=np.float64) for a in (x, y, m, h)]
    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("x, y, m and h must be one-dimensional arrays of equal length")
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("x, y, m and h must contain only finite values")
    x, y, m, h = arrays
    if (
        np.any(h < 0)
        or not np.all(np.isfinite([box_x, box_y]))
        or min(box_x, box_y) < 0
    ):
        raise ValueError(
            "Smoothing lengths and box sizes must be finite and nonnegative"
        )
    kernel_id = get_kernel_id(kernel_id)
    image = np.zeros((res, res), dtype=np.float32)
    if not len(x) or not np.any(m):
        return image

    device, pipeline = _get_pipeline()
    radius = np.maximum(kernel_gammas[kernel_id] * h, 2.0 / res)
    coefficients = m / radius**2
    if kernel_id != 0:
        # The shader holds the Wendland-C2 normalisation, the others are applied here.
        coefficients *= kernel_constant_2d[kernel_id]
    scale = np.max(np.abs(coefficients))
    particles = np.column_stack((x, y, radius, coefficients / scale)).astype(np.float32)
    # Tile exactly as the CPU backends, without duplicating particle buffers.
    nx, ny = [1 if b == 0 else int(np.ceil(1 / b)) + 2 for b in (box_x, box_y)]
    params = np.zeros(8, dtype=np.float32)
    params[:4] = box_x, box_y, -int(box_x > 0), -int(box_y > 0)
    params.view(np.uint32)[4:8] = len(x), nx, res, kernel_id
    particle_buffer = device.create_buffer_with_data(
        data=particles, usage=wgpu.BufferUsage.STORAGE
    )
    uniform = device.create_buffer_with_data(
        data=params, usage=wgpu.BufferUsage.UNIFORM
    )
    texture = device.create_texture(
        size=(res, res, 1),
        format="r32float",
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    try:
        bind = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                dict(binding=0, resource=dict(buffer=uniform)),
                dict(binding=1, resource=dict(buffer=particle_buffer)),
            ],
        )
        encoder = device.create_command_encoder()
        render = encoder.begin_render_pass(
            color_attachments=[
                dict(
                    view=texture.create_view(),
                    resolve_target=None,
                    clear_value=(0, 0, 0, 0),
                    load_op="clear",
                    store_op="store",
                )
            ]
        )
        render.set_pipeline(pipeline)
        render.set_bind_group(0, bind)
        render.draw(6, len(x) * nx * ny)
        render.end()
        device.queue.submit([encoder.finish()])
        pixels = device.queue.read_texture(
            dict(texture=texture), dict(bytes_per_row=res * 4), (res, res, 1)
        )
        # WebGPU returns (y, x); the projection API expects (x, y).
        image[:] = np.frombuffer(pixels, np.float32).reshape(res, res).T * scale
    finally:
        texture.destroy()
        uniform.destroy()
        particle_buffer.destroy()
    return image


scatter_parallel = scatter
