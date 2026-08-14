"""
GPU specification and workload description — the shared contract for sim/gpu/.

This module is deliberately small and dependency-free: it defines WHAT a GPU is
configured as (``GPUSpec``) and HOW MUCH work a frame represents (``Workload``).
Everything else in the package consumes these two types:

    config.py     <- you are here (the contract)
    cost_model.py  reads GPUSpec + Workload -> per-stage times, frame time,
                   bottleneck.  NO rendering required, so it is instant.
    geometry.py    transform / project / clip / cull -> triangles + a Workload
    scene.py       built-in scenes (vertex/triangle heavy vs fill heavy)
    raster.py      actually draws triangles into a Framebuffer
    frame.py       Framebuffer (resolution-parameterised RGBA)

WHY THE SPLIT MATTERS
---------------------
The cost model is ANALYTICAL: it needs only counts, never pixels. That is what
lets the UI report frame time / FPS / bottleneck for 1920x1080 while only ever
*rendering* a small preview. The core simulator is zero-dependency (no numpy),
so a pure-Python rasterizer cannot keep up at high resolution — decoupling
"what it looks like" from "how hard the GPU works" is the design decision that
makes the whole section usable. See GRAPHICS_DESIGN.md.

UNITS (stated once, used everywhere)
------------------------------------
    clock_mhz            10^6 clocks / second
    mem_bandwidth_gbps   10^9 BYTES / second (not bits — bytes, throughout)
    all stage times      MILLISECONDS
    frame_time_ms        milliseconds; fps = 1000 / frame_time_ms
"""
from __future__ import annotations

# ── Fixed workload constants ──────────────────────────────────────────────
# A vertex carries position + normal + uv (roughly 8 floats), a fragment costs
# a colour write plus a depth read-modify-write. These are the "how many bytes
# does one unit of work move" numbers the bandwidth stages divide by; they are
# deliberately named so a reader can see they are modelling choices, not laws.
BYTES_PER_VERTEX = 32
BYTES_PER_FRAGMENT = 8

# Clamp ranges. Mirrors how api/app.py clamps cycle_costs / cache geometry, so
# a hostile or fat-fingered request cannot ask for an absurd machine.
MIN_CORES, MAX_CORES = 1, 1024
MIN_CLOCK_MHZ, MAX_CLOCK_MHZ = 1, 10_000
MIN_BANDWIDTH, MAX_BANDWIDTH = 0.1, 10_000.0
MIN_ROPS, MAX_ROPS = 1, 512
MIN_TEXTURE_UNITS, MAX_TEXTURE_UNITS = 0, 512
MIN_SETUP_RATE, MAX_SETUP_RATE = 1, 64
MIN_WARP, MAX_WARP = 1, 128
MIN_SHADER_COST, MAX_SHADER_COST = 1, 4096

# Resolution limits. The COST MODEL accepts anything up to MAX_*; the RENDERER
# is capped separately (see frame.py) because pure-Python rasterisation of a
# 1080p frame is not viable.
MIN_WIDTH, MIN_HEIGHT = 1, 1
MAX_WIDTH, MAX_HEIGHT = 3840, 2160


def _clamp(value, lo, hi):
    return max(lo, min(value, hi))


class GPUSpec:
    """A configurable graphics processor.

    Every field maps to exactly one thing the cost model can be limited by, so
    that changing a knob has an explainable effect rather than a mysterious one:

    ``cores`` x ``clock_mhz``      shader throughput (vertex + fragment shading)
    ``mem_bandwidth_gbps``         vertex fetch + framebuffer traffic
    ``rops`` x ``clock_mhz``       fill rate (rasterise + blend/depth)
    ``setup_rate`` x ``clock_mhz`` triangle setup rate
    ``texture_units``              texture fetch capacity (fragment stage)
    ``warp_size``                  SIMT group width; informational for the
                                   analytical model, load-bearing if/when
                                   cost_model is replaced by real core sim.
    """

    __slots__ = ("cores", "clock_mhz", "mem_bandwidth_gbps", "rops",
                 "texture_units", "setup_rate", "warp_size",
                 "vertex_shader_cost", "fragment_shader_cost")

    def __init__(self, cores: int = 16, clock_mhz: int = 800,
                 mem_bandwidth_gbps: float = 25.6, rops: int = 8,
                 texture_units: int = 8, setup_rate: int = 1,
                 warp_size: int = 8, vertex_shader_cost: int = 16,
                 fragment_shader_cost: int = 8):
        self.cores = int(_clamp(int(cores), MIN_CORES, MAX_CORES))
        self.clock_mhz = int(_clamp(int(clock_mhz), MIN_CLOCK_MHZ, MAX_CLOCK_MHZ))
        self.mem_bandwidth_gbps = float(
            _clamp(float(mem_bandwidth_gbps), MIN_BANDWIDTH, MAX_BANDWIDTH))
        self.rops = int(_clamp(int(rops), MIN_ROPS, MAX_ROPS))
        self.texture_units = int(
            _clamp(int(texture_units), MIN_TEXTURE_UNITS, MAX_TEXTURE_UNITS))
        self.setup_rate = int(_clamp(int(setup_rate), MIN_SETUP_RATE, MAX_SETUP_RATE))
        self.warp_size = int(_clamp(int(warp_size), MIN_WARP, MAX_WARP))
        self.vertex_shader_cost = int(
            _clamp(int(vertex_shader_cost), MIN_SHADER_COST, MAX_SHADER_COST))
        self.fragment_shader_cost = int(
            _clamp(int(fragment_shader_cost), MIN_SHADER_COST, MAX_SHADER_COST))

    # ── derived capacities (all per second) ───────────────────────────
    @property
    def shader_cycles_per_s(self) -> float:
        """Total shader cycles available per second across all cores."""
        return self.cores * self.clock_mhz * 1e6

    @property
    def pixels_per_s(self) -> float:
        """Fill rate: ROPs each retire one pixel per clock."""
        return self.rops * self.clock_mhz * 1e6

    @property
    def triangles_per_s(self) -> float:
        """Setup rate: triangles per second the front end can admit."""
        return self.setup_rate * self.clock_mhz * 1e6

    @property
    def bytes_per_s(self) -> float:
        """Memory bandwidth in BYTES per second."""
        return self.mem_bandwidth_gbps * 1e9

    def to_dict(self) -> dict:
        return {
            "cores": self.cores,
            "clock_mhz": self.clock_mhz,
            "mem_bandwidth_gbps": self.mem_bandwidth_gbps,
            "rops": self.rops,
            "texture_units": self.texture_units,
            "setup_rate": self.setup_rate,
            "warp_size": self.warp_size,
            "vertex_shader_cost": self.vertex_shader_cost,
            "fragment_shader_cost": self.fragment_shader_cost,
        }

    @classmethod
    def from_dict(cls, raw) -> "GPUSpec":
        """Build from an API payload, ignoring unknown keys.

        Raises ValueError for a non-object or a non-numeric field; every valid
        number is CLAMPED rather than rejected, so a slider can never produce
        an error.
        """
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("gpu spec must be an object")
        kwargs = {}
        for key in cls.__slots__:
            if key not in raw:
                continue
            val = raw[key]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"spec['{key}'] must be a number")
            if val != val or val in (float("inf"), float("-inf")):
                raise ValueError(f"spec['{key}'] must be finite")
            kwargs[key] = val
        return cls(**kwargs)

    def __repr__(self) -> str:                       # pragma: no cover - debug
        return (f"<GPUSpec {self.cores} cores @ {self.clock_mhz}MHz, "
                f"{self.rops} ROPs, {self.mem_bandwidth_gbps}GB/s>")


# ── Era presets ───────────────────────────────────────────────────────────
# Rough, deliberately round numbers chosen so the BOTTLENECK DIFFERS between
# them on the same scene — that contrast is the teaching point, not historical
# accuracy. Each is documented with what it is meant to demonstrate.
SPEC_PRESETS = {
    "sw_1995": {
        "label": "Software renderer (1995)",
        "note": "No dedicated fill hardware: 1 ROP. Fill-rate bound almost "
                "immediately as resolution rises.",
        "spec": {"cores": 1, "clock_mhz": 100, "mem_bandwidth_gbps": 0.4,
                 "rops": 1, "texture_units": 0, "setup_rate": 1,
                 "vertex_shader_cost": 32, "fragment_shader_cost": 16},
    },
    "fixed_2001": {
        "label": "Fixed-function GPU (2001)",
        "note": "Real ROPs and texture units, but few shader cores — geometry "
                "heavy scenes stall in vertex shading.",
        "spec": {"cores": 4, "clock_mhz": 250, "mem_bandwidth_gbps": 8.0,
                 "rops": 4, "texture_units": 4, "setup_rate": 1,
                 "vertex_shader_cost": 24, "fragment_shader_cost": 12},
    },
    "programmable_2008": {
        "label": "Unified shaders (2008)",
        "note": "Plenty of cores; bandwidth becomes the limit at high "
                "resolution.",
        "spec": {"cores": 32, "clock_mhz": 650, "mem_bandwidth_gbps": 64.0,
                 "rops": 16, "texture_units": 16, "setup_rate": 1,
                 "vertex_shader_cost": 16, "fragment_shader_cost": 8},
    },
    "modern": {
        "label": "Modern GPU",
        "note": "Wide and fast — most built-in scenes cannot saturate it, "
                "which is itself the lesson about workload scaling.",
        "spec": {"cores": 256, "clock_mhz": 1800, "mem_bandwidth_gbps": 500.0,
                 "rops": 64, "texture_units": 64, "setup_rate": 4,
                 "vertex_shader_cost": 16, "fragment_shader_cost": 8},
    },
}
DEFAULT_SPEC_PRESET = "programmable_2008"


def spec_preset(name: str) -> GPUSpec:
    if not isinstance(name, str) or name not in SPEC_PRESETS:
        raise ValueError(f"unknown spec preset {name!r}; "
                         f"available: {sorted(SPEC_PRESETS)}")
    return GPUSpec(**SPEC_PRESETS[name]["spec"])


def spec_preset_catalog() -> list:
    """Serialisable preset list for GET /gpu/specs."""
    return [
        {"name": name, "label": entry["label"], "note": entry["note"],
         "spec": GPUSpec(**entry["spec"]).to_dict()}
        for name, entry in SPEC_PRESETS.items()
    ]


class Workload:
    """How much work one frame represents.

    Produced either by actually walking a scene's geometry (``geometry.py``) or
    estimated analytically for a resolution the renderer would never attempt.
    The cost model consumes ONLY these counts — it never needs pixels — which
    is what makes ``/gpu/analyze`` instant at any resolution.

    ``fragments`` is *shaded* fragments (post-depth-test, post-cull), while
    ``covered_pixels`` is raw coverage before the depth test. Overdraw is the
    ratio, and it is the number that explains why a scene costs more than its
    silhouette suggests.
    """

    __slots__ = ("vertices", "triangles", "covered_pixels", "fragments",
                 "width", "height", "scene")

    def __init__(self, vertices: int = 0, triangles: int = 0,
                 covered_pixels: int = 0, fragments: int | None = None,
                 width: int = 0, height: int = 0, scene: str = ""):
        self.vertices = max(0, int(vertices))
        self.triangles = max(0, int(triangles))
        self.covered_pixels = max(0, int(covered_pixels))
        self.fragments = (self.covered_pixels if fragments is None
                          else max(0, int(fragments)))
        self.width = max(0, int(width))
        self.height = max(0, int(height))
        self.scene = scene

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def overdraw(self) -> float:
        """Covered pixels per screen pixel. 1.0 == every pixel touched once."""
        return (self.covered_pixels / self.pixels) if self.pixels else 0.0

    def scaled_to(self, width: int, height: int) -> "Workload":
        """Same scene at a different resolution.

        Coverage is AREA-proportional (2x the width and height is 4x the
        pixels) while vertex and triangle counts are resolution-INDEPENDENT.
        That asymmetry is exactly why raising the resolution moves the
        bottleneck from geometry to fill, so it is modelled explicitly rather
        than by scaling everything uniformly.
        """
        width = int(_clamp(int(width), MIN_WIDTH, MAX_WIDTH))
        height = int(_clamp(int(height), MIN_HEIGHT, MAX_HEIGHT))
        if not self.pixels:
            return Workload(self.vertices, self.triangles, 0, 0,
                            width, height, self.scene)
        ratio = (width * height) / self.pixels
        return Workload(
            vertices=self.vertices,
            triangles=self.triangles,
            covered_pixels=round(self.covered_pixels * ratio),
            fragments=round(self.fragments * ratio),
            width=width, height=height, scene=self.scene,
        )

    def to_dict(self) -> dict:
        return {
            "scene": self.scene,
            "width": self.width,
            "height": self.height,
            "pixels": self.pixels,
            "vertices": self.vertices,
            "triangles": self.triangles,
            "covered_pixels": self.covered_pixels,
            "fragments": self.fragments,
            "overdraw": round(self.overdraw, 4),
        }

    def __repr__(self) -> str:                       # pragma: no cover - debug
        return (f"<Workload {self.scene} {self.width}x{self.height} "
                f"{self.triangles} tris, {self.covered_pixels} covered>")
