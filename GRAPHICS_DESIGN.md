# Graphics / GPU Section — Design

Status: **design approved, not yet implemented**. Decision: **analytical model
+ live display** — a configurable GPU spec drives a real rasterizer rendering to
a resolution you choose, with a cost model reporting frame time / FPS /
bottleneck. Internals designed so real shader-core simulation can replace the
cost model later without redoing the UI.

## The teaching goal

The CPU side of this project teaches *"cycles != time"* (the configurable
model's clock-period readout). The graphics section should teach the parallel
lesson: **throughput vs. latency, and where the bottleneck actually is.** Crank
the resolution and watch fill rate saturate; crank the cores and watch it stop
helping because you're now memory-bandwidth bound. That's the payoff, and it
argues for the cost model being *honest about which stage is the constraint*
rather than just multiplying a fudge factor.

## Architecture

New package `sim/gpu/`:

```
config.py       GPUSpec — cores, clock_mhz, mem_bandwidth_gbps, fillrate,
                texture_units, warp_size, cache_kb
raster.py       the actual rasterizer (pure Python, zero-dep like the core sim)
geometry.py     transform / project / clip / backface-cull
scene.py        built-in scenes (spinning cube, teapot, terrain, particles)
cost_model.py   stage cost estimator -> frame_time, fps, bottleneck
frame.py        Framebuffer — resolution-parameterized, emits the display buffer
```

### The pipeline, staged (this is what gets costed)

Model it as the classic stages so each can be independently costed *and* later
independently replaced by a real simulation:

| Stage | Work driven by | Limited by |
|---|---|---|
| Vertex fetch | vertex count | memory bandwidth |
| Vertex shading | vertex count | core count x clock |
| Clip / cull / project | vertex count | fixed function |
| Triangle setup | triangle count | setup rate |
| Rasterization | covered pixels | fill rate |
| Fragment shading | covered pixels x shader cost | core count x clock |
| ROP / blend / depth | covered pixels | fill rate + bandwidth |

Frame time = **max over stages** (they pipeline, so the slowest stage governs
steady-state throughput) — not the sum. The stage attaining that max IS the
reported bottleneck, which is what makes the readout educational rather than
decorative. Report per-stage utilization so the user can see the second-place
stage waiting to become the bottleneck.

This staged decomposition is the key design decision: it's exactly the seam
where `cost_model.py` can later be swapped for real SIMT core simulation
built on the existing `sim/component/` framework (SIMT lanes, warp scheduling,
memory coalescing) with no change to the API surface or the UI.

### Zero-dependency constraint

The core sim is dependency-free and the rasterizer must stay that way — pure
Python, no numpy. Consequence: **the Python rasterizer will be slow at high
resolutions.** Do not fight this by adding numpy. Instead:

- Render at the *simulated* resolution (which is small — 160x120 to 640x480),
  and let the browser scale the image up for display. A 320x240 frame is 76,800
  pixels, which is fine; 1280x720 is 921,600 and is not.
- The **cost model is analytical and instant** — it does not require actually
  rendering. So the FPS/bottleneck readout can cover resolutions far beyond what
  we actually rasterize. Decouple these two paths: "what it looks like" (render,
  capped resolution) and "how hard the GPU works" (cost model, any resolution).
- Cap the render resolution server-side and return the cost-model numbers for
  the requested resolution regardless. Clamp like every other route does.

### Transport

Frames go to the browser as a base64 RGBA buffer or a packed palette + indices,
drawn to a `<canvas>` with `imageSmoothingEnabled=false` for the chunky
pixel-art look the mock implies. For animation, render a short loop of frames
server-side and let the UI scrub/play them the way the cycle scrubber already
works — reuse that interaction model rather than inventing a new one.

## API

```
GET  /gpu/scenes          built-in scene catalog
GET  /gpu/specs           preset GPU configs (e.g. "1998 era", "modern")
POST /gpu/render          {spec, scene, width, height, frames} -> frames + stats
POST /gpu/analyze         {spec, scene, width, height} -> cost model only, instant
```

Same contract as everything else: validate + clamp, `400 {error}` on bad input,
never a raw 500. `/gpu/analyze` being render-free is what makes the sliders feel
live — the UI calls it on every knob drag, and only calls `/gpu/render` when the
user asks for a picture.

## UI — the third section

Add `/graphics` alongside `/` and `/lab`, following the established page shape
(the three pages cross-link via `url_for`, which is prefix-safe — see the
mountable-under-a-prefix commit).

- **Spec panel**: sliders for cores, clock, bandwidth, fillrate, texture units —
  same visual language as `#configurable-panel`'s cycle-budget knobs.
- **Display**: the mini canvas, with a resolution slider (160x120 -> 1280x720).
- **Readout**: frame time, FPS, and a **stacked per-stage bar** showing where the
  time goes, with the bottleneck stage highlighted. This is the money shot.
- **Scene picker**: cube / teapot / terrain / particles, varying vertex-heavy vs
  fill-heavy so the bottleneck actually moves between stages as you switch.
- A "race" grid like `/lab`'s Compare: same scene across several GPU specs.

## Correctness strategy

A renderer is easy to test badly ("it produced pixels"). Test it properly:

- **Golden-image tests** with an exact checksum per (scene, spec, resolution),
  so any change to the rasterizer is caught. Keep the images tiny (64x64).
- **Geometric invariants**: a triangle covering the whole viewport fills every
  pixel; a backfacing triangle renders nothing; a degenerate (zero-area)
  triangle renders nothing; a triangle fully outside the frustum is clipped;
  clipped and unclipped renderings agree on the visible region.
- **Depth correctness**: draw two overlapping triangles in both orders — the
  nearer one must win both times. This catches z-buffer bugs that eyeballing
  misses.
- **Cost-model monotonicity** (the analogue of the cache's monotonicity rule):
  more cores must never *increase* frame time; higher resolution must never
  *decrease* it; the reported bottleneck must actually be the argmax stage.
- **No-crash fuzzing** over degenerate specs: 0 cores, 1x1 resolution,
  enormous resolution, empty scene.

## Build order

1. `GPUSpec` + `cost_model.py` + `/gpu/analyze` — **no rendering at all**. The
   whole "crank the knobs, watch the bottleneck move" lesson works with just
   this, and it's fast to get right.
2. `raster.py` + `frame.py` + one scene (cube) + `/gpu/render`.
3. The `/graphics` page.
4. More scenes, spec presets, the race grid.
5. (Later, optional) replace `cost_model.py`'s estimates with real SIMT core
   simulation on the component framework.

Step 1 delivers most of the educational value for a fraction of the work — ship
it before writing a single line of rasterizer.
