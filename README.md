# Clearwater — CUDA infinite water

A CUDA reimplementation of [Aurélien / Lumaris's Clearwater](https://github.com/Aureliengmz/clearwater), extended with three FFT ocean scales, a moving weather front, and unrestricted camera travel.

**[Launch the live demo](https://samg-coder.github.io/clearwater/)** · [Native Windows version](Native/README.md) · [Pages deployment status](https://github.com/SamG-Coder/clearwater/actions/workflows/pages.yml)

**[Explore the pool study](https://samg-coder.github.io/clearwater/pool/)** — a separate CUDA demo focused on a curved swimming pool and its tilework.

## Water cube test

**[Open the water cube](https://samg-coder.github.io/clearwater/cube/)** or visit `/cube/` locally. This is a two-metre cubic optical volume testing the lotus-pond reference's clear jade water, visible weathered stone, rippled refraction, and moving caustics. Drag to orbit, scroll to zoom, and click the top surface to disturb it. The existing weather and view controls are available.

The CUDA renderer in [`cube/cube.cu`](cube/cube.cu) traces entry and exit through the cube, with wavelength-dependent absorption and up to four internal reflection segments. The top uses the shared short-wave FFT; rain and click impulses use a separate bounded 256² ripple field. The test reuses the pool's browser host rather than duplicating its GPU setup. The cubic boundary is an optical test constraint, not a freely suspended fluid simulation. The caustic map approximates overhead illumination of the stone base. Run `npm run test:cube` with the local server on port 5186.

## The pool study

![Pool study with CUDA water and tilework](previews/pool-ui.png)

Open `/pool/` on the local server or use the link above. This scene concentrates on the pool, pale segmented coping, submerged entry steps, ceramic mosaic lining, slate deck, and stepped terrace. Drag to orbit, scroll to zoom, or click the water to make a ripple. **Overview**, **Waterline**, and **Tile detail** provide three camera presets; **H** restores hidden controls.

Sunlit, Breeze, and Rainstorm presets control wind, rainfall, and cloud cover. **Let the weather change** sends the shared moving storm across the pool at an accelerated showcase rate. Wind energy takes time to build and settle; rain injects individual impulses into a fixed 120 Hz wave equation with reflecting basin boundaries. Wet paving darkens and reflects the sky. Pause freezes simulation time while leaving the camera usable.

Rain falls downward with a wind slant. Deterministic world-space drop events drive both visible contact crowns/expanding rings and the numerical surface impulses, making each splash visible where its ripple begins. The crowns are an analytic visual approximation, layered over the bounded wave field.

[`pool/pool.cu`](pool/pool.cu) supplies the geometry, procedural stone and ceramic materials, pool dynamics, ray-traced reflection/refraction, and shallow-water optics. It compiles together with the existing [`src/clearwater.cu`](src/clearwater.cu), reusing the FFT, spectral weather response, sky, Fresnel optics, photon filtering, and tone mapping. Only the short 4.6 m FFT band contributes to pool wind ripples; ocean swell is excluded. The scene contains no terrain or imported models/textures.

This is a real-time visual approximation: FFT wind ripples taper near the wall, the bounded solver handles rain and click impulses, and the RGB photon map uses a representative 1.5 m depth rather than a full light-transport solve on every step and wall. The pool is currently a browser sub-demo; the native application runs the main ocean demo. Actions compile all CUDA demo paths for WebGPU and package all three Pages entry points; they do not build native CUDA.

Run `npm run test:pool` against the local server on port 5186 for GPU residency, bounded-ripple, weather, pause, controls, and screenshot checks.

The live demo runs directly in a WebGPU-capable browser; no installation is needed.

![Clearwater CUDA](previews/clearwater-ui.png)

Simulation and image formation live in [`src/clearwater.cu`](src/clearwater.cu), with the pool-specific geometry and optics in [`pool/pool.cu`](pool/pool.cu). The browser executes it through [SamG-Coder/cuda-webshader](https://github.com/SamG-Coder/cuda-webshader): CUDA source → generated WGSL → WebGPU. JavaScript handles DOM controls, asset decoding, resources, dispatch and presentation. The active application has no WebGL, Three.js, handwritten WGSL, CPU wave simulation or CPU FFT.

## Run

### 10,000 rubber ducks

Click **Add 10,000 rubber ducks** in the default browser demo. **Duck overview** frames the whole flock; **Ducks up close** moves down to the water. Each roughly half-metre, 140–240 gram hollow toy duck has its own GPU-resident position, velocity, quaternion, and angular velocity. Pause freezes the bodies while the camera remains usable. Starting a tornado with the ducks present places it in the flock.

[`src/ducks.cu`](src/ducks.cu) contains the simulation and solid procedural duck geometry. Four buoyancy samples estimate submerged volume against the three FFT ocean scales and the tornado's local water patch. Their force arms generate pitch and roll, with water damping, gravity, distributed quadratic air drag, wind torque, and frictional collision impulses. Four exposed body points sample the air and produce rotational drag, with orientation-dependent exposed area. Evolving Fourier gust modes drive resolved eddies in the shared airflow; spatially correlated metre-scale gusts approximate unresolved shear across each duck. The time-varying forces, mass variation, and orientation-dependent surface lift break up uniform rings without prescribing duck trajectories or randomizing their poses. A bounded near-surface pressure-lift approximation bridges the six-metre air grid; the resolved converging airflow and core updraft then entrain the light toys. A chained spatial hash finds nearby bodies without an all-pairs collision pass. The fixed 120 Hz solver allows at most four catch-up steps per frame. Moving bodies inject small wakes into the existing near-camera ripple equation. No body state is downloaded during normal frames.

Rendering uses smoothly blended moulded body geometry within 18 metres, with analytical ellipsoid intersections for distant ducks and crisp bills/eyes, binned into 8 × 8 screen tiles. The model has a rounded neck, a two-part bill, flattened tail, softly joined wings, subtle colour variation, and a satin rubber finish. There is no per-tile occupancy limit; a global bin overflow uses a slower exact fallback rather than dropping ducks. Geometry and shading are CUDA source compiled to WebGPU, without meshes or imported duck textures. The browser host supplies buffers, dispatches, controls, and presentation.

These are approximate rigid floating bodies: collisions use a bounding sphere, buoyancy uses four volume samples, and the rubber does not deform. Duck wakes feed the small ripple patch; the ducks do not displace the whole FFT ocean or feed momentum back into the air solver. Individual duck reflections/refraction are not yet included. The native build compiles these kernels, but interactive duck simulation and controls are currently in the browser host.

With the local server on port 5173, `npm run test:ducks` checks all 10,000 bodies, quaternion normalization, sustained floating, a collision negative control, gravity, pause, tornado acceleration and sustained lofting, screen-bin overflow, and the absence of live state readbacks. It saves screenshots and GPU timing evidence to `previews/ducks/`. GPU timings exclude browser presentation and are not end-to-end frame rates.

Run `node scripts/record-ducks-showcase.mjs --preview` to inspect the six shots, or omit `--preview` to record a 45-second 1080p/30 FPS MP4 with FFmpeg on PATH. The video follows rainy ducks, the storm, tornado activation at 23 seconds, and the resulting flock motion, ending with project and demo URLs. Capture advances simulation at fixed timesteps and is not a real-time performance measurement. Outputs and the casual post caption are saved under `recordings/ClearWater-ducks-showcase-2026-09-29/`.

### Tornado / waterspout in the default browser demo

Use **Start tornado** in the weather panel. It places a vortex 650 metres ahead of the camera, aims the view toward it, and increases cloud cover. Fly freely around it with the existing controls; **Pause** freezes the fluid while the camera remains usable. **Clear skies** disables it. Intensity changes the circulation forcing. Simulation time stays real-time, independently of the weather showcase speed.

The CUDA source now contains a local 48 × 96 × 48 moist-flow solver covering 288 × 576 × 288 metres. It uses midpoint semi-Lagrangian advection, buoyancy, vorticity confinement, 12 warm-started pressure iterations, velocity projection, saturation/evaporation with latent heat, and transported spray. A transported material-coordinate field deforms cloud density and shadows across a broader entrainment region; the outer circulation is an extrapolation of the local fluid. A 256 × 256 local FFT patch receives atmospheric pressure and wind work sampled from the existing medium-scale ocean waves. Its modes evolve with finite-depth gravity/capillary dispersion and damping; height and spectral slopes blend to zero before the periodic boundary. A separate 96 × 96 shallow-water patch handles wind-driven currents and transported foam. Up to 8,192 spray parcels are emitted from windy wave crests and follow air drag and gravity, supplementing the Eulerian mist. The funnel and its reflected radiance are volumetric, with cached lighting and half-resolution integration.

The fluid runs at a fixed 30 Hz, with at most two catch-up steps per rendered frame; slow devices can therefore fall behind real time rather than enter an unbounded catch-up loop. All live simulation and rendering remain GPU-resident. With a stationary camera, environment lighting refreshes at 10 Hz and the visible cloud cache refreshes at up to 30 Hz (every other fluid step during accelerated diagnostics). Camera or weather-control changes invalidate the caches immediately; active storm weather bypasses this caching to preserve lightning. The volume cache is also reused when neither the flow nor the camera changes. The vortex starts from a mature seeded state. This is a driven local Boussinesq simulation with approximate moist thermodynamics and one-way air-to-water coupling, not a compressible storm-scale weather model. Six-metre fluid cells do not resolve individual droplets; reflection uses a displaced planar approximation, and the simulation footprint is finite. Native CUDA call sites retain their existing tornado-disabled behavior; the new controls are in the browser demo.

With `npm start` running on port 5173, run `npm run test:tornado` to check divergence reduction, finite fields, cloud transport, surface response, local FFT boundary/zero-mode behavior, an FFT-input negative control, airborne spray, sustained operation, strong-wind/shallow-water stability, and GPU timings. Run `node scripts/tornado-profile.mjs <label>` for a comparable 60-sample stationary-camera GPU profile. Evidence is written to `previews/tornado/` without replacing the existing demo previews.

The browser version and the Windows native application share `src/clearwater.cu`. For the native build with a separate control window, see [`Native/README.md`](Native/README.md).

Requires Node.js 20+ and a browser with WebGPU enabled. Serve over localhost or HTTPS.

```powershell
npm ci
npm start
```

Open **http://localhost:5173**. For another port: `$env:PORT='5186'; npm start`.

- Drag or use arrow keys to look: right/left and up/down follow your input direction.
- WASD flies relative to the camera; W follows the direction you are looking, including pitch. E rises and Q descends, with a minimum camera height of 0.65 m.
- Scroll up to increase travel speed, down to decrease it (0.1–200 m/s). Hold either Shift for a temporary 6× boost. Current speed appears in the footer.
- Click nearby water to generate ripples.
- Space pauses; H hides controls; Reset view returns to the starting point.
- Clearwater and Open water presets set wave energy, depth and camera position.
- Depth, exposure, resolution, caustic/normal diagnostics, lens glare and continuous drift are adjustable.
- PNG saves the rendered image. `?t=5` opens at a fixed wave time.

## A passing storm

Click **Send a storm** and watch the horizon. The default showcase advances weather 30 times faster while the waves and rain continue moving at normal simulation speed. Near the starting position, the strongest conditions arrive after about 30 seconds, with clearing skies around 70 seconds. Select **Real time** for uncompressed weather evolution. **Clear skies** removes the storm forcing; the waves relax gradually.

- A moving world-space storm band drives local rain, cloud cover, visibility and wind forcing at the camera.
- A JONSWAP-shaped directional weighting redistributes energy in the short and medium FFT bands. Stored spectral energy responds gradually; decay is slower than growth, and the independent long swell is preserved.
- World-space impact events drive visible crowns/rings and impulses in the 120 Hz ripple solver. Nearby rain uses wind-advected world planes with ray-depth occlusion; distant rain curtains soften the horizon. Airborne streaks are a statistical approximation, not individually tracked drops tied to every impact.
- Irregular seeded lightning lights nearby cloud density and produces a high-resolution bolt and water reflection. Pause freezes clouds, rain and lightning as well as the waves.
- Three-dimensional cloud density is integrated with 72 bounded ray steps, rounded cellular detail, sun self-shadowing and approximate multiple scattering. A half-resolution camera pass resolves visible clouds; a 512 x 128 environment map supplies reflections. A 128 x 128 world-space shadow map uses the same density to dim sunlight and caustics.
- A second packed FFT supplies horizontal crest displacement. Surface compression generates persistent, advected whitecaps, with a faster-decaying fresh-foam channel feeding a small near-surface spray volume. Horizontal displacement is limited at high wave-energy settings. Spectral displacement derivatives share the existing FFT channels, adding no FFT passes. Foam builds from compression and slope, with a transported underwater bubble layer and no independent procedural foam mask. See the [FFT foam implementation and verification notes](previews/foam/FFT-FOAM.md) for performance results and the remaining close-up visual limitations.
- Wind, direction, storm strength, rain and cloud cover have separate controls. The optional marker buoy follows the sampled height and slope; it is a visual scale reference.

![Passing storm](previews/weather-storm.png)

## CUDA pipeline

| Stage | Implementation |
|---|---|
| Spectrum | Seeded Gaussian complex coefficients, directional spectral bumps and GPU RMS slope normalization |
| Wave evolution | Gravity/capillary dispersion with finite-depth tanh(k·depth), per-mode wind-energy memory |
| Weather | Irregular moving storm band, lagged local wind, cached volumetric clouds, shared-density shadows, depth-aware rain and local lightning |
| Foam | Persistent 512² coverage, freshness and age; spectral Jacobian + slope breaking, transport, submerged bubble density and decay |
| Infinite surface | Three independently seeded 256² periodic cascades spanning 4.6 m, 37 m and 293 m, sampled in world space |
| FFT | 16 Stockham butterfly passes per 2D transform, two packed complex fields; height, analytic slopes and horizontal displacement |
| Interaction | 256² camera-relative ripple field, 16 m wide, fixed 120 Hz wave equation and integer-cell recentering |
| Caustics | 1024² refracted rays, three refractive indices, bilinear fixed-point atomic splats into a 512² RGB field |
| Water optics | Height-field intersection, Fresnel reflection, Snell refraction, Beer–Lambert extinction, underwater scattering, pebble/sand seabed and sun highlights |
| Lens | CUDA aperture rasterization, wavelength-dependent diffraction PSF, forward FFT / multiplication / inverse FFT convolution, padded image to avoid wrapping ghosts |
| Output | Bloom and filmic tone curve in CUDA; direct GPU buffer-to-canvas copy |

The normal frame loop performs **zero GPU-to-CPU readbacks**. Explicit inspection and PNG export read data back on request. The full compiler/runtime module graph is vendored; no CDN is needed.

## Validation

```powershell
npm run check
# Start the server on port 5186 in another terminal, then:
npm test
npm run test:weather
```

`npm test` launches installed Microsoft Edge through Playwright with WebGPU. The validation port is 5186. Latest evidence is in [`previews/verification.json`](previews/verification.json).

- All 36 CUDA entries (28 shared/main, plus pool and cube kernels) compile through the vendored CUDA frontend.
- The native application built with CUDA Toolkit 13.3 and passed its GPU smoke test on an RTX 5080: FFT error **2.47e-7**, ripple generation, 6x Shift boost, separate windows, resizing, diagnostic views and finite values at 10 km. See [`previews/native-smoke.json`](previews/native-smoke.json). Full manual control-window QA remains incomplete.
- Weather validation passes in Edge/WebGPU and native CUDA: the medium-wave band RMS rises from **0.086 m to 0.194 m** in the recorded browser scenario, rain generates nonzero ripple heights, foam remains bounded, pause freezes both clocks, and stored wave energy remains elevated after the wind eases. See [`previews/weather-verification.json`](previews/weather-verification.json) and [`previews/native-weather.json`](previews/native-weather.json). These are implementation checks, not oceanographic calibration.
- GPU 2D FFT compared against five analytic Fourier modes across both axes, all three cascades and both complex fields: maximum absolute error **3.89e-7**.
- Forward/inverse round trip error: **2.99e-7**.
- RGB caustic mean energy: **0.9922** (small fixed-point splat truncation loss).
- Lens point-spread function normalized to unity within **2.4e-7**.
- Finite wave state, HDR and diffraction buffers; click-generated ripples; WASD travel; finite state after travel to `(10000, -10000)` metres.
- Pause/reset, window resize, resolution selection, caustic/normal views and glare switching exercised.
- No browser console errors, page errors, failed HTTP requests or WebGPU validation errors in the recorded run.
- Visual captures inspected for shallow water and open water on an NVIDIA Blackwell adapter in Edge. Other hardware/browser combinations have not been tested.

## Main-demo quality and performance checks

Run `npm run test:quality` with the server on port 5186 for fixed-camera calm/front/storm screenshots, GPU timestamps, resolution checks, bit-identical pause checks, and maximum-energy/wind stress views. `node scripts/frame-profile.mjs host-current` measures 100 live frames including CPU command preparation and GPU completion. These scripts use installed Edge and real WebGPU; the normal render loop performs no readbacks.

The baseline and stage captures are preserved under [`previews/quality/`](previews/quality/). Baseline rendering at 1152 x 720 took about **4.30–4.34 ms median GPU time** on the tested NVIDIA RTX 5080 / Edge system. The expanded renderer is roughly **2.1–2.4 ms** for the same rendering scope, or **2.2–2.5 ms** including wave computation and two ripple substeps in the fixed-camera cases. See [`final.json`](previews/quality/final.json) for exact medians, p95 values and measurement scope. These are GPU workload measurements, not delivered browser FPS or a promise for other hardware.

Reusing GPU bindings lowered measured live frame work from **5.50 ms to 4.78 ms median** in separate 100-frame runs. Both report zero readbacks. Resolution and camera angle materially change cost; the quality check records both. Native builds and smoke checks remain local-only.

We deliberately avoid temporal cloud history: the visible sky is recomputed each frame, so camera turns and lightning cannot drag old cloud images across the screen. Reflections use a lower-resolution environment map centred on the camera, so they approximate cloud parallax at distant water points. The shadow map covers 4 km around the camera and clamps outside that region. Impact crowns perturb shading; they are not fully meshed airborne splash geometry.

## Scope and tradeoffs

“Infinite” means there is no finite mesh edge or camera travel boundary. The wave fields remain periodic, as FFT oceans are; combining three scales reduces obvious repetition. It is not an infinitely large stored simulation. World coordinates and GPU math use 32-bit floats, so precision eventually deteriorates at extreme travel distances; 10 km coordinates are covered by the test.

The weather system is a visual, physically motivated approximation, not a forecast or a calibrated wind/fetch model. Wind forcing is uniform across the FFT tiles and sampled from the front at the camera; spatial gust shading does not solve local fluid momentum. Foam is sourced from the horizontal-displacement Jacobian with height and wind gates. Clouds integrate procedural density with approximate scattering. The buoy uses surface-following animation rather than rigid-body buoyancy.

This is a spectral height-field ocean with bounded horizontal crest displacement. It does not simulate overturning breakers, volumetric water or an underwater camera. Spray is a small visual volume sourced by fresh foam, not a particle-based fluid solve. Shallow caustics are driven by the short-wave cascade at the selected mean depth, with local ripple curvature added during shading; the long-wave cascades are not included in the photon map. The caustic map and seabed remain periodic. Distant headlands are a procedural sky silhouette, not traversable terrain.

The optical design is reimplemented, not a pixel-identical port: the lens uses three representative wavelengths, and compute filtering replaces WebGL derivatives and texture mipmaps. The unused original WebGL application, old media/tools and unused vendor helpers have been removed; they remain available in Git history. The browser targets CUDA WebShader, while `Native/main.cu` directly includes the same kernels for native CUDA execution.

## Actions and Pages

Published site: **[samg-coder.github.io/clearwater](https://samg-coder.github.io/clearwater/)**.

`npm run build` stages the browser entry, all 37 required JavaScript modules, shared CUDA source, seabed asset and licenses into `dist/`, with relative URLs and `.nojekyll` for Pages. Actions runs browser checks and Pages deployment only: pull requests are checked and built, and pushes to `main` deploy the site. Build the native application locally using `Native/build.ps1`; native CUDA builds do not run on Actions.

## Provenance

- Original Clearwater: [Aureliengmz/clearwater](https://github.com/Aureliengmz/clearwater), commit `4bc826134321043a25df3c2b6fed16fb7b9241e8`, MIT, copyright 2026 Lumaris. Original license retained at [`LICENSE`](LICENSE).
- Pebble image: extracted without modification from the original embedded asset into `assets/seabed.jpg`.
- CUDA WebShader: vendored compiler and runtime from local `D:\cuda-webshader`, commit `9011955806cee30636ba24ae34b22d218e84196f`, MIT; license at [`vendor/cuda-webshader/LICENSE`](vendor/cuda-webshader/LICENSE).
- Original design references: Tessendorf (FFT water), Evan Wallace (refracted-grid caustics), Inigo Quilez (texture repetition), Olano & Baker (LEAN mapping).
