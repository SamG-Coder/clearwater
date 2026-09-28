# FFT-driven foam

Foam production uses the horizontal displacement Jacobian and slope. Absolute
wave height and independent procedural bubble/cellular masks are not inputs.
The deformation criterion follows the idea in Dupuy and Bruneton's
[Real-time Animation and Rendering of Ocean Whitecaps](https://liris.cnrs.fr/Documents/Liris-5812.pdf).
This is not an implementation of their complete statistical filtering model.

## Shared CUDA implementation

The two existing packed IFFT paths now produce eight real fields:

- height, X slope, Z slope, symmetric cross displacement derivative;
- X displacement, Z displacement, X diagonal derivative, Z diagonal derivative.

Derivatives are spectral multiplications. The existing inverse-transform sign
convention is retained and checked against analytic oblique and axis-aligned
waves. There are **no additional FFT passes**. The inverse surface mapping
stores deformation separately from the slope second moment used by lighting.

A 512 x 512 periodic history over 37 metres transports surface concentration,
freshness and age. Combined wind-wave and short-wave deformation drives
entrainment. Displacement history estimates transport velocity; limited
anti-diffusion reduces smearing. Wind changes the wave spectrum rather than
switching foam production off directly.

Two transported, mixed and decaying bubble reservoirs approximate shallow and
deeper entrained air. Refracted samples below the surface create a soft submerged
layer. Surface optical thickness responds to concentration, age and tensile
stretch of the short-wave field. This last film-rupture response is an optical
approximation, not a resolved bubble simulation.

Foam history consumes 24 MiB (two buffers, three float4 planes). Extra deformation
storage consumes 3 MiB. Together this is 23 MiB more than the old foam/surface
buffers. No image assets, CPU readbacks or separate fluid solver are added to
the render loop. Native and browser hosts allocate the same layouts.

## Reproduce validation

With the local server on port 5186:

```
node scripts/foam-spectral-verify.mjs
node scripts/foam-verify.mjs spectral-flat --flat-waves
node scripts/foam-verify.mjs spectral-final --motion
npm run check
npm run test:weather
```

The spectral oracle checks eight fields over 196,608 samples against independent
analytic formulas. Flat FFT water under storm wind must produce exactly zero
foam and zero bubble density. Lifecycle checks stop wave forcing explicitly,
then check that surface and submerged densities dissipate; lowering wind alone
should not erase a still-energetic sea. Native build and smoke tests run locally.

See `spectral-oracle.json`, `spectral-flat.json`, `spectral-final.json` and
`spectral-stability.json` for numerical evidence. GPU timings cover the complete
compute frame including ripple substeps, and exclude presentation.

## Visual boundary

The implementation follows the requested deformation/history mechanism. It does
not resolve individual bubbles, overturning geometry or three-dimensional fluid
motion. Close views still show broad, smooth interiors rather than the fine
bubble network of the supplied photographs. The submerged layer and froth
relief are optical approximations. The 37-metre history is periodic, and its
finite resolution limits filament detail. Screenshots and motion are evidence
of the current render, not a claim of photographic equivalence.

## Recorded result (local NVIDIA Blackwell GPU)

- Analytic packed-field maximum absolute error: 8.3504e-7.
- 1920 x 1080 complete GPU compute frame: median 3.639 ms, p95 3.888 ms,
  40 samples; excludes presentation and is not an end-to-end FPS claim.
- Ten seconds after wave forcing stops: surface mean drops 99.25%, deep bubble
  mean drops 97.76%.
- Flat-wave storm control: exactly zero surface and submerged foam.
- `spectral-final-motion.mp4`: 1920 x 1080, 30 fps, 90 frames; decoded with ffmpeg.
