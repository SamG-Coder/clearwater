# Persistent crest foam

This replaces the rejected speckle-based appearance in commit 7ffc09c. The old captures and README document that earlier iteration, not the current rendering model.

The source is still shared CUDA. Foam is deposited at compressed positive crests, using the same inverse displacement map as the rendered water. Coverage, fresh foam and density-weighted age persist between frames. A second buffer plane stores displacement history and advected material offsets, so residual patches deform with an approximate surface velocity derived from displacement changes. Velocity is bounded around time jumps. This is a surface approximation, not an overturning fluid solver.

Fresh foam is optically denser. Aging residue thins unevenly, with connected films around variable-size rounded openings. Fine bubbles affect shading rather than creating detached white speckles. Unresolved detail blends to average coverage. Fine shading is skipped in foam-free regions. Both browser and native hosts allocate the extra history plane: 2 MiB total additional GPU memory, with no additional dispatch.

## Evidence

- `accepted-close.png` / `accepted-close-next.png`: matched close views one simulated second apart.
- `accepted-storm.png`: wider showcase view.
- `verified-motion.mp4`: 3 seconds, 1920 x 1080, 30 fps, 90 fixed-timestep frames from actual browser output. It is an inspection clip, not a real-time performance recording. Decoded fully without errors; frames at 0, 1 and 2 seconds were visually inspected.
- `accepted.json`: 40-sample GPU compute median 3.781 ms; previous implementation measured 3.863 ms in `detail.json`. These separate runs are similar, not evidence of a statistically established speedup. Presentation and CPU overhead are excluded.
- Zero foam in calm water. After breaking was disabled, mean density fell from 0.020642 to 0.001411 in ten simulated seconds (about 93%). All state remained finite.
- All 36 browser kernels compiled. Browser weather integration and quality stability checks passed, including pause image equality, three resolutions and camera stress views. Local native build and smoke test passed. Native CI remains disabled.

Reproduce with `node scripts/foam-verify.mjs review --motion` while serving the project on port 5186. Omit `--motion` for screenshots and lifecycle/timing checks only.

Visual direction was informed by the distinction between active breaking fronts and residual patches in [Kleiss and Melville, 2011](https://airsea.ucsd.edu/wp-content/uploads/sites/10/2019/06/2011_Kleiss_Melville-Journal_of_Atmospheric_and_Oceanic_Technology_vol_28.pdf), and variable foam decay and deformation in [Callaghan et al., 2012](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2012JC008147). The implementation is an artistic real-time approximation, not a reproduction of those physical measurements.
