# Foam and rain refinement

`detail-close.png` and `detail-close-next.png` are actual 1920 x 1080 browser renders one simulated second apart. `detail-storm.png` uses the original showcase angle. `before-storm.png` is the matched pre-change view.

The shared CUDA renderer now reconstructs the foam field with cubic B-spline filtering, uses triangular gradient noise for finer breakup, filters unresolved coverage by distance, and removes the minimum opacity that made old trails milky. Breaking is concentrated on compressed positive crests. This remains a surface-foam approximation, not a volumetric breaking-wave simulation.

Rain events use rotated spatial cells, full-cell random positions and independent cell periods. Visible rings and ripple impulses share those events. Impulses include neighbouring cells so cell boundaries do not clip splashes.

Run `node scripts/foam-verify.mjs review` with the local server on port 5186 to reproduce the review views, foam lifecycle assertions and GPU timing. In this run, the 40-sample 1080p full-compute median was 3.863 ms against 3.685 ms before changes (about 0.18 ms added). Timing excludes presentation and is not an FPS claim. The underlying foam field was zero in calm conditions and lost about 98% of its mean density after ten simulated seconds with breaking disabled.

Validation: all 36 kernels compiled through the browser compiler; weather integration checks passed; local native CUDA build and native smoke test passed. The close views were inspected at two simulation times; a new continuous showcase recording has not been made.
