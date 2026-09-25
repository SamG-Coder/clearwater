# Clearwater Native

Native Windows CUDA application with two independent windows: a water view and a control panel. The application directly includes `../src/clearwater.cu`; it does not fork or translate the water implementation. No browser or WebGPU is used in this build.

## Build and run

Native builds are local only; GitHub Actions handles the browser version and Pages deployment.

Requires Windows 10/11, an NVIDIA CUDA-capable GPU with an up-to-date driver, CUDA Toolkit 13.x, and Visual Studio C++ build tools with the Windows SDK.

```powershell
pwsh -File Native/build.ps1
./Native/build/ClearwaterNative.exe
```

The default architecture is the installed GPU. For a distributable forward-compatible PTX build use `pwsh -File Native/build.ps1 -Architecture compute_75` (Turing or newer). CUDA Toolkit 13 requires an appropriately recent NVIDIA driver. The CUDA runtime and MSVC runtime are linked statically. Keep `assets/seabed.jpg` next to the executable inside `assets/`.

## Controls

- Drag the water view or use arrow keys to look. Right/left and up/down follow the input direction.
- WASD flies; W follows the viewing direction. E rises, Q descends.
- Scroll up/down changes movement speed; Shift temporarily gives 6x boost.
- Click nearby water to add ripples. Space pauses the waves.
- H hides/restores the controls window. Closing only the controls hides it; closing the water view or pressing Escape exits.
- The separate panel exposes presets, energy, depth, exposure, flight speed, resolution, water/caustic/normal views, lens glare, continuous drift, pause, reset and PNG export.
- Save PNG writes a timestamped file under `output/` beside the executable.

CUDA renders the complete image into a device buffer. CUDA–D3D11 interop transfers it to a registered GPU texture, and Direct3D copies it to the swap chain. There are no native vertex/pixel shaders and no per-frame CPU image readback. Windows/WIC only provide windows, controls, asset decoding and explicit PNG export.

## Validation

```powershell
./Native/build/ClearwaterNative.exe --smoke
```

The smoke run opens both windows, tests the shared FFT against analytic reference modes, tests fly-camera directions/speed, generates a ripple, verifies finite simulation/HDR values at 10 km coordinates, exercises resize/diagnostic/glare paths, and writes PNG captures plus `output/native-smoke.json`. A failure returns exit code 1 and writes `output/error.log`.

The browser and native hosts use the same 20 kernels, spectrum dimensions and optical settings. GPU float math/compiler differences can cause small image differences. The same height-field and extreme-distance precision limitations as the browser version apply.
