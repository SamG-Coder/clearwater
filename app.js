import { GpuRuntime } from "./vendor/cuda-webshader/runtime/runtime.js";
const $ = (id) => document.getElementById(id),
  canvas = $("water"),
  q = new URLSearchParams(location.search);
const state = {
  x: 0,
  z: 0,
  y: 1.55,
  yaw: 0,
  pitch: -0.4,
  speed: 2.2,
  time: q.has("t") ? Number(q.get("t")) : 0,
  playing: !q.has("t"),
  frames: 0,
};
const diag = (window.clearwaterDiagnostics = {
  ready: false,
  errors: [],
  state,
  frames: 0,
  cascades: [4.6, 37, 293],
  fftSize: 256,
});
let rt,
  ctx,
  k = {},
  seed,
  scales,
  rows,
  fft,
  surface,
  rip,
  ripNormals,
  photons,
  caustics,
  pebbles,
  hdr,
  bloom,
  pixels,
  width = 0,
  height = 0,
  ripIndex = 0,
  centerX = 0,
  centerZ = 0,
  resizePending = true,
  lensKernel,
  lensFFT,
  last = 0,
  accumulator = 0,
  tap = null,
  failed = false,
  locked = false;
const keys = new Set(),
  WG = [32, 32, 3],
  RG = [32, 32, 1];
function fail(e) {
  failed = true;
  diag.errors.push(String(e.message || e));
  console.error(e);
  $("error").hidden = false;
  $("error").textContent = diag.errors.at(-1);
  $("loading").hidden = true;
  $("status").textContent = "UNAVAILABLE";
}
function labels() {
  for (const id of ["energy", "depth", "exposure"])
    $(id + "Value").textContent =
      Number($(id).value).toFixed(1) + (id === "depth" ? " m" : "");
}
for (const id of ["energy", "depth", "exposure"]) $(id).oninput = labels;
$("quality").onchange = () => (resizePending = true);
addEventListener("resize", () => (resizePending = true));
function play(value) {
  state.playing = value;
  $("pause").textContent = value ? "Ⅱ Pause" : "▶ Resume";
  $("status").textContent = value ? "LIVE / INFINITE SURFACE" : "PAUSED";
}
$("pause").onclick = () => play(!state.playing);
$("toggle").onclick = () => {
  document.body.classList.toggle("clean");
  $("toggle").textContent = document.body.classList.contains("clean")
    ? "Show controls ↙"
    : "Hide controls ↗";
};
$("reset").onclick = () =>
  Object.assign(state, {
    x: 0,
    z: 0,
    y: 1.55,
    yaw: 0,
    pitch: -0.4,
    speed: 2.2,
  });
for (const button of document.querySelectorAll("[data-preset]"))
  button.onclick = () => {
    const open = button.dataset.preset === "swell";
    $("energy").value = open ? 2.4 : 1;
    $("depth").value = open ? 12 : 1.6;
    state.y = open ? 3 : 1.55;
    state.pitch = open ? -0.19 : -0.4;
    document
      .querySelectorAll("[data-preset]")
      .forEach((b) => b.classList.toggle("active", b === button));
    labels();
  };
let drag = null;
canvas.onpointerdown = (e) => {
  drag = { x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY };
  canvas.setPointerCapture(e.pointerId);
};
canvas.onpointermove = (e) => {
  if (!drag) return;
  state.yaw += (e.clientX - drag.x) * 0.003;
  state.pitch = Math.max(
    -1.55,
    Math.min(1.55, state.pitch - (e.clientY - drag.y) * 0.003),
  );
  drag.x = e.clientX;
  drag.y = e.clientY;
};
canvas.onpointerup = (e) => {
  if (drag && Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) < 6)
    tap = [(e.clientX / innerWidth) * 2 - 1, 1 - (e.clientY / innerHeight) * 2];
  drag = null;
};
canvas.onpointercancel = () => (drag = null);
canvas.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const delta =
      e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? innerHeight : 1);
    state.speed = Math.max(
      0.1,
      Math.min(200, state.speed * Math.exp(-delta * 0.002)),
    );
  },
  { passive: false },
);
addEventListener("keydown", (e) => {
  if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
  keys.add(e.code);
  if (
    ["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(
      e.code,
    )
  )
    e.preventDefault();
  if (e.code === "Space" && !e.repeat) play(!state.playing);
  if (e.code === "KeyH" && !e.repeat) $("toggle").click();
});
addEventListener("keyup", (e) => keys.delete(e.code));
addEventListener("blur", () => keys.clear());
async function resize() {
  if (!resizePending) return;
  resizePending = false;
  await rt.idle();
  width = +$("quality").value;
  height = Math.ceil((width * innerHeight) / innerWidth / 8) * 8;
  canvas.width = width;
  canvas.height = height;
  for (const b of [hdr, ...(bloom || []), pixels]) if (b) rt.destroyBuffer(b);
  hdr = rt.createBuffer(width * height * 16);
  bloom = [
    rt.createBuffer(width * height * 16),
    rt.createBuffer(width * height * 16),
  ];
  pixels = rt.createBuffer(width * height * 4);
  ctx.configure({
    device: rt.device,
    format: "rgba8unorm",
    alphaMode: "opaque",
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  Object.assign(diag, { width, height });
}
function waves(batch) {
  batch.dispatch(
    k.evolve_spectrum.bind(
      { seed, scales, output: fft[0] },
      { time: state.time, sea: +$("energy").value, depth: +$("depth").value },
    ),
    WG,
  );
  let src = 0;
  for (let axis = 0; axis < 2; axis++)
    for (let p = 1; p < 256; p *= 2) {
      batch.dispatch(
        k.fft_pass.bind(
          { input: fft[src], output: fft[1 - src] },
          { p, axis, sign: 1 },
        ),
        WG,
      );
      src = 1 - src;
    }
  batch.dispatch(k.resolve_surface.bind({ input: fft[src], surface }), WG);
}
function ripples(batch, dt) {
  accumulator = Math.min(0.1, accumulator + dt);
  const steps = Math.floor(accumulator * 120);
  accumulator -= steps / 120;
  for (let i = 0; i < steps; i++) {
    const nx = Math.round(state.x * 16) / 16,
      nz = Math.round(state.z * 16) / 16,
      shiftX = Math.round((nx - centerX) * 16),
      shiftZ = Math.round((nz - centerZ) * 16);
    centerX = nx;
    centerZ = nz;
    batch.dispatch(
      k.ripple_step.bind(
        { previous: rip[ripIndex], next: rip[1 - ripIndex] },
        {
          shiftX,
          shiftZ,
          centerX,
          centerZ,
          camX: state.x,
          camZ: state.z,
          camY: state.y,
          yaw: state.yaw,
          pitch: state.pitch,
          aspect: width / height,
          tapX: tap?.[0] || 0,
          tapY: tap?.[1] || 0,
          drop: tap ? 1 : 0,
        },
      ),
      RG,
    );
    tap = null;
    ripIndex = 1 - ripIndex;
  }
  batch.dispatch(
    k.ripple_normals.bind({ input: rip[ripIndex], output: ripNormals }),
    RG,
  );
}
function transform(batch, pair, sign) {
  let src = 0;
  for (let axis = 0; axis < 2; axis++)
    for (let p = 1; p < 256; p *= 2) {
      batch.dispatch(
        k.fft_pass.bind(
          { input: pair[src], output: pair[1 - src] },
          { p, axis, sign },
        ),
        WG,
      );
      src = 1 - src;
    }
  return pair[src];
}
function setupLens() {
  const batch = rt.batch();
  batch.dispatch(k.lens_aperture.bind({ output: lensFFT[0] }), WG);
  transform(batch, lensFFT, -1);
  batch
    .dispatch(k.lens_power.bind({ amplitude: lensFFT[0], psf: lensFFT[1] }), WG)
    .dispatch(k.lens_rows.bind({ psf: lensFFT[1], sums: rows }), [12, 1, 1])
    .dispatch(
      k.lens_normalize.bind({
        psf: lensFFT[1],
        sums: rows,
        output: lensFFT[0],
      }),
      WG,
    );
  transform(batch, lensFFT, -1);
  batch.submit();
  const enc = rt.device.createCommandEncoder();
  enc.copyBufferToBuffer(
    lensFFT[0].gpuBuffer,
    0,
    lensKernel.gpuBuffer,
    0,
    3 * 65536 * 16,
  );
  rt.device.queue.submit([enc.finish()]);
}
function render() {
  const batch = rt.batch(),
    grid = [width / 8, height / 8, 1];
  batch
    .dispatch(k.clear_caustics.bind({ photons }), [64, 64, 1])
    .dispatch(
      k.trace_caustics.bind({ surface, photons }, { depth: +$("depth").value }),
      [128, 128, 1],
    )
    .dispatch(k.filter_caustics.bind({ photons, caustics }), [64, 64, 1]);
  batch.dispatch(
    k.render_water.bind(
      { surface, rip: ripNormals, caustics, pebbles, hdr },
      {
        width,
        height,
        camX: state.x,
        camZ: state.z,
        camY: state.y,
        yaw: state.yaw,
        pitch: state.pitch,
        centerX,
        centerZ,
        depth: +$("depth").value,
        time: state.time,
        view: +$("view").value,
      },
    ),
    grid,
  );
  if ($("glare").checked) {
    batch.dispatch(
      k.glare_source.bind({ hdr, output: lensFFT[0] }, { width, height }),
      WG,
    );
    transform(batch, lensFFT, -1);
    batch.dispatch(
      k.glare_multiply.bind({
        input: lensFFT[0],
        kernel: lensKernel,
        output: lensFFT[1],
      }),
      WG,
    );
    transform(batch, [lensFFT[1], lensFFT[0]], 1);
  }
  batch
    .dispatch(
      k.bloom_pass.bind(
        { input: hdr, output: bloom[0] },
        { width, height, axis: 0 },
      ),
      grid,
    )
    .dispatch(
      k.bloom_pass.bind(
        { input: bloom[0], output: bloom[1] },
        { width, height, axis: 1 },
      ),
      grid,
    )
    .dispatch(
      k.present.bind(
        { hdr, bloom: bloom[1], diffraction: lensFFT[1], image: pixels },
        {
          width,
          height,
          exposure: +$("exposure").value,
          glare: $("glare").checked ? 1 : 0,
        },
      ),
      grid,
    )
    .submit();
  const enc = rt.device.createCommandEncoder();
  enc.copyBufferToTexture(
    { buffer: pixels.gpuBuffer, bytesPerRow: width * 4, rowsPerImage: height },
    { texture: ctx.getCurrentTexture() },
    [width, height],
  );
  rt.device.queue.submit([enc.finish()]);
}
async function frame(now) {
  if (failed) return;
  try {
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 1 / 60;
    last = now;
    if (!locked && !document.hidden) {
      await resize();
      const start = performance.now(),
        boost = keys.has("ShiftLeft") || keys.has("ShiftRight") ? 6 : 1,
        speed = state.speed * boost * dt;
      state.yaw +=
        ((keys.has("ArrowRight") ? 1 : 0) - (keys.has("ArrowLeft") ? 1 : 0)) *
        dt;
      state.pitch = Math.max(
        -1.55,
        Math.min(
          1.55,
          state.pitch +
            ((keys.has("ArrowUp") ? 1 : 0) - (keys.has("ArrowDown") ? 1 : 0)) *
              dt,
        ),
      );
      let forward =
          (keys.has("KeyW") ? 1 : 0) -
          (keys.has("KeyS") ? 1 : 0) +
          ($("cruise").checked ? 1 : 0),
        side = (keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0),
        up = (keys.has("KeyE") ? 1 : 0) - (keys.has("KeyQ") ? 1 : 0);
      const length = Math.max(1, Math.hypot(forward, side, up));
      forward /= length;
      side /= length;
      up /= length;
      state.x +=
        speed *
        (Math.sin(state.yaw) * Math.cos(state.pitch) * forward +
          Math.cos(state.yaw) * side);
      state.z +=
        speed *
        (-Math.cos(state.yaw) * Math.cos(state.pitch) * forward +
          Math.sin(state.yaw) * side);
      state.y = Math.max(
        0.65,
        state.y + speed * (Math.sin(state.pitch) * forward + up),
      );
      if (state.playing) state.time += dt;
      const batch = rt.batch();
      waves(batch);
      ripples(batch, state.playing ? dt : 0);
      batch.submit();
      render();
      await rt.idle();
      diag.frameMs = performance.now() - start;
      diag.frames = ++state.frames;
      diag.ready = true;
      diag.readbackBytes = rt.stats.readbackBytes;
      $("loading").hidden = true;
      $("status").textContent = state.playing
        ? "LIVE / INFINITE SURFACE"
        : "PAUSED";
      if (state.frames % 15 === 0)
        $("metrics").textContent =
          `${Math.round(1 / dt)} FPS · ${width} × ${height} · SPEED ${(state.speed * boost).toFixed(1)} m/s · ${Math.round(state.x)}, ${Math.round(state.z)} m`;
    }
    requestAnimationFrame(frame);
  } catch (e) {
    fail(e);
  }
}
async function exclusive(fn) {
  locked = true;
  try {
    await rt.idle();
    return await fn();
  } finally {
    locked = false;
  }
}
window.clearwaterLab = {
  state,
  pause: () => play(false),
  resume: () => play(true),
  async inspect() {
    return exclusive(async () => {
      const a = await rt.read(surface),
        r = await rt.read(rip[ripIndex]);
      let min = Infinity,
        max = -Infinity,
        sum = 0,
        finite = true,
        ripplePeak = 0;
      for (let i = 0; i < a.length; i++) {
        finite &&= Number.isFinite(a[i]);
        if (i % 4 === 0) {
          min = Math.min(min, a[i]);
          max = Math.max(max, a[i]);
          sum += a[i] * a[i];
        }
      }
      for (let i = 0; i < r.length; i += 4)
        ripplePeak = Math.max(ripplePeak, Math.abs(r[i]));
      return {
        finite,
        min,
        max,
        rms: Math.sqrt(sum / (3 * 65536)),
        ripplePeak,
        adapter: rt.describe(),
        errors: diag.errors,
        stats: { ...rt.stats },
      };
    });
  },
  async seek(t) {
    return exclusive(async () => {
      state.time = t;
      play(false);
      const b = rt.batch();
      waves(b);
      b.submit();
      render();
      await rt.idle();
    });
  },
  async fftTest() {
    return exclusive(async () => {
      const n = 256,
        data = new Float32Array(n * n * 3 * 4);
      const modes = [
        { x: 1, z: 0, re: 0.5, im: 0, c: 0, f: 0 },
        { x: 255, z: 0, re: 0.5, im: 0, c: 0, f: 0 },
        { x: 3, z: 7, re: 0.3, im: -0.2, c: 1, f: 0 },
        { x: 11, z: 253, re: -0.4, im: 0.15, c: 2, f: 2 },
        { x: 51, z: 89, re: 0.07, im: 0.11, c: 0, f: 2 },
      ];
      for (const m of modes) {
        const i = (m.c * n * n + m.z * n + m.x) * 4 + m.f;
        data[i] = m.re;
        data[i + 1] = m.im;
      }
      const a = rt.createBuffer(data),
        b = rt.createBuffer(data.byteLength),
        batch = rt.batch();
      transform(batch, [a, b], 1);
      batch.submit();
      const out = await rt.read(a);
      let maxError = 0;
      for (let c = 0; c < 3; c++)
        for (let z = 0; z < n; z++)
          for (let x = 0; x < n; x++)
            for (let f = 0; f < 4; f += 2) {
              let re = 0,
                im = 0;
              for (const m of modes) {
                if (m.c !== c || m.f !== f) continue;
                const angle = (2 * Math.PI * (m.x * x + m.z * z)) / n;
                re += m.re * Math.cos(angle) - m.im * Math.sin(angle);
                im += m.re * Math.sin(angle) + m.im * Math.cos(angle);
              }
              const id = (c * n * n + z * n + x) * 4 + f;
              maxError = Math.max(
                maxError,
                Math.abs(out[id] - re),
                Math.abs(out[id + 1] - im),
              );
            }
      const back = rt.batch();
      transform(back, [a, b], -1);
      back.submit();
      const round = await rt.read(a);
      let roundtripError = 0;
      for (let i = 0; i < data.length; i++)
        roundtripError = Math.max(
          roundtripError,
          Math.abs(round[i] / (n * n) - data[i]),
        );
      rt.destroyBuffer(a);
      rt.destroyBuffer(b);
      return {
        maxError,
        roundtripError,
        modes: modes.length,
        axes: 2,
        cascades: 3,
        complexFields: 2,
      };
    });
  },
  async inspectOptics() {
    return exclusive(async () => {
      const ca = await rt.read(caustics),
        im = await rt.read(hdr),
        psf = await rt.read(lensKernel),
        gl = await rt.read(lensFFT[1]);
      const means = [0, 0, 0];
      for (let i = 0; i < ca.length; i += 4)
        for (let c = 0; c < 3; c++) means[c] += ca[i + c] / (512 * 512);
      return {
        causticMean: means,
        hdrFinite: im.every(Number.isFinite),
        glareFinite: gl.every(Number.isFinite),
        psfEnergy: [0, 1, 2].map((c) => psf[c * 65536 * 4] * 65536),
      };
    });
  },
};

$("capture").onclick = () =>
  exclusive(async () => {
    const data = await rt.read(pixels, Uint32Array),
      c = document.createElement("canvas");
    c.width = width;
    c.height = height;
    c.getContext("2d").putImageData(
      new ImageData(new Uint8ClampedArray(data.buffer), width, height),
      0,
      0,
    );
    c.toBlob((blob) => {
      const url = URL.createObjectURL(blob),
        a = document.createElement("a");
      a.href = url;
      a.download = "Clearwater.png";
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    });
  });
try {
  rt = await GpuRuntime.create({ onError: fail });
  ctx = canvas.getContext("webgpu");
  const source = await (await fetch("./src/clearwater.cu")).text();
  for (const name of [...source.matchAll(/__global__ void (\w+)/g)].map(
    (m) => m[1],
  )) {
    $("loadText").textContent = `Compiling ${name.replaceAll("_", " ")}…`;
    k[name] = await rt.kernel(source, {
      entry: name,
      workgroupSize: ["spectrum_rows", "spectrum_norm", "lens_rows"].includes(
        name,
      )
        ? [64, 1, 1]
        : [8, 8, 1],
    });
  }
  lensKernel = rt.createBuffer(3 * 65536 * 16);
  lensFFT = [rt.createBuffer(3 * 65536 * 16), rt.createBuffer(3 * 65536 * 16)];
  seed = rt.createBuffer(3 * 65536 * 8);
  rows = rt.createBuffer(768 * 4);
  scales = rt.createBuffer(3 * 4);
  fft = [rt.createBuffer(3 * 65536 * 16), rt.createBuffer(3 * 65536 * 16)];
  surface = rt.createBuffer(3 * 65536 * 16);
  rip = [rt.createBuffer(65536 * 16), rt.createBuffer(65536 * 16)];
  ripNormals = rt.createBuffer(65536 * 16);
  photons = rt.createBuffer(512 * 512 * 3 * 4);
  caustics = rt.createBuffer(512 * 512 * 16);
  // Asset decoding only: the source asset is converted to linear floats once.
  const bitmap = await createImageBitmap(
      await (await fetch("./assets/seabed.jpg")).blob(),
    ),
    off = new OffscreenCanvas(1024, 1024),
    dc = off.getContext("2d");
  dc.drawImage(bitmap, 0, 0, 1024, 1024);
  const rgba = dc.getImageData(0, 0, 1024, 1024).data,
    linear = new Float32Array(1024 * 1024 * 4);
  for (let i = 0; i < rgba.length; i++)
    linear[i] = i % 4 === 3 ? 1 : Math.pow(rgba[i] / 255, 2.2);
  pebbles = rt.createBuffer(linear);
  bitmap.close();
  rt.batch()
    .dispatch(k.seed_spectrum.bind({ seed }, { seedValue: 7 }), WG)
    .dispatch(k.spectrum_rows.bind({ seed, rows }), [12, 1, 1])
    .dispatch(k.spectrum_norm.bind({ rows, scales }), [1, 1, 1])
    .submit();
  await rt.idle();
  setupLens();
  await rt.idle();
  labels();
  requestAnimationFrame(frame);
} catch (e) {
  fail(e);
}
