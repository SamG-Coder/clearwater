import { GpuRuntime } from "../vendor/cuda-webshader/runtime/runtime.js";
const $ = (id) => document.getElementById(id),
  canvas = $("pool");
const state = {
  time: 0,
  playing: true,
  wind: 4,
  rain: 0,
  clouds: 0.12,
  age: -1,
  angle: 0.32,
  elevation: 0.65,
  distance: 11.8,
  targetX: -1.7,
  targetZ: 0,
  frames: 0,
};
const diag = (window.poolDiagnostics = { ready: false, errors: [], state });
let rt,
  k = {},
  width = 0,
  height = 0,
  pixels,
  hdr,
  resize = true,
  last = 0,
  accumulator = 0,
  index = 0,
  tap = null;
const fail = (e) => {
  diag.errors.push(String(e.message || e));
  $("error").hidden = false;
  $("error").textContent = diag.errors.join("\n");
  $("loading").hidden = true;
  console.error(e);
};
function labels() {
  for (const [id, key, unit] of [
    ["wind", "wind", " m/s"],
    ["rain", "rain", "%"],
    ["cloud", "clouds", "%"],
  ]) {
    $(id).value = state[key];
    $(id + "Value").textContent =
      (key === "wind" ? state[key].toFixed(1) : Math.round(state[key] * 100)) +
      unit;
  }
}
for (const [id, key] of [
  ["wind", "wind"],
  ["rain", "rain"],
  ["cloud", "clouds"],
])
  $(id).oninput = () => {
    state[key] = +$(id).value;
    labels();
  };
function preset(w, r, c, id) {
  state.wind = w;
  state.rain = r;
  state.clouds = c;
  state.age = -1;
  $("auto").checked = false;
  for (const p of ["sunny", "breeze", "storm"])
    $(p).classList.toggle("active", p === id);
  labels();
}
$("sunny").onclick = () => preset(4, 0, 0.12, "sunny");
$("breeze").onclick = () => preset(12, 0, 0.35, "breeze");
$("storm").onclick = () => preset(21, 0.9, 0.88, "storm");
$("auto").onchange = () => {
  state.age = $("auto").checked ? 0 : -1;
};
function view(id) {
  Object.assign(
    state,
    id === "hero"
      ? {
          angle: 0.32,
          elevation: 0.65,
          distance: 11.8,
          targetX: -1.7,
          targetZ: 0,
        }
      : id === "waterline"
        ? { angle: -0.55, elevation: 0.18, distance: 9, targetX: 0, targetZ: 0 }
        : {
            angle: 0.65,
            elevation: 0.78,
            distance: 7,
            targetX: -3.3,
            targetZ: 0.2,
          },
  );
  for (const p of ["hero", "waterline", "tiles"])
    $(p).classList.toggle("active", p === id);
}
for (const id of ["hero", "waterline", "tiles"]) $(id).onclick = () => view(id);
$("pause").onclick = () => {
  state.playing = !state.playing;
  $("pause").textContent = state.playing ? "Pause" : "Resume";
};
$("hide").onclick = () => document.body.classList.add("clean");
$("restore").onclick = () => document.body.classList.remove("clean");
addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() === "h" || e.key === "Escape")
    document.body.classList.toggle("clean");
});
$("quality").onchange = () => (resize = true);
addEventListener("resize", () => (resize = true));
let drag = null;
canvas.onpointerdown = (e) => {
  drag = { x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY };
  canvas.setPointerCapture(e.pointerId);
};
canvas.onpointermove = (e) => {
  if (!drag) return;
  state.angle -= (e.clientX - drag.x) * 0.005;
  state.elevation = Math.max(
    0.1,
    Math.min(1.4, state.elevation + (e.clientY - drag.y) * 0.004),
  );
  drag.x = e.clientX;
  drag.y = e.clientY;
};
canvas.onpointerup = (e) => {
  if (
    drag &&
    Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) < 5
  ) {
    const c = camera(),
      sx = (e.clientX / innerWidth) * 2 - 1,
      sy = 1 - (e.clientY / innerHeight) * 2,
      aspect = width / height,
      cy = Math.cos(c.yaw),
      sn = Math.sin(c.yaw),
      cp = Math.cos(c.pitch),
      sp = Math.sin(c.pitch);
    const dx = sn * cp + sx * aspect * 0.62487 * cy - sy * 0.62487 * sn * sp,
      dy = sp + sy * 0.62487 * cp,
      dz = -cy * cp + sx * aspect * 0.62487 * sn + sy * 0.62487 * cy * sp;
    if (dy < 0)
      tap = { x: c.camX - (dx * c.camY) / dy, z: c.camZ - (dz * c.camY) / dy };
  }
  drag = null;
};
canvas.onwheel = (e) => {
  e.preventDefault();
  state.distance = Math.max(
    3,
    Math.min(24, state.distance * Math.exp(e.deltaY * 0.001)),
  );
};
function camera() {
  return {
    camX:
      state.targetX +
      Math.sin(state.angle) * state.distance * Math.cos(state.elevation),
    camY: Math.sin(state.elevation) * state.distance,
    camZ:
      state.targetZ +
      Math.cos(state.angle) * state.distance * Math.cos(state.elevation),
    yaw: -state.angle,
    pitch: -state.elevation,
  };
}
try {
  rt = await GpuRuntime.create({ onError: fail });
  const source = (
    await Promise.all(
      ["../src/clearwater.cu", "./pool.cu"].map((p) =>
        fetch(p).then((r) => {
          if (!r.ok) throw Error(p);
          return r.text();
        }),
      ),
    )
  ).join("\n");
  for (const entry of [
    "weather_update",
    "pool_wetness",
    "seed_spectrum",
    "spectrum_rows",
    "spectrum_norm",
    "evolve_spectrum",
    "fft_pass",
    "resolve_surface",
    "pool_step",
    "ripple_normals",
    "clear_caustics",
    "pool_caustics",
    "filter_caustics",
    "pool_render",
    "present",
  ])
    k[entry] = await rt.kernel(source, {
      entry,
      workgroupSize: ["spectrum_rows", "spectrum_norm"].includes(entry)
        ? [64, 1, 1]
        : [8, 8, 1],
    });
  const buf = (n) => rt.createBuffer(n),
    weather = buf(48),
    seed = buf(3 * 65536 * 8),
    rows = buf(768 * 4),
    scales = buf(12),
    spectralEnergy = rt.createBuffer(new Float32Array(3 * 65536).fill(1)),
    fft = [buf(3 * 65536 * 16), buf(3 * 65536 * 16)],
    surface = buf(3 * 65536 * 16),
    rip = [buf(65536 * 16), buf(65536 * 16)],
    normals = buf(65536 * 16),
    photons = buf(512 * 512 * 12),
    caustics = buf(512 * 512 * 16),
    dummy = buf(16);
  const WG = [32, 32, 3],
    RG = [32, 32, 1];
  rt.batch()
    .dispatch(k.seed_spectrum.bind({ seed }, { seedValue: 7 }), WG)
    .dispatch(k.spectrum_rows.bind({ seed, rows }), [12, 1, 1])
    .dispatch(k.spectrum_norm.bind({ rows, scales }), [1, 1, 1])
    .submit();
  await rt.idle();
  const ctx = canvas.getContext("webgpu");
  ctx.configure({
    device: rt.device,
    format: "rgba8unorm",
    alphaMode: "opaque",
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  async function frame(now) {
    try {
      const dt = last ? Math.min(0.05, (now - last) / 1000) : 1 / 60;
      last = now;
      if (resize) {
        await rt.idle();
        if (hdr) {
          rt.destroyBuffer(hdr);
          rt.destroyBuffer(pixels);
        }
        width = +$("quality").value;
        height = Math.ceil((width * innerHeight) / innerWidth / 8) * 8;
        canvas.width = width;
        canvas.height = height;
        hdr = buf(width * height * 16);
        pixels = buf(width * height * 4);
        resize = false;
      }
      const elapsed = state.playing ? dt : 0;
      state.time += elapsed;
      if (state.age >= 0) state.age += elapsed * 50;
      const batch = rt.batch(),
        weatherDt = state.frames === 0 ? 100 : elapsed * 5;
      batch.dispatch(
        k.weather_update.bind(
          { weather },
          {
            age: state.age,
            direction: 0.64,
            strength: 1,
            wind: state.wind,
            rain: state.rain,
            clouds: state.clouds,
            dt: weatherDt,
            camX: 0,
            camZ: 0,
          },
        ),
        [1, 1, 1],
      );
      batch.dispatch(
        k.pool_wetness.bind({ weather }, { dt: weatherDt }),
        [1, 1, 1],
      );
      batch.dispatch(
        k.evolve_spectrum.bind(
          { seed, scales, output: fft[0], weather, spectralEnergy },
          { time: state.time, sea: 1, depth: 1.5, weatherDt },
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
      accumulator += elapsed;
      let steps = 0;
      while (accumulator >= 1 / 120 && steps < 6) {
        batch.dispatch(
          k.pool_step.bind(
            { previous: rip[index], next: rip[1 - index], weather },
            {
              time: state.time - accumulator,
              tapX: tap?.x ?? 0,
              tapZ: tap?.z ?? 0,
              drop: tap ? 1 : 0,
            },
          ),
          RG,
        );
        tap = null;
        index = 1 - index;
        accumulator -= 1 / 120;
        steps++;
      }
      batch
        .dispatch(
          k.ripple_normals.bind({ input: rip[index], output: normals }),
          RG,
        )
        .dispatch(k.clear_caustics.bind({ photons }), [64, 64, 1])
        .dispatch(
          k.pool_caustics.bind({ surface, photons }, { depth: 1.5 }),
          [128, 128, 1],
        )
        .dispatch(k.filter_caustics.bind({ photons, caustics }), [64, 64, 1]);
      const grid = [width / 8, height / 8, 1];
      batch
        .dispatch(
          k.pool_render.bind(
            { surface, rip: normals, caustics, weather, hdr },
            { width, height, ...camera(), time: state.time },
          ),
          grid,
        )
        .dispatch(
          k.present.bind(
            { hdr, bloom: hdr, diffraction: dummy, image: pixels },
            { width, height, exposure: 0.95, glare: 0 },
          ),
          grid,
        )
        .submit();
      const enc = rt.device.createCommandEncoder();
      enc.copyBufferToTexture(
        {
          buffer: pixels.gpuBuffer,
          bytesPerRow: width * 4,
          rowsPerImage: height,
        },
        { texture: ctx.getCurrentTexture() },
        [width, height],
      );
      rt.device.queue.submit([enc.finish()]);
      await rt.idle();
      state.frames++;
      diag.ready = true;
      diag.readbackBytes = rt.stats.readbackBytes;
      diag.width = width;
      diag.height = height;
      $("loading").hidden = true;
      $("status").textContent = state.playing ? "LIVE / POOL STUDY" : "PAUSED";
      if (state.frames % 30 === 0)
        $("meter").textContent = `${width} × ${height} · CUDA / WEBGPU`;
      requestAnimationFrame(frame);
    } catch (e) {
      fail(e);
    }
  }
  window.poolLab = {
    state,
    preset,
    view,
    inspect: async () => {
      await rt.idle();
      const data = await rt.read(rip[index]);
      let finite = true,
        peak = 0,
        borderPeak = 0;
      for (let i = 0; i < data.length; i += 4) {
        finite &&= Number.isFinite(data[i]);
        peak = Math.max(peak, Math.abs(data[i]));
        const x = (i / 4) % 256,
          z = Math.floor(i / 1024);
        if (x < 16 || x > 240 || z < 16 || z > 240)
          borderPeak = Math.max(borderPeak, Math.abs(data[i]));
      }
      return {
        finite,
        peak,
        borderPeak,
        weather: Array.from(await rt.read(weather)),
        readbackBytes: rt.stats.readbackBytes,
      };
    },
  };
  labels();
  requestAnimationFrame(frame);
} catch (e) {
  fail(e);
}
