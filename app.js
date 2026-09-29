import { GpuRuntime } from "./vendor/cuda-webshader/runtime/runtime.js";
const $ = (id) => document.getElementById(id),
  canvas = $("water"),
  q = new URLSearchParams(location.search);
const state = {
  x: 0,
  z: 0,
  y: 1.55,
  yaw: 0,
  pitch: -0.24,
  speed: 2.2,
  time: q.has("t") ? Number(q.get("t")) : 0,
  playing: !q.has("t"),
  weatherAge: -1,
  weatherRate: 30,
  stormStrength: 1,
  wind: 5,
  rain: 0,
  clouds: 0.28,
  direction: 0.64,
  buoy: false,
  ducks: false, duckX: 0, duckZ: -60,
  tornado: false,
  tornadoX: 0,
  tornadoZ: -650,
  tornadoStrength: 1,
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
let chopFFT, linearSurface, environment, weather,
  spectralEnergy,
  foam,
  foamIndex = 0,
  weatherDt = 0,
  frameDt = 0;
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
const tornadoGrid = [6, 576, 1], tornadoCells = 48 * 96 * 48;
let tornadoVelocity, tornadoMoisture, tornadoPressure, tornadoRhs, tornadoSeaState, tornadoVolumeCache, tornadoCurl;
let tornadoCloudMap, tornadoLighting, tornadoLightingDirty = true;
let tornadoWaveFFT, tornadoWaveModes, tornadoWaveReset = true;
let tornadoSpray, tornadoSprayImage, tornadoSprayReset = true;
let skyEnvironmentCache = null, skyViewCache = null, tornadoViewCache = null;
let tornadoMoistureIndex = 0, tornadoSeaIndex = 0, tornadoPressureIndex = 0;
let tornadoAccumulator = 0, tornadoReset = true, tornadoSteps = 0;
function tornadoStep(batch, dt) {
  if (!state.tornado) return;
  if (tornadoReset) {
    skyEnvironmentCache=skyViewCache=tornadoViewCache=null;
    tornadoSeaIndex = 0;
    batch.dispatch(k.tornado_init.bind({velocity:tornadoVelocity[0],moisture:tornadoMoisture[0],pressure:tornadoPressure[0],cloudMap:tornadoCloudMap[0],sea:tornadoSeaState[0]}, {strength:state.tornadoStrength}), tornadoGrid);
    tornadoMoistureIndex = tornadoPressureIndex = 0;
    tornadoReset = false;
    tornadoLightingDirty = true;
    tornadoWaveReset = true;
    tornadoSprayReset = true;
  }
  tornadoAccumulator = Math.min(tornadoAccumulator + dt, 2 / 30);
  while (tornadoAccumulator >= 1 / 30) {
    batch.dispatch(k.tornado_curl.bind({velocity:tornadoVelocity[0],curl:tornadoCurl}), tornadoGrid);
    batch.dispatch(k.tornado_advect.bind({velocity:tornadoVelocity[0],moisture:tornadoMoisture[tornadoMoistureIndex],curl:tornadoCurl,cloudMap:tornadoCloudMap[tornadoMoistureIndex],nextVelocity:tornadoVelocity[1],nextMoisture:tornadoMoisture[1-tornadoMoistureIndex],nextCloudMap:tornadoCloudMap[1-tornadoMoistureIndex]}, {dt:1/30,strength:state.tornadoStrength,time:tornadoSteps/30}), tornadoGrid);
    tornadoMoistureIndex = 1 - tornadoMoistureIndex;
    batch.dispatch(k.tornado_divergence.bind({velocity:tornadoVelocity[1],rhs:tornadoRhs}), tornadoGrid);
    for (let i=0;i<12;i++) {
      batch.dispatch(k.tornado_pressure.bind({previous:tornadoPressure[tornadoPressureIndex],rhs:tornadoRhs,next:tornadoPressure[1-tornadoPressureIndex]}), tornadoGrid);
      tornadoPressureIndex = 1 - tornadoPressureIndex;
    }
    batch.dispatch(k.tornado_project.bind({input:tornadoVelocity[1],pressure:tornadoPressure[tornadoPressureIndex],output:tornadoVelocity[0]}), tornadoGrid);
    batch.dispatch(k.tornado_water.bind({previous:tornadoSeaState[tornadoSeaIndex],next:tornadoSeaState[1-tornadoSeaIndex],velocity:tornadoVelocity[0],pressure:tornadoPressure[tornadoPressureIndex]}, {dt:1/30,depth:+$("depth").value}), [12,12,1]);
    tornadoSeaIndex = 1 - tornadoSeaIndex;
    batch.dispatch(k.tornado_wave_force.bind({velocity:tornadoVelocity[0],pressure:tornadoPressure[tornadoPressureIndex],surface,forcing:tornadoWaveFFT[0]}, {tx:state.tornadoX,tz:state.tornadoZ,dt:1/30}),[32,32,1]);
    transform(batch,tornadoWaveFFT,-1,[32,32,1]);
    batch.dispatch(k.tornado_wave_evolve.bind({forcing:tornadoWaveFFT[0],modes:tornadoWaveModes,output:tornadoWaveFFT[1]}, {dt:1/30,depth:+$("depth").value,reset:tornadoWaveReset?1:0}),[32,32,1]);
    transform(batch,[tornadoWaveFFT[1],tornadoWaveFFT[0]],1,[32,32,1]);
    batch.dispatch(k.tornado_wave_resolve.bind({input:tornadoWaveFFT[1],sea:tornadoSeaState[tornadoSeaIndex]}),[32,32,1]);
    batch.dispatch(k.tornado_spray_step.bind({particles:tornadoSpray,velocity:tornadoVelocity[0],sea:tornadoSeaState[tornadoSeaIndex],surface}, {tx:state.tornadoX,tz:state.tornadoZ,dt:1/30,tick:tornadoSteps,reset:tornadoSprayReset?1:0}),[32,4,1]);
    tornadoSprayReset=false;
    tornadoWaveReset=false;
    tornadoAccumulator -= 1 / 30;
    tornadoSteps++;
    tornadoLightingDirty = true;
  }
}
const duckGrid=[16,10,1];
let duckBodies,duckHeads,duckLinks,duckDepth,duckNodes,duckIndex=0,duckReset=true,duckAccumulator=0,duckSteps=0;
function ducksHash(batch) {
  batch.dispatch(k.ducks_hash_clear.bind({heads:duckHeads}),[32,32,1]);
  batch.dispatch(k.ducks_hash.bind({bodies:duckBodies[duckIndex],heads:duckHeads,links:duckLinks}),duckGrid);
}
function ducksStep(batch,dt) {
  if(!state.ducks)return;
  if(duckReset) {
    duckIndex=0;duckAccumulator=0;duckSteps=0;
    batch.dispatch(k.ducks_init.bind({bodies:duckBodies[0],surface,sea:tornadoSeaState[tornadoSeaIndex]}, {originX:state.duckX,originZ:state.duckZ,tornadoOn:state.tornado?1:0,tx:state.tornadoX,tz:state.tornadoZ}),duckGrid);
    duckReset=false;
  }
  duckAccumulator=Math.min(duckAccumulator+dt,4/120);
  while(duckAccumulator>=1/120) {
    ducksHash(batch);
    batch.dispatch(k.ducks_step.bind({previous:duckBodies[duckIndex],next:duckBodies[1-duckIndex],heads:duckHeads,links:duckLinks,surface,sea:tornadoSeaState[tornadoSeaIndex],air:tornadoVelocity[0],weather},{dt:1/120,tornadoOn:state.tornado?1:0,tx:state.tornadoX,tz:state.tornadoZ,time:duckSteps/120}),duckGrid);
    duckIndex=1-duckIndex;duckAccumulator-=1/120;duckSteps++;
  }
  if(dt>0){
    ducksHash(batch);
    batch.dispatch(k.ducks_wakes.bind({bodies:duckBodies[duckIndex],heads:duckHeads,links:duckLinks,ripples:rip[ripIndex]},{centerX,centerZ,dt:Math.min(dt,4/120)}),RG);
  }
}
function duckCamera(close=false) {
  state.x=state.duckX;state.z=state.duckZ+(close?51:90);state.y=close?.8:38;
  state.yaw=0;state.pitch=close?-.11:-.48;
}
$("ducks").onclick=()=>{
  state.ducks=!state.ducks;$("ducks").textContent=state.ducks?"Remove 10,000 ducks":"Add 10,000 rubber ducks";
  $("duckStatus").textContent=state.ducks?"10,000 bodies \u00b7 120 Hz physics":"Floating rigid bodies \u00b7 wave buoyancy \u00b7 collisions";
  if(state.ducks){state.duckX=state.tornado?state.tornadoX:state.x+Math.sin(state.yaw)*60;state.duckZ=state.tornado?state.tornadoZ:state.z-Math.cos(state.yaw)*60;duckReset=true;duckCamera();play(true);}
};
$("duckOverview").onclick=()=>{if(state.ducks)duckCamera();};
$("duckClose").onclick=()=>{if(state.ducks)duckCamera(true);};
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
    pitch: -0.24,
    speed: 2.2,
  });
for (const button of document.querySelectorAll("[data-preset]"))
  button.onclick = () => {
    const open = button.dataset.preset === "swell";
    $("energy").value = open ? 2.4 : 1;
    $("depth").value = open ? 12 : 1.6;
    state.y = open ? 3 : 1.55;
    state.pitch = open ? -0.19 : -0.24;
    document
      .querySelectorAll("[data-preset]")
      .forEach((b) => b.classList.toggle("active", b === button));
    labels();
  };
$("storm").onclick = () => {
  state.weatherAge = 0;
  play(true);
};
$("clearWeather").onclick = () => {
  state.tornado = false;
  $("tornado").textContent = "Start tornado";
  state.weatherAge = -1;
  state.wind = 5;
  state.rain = 0;
  state.clouds = 0.28;
  $("wind").value = 5;
  $("rain").value = 0;
  $("clouds").value = 0.28;
};
for (const id of ["wind", "rain", "clouds", "direction", "stormStrength"])
  $(id).oninput = () => {
    state[id] = +$(id).value;
  };
$("weatherRate").onchange = () => (state.weatherRate = +$("weatherRate").value);
$("buoy").onchange = () => (state.buoy = $("buoy").checked);
$("tornado").onclick = () => {
  state.tornado = !state.tornado;
  $("tornado").textContent = state.tornado ? "Stop tornado" : "Start tornado";
  if (state.tornado) {
    state.tornadoX = state.ducks ? state.duckX : state.x + Math.sin(state.yaw) * 650;
    state.tornadoZ = state.ducks ? state.duckZ : state.z - Math.cos(state.yaw) * 650;
    if(!state.ducks)state.pitch = .22;
    state.clouds = .62;
    $("clouds").value = state.clouds;
    tornadoReset = true;
    tornadoAccumulator = 0;
    play(true);
  }
};
$("tornadoStrength").oninput = () => (state.tornadoStrength = +$("tornadoStrength").value);
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
  for (const kernel of Object.values(k)) kernel.clearBindings();
  width = +$("quality").value;
  height = Math.ceil((width * innerHeight) / innerWidth / 8) * 8;
  canvas.width = width;
  canvas.height = height;
  for (const b of [environment, hdr, ...(bloom || []), pixels, tornadoVolumeCache,tornadoSprayImage,duckDepth]) if (b) rt.destroyBuffer(b);
  skyEnvironmentCache=skyViewCache=tornadoViewCache=null;
  duckDepth=rt.createBuffer(width*height*4);
  tornadoSprayImage=rt.createBuffer(width*height*4);
  tornadoVolumeCache=rt.createBuffer(width*height/2*16);
  environment=rt.createBuffer((81920+width*height/4)*16);
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
    k.weather_update.bind(
      { weather },
      {
        age: state.weatherAge,
        direction: state.direction,
        strength: state.stormStrength,
        wind: state.wind,
        rain: state.rain,
        clouds: state.clouds,
        dt: weatherDt,
        camX: state.x,
        camZ: state.z,
      },
    ),
    [1, 1, 1],
  );
  batch.dispatch(
    k.evolve_spectrum.bind(
      { seed, scales, output: fft[0], weather, spectralEnergy },
      {
        time: state.time,
        sea: +$("energy").value,
        depth: +$("depth").value,
        weatherDt,
      },
    ),
    WG,
  );
  batch.dispatch(k.chop_spectrum.bind({input:fft[0],output:chopFFT[0]},{sea:+$("energy").value}),WG);
  transform(batch,chopFFT,1);
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
  batch.dispatch(k.resolve_surface.bind({ input: fft[src], surface:linearSurface }), WG);
  batch.dispatch(k.chop_surface.bind({input:linearSurface,displacement:chopFFT[0],surface}),WG);
  batch.dispatch(
    k.ocean_foam.bind(
      {
        surface,
        displacement:chopFFT[0],
        previous: foam[foamIndex],
        next: foam[1 - foamIndex],
        weather,
      },
      { dt: frameDt },
    ),
    [64,64,1],
  );
  foamIndex = 1 - foamIndex;
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
        { previous: rip[ripIndex], next: rip[1 - ripIndex], weather },
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
          time: state.time - accumulator + i / 120,
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
function transform(batch, pair, sign, grid = WG) {
  let src = 0;
  for (let axis = 0; axis < 2; axis++)
    for (let p = 1; p < 256; p *= 2) {
      batch.dispatch(
        k.fft_pass.bind(
          { input: pair[src], output: pair[1 - src] },
          { p, axis, sign },
        ),
        grid,
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
function render(timestampWrites, simulate = false) {
  const batch = rt.batch({ timestampWrites }),
    grid = [width / 8, height / 8, 1];
  if (simulate) { waves(batch); ripples(batch, 1/60); tornadoStep(batch,1/30); ducksStep(batch,1/60); }
  if(state.tornado) {
    const volumeKey=[state.x,state.y,state.z,state.yaw,state.pitch,width,height,state.tornadoX,state.tornadoZ,tornadoSteps].join(':');
    if(tornadoLightingDirty) {
      batch.dispatch(k.tornado_light.bind({moisture:tornadoMoisture[tornadoMoistureIndex],cloudMap:tornadoCloudMap[tornadoMoistureIndex],lighting:tornadoLighting}),tornadoGrid);
      tornadoLightingDirty=false;
    }
    if(tornadoViewCache!==volumeKey) {
      batch.dispatch(k.tornado_view.bind({lighting:tornadoLighting,volume:tornadoVolumeCache}, {width,height,camX:state.x,camY:state.y,camZ:state.z,yaw:state.yaw,pitch:state.pitch,tx:state.tornadoX,tz:state.tornadoZ}),[Math.ceil(width/16),Math.ceil(height/16),1]);
      tornadoViewCache=volumeKey;
    }
  }
  const cloudBuffers={environment,weather,cloudMap:tornadoCloudMap[tornadoMoistureIndex]};
  const cloudParams={camX:state.x,camY:state.y,camZ:state.z,time:state.time,tornadoOn:state.tornado?1:0,tx:state.tornadoX,tz:state.tornadoZ};
  const {camY: unusedCloudHeight,...shadowParams}=cloudParams;
  const skyKey=[state.x,state.y,state.z,state.yaw,state.pitch,width,height,state.clouds,state.wind,state.rain,state.direction,state.stormStrength,state.tornado,state.tornadoX,state.tornadoZ,state.weatherAge>=0].join(':');
  const refreshSky=(cache,seconds,steps)=>!cache||cache.key!==skyKey||state.time<cache.time||state.time-cache.time>=seconds||tornadoSteps-cache.steps>=steps||state.weatherAge>=0;
  if(refreshSky(skyEnvironmentCache,.1,3)) {
    batch.dispatch(k.sky_environment.bind(cloudBuffers,cloudParams),[64,16,1]);
    batch.dispatch(k.cloud_shadow.bind(cloudBuffers,shadowParams),[16,16,1]);
    skyEnvironmentCache={key:skyKey,time:state.time,steps:tornadoSteps};
  }
  if(refreshSky(skyViewCache,1/30,2)) {
    batch.dispatch(k.sky_view.bind(cloudBuffers,{...cloudParams,width,height,yaw:state.yaw,pitch:state.pitch}),[Math.ceil(width/16),Math.ceil(height/16),1]);
    skyViewCache={key:skyKey,time:state.time,steps:tornadoSteps};
  }
  batch
    .dispatch(k.clear_caustics.bind({ photons }), [64, 64, 1])
    .dispatch(
      k.trace_caustics.bind({ surface, photons }, { depth: +$("depth").value }),
      [128, 128, 1],
    )
    .dispatch(k.filter_caustics.bind({ photons, caustics }), [64, 64, 1]);
  batch.dispatch(
    k.render_water.bind(
      {
        surface,
        rip: ripNormals,
        caustics,
        pebbles,
        hdr,
        weather,
        foam: foam[foamIndex],
        environment,
        tornadoVolumeCache,
        tornadoSeaState: tornadoSeaState[tornadoSeaIndex],
      },
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
        buoy: state.buoy ? 1 : 0,
        tornadoOn: state.tornado ? 1 : 0,
        tornadoX: state.tornadoX,
        tornadoZ: state.tornadoZ,
      },
    ),
    grid,
  );
  if(state.ducks && +$("view").value===0) {
    const camera={width,height,camX:state.x,camY:state.y,camZ:state.z,yaw:state.yaw,pitch:state.pitch};
    batch.dispatch(k.ducks_pixels_clear.bind({depth:duckDepth},{width,height}),grid);
    batch.dispatch(k.ducks_project.bind({bodies:duckBodies[duckIndex],depth:duckDepth,nodes:duckNodes},camera),duckGrid);
    batch.dispatch(k.ducks_shade.bind({bodies:duckBodies[duckIndex],depth:duckDepth,nodes:duckNodes,hdr,surface,sea:tornadoSeaState[tornadoSeaIndex],environment},{...camera,tornadoOn:state.tornado?1:0,tx:state.tornadoX,tz:state.tornadoZ}),grid);
  }
  if(state.tornado && +$("view").value===0) {
    batch.dispatch(k.tornado_spray_clear.bind({spray:tornadoSprayImage},{width,height}),grid);
    batch.dispatch(k.tornado_spray_draw.bind({particles:tornadoSpray,spray:tornadoSprayImage},{width,height,camX:state.x,camY:state.y,camZ:state.z,yaw:state.yaw,pitch:state.pitch,tx:state.tornadoX,tz:state.tornadoZ}),[32,4,1]);
    batch.dispatch(k.tornado_spray_composite.bind({spray:tornadoSprayImage,hdr},{width,height}),grid);
  }
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
      frameDt = state.playing ? dt : 0;
      weatherDt = frameDt * state.weatherRate;
      if (state.playing) {
        state.time += dt;
        if (state.weatherAge >= 0) state.weatherAge += weatherDt;
      }
      $("weatherPhase").textContent =
        state.weatherAge < 0
          ? "FAIR WEATHER"
          : state.weatherAge < 400
            ? "FRONT APPROACHING"
            : state.weatherAge < 1200
              ? "SQUALL PASSING"
              : state.weatherAge < 2100
                ? "CLEARING / RESIDUAL SWELL"
                : "AFTER THE STORM";
      const batch = rt.batch();
      waves(batch);
      ripples(batch, state.playing ? dt : 0);
      tornadoStep(batch, frameDt);
      ducksStep(batch,frameDt);
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
  async ducksAdvance(steps=120) {
    return exclusive(async()=>{play(false);for(let i=0;i<steps;i++){
      frameDt=1/120;weatherDt=0;state.time+=frameDt;
      const b=rt.batch();waves(b);ripples(b,frameDt);tornadoStep(b,frameDt);ducksStep(b,frameDt);b.submit();
      if(i%12===11)await rt.idle();
    }render();await rt.idle();});
  },
  async ducksBins() {return exclusive(async()=>{const a=await rt.read(duckDepth,Uint32Array);return {nodes:a[(width/8)*(height/8)],overflow:a[(width/8)*(height/8)+1]};});},
  async ducksInspect() {return exclusive(async()=>({steps:duckSteps,bodies:Array.from(await rt.read(duckBodies[duckIndex]))}));},
  async ducksWrite(data) {return exclusive(async()=>{rt.write(duckBodies[duckIndex],new Float32Array(data));});},
  async tornadoCouplingTest() {
    return exclusive(async () => {
      const calm=rt.createBuffer(3*65536*16),coupled=rt.createBuffer(65536*16),control=rt.createBuffer(65536*16);
      try {
        const batch=rt.batch(),params={tx:state.tornadoX,tz:state.tornadoZ,dt:1/30};
        batch.dispatch(k.tornado_wave_force.bind({velocity:tornadoVelocity[0],pressure:tornadoPressure[tornadoPressureIndex],surface,forcing:coupled},params),[32,32,1]);
        batch.dispatch(k.tornado_wave_force.bind({velocity:tornadoVelocity[0],pressure:tornadoPressure[tornadoPressureIndex],surface:calm,forcing:control},params),[32,32,1]);
        batch.submit();await rt.idle();
        const a=await rt.read(coupled),b=await rt.read(control);
        let windWork=0,controlWindWork=0,pressureDifference=0,farForce=0;
        for(let z=0;z<256;z++)for(let x=0;x<256;x++) {
          const i=(z*256+x)*4;
          windWork+=a[i+2]*a[i+2];controlWindWork+=b[i+2]*b[i+2];
          pressureDifference=Math.max(pressureDifference,Math.abs(a[i]-b[i]));
          if(Math.hypot(x-128,z-128)*1.125>=125)farForce=Math.max(farForce,Math.abs(a[i]),Math.abs(a[i+2]));
        }
        return {windWorkRms:Math.sqrt(windWork/65536),controlWindWorkRms:Math.sqrt(controlWindWork/65536),pressureDifference,farForce};
      } finally {
        for(const b of [calm,coupled,control])rt.destroyBuffer(b);
        k.tornado_wave_force.clearBindings();
      }
    });
  },
  async tornadoAdvance(steps = 30) {
    return exclusive(async () => {
      play(false);
      for (let i=0;i<steps;i++) {
        const batch=rt.batch();tornadoStep(batch,1/30);batch.submit();
        if(i%10===9)await rt.idle();
      }
      render();await rt.idle();
    });
  },
  async tornadoInspect() {
    return exclusive(async () => ({
      velocity:Array.from(await rt.read(tornadoVelocity[0])),
      unprojected:Array.from(await rt.read(tornadoVelocity[1])),
      moisture:Array.from(await rt.read(tornadoMoisture[tornadoMoistureIndex])),
      sea:Array.from(await rt.read(tornadoSeaState[tornadoSeaIndex])),
      cloudMap:Array.from(await rt.read(tornadoCloudMap[tornadoMoistureIndex])),
      sprayParticles:Array.from(await rt.read(tornadoSpray)),
      waveModes:Array.from(await rt.read(tornadoWaveModes)),
      steps:tornadoSteps,
    }));
  },
  // Explicit diagnostic only: timestamp readbacks never enter the live frame loop.
  async benchmark(samples = 40, simulate = false) {
    return exclusive(async () => {
      play(false);
      if (!rt.device.features.has("timestamp-query")) throw Error("GPU timestamps unavailable");
      const querySet = rt.device.createQuerySet({type: "timestamp", count: 2});
      const resolve = rt.device.createBuffer({size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC});
      const read = rt.device.createBuffer({size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
      const times = [];
      try {
        for (let i = 0; i < samples + 8; i++) {
          render({querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1}, simulate);
          const enc = rt.device.createCommandEncoder();
          enc.resolveQuerySet(querySet, 0, 2, resolve, 0);
          enc.copyBufferToBuffer(resolve, 0, read, 0, 16);
          rt.device.queue.submit([enc.finish()]);
          await read.mapAsync(GPUMapMode.READ);
          const t = new BigUint64Array(read.getMappedRange());
          if (i >= 8) times.push(Number(t[1] - t[0]) / 1e6);
          read.unmap();
        }
        times.sort((a,b) => a-b);
        return {width, height, samples, renderGpuMedianMs: times[Math.floor(times.length/2)], renderGpuP95Ms: times[Math.floor(times.length*.95)], adapter: rt.describe(), scope: simulate ? "Stationary-camera GPU frame with ocean and enabled effects (one 30 Hz tornado step, two 120 Hz duck steps); sky-cache cadence; excludes presentation" : "Paused stationary-camera GPU rendering with reused sky/volume caches; excludes simulation and presentation"};
      } finally {querySet.destroy(); resolve.destroy(); read.destroy();}
    });
  },
  async weatherInspect() {
    return exclusive(async () => ({
      weather: Array.from(await rt.read(weather)),
      spectrum: Array.from(await rt.read(spectralEnergy)).reduce(
        (a, v) => ({
          min: Math.min(a.min, v),
          max: Math.max(a.max, v),
          finite: a.finite && Number.isFinite(v),
        }),
        { min: Infinity, max: -Infinity, finite: true },
      ),
      compressionMin: (await rt.read(foam[foamIndex])).slice(0,262144*4).reduce((a,v,i)=>i%4===2?Math.min(a,v):a,Infinity),
      foamPeak: (await rt.read(foam[foamIndex])).slice(0,262144*4).reduce((a,v,i)=>i%4===0?Math.max(a,v):a,0),
    }));
  },
  async weatherAdvance(seconds) {
    return exclusive(async () => {
      play(false);
      for (let i = 0; i < Math.ceil(seconds); i++) {
        frameDt = 1 / 30;
        weatherDt = 1;
        if (state.weatherAge >= 0) state.weatherAge += 1;
        state.time += 1 / 30;
        const batch = rt.batch();
        waves(batch);
        ripples(batch, 1 / 30);
        batch.submit();
        if (i % 30 === 0) await rt.idle();
      }
      frameDt = weatherDt = 0;
      render();
      await rt.idle();
    });
  },
  resume: () => play(true),
  async inspect() {
    return exclusive(async () => {
      const a = (await rt.read(surface)).slice(0,3*65536*4),
        r = await rt.read(rip[ripIndex]);
      const bandSquares = [0, 0, 0];
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
          bandSquares[Math.floor(i / (65536 * 4))] += a[i] * a[i];
        }
      }
      for (let i = 0; i < r.length; i += 4)
        ripplePeak = Math.max(ripplePeak, Math.abs(r[i]));
      return {
        finite,
        min,
        max,
        rms: Math.sqrt(sum / (3 * 65536)),
        bandRms: bandSquares.map((v) => Math.sqrt(v / 65536)),
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
      frameDt = weatherDt = 0;
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
      k.fft_pass.clearBindings();
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
  const source = await (await fetch("./src/clearwater.cu")).text()+"\n"+await (await fetch("./src/ducks.cu")).text();
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
    const kernel = k[name], bindings = new Map();
    k[name] = {
      bind(buffers, scalars = {}) {
        const key = kernel.artifact.metadata.bindings.map(binding => buffers[binding.name].id).join(":");
        let invocation = bindings.get(key);
        if (!invocation) {
          invocation = kernel.bind(buffers, scalars);
          bindings.set(key, invocation);
        } else invocation.setScalars(scalars);
        return invocation;
      },
      clearBindings() { bindings.clear(); },
    };
  }
  chopFFT=[rt.createBuffer(3*65536*16),rt.createBuffer(3*65536*16)];
  linearSurface=rt.createBuffer(3*65536*16);
  weather = rt.createBuffer(2 * 16);
  duckBodies=[rt.createBuffer(10000*5*16),rt.createBuffer(10000*5*16)];
  duckNodes=rt.createBuffer(1048576*2*4);
  duckHeads=rt.createBuffer(65536*4);duckLinks=rt.createBuffer(10000*4);
  tornadoVelocity = [rt.createBuffer(tornadoCells*16),rt.createBuffer(tornadoCells*16)];
  tornadoMoisture = [rt.createBuffer(tornadoCells*16),rt.createBuffer(tornadoCells*16)];
  tornadoPressure = [rt.createBuffer(tornadoCells*16),rt.createBuffer(tornadoCells*16)];
  tornadoRhs = rt.createBuffer(tornadoCells*16);
  tornadoCurl = rt.createBuffer(tornadoCells*16);
  tornadoCloudMap = [rt.createBuffer(tornadoCells*16),rt.createBuffer(tornadoCells*16)];
  tornadoLighting = rt.createBuffer(tornadoCells*16);
  tornadoSeaState = [rt.createBuffer((96*96+65536)*16),rt.createBuffer((96*96+65536)*16)];
  tornadoWaveFFT = [rt.createBuffer(65536*16),rt.createBuffer(65536*16)];
  tornadoWaveModes = rt.createBuffer(65536*16);
  tornadoSpray = rt.createBuffer(8192*2*16);
  spectralEnergy = rt.createBuffer(new Float32Array(3 * 65536).fill(1));
  foam = [rt.createBuffer(3 * 262144 * 16), rt.createBuffer(3 * 262144 * 16)];
  lensKernel = rt.createBuffer(3 * 65536 * 16);
  lensFFT = [rt.createBuffer(3 * 65536 * 16), rt.createBuffer(3 * 65536 * 16)];
  seed = rt.createBuffer(3 * 65536 * 8);
  rows = rt.createBuffer(768 * 4);
  scales = rt.createBuffer(3 * 4);
  fft = [rt.createBuffer(3 * 65536 * 16), rt.createBuffer(3 * 65536 * 16)];
  surface = rt.createBuffer(6 * 65536 * 16);
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
