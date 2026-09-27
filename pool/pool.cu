// SPDX-License-Identifier: MIT
// Append to src/clearwater.cu. All pool geometry, optics and dynamics live
// here.
__device__ float poolEdge(float x, float z) {
  float a = sqrtf(square((x + 2.15f) / 3.55f) + square(z / 2.8f));
  float b = sqrtf(square((x - 2.0f) / 3.4f) + square((z - .35f) / 3.35f));
  float h = sat(.5f + .5f * (b - a) / .42f);
  return (lerp(b, a, h) - .42f * h * (1 - h) - 1) * 2.8f;
}
__device__ float poolBottom(float x, float z) {
  float floorY = -1.65f + .32f * smooth(-3, 4, x);
  // Broad submerged entry treads in the small end of the basin.
  if (x < -4.7f)
    floorY = -.24f;
  else if (x < -4.25f)
    floorY = -.49f;
  else if (x < -3.8f)
    floorY = -.76f;
  else if (x < -3.35f)
    floorY = -1.02f;
  return floorY;
}
__device__ float poolGround(float x, float z) {
  float edge = poolEdge(x, z);
  if (edge < 0)
    return poolBottom(x, z);
  if (edge < .38f)
    return .22f;
  if (z < -5.0f)
    return .64f;
  if (z < -4.55f)
    return .42f;
  if (z < -4.1f)
    return .23f;
  return .16f;
}
__device__ float poolSolid(float3 p) {
  // Conservative distance estimate for the basin, coping and stepped terrace.
  float edge = poolEdge(p.x, p.z);
  float basin = fmaxf(-edge, p.y - .16f);
  float floorD = p.y - (-1.65f + .32f * smooth(-3, 4, p.x));
  floorD = fminf(floorD, fmaxf(p.x + 3.35f, p.y + 1.02f));
  floorD = fminf(floorD, fmaxf(p.x + 3.8f, p.y + .76f));
  floorD = fminf(floorD, fmaxf(p.x + 4.25f, p.y + .49f));
  floorD = fminf(floorD, fmaxf(p.x + 4.7f, p.y + .24f));
  float coping = fmaxf(fmaxf(-edge, edge - .38f), p.y - .22f);
  float terrace = fminf(
      fmaxf(p.z + 4.1f, p.y - .23f),
      fminf(fmaxf(p.z + 4.55f, p.y - .42f), fmaxf(p.z + 5.0f, p.y - .64f)));
  return fminf(fminf(basin, floorD), fminf(coping, terrace));
}
__device__ float poolTrace(float3 o, float3 d) {
  float t = .015f;
  // Start above the highest surface when tracing from an elevated camera.
  if (o.y > .67f && d.y < -.001f)
    t = fmaxf(t, (.67f - o.y) / d.y);
  for (int i = 0; i < 150; i++) {
    float3 p = add(o, mul(d, t));
    float s = poolSolid(p);
    if (s < .003f)
      return t;
    t += fmaxf(.002f, s * .55f);
    if (t > 65)
      break;
  }
  if (d.y < -.001f) {
    float farT = (.64f - o.y) / d.y;
    if (farT > 0 && o.z + d.z * farT < -5.0f)
      return farT;
    farT = (.16f - o.y) / d.y;
    if (farT > 0 && poolEdge(o.x + d.x * farT, o.z + d.z * farT) > .4f)
      return farT;
  }
  return 65;
}
__device__ float3 poolNormal(float3 p) {
  float e = .004f;
  return norm(
      v3(poolSolid(add(p, v3(e, 0, 0))) - poolSolid(sub(p, v3(e, 0, 0))),
         poolSolid(add(p, v3(0, e, 0))) - poolSolid(sub(p, v3(0, e, 0))),
         poolSolid(add(p, v3(0, 0, e))) - poolSolid(sub(p, v3(0, 0, e)))));
}
__device__ float3 poolSky(float3 d, const float4 *weather, float time) {
  // Shared moving cloud and lightning model; no landscape geometry.
  return weatherSky(norm(v3(d.x, fmaxf(.12f, d.y), d.z)), weather, 0, 0, time);
}
__device__ float3 poolMaterial(float3 p, float3 n, float3 view,
                               const float4 *weather, const float4 *caustics,
                               float time) {
  float edge = poolEdge(p.x, p.z), rain = weather[2].x;
  float u = p.x, v = p.z;
  if (fabsf(n.y) < .5f) {
    u = atan2f(p.z / 3, p.x / 5) * 4;
    v = p.y;
  }
  float3 albedo = v3(.35f, .39f, .40f);
  float grout = 0;
  if (p.y < .08f && edge < .05f) {
    float tx = u * 11, tz = v * 11;
    grout = 1 - smooth(.025f, .055f,
                       fminf(fminf(frac(tx), 1 - frac(tx)),
                             fminf(frac(tz), 1 - frac(tz))));
    float tile = .32f + .36f * hash(floorf(tx), floorf(tz));
    albedo = mix3(v3(.055f, .22f, .30f), v3(.25f, .45f, .48f), tile);
    if (p.y > -.18f)
      albedo = mul(v3(.06f, .22f, .28f), .7f + tile * .7f);
    albedo = mix3(albedo, v3(.28f, .39f, .37f), grout * .65f);
  } else if (edge < .4f) {
    float seam = frac(atan2f(p.z / 3, p.x / 5) * 18);
    grout = 1 - smooth(.012f, .035f, fminf(seam, 1 - seam));
    albedo = mul(v3(.72f, .70f, .61f), .91f + .14f * noise(p.x * 35, p.z * 35));
    albedo = mix3(albedo, v3(.38f, .40f, .37f), grout * .7f);
  } else {
    float tx = u / .92f + floorf(v / .61f) * .5f, tz = v / .61f;
    float id = hash(floorf(tx), floorf(tz));
    float border = fminf(fminf(frac(tx), 1 - frac(tx)) * .92f,
                         fminf(frac(tz), 1 - frac(tz)) * .61f);
    grout = 1 - smooth(.002f, .007f, border);
    albedo = mix3(v3(.105f, .14f, .16f), v3(.27f, .28f, .27f), id);
    albedo = mul(albedo, .85f + .22f * fbm(u * 12 + id * 40, v * 12));
    albedo = mix3(albedo, v3(.38f, .39f, .36f), grout * .8f);
    albedo = mul(albedo, 1 - rain * .3f);
  }
  float3 sun = norm(v3(.08959f, .51504f, -.85247f));
  float cloud = cloudAt(weather, 0, 0), direct = expf(-cloud * 2.8f);
  float shadow = 1, travel = .04f;
  for (int i = 0; i < 20; i++) {
    float dist = poolSolid(add(add(p, mul(n, .012f)), mul(sun, travel)));
    shadow = fminf(shadow, 12 * dist / travel);
    travel += fmaxf(.03f, dist);
    if (travel > 9)
      break;
  }
  float light = .38f + .9f * sat(dot3(n, sun)) * sat(shadow) * direct;
  float3 color = prod(albedo, mul(v3(1.12f, 1.07f, .97f), light));
  if (p.y < 0) {
    float4 c = sample4(caustics, p.x * 512 / 4.6f, p.z * 512 / 4.6f, 512, 0);
    float caustic = fminf(4, c.x * .7f + c.y * .2f + c.z * .1f);
    color = mul(color, .8f + caustic * .32f * direct);
  } else {
    float3 r = sub(mul(n, 2 * dot3(n, view)), view);
    float spec = powf(sat(dot3(r, sun)), 90);
    color =
        add(color, mul(v3(1, .91f, .74f), spec * (.12f + rain * .9f) * direct));
    color = mix3(color, poolSky(r, weather, time),
                 rain * .12f * (1 - grout) * powf(1 - sat(dot3(n, view)), 3));
  }
  return color;
}
// One deterministic impact stream drives both the wave solver and visible
// splashes. Each world-space cell has its own phase, drop position and rain
// threshold.
__device__ float4 poolDrop(float cx, float cz, float time, float rain) {
  float cycle = time * 1.7f + hash(cx + 53, cz - 19);
  float tick = floorf(cycle), age = frac(cycle) / 1.7f;
  float px = (cx + .15f + .7f * hash(cx + tick * 17, cz + 31)) * .42f;
  float pz = (cz + .15f + .7f * hash(cx + 47, cz + tick * 13)) * .42f;
  float active = hash(cx + tick * 7, cz - tick * 23) < rain * .8f ? 1.0f : 0.0f;
  return make_float4(px, pz, age,
                     active * (poolEdge(px, pz) < -.06f ? 1.0f : 0.0f));
}
__device__ float4 poolImpact(float x, float z, float time, float rain) {
  if (rain < .001f)
    return make_float4(0, 0, 0, 0);
  float cellX = floorf(x / .42f), cellZ = floorf(z / .42f);
  float height = 0, dx = 0, dz = 0, highlight = 0;
  for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
      float4 drop = poolDrop(cellX + i, cellZ + j, time, rain);
      if (drop.w < .5f)
        continue;
      float vx = x - drop.x, vz = z - drop.y, r = sqrtf(vx * vx + vz * vz),
            age = drop.z;
      // A short-lived crown rises and collapses; the expanding rim catches the
      // sky.
      float life = sinf(3.14159265f * sat(age / .24f));
      float radius = .018f + age * .36f;
      float q = (r - radius) / .014f, rim = expf(-q * q);
      float lobes = .78f + .22f * cosf(atan2f(vz, vx) * 9 + drop.x * 17);
      float amplitude = .032f * life * lobes;
      float h = amplitude * rim, slope = amplitude * rim * (-2 * q / .014f);
      // A broader ring follows the crown; the solver supplies subsequent
      // propagation.
      float rq = (r - (.035f + age * .62f)) / .023f;
      float ring = expf(-rq * rq),
            decay = expf(-age * 5) * smooth(0, .035f, age);
      h += .0028f * ring * decay;
      slope += .0028f * ring * decay * (-2 * rq / .023f);
      height += h;
      dx += slope * vx / fmaxf(r, .002f);
      dz += slope * vz / fmaxf(r, .002f);
      highlight += rim * life * lobes * .75f + ring * decay * .18f;
    }
  return make_float4(height, dx, dz, sat(highlight));
}
__global__ void pool_step(const float4 *previous, float4 *next,
                          const float4 *weather, float time, float tapX,
                          float tapZ, int drop) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 256 || z >= 256)
    return;
  int id = z * 256 + x;
  float wx = (x - 128) * .0625f, wz = (z - 128) * .0625f;
  if (x < 1 || z < 1 || x > 254 || z > 254 || poolEdge(wx, wz) >= 0) {
    next[id] = make_float4(0, 0, 0, 0);
    return;
  }
  float4 a = previous[id];
  float l = poolEdge(wx - .0625f, wz) < 0 ? previous[id - 1].x : a.x;
  float r = poolEdge(wx + .0625f, wz) < 0 ? previous[id + 1].x : a.x;
  float b = poolEdge(wx, wz - .0625f) < 0 ? previous[id - 256].x : a.x;
  float f = poolEdge(wx, wz + .0625f) < 0 ? previous[id + 256].x : a.x;
  float vel = (a.y + .18f * (l + r + b + f - 4 * a.x)) * .994f;
  float h = (a.x + vel) * .9995f;
  float rain = sat(weather[0].y + stormAt(weather, 0, 0));
  float cellX = floorf(wx / .42f), cellZ = floorf(wz / .42f);
  if (rain > .001f)
    for (int j = -1; j <= 1; j++)
      for (int i = -1; i <= 1; i++) {
        float4 event = poolDrop(cellX + i, cellZ + j, time, rain);
        if (event.w > .5f && event.z < 1.0f / 120)
          h -= .012f *
               expf(-(square(wx - event.x) + square(wz - event.y)) / .003f);
      }
  if (drop != 0)
    h -= .055f * expf(-(square(wx - tapX) + square(wz - tapZ)) / .024f);
  next[id] = make_float4(h, vel, 0, 0);
}
__global__ void pool_wetness(float4 *weather, float dt) {
  if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0 ||
      threadIdx.y != 0)
    return;
  float4 old = weather[2], current = weather[0];
  float response = 1 - expf(-dt / 4);
  float rain = lerp(old.y, current.y, response),
        clouds = lerp(old.z, current.z, response);
  float localRain = sat(rain + stormAt(weather, 0, 0));
  float wet = sat(old.x + dt * (localRain * .045f - (1 - localRain) * .003f));
  weather[2] = make_float4(wet, rain, clouds, 0);
  weather[0] = make_float4(current.x, rain, clouds, current.w);
}
__device__ float4 poolWaves(const float4 *surface, const float4 *rip, float x,
                            float z) {
  float4 a = sample4(surface, x * 256 / 4.6f, z * 256 / 4.6f, 256, 0);
  float4 b = sample4(rip, x * 16 + 128, z * 16 + 128, 256, 0);
  float shore = smooth(0, .45f, -poolEdge(x, z));
  return make_float4((a.x * .3f + b.x) * shore, (a.y * .3f + b.y) * shore,
                     (a.z * .3f + b.z) * shore, 0);
}
__global__ void pool_render(const float4 *surface, const float4 *rip,
                            const float4 *caustics, const float4 *weather,
                            float4 *hdr, int width, int height, float camX,
                            float camY, float camZ, float yaw, float pitch,
                            float time) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height)
    return;
  float3 o = v3(camX, camY, camZ),
         d = ray(((x + .5f) / width) * 2 - 1, 1 - ((y + .5f) / height) * 2,
                 (float)width / height, yaw, pitch);
  float t = poolTrace(o, d);
  float3 p = add(o, mul(d, t));
  float3 color = poolSky(d, weather, time);
  if (t != 65)
    color = poolMaterial(p, poolNormal(p), mul(d, -1), weather, caustics, time);
  float wt = -o.y / d.y;
  if (d.y < -.001f && wt > 0 && wt < t) {
    float3 w = add(o, mul(d, wt));
    for (int i = 0; i < 3; i++) {
      float4 h = poolWaves(surface, rip, w.x, w.z);
      float4 impact = poolImpact(w.x, w.z, time,
                                 sat(weather[0].y + stormAt(weather, 0, 0)));
      wt = (h.x + impact.x - o.y) / d.y;
      w = add(o, mul(d, wt));
    }
    if (poolEdge(w.x, w.z) < -.004f) {
      float4 s = poolWaves(surface, rip, w.x, w.z);
      float4 impact = poolImpact(w.x, w.z, time,
                                 sat(weather[0].y + stormAt(weather, 0, 0)));
      float3 n = norm(v3(-s.y - impact.y, 1, -s.z - impact.z));
      float3 rd = refract3(d, n, 1 / 1.3335f), ro = add(w, mul(rd, .012f));
      float td = poolTrace(ro, rd);
      float3 bottom = add(ro, mul(rd, td));
      float3 under = poolMaterial(bottom, poolNormal(bottom), mul(rd, -1),
                                  weather, caustics, time);
      float3 absorption = exp3(mul(v3(.29f, .075f, .033f), -td));
      under = add(prod(under, absorption),
                  prod(v3(.025f, .20f, .24f), sub(v3(1, 1, 1), absorption)));
      float3 reflected = sub(d, mul(n, 2 * dot3(d, n))),
             origin = add(w, mul(n, .015f));
      float rt = poolTrace(origin, reflected);
      float3 refl = poolSky(reflected, weather, time);
      if (rt < 35) {
        float3 rp = add(origin, mul(reflected, rt));
        refl = poolMaterial(rp, poolNormal(rp), mul(reflected, -1), weather,
                            caustics, time);
      }
      float fres = fresnel(sat(-dot3(d, n)));
      color = mix3(under, refl, fres);
      color = mix3(color, v3(.59f, .73f, .76f), impact.w * .6f);
      float3 sun = norm(v3(.08959f, .51504f, -.85247f));
      float sparkle = powf(sat(dot3(reflected, sun)), 700);
      color = add(color, mul(v3(1, .89f, .64f),
                             sparkle * 5 * expf(-cloudAt(weather, 0, 0) * 3)));
    }
  }
  // Atmospheric rain catches light in front of the pool, alongside physical
  // impacts.
  float rain = sat(weather[0].y + stormAt(weather, 0, 0));
  float sx = (float)x / width, sy = (float)y / height;
  float streak = 0;
  for (int layer = 0; layer < 2; layer++) {
    float u = (sx + sy * .14f) * (270 + layer * 133),
          v = (sy - time * (1.35f + layer * .65f)) * (31 + layer * 11);
    float ix = floorf(u), iy = floorf(v), seed = hash(ix + layer * 91, iy);
    float cx = frac(u) - (.2f + .6f * hash(ix, iy + 83));
    float cy = frac(v) - .5f;
    streak += (1 - smooth(.025f, .17f, fabsf(cx))) *
              (1 - smooth(.12f, .46f, fabsf(cy))) * smooth(.87f, .99f, seed);
  }
  color = add(color, mul(v3(.42f, .51f, .58f), rain * streak * .22f));
  hdr[y * width + x] = make_float4(color.x, color.y, color.z, 1);
}

// Periodic RGB photon map at representative basin depth (wall projection is
// approximate).
__global__ void pool_caustics(const float4 *surface, unsigned *photons,
                              float depth) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 1024 || z >= 1024)
    return;
  float wx = ((float)x + .5f) * 4.6f / 1024,
        wz = ((float)z + .5f) * 4.6f / 1024;
  float4 a = sample4(surface, wx * 256 / 4.6f, wz * 256 / 4.6f, 256, 0);
  float3 n = norm(v3(-a.y * .3f, 1, -a.z * .3f)),
         sun = v3(.08959f, .51504f, -.85247f);
  for (int c = 0; c < 3; c++) {
    float ior = c == 0 ? 1.3315f : (c == 1 ? 1.3335f : 1.3365f);
    float3 d = refract3(mul(sun, -1), n, 1 / ior);
    float travel = (-depth - a.x * .3f) / d.y,
          px = (wx + d.x * travel) * 512 / 4.6f,
          pz = (wz + d.z * travel) * 512 / 4.6f;
    int ix = (int)floorf(px), iz = (int)floorf(pz);
    float fx = frac(px), fz = frac(pz);
    atomicAdd(&photons[(wrap(iz, 512) * 512 + wrap(ix, 512)) * 3 + c],
              (unsigned)(256 * (1 - fx) * (1 - fz)));
    atomicAdd(&photons[(wrap(iz, 512) * 512 + wrap(ix + 1, 512)) * 3 + c],
              (unsigned)(256 * fx * (1 - fz)));
    atomicAdd(&photons[(wrap(iz + 1, 512) * 512 + wrap(ix, 512)) * 3 + c],
              (unsigned)(256 * (1 - fx) * fz));
    atomicAdd(&photons[(wrap(iz + 1, 512) * 512 + wrap(ix + 1, 512)) * 3 + c],
              (unsigned)(256 * fx * fz));
  }
}
