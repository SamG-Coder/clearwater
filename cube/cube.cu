// SPDX-License-Identifier: MIT
// Optical test volume: 2 x 2 x 2 m, surface y=0, stone bed y=-2.
// Shares FFT/weather/optical helpers with clearwater.cu and pool.cu.
__device__ float4 cubeBox(float3 o, float3 d) {
  float ix = 1 / (fabsf(d.x) < .000001f ? .000001f : d.x),
        iy = 1 / (fabsf(d.y) < .000001f ? .000001f : d.y),
        iz = 1 / (fabsf(d.z) < .000001f ? .000001f : d.z);
  float ax = (-1 - o.x) * ix, bx = (1 - o.x) * ix, ay = (-2 - o.y) * iy,
        by = -o.y * iy, az = (-1 - o.z) * iz, bz = (1 - o.z) * iz;
  float nearT = fmaxf(fmaxf(fminf(ax, bx), fminf(ay, by)), fminf(az, bz));
  float farT = fminf(fminf(fmaxf(ax, bx), fmaxf(ay, by)), fmaxf(az, bz));
  return make_float4(nearT, farT, 0, 0);
}
__device__ float4 cubeWave(const float4 *surface, const float4 *rip, float x,
                           float z) {
  float4 s = sample4(surface, x * 256 / 4.6f, z * 256 / 4.6f, 256, 0);
  float4 r = sample4(rip, (x + 1) * 128, (z + 1) * 128, 256, 0);
  float fade = smooth(0, .09f, 1 - fmaxf(fabsf(x), fabsf(z)));
  return make_float4((s.x * .65f + r.x) * fade, (s.y * .65f + r.y) * fade,
                     (s.z * .65f + r.z) * fade, 0);
}
__device__ float3 cubeNormal(float3 p, const float4 *surface,
                             const float4 *rip) {
  if (p.y > -.02f) {
    float4 s = cubeWave(surface, rip, p.x, p.z);
    return norm(v3(-s.y, 1, -s.z));
  }
  if (p.y < -1.999f)
    return v3(0, -1, 0);
  if (fabsf(p.x) > fabsf(p.z))
    return v3(p.x > 0 ? 1 : -1, 0, 0);
  return v3(0, 0, p.z > 0 ? 1 : -1);
}
__device__ float3 cubeSky(float3 d, const float4 *weather, float time) {
  float3 c =
      weatherSky(norm(v3(d.x, fmaxf(.1f, d.y), d.z)), weather, 0, 0, time);
  // Soft foliage-coloured reflection card, not garden geometry or a backdrop
  // image.
  float a = atan2f(d.z, d.x), leaves = fbm(a * 8, d.y * 12);
  float foliage = (1 - smooth(.08f, .65f, d.y)) * smooth(.35f, .65f, leaves);
  return mix3(c, mul(v3(.16f, .22f, .10f), .6f + leaves), foliage * .8f);
}
__device__ float3 cubeStone(float3 p, const float4 *caustics, int under,
                            const float4 *weather) {
  float tx = p.x / .43f + floorf(p.z / .34f) * .5f, tz = p.z / .34f;
  float id = hash(floorf(tx), floorf(tz));
  float seam = fminf(fminf(frac(tx), 1 - frac(tx)) * .43f,
                     fminf(frac(tz), 1 - frac(tz)) * .34f);
  float grain = fbm(p.x * 28 + id * 30, p.z * 28),
        vein = powf(1 - fabsf(noise(p.x * 9, p.z * 9) * 2 - 1), 12);
  float3 stone = mix3(v3(.19f, .22f, .16f), v3(.43f, .43f, .32f),
                      id * .65f + grain * .25f);
  stone = mul(stone, .76f + grain * .35f + vein * .10f);
  stone = mix3(v3(.08f, .105f, .07f), stone, smooth(.002f, .009f, seam));
  if (under != 0) {
    float4 c = sample4(caustics, (p.x + 1) * 256, (p.z + 1) * 256, 512, 0);
    stone =
        prod(stone, v3(.5f + fminf(4, c.x) * .6f, .5f + fminf(4, c.y) * .62f,
                       .5f + fminf(4, c.z) * .52f));
  }
  return mul(stone, .5f + .65f * expf(-cloudAt(weather, 0, 0) * 2));
}
__device__ float3 cubeOutside(float3 o, float3 d, const float4 *caustics,
                              const float4 *weather, float time) {
  if (d.y >= -.001f)
    return cubeSky(d, weather, time);
  float t = (-2.035f - o.y) / d.y;
  float3 p = add(o, mul(d, t));
  float radius = sqrtf(p.x * p.x + p.z * p.z);
  float3 floor =
      mix3(v3(.18f, .205f, .18f), v3(.31f, .33f, .29f), smooth(1, 9, radius));
  if (fabsf(p.x) < 1.08f && fabsf(p.z) < 1.08f)
    return cubeStone(p, caustics, 0, weather);
  float shadow =
      1 - .38f * expf(-square((p.x - .5f) / 1.2f) - square((p.z - .8f) / 1.4f));
  float3 paving = cubeStone(p, caustics, 0, weather);
  floor = mix3(mul(paving, .8f), floor, smooth(3, 14, radius));
  return mul(floor, shadow);
}
__global__ void cube_step(const float4 *previous, float4 *next,
                          const float4 *weather, float time, float tapX,
                          float tapZ, int drop) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 256 || z >= 256)
    return;
  int id = z * 256 + x;
  float4 a = previous[id];
  float lap = previous[z * 256 + max(0, x - 1)].x +
              previous[z * 256 + min(255, x + 1)].x +
              previous[max(0, z - 1) * 256 + x].x +
              previous[min(255, z + 1) * 256 + x].x - 4 * a.x;
  float v = (a.y + .18f * lap) * .995f, h = (a.x + v) * .9995f,
        wx = (x + .5f) / 128 - 1, wz = (z + .5f) / 128 - 1;
  float rain = sat(weather[0].y + stormAt(weather, 0, 0));
  unsigned tick = (unsigned)floorf(time * 120);
  if (random((unsigned)id + tick * 65537u) < rain * .00025f)
    h -= .003f;
  if (drop != 0)
    h -= .018f * expf(-(square(wx - tapX) + square(wz - tapZ)) / .006f);
  next[id] = make_float4(h, v, 0, 0);
}
__global__ void cube_normals(const float4 *input, float4 *output) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 256 || z >= 256)
    return;
  float l = input[z * 256 + max(0, x - 1)].x,
        r = input[z * 256 + min(255, x + 1)].x,
        b = input[max(0, z - 1) * 256 + x].x,
        f = input[min(255, z + 1) * 256 + x].x;
  output[z * 256 + x] =
      make_float4(input[z * 256 + x].x, (r - l) * 64, (f - b) * 64, 0);
}
__global__ void cube_caustics(const float4 *surface, unsigned *photons,
                              float depth) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 1024 || z >= 1024)
    return;
  float wx = (x + .5f) / 512 - 1, wz = (z + .5f) / 512 - 1;
  float4 s = sample4(surface, wx * 256 / 4.6f, wz * 256 / 4.6f, 256, 0);
  float3 n = norm(v3(-s.y * .65f, 1, -s.z * .65f));
  for (int c = 0; c < 3; c++) {
    float3 d = refract3(v3(0, -1, 0), n, 1 / (1.3315f + c * .0025f));
    float t = (-2 - s.x * .65f) / d.y, px = (wx + d.x * t + 1) * 256,
          pz = (wz + d.z * t + 1) * 256;
    int ix = (int)floorf(px), iz = (int)floorf(pz);
    float fx = frac(px), fz = frac(pz);
    if (ix < 0 || iz < 0 || ix > 510 || iz > 510)
      continue;
    atomicAdd(&photons[(iz * 512 + ix) * 3 + c],
              (unsigned)(256 * (1 - fx) * (1 - fz)));
    atomicAdd(&photons[(iz * 512 + ix + 1) * 3 + c],
              (unsigned)(256 * fx * (1 - fz)));
    atomicAdd(&photons[((iz + 1) * 512 + ix) * 3 + c],
              (unsigned)(256 * (1 - fx) * fz));
    atomicAdd(&photons[((iz + 1) * 512 + ix + 1) * 3 + c],
              (unsigned)(256 * fx * fz));
  }
}
__global__ void cube_render(const float4 *surface, const float4 *rip,
                            const float4 *caustics, const float4 *weather,
                            float4 *hdr, int width, int height, float camX,
                            float camY, float camZ, float yaw, float pitch,
                            float time) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height)
    return;
  float3 o = v3(camX, camY, camZ),
         d = ray((x + .5f) / width * 2 - 1, 1 - (y + .5f) / height * 2,
                 (float)width / height, yaw, pitch);
  float3 color = cubeOutside(o, d, caustics, weather, time);
  float4 hit = cubeBox(o, d);
  if (hit.y > fmaxf(hit.x, 0)) {
    float3 p = add(o, mul(d, fmaxf(hit.x, 0))), n = cubeNormal(p, surface, rip);
    float fres = fresnel(sat(-dot3(d, n)));
    float3 reflected =
        cubeOutside(add(p, mul(n, .002f)), sub(d, mul(n, 2 * dot3(d, n))),
                    caustics, weather, time);
    float3 direction = refract3(d, n, 1 / 1.3335f),
           origin = add(p, mul(direction, .002f));
    float3 through = v3(0, 0, 0), transmission = v3(1, 1, 1);
    for (int bounce = 0; bounce < 4; bounce++) {
      float4 inside = cubeBox(origin, direction);
      float travel = fmaxf(.001f, inside.y);
      float3 q = add(origin, mul(direction, travel));
      float3 atten = exp3(mul(v3(.16f, .048f, .105f), -travel));
      through = add(through, prod(transmission, prod(sub(v3(1, 1, 1), atten),
                                                     v3(.11f, .21f, .13f))));
      transmission = prod(transmission, atten);
      if (q.y < -1.998f) {
        through = add(through,
                      prod(transmission, cubeStone(q, caustics, 1, weather)));
        break;
      }
      float3 outward = cubeNormal(q, surface, rip), inward = mul(outward, -1);
      float cosine = dot3(direction, outward);
      float discriminant = 1 - square(1.3335f) * (1 - cosine * cosine);
      if (discriminant > 0) {
        float3 exitDirection = refract3(direction, inward, 1.3335f);
        through = add(
            through, prod(transmission,
                          cubeOutside(add(q, mul(outward, .002f)),
                                      exitDirection, caustics, weather, time)));
        break;
      }
      direction = sub(direction, mul(outward, 2 * cosine));
      origin = add(q, mul(direction, .002f));
    }
    color = mix3(through, reflected, fres);
    float glint = powf(
        sat(dot3(sub(d, mul(n, 2 * dot3(d, n))), norm(v3(-.3f, .9f, -.3f)))),
        850);
    color = add(color, mul(v3(1, .91f, .68f), glint * 3));
  }
  float vignette = 1 - .12f * (square((x + .5f) / width * 2 - 1) +
                               square((y + .5f) / height * 2 - 1));
  color = mul(color, vignette);
  hdr[y * width + x] = make_float4(color.x, color.y, color.z, 1);
}
