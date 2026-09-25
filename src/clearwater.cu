// SPDX-License-Identifier: MIT
// Clearwater CUDA reimplementation. Original optical design: Lumaris (2026).
// All spectrum generation, FFT, ripples, ray projection, optics and post are CUDA.
// The browser host only supplies inputs, resources and dispatches.
__device__ float sat(float x) { return fminf(1.0f, fmaxf(0.0f, x)); }
__device__ float frac(float x) { return x - floorf(x); }
__device__ float lerp(float a, float b, float t) { return a + (b - a) * t; }
__device__ float smooth(float a, float b, float x) {
  float t = sat((x - a) / (b - a));
  return t * t * (3.0f - 2.0f * t);
}
__device__ float3 v3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ float3 add(float3 a, float3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ float3 sub(float3 a, float3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ float3 mul(float3 a, float b) { return v3(a.x * b, a.y * b, a.z * b); }
__device__ float3 prod(float3 a, float3 b) { return v3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 norm(float3 a) { return mul(a, rsqrtf(fmaxf(dot3(a, a), 0.00000001f))); }
__device__ float3 mix3(float3 a, float3 b, float t) { return add(mul(a, 1 - t), mul(b, t)); }
__device__ float3 exp3(float3 a) { return v3(expf(a.x), expf(a.y), expf(a.z)); }
__device__ float3 refract3(float3 d, float3 n, float eta) {
  float c = dot3(n, d);
  return sub(mul(d, eta), mul(n, eta * c + sqrtf(fmaxf(0, 1 - eta * eta * (1 - c * c)))));
}
__device__ float4 mix4(float4 a, float4 b, float t) {
  return make_float4(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t), lerp(a.w, b.w, t));
}
__device__ unsigned hashU(unsigned x) {
  x ^= x >> 16;
  x *= 2146121005u;
  x ^= x >> 15;
  x *= 2221713035u;
  x ^= x >> 16;
  return x;
}
__device__ float random(unsigned x) { return ((float)(hashU(x) & 16777215u) + 1.0f) / 16777217.0f; }
__device__ float hash(float x, float z) {
  return random((unsigned)((int)x * 1973 + (int)z * 9277 + 89173));
}
__device__ float noise(float x, float z) {
  float ix = floorf(x), iz = floorf(z), u = frac(x), w = frac(z);
  u = u * u * (3 - 2 * u);
  w = w * w * (3 - 2 * w);
  return lerp(lerp(hash(ix, iz), hash(ix + 1, iz), u),
              lerp(hash(ix, iz + 1), hash(ix + 1, iz + 1), u), w);
}
__device__ float fbm(float x, float z) {
  return .55f * noise(x, z) + .28f * noise(x * 2.03f + 17.1f, z * 2.03f + 17.1f) +
         .12f * noise(x * 4.12f, z * 4.12f) + .05f * noise(x * 8.36f, z * 8.36f);
}
__device__ int wrap(int x, int n) { return (x % n + n) % n; }
__device__ float lengthL(int c) { return c == 0 ? 4.6f : (c == 1 ? 37.0f : 293.0f); }
__device__ float4 sample4(const float4 *data, float x, float z, int n, int offset) {
  int ix = (int)floorf(x), iz = (int)floorf(z);
  float fx = frac(x), fz = frac(z);
  return mix4(mix4(data[offset + wrap(iz, n) * n + wrap(ix, n)],
                   data[offset + wrap(iz, n) * n + wrap(ix + 1, n)], fx),
              mix4(data[offset + wrap(iz + 1, n) * n + wrap(ix, n)],
                   data[offset + wrap(iz + 1, n) * n + wrap(ix + 1, n)], fx),
              fz);
}
// Three independent narrow-band spectra: capillary detail, wind waves and swell.
// A GPU reduction normalizes each cascade by expected RMS slope (as upstream).
__global__ void seed_spectrum(float2 *seed, unsigned seedValue) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y), c = (int)blockIdx.z;
  if (x >= 256 || z >= 256)
    return;
  int id = c * 65536 + z * 256 + x;
  float L = lengthL(c), kx = 6.283185307f * (float)(x < 128 ? x : x - 256) / L,
        kz = 6.283185307f * (float)(z < 128 ? z : z - 256) / L, k = sqrtf(kx * kx + kz * kz);
  float P = 0;
  if (k > 0.00001f) {
    float peak = c == 0 ? .62f : (c == 1 ? 7.0f : 65.0f), kp = 6.283185307f / peak,
          lk = logf(k / kp), bump = expf(-.5f * lk * lk / (.36f * .36f));
    float tail = .035f * expf(-kp * kp / (k * k)) * expf(-k * k / (kp * kp * 180));
    float swell = .35f * expf(-.5f * powf(logf(k / (kp * .3875f)) / .3f, 2));
    float dir = (kx * .8f + kz * .6f) / k;
    P = (bump + tail + swell) * (.3f + .7f * dir * dir) * (dir < 0 ? .35f : 1.0f) / (k * k * k * k);
  }
  unsigned h = (unsigned)id + seedValue * 19391u;
  float radius = sqrtf(-2 * logf(random(h * 2u + 1u))), angle = 6.283185307f * random(h * 2u + 2u),
        amp = sqrtf(P * .5f);
  seed[id] = make_float2(radius * cosf(angle) * amp, radius * sinf(angle) * amp);
}
__global__ void spectrum_rows(const float2 *seed, float *rows) {
  int z = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (z >= 768)
    return;
  int c = z / 256, zz = z % 256;
  float L = lengthL(c), sum = 0;
  for (int x = 0; x < 256; x++) {
    float kx = 6.283185307f * (float)(x < 128 ? x : x - 256) / L,
          kz = 6.283185307f * (float)(zz < 128 ? zz : zz - 256) / L;
    float2 h = seed[z * 256 + x];
    sum += 2 * (kx * kx + kz * kz) * (h.x * h.x + h.y * h.y);
  }
  rows[z] = sum;
}
__global__ void spectrum_norm(const float *rows, float *scales) {
  int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= 3)
    return;
  float sum = 0;
  for (int j = 0; j < 256; j++)
    sum += rows[c * 256 + j];
  scales[c] = (c == 0 ? .078f : (c == 1 ? .047f : .021f)) / sqrtf(fmaxf(sum, .0000000001f));
}
__global__ void evolve_spectrum(const float2 *seed, const float *scales, float4 *output, float time,
                                float sea, float depth) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y), c = (int)blockIdx.z;
  if (x >= 256 || z >= 256)
    return;
  int id = c * 65536 + z * 256 + x, j = c * 65536 + wrap(-z, 256) * 256 + wrap(-x, 256);
  float2 a = seed[id], b = seed[j];
  float L = lengthL(c), kx = 6.283185307f * (float)(x < 128 ? x : x - 256) / L,
        kz = 6.283185307f * (float)(z < 128 ? z : z - 256) / L, k = sqrtf(kx * kx + kz * kz);
  float kd = fminf(20, k * depth), th = (1 - expf(-2 * kd)) / (1 + expf(-2 * kd));
  float omega = sqrtf((9.81f * k + .000074f * k * k * k) * th), co = cosf(omega * time),
        si = sinf(omega * time), scale = scales[c] * sea;
  float re = ((a.x + b.x) * co - (a.y + b.y) * si) * scale,
        im = ((a.x - b.x) * si + (a.y - b.y) * co) * scale;
  // Nyquist derivatives must vanish to keep the packed slopes real.
  if (x == 128)
    kx = 0;
  if (z == 128)
    kz = 0;
  output[id] = make_float4((1 - kx) * re, (1 - kx) * im, -kz * im, kz * re);
}
// Stockham autosort IFFT, two complex fields packed in float4, no CPU FFT.
// The unnormalized inverse matches the slope-normalized Fourier coefficients.
__global__ void fft_pass(const float4 *input, float4 *output, int p, int axis, float sign) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y), c = (int)blockIdx.z;
  if (x >= 256 || z >= 256)
    return;
  int j = axis == 0 ? x : z, k = j & (p - 1), i = ((j - (j & (2 * p - 1))) >> 1) + k,
      ia = c * 65536 + (axis == 0 ? z * 256 + i : i * 256 + x), ib = ia + (axis == 0 ? 128 : 32768);
  float4 a = input[ia], b = input[ib];
  float ang = sign * 3.14159265359f * (float)k / (float)p, co = cosf(ang), si = sinf(ang),
        sgn = (j & p) != 0 ? -1.0f : 1.0f;
  output[c * 65536 + z * 256 + x] =
      make_float4(a.x + sgn * (co * b.x - si * b.y), a.y + sgn * (si * b.x + co * b.y),
                  a.z + sgn * (co * b.z - si * b.w), a.w + sgn * (si * b.z + co * b.w));
}
__global__ void resolve_surface(const float4 *input, float4 *surface) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y), c = (int)blockIdx.z;
  if (x >= 256 || z >= 256)
    return;
  int id = c * 65536 + z * 256 + x;
  float4 s = input[id];
  surface[id] = make_float4(s.x, s.y, s.z, s.y * s.y + s.z * s.z);
}
__device__ float3 ray(float sx, float sy, float aspect, float yaw, float pitch) {
  float cy = cosf(yaw), syaw = sinf(yaw), cp = cosf(pitch), sp = sinf(pitch);
  return norm(v3(syaw * cp + sx * aspect * .62487f * cy - sy * .62487f * syaw * sp,
                 sp + sy * .62487f * cp,
                 -cy * cp + sx * aspect * .62487f * syaw + sy * .62487f * cy * sp));
}
// Fixed 120 Hz damped wave equation, camera-relative grid with integer recentering.
__global__ void ripple_step(const float4 *previous, float4 *next, int shiftX, int shiftZ,
                            float centerX, float centerZ, float camX, float camZ, float camY,
                            float yaw, float pitch, float aspect, float tapX, float tapY,
                            int drop) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 256 || z >= 256)
    return;
  int sx = x + shiftX, sz = z + shiftZ, id = z * 256 + x;
  float h = 0, vel = 0;
  if (sx > 0 && sx < 255 && sz > 0 && sz < 255) {
    int old = sz * 256 + sx;
    float4 a = previous[old];
    float lap = previous[old - 1].x + previous[old + 1].x + previous[old - 256].x +
                previous[old + 256].x - 4 * a.x;
    vel = (a.y + .21f * lap) * .994f;
    h = (a.x + vel) * .999f;
  }
  if (drop != 0) {
    float3 d = ray(tapX, tapY, aspect, yaw, pitch);
    if (d.y < -.01f) {
      float t = -camY / d.y, px = camX + d.x * t, pz = camZ + d.z * t,
            wx = centerX + ((float)x - 128) * .0625f, wz = centerZ + ((float)z - 128) * .0625f,
            dist = sqrtf((wx - px) * (wx - px) + (wz - pz) * (wz - pz));
      h -= .065f * expf(-dist * dist / 0.0225f);
    }
  }
  float edge = smooth(0, 16, (float)min(min(x, 255 - x), min(z, 255 - z)));
  next[id] = make_float4(h * lerp(.85f, 1, edge), vel * lerp(.85f, 1, edge), 0, 0);
}
__global__ void ripple_normals(const float4 *input, float4 *output) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 256 || z >= 256)
    return;
  int id = z * 256 + x;
  float l = input[z * 256 + max(0, x - 1)].x, r = input[z * 256 + min(255, x + 1)].x,
        b = input[max(0, z - 1) * 256 + x].x, f = input[min(255, z + 1) * 256 + x].x,
        h = input[id].x;
  output[id] = make_float4(h, (r - l) * 8, (f - b) * 8, (r + l + b + f - 4 * h) * 256);
}
__device__ float4 rippleAt(const float4 *rip, float x, float z, float cx, float cz) {
  float u = (x - cx) * 16 + 128, w = (z - cz) * 16 + 128;
  if (u < 1 || u > 254 || w < 1 || w > 254)
    return make_float4(0, 0, 0, 0);
  return sample4(rip, u, w, 256, 0);
}
__device__ float4 water(const float4 *surf, const float4 *rip, float x, float z, float cx, float cz,
                        float distance) {
  float4 a = sample4(surf, x * 256 / 4.6f, z * 256 / 4.6f, 256, 0),
         b = sample4(surf, x * 256 / 37, z * 256 / 37, 256, 65536),
         c = sample4(surf, x * 256 / 293, z * 256 / 293, 256, 131072),
         r = rippleAt(rip, x, z, cx, cz);
  float fade = 1 / (1 + distance * .018f);
  return make_float4(a.x + b.x + c.x + r.x, (a.y * fade + b.y + c.y + r.y),
                     (a.z * fade + b.z + c.z + r.z),
                     fmaxf(0, a.w - a.y * a.y - a.z * a.z) + .006f * (1 - fade));
}
// RGB photon splats. Fixed-point atomics avoid floating-point atomic requirements.
__global__ void clear_caustics(unsigned *photons) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x < 512 && z < 512) {
    int id = (z * 512 + x) * 3;
    photons[id] = 0;
    photons[id + 1] = 0;
    photons[id + 2] = 0;
  }
}
__global__ void trace_caustics(const float4 *surface, unsigned *photons, float depth) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 1024 || z >= 1024)
    return;
  float wx = ((float)x + .5f) * 4.6f / 1024, wz = ((float)z + .5f) * 4.6f / 1024;
  float4 a = sample4(surface, wx * 256 / 4.6f, wz * 256 / 4.6f, 256, 0);
  float3 n = norm(v3(-a.y, 1, -a.z)), sun = v3(.08959f, .51504f, -.85247f);
  for (int c = 0; c < 3; c++) {
    float ior = c == 0 ? 1.3315f : (c == 1 ? 1.3335f : 1.3365f);
    float3 d = refract3(mul(sun, -1), n, 1 / ior);
    float travel = (-depth - a.x) / d.y, px = (wx + d.x * travel) * 512 / 4.6f,
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
__global__ void filter_caustics(const unsigned *photons, float4 *caustics) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 512 || z >= 512)
    return;
  float3 sum = v3(0, 0, 0);
  for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
      int id = (wrap(z + j, 512) * 512 + wrap(x + i, 512)) * 3;
      float w = (i == 0 ? 2.0f : 1.0f) * (j == 0 ? 2.0f : 1.0f) / 16384;
      sum = add(sum,
                v3((float)photons[id] * w, (float)photons[id + 1] * w, (float)photons[id + 2] * w));
    }
  caustics[z * 512 + x] = make_float4(sum.x, sum.y, sum.z, 1);
}
__device__ float fresnel(float ci) {
  ci = sat(ci);
  float ct = sqrtf(1 - (1 - ci * ci) / (1.3335f * 1.3335f)),
        rs = (ci - 1.3335f * ct) / (ci + 1.3335f * ct),
        rp = (1.3335f * ci - ct) / (1.3335f * ci + ct);
  return .5f * (rs * rs + rp * rp);
}
__device__ float3 sky(float3 d) {
  float3 sun = v3(.08959f, .51504f, -.85247f);
  float mu = fmaxf(0, dot3(d, sun)), e = d.y;
  float3 col = mix3(v3(.66f, .78f, .9f), v3(.11f, .27f, .62f), powf(sat(e), .42f));
  col = add(
      col, mul(v3(1, .86f, .66f), .22f * powf(mu, 6) + .3f * powf(mu, 64) + 1.6f * powf(mu, 2400)));
  float a = atan2f(d.z, d.x), ridge = .04f + .016f * sinf(a * 2 + .7f) +
                                      .011f * sinf(a * 5 + 2.1f) + .006f * sinf(a * 11 + .3f) +
                                      .003f * sinf(a * 23 + 1.7f) +
                                      .0045f * (noise(a * 260, 0) - .5f);
  float tex = fbm(a * 420, e * 420),
        cliff = smooth(.42f, .18f, e / fmaxf(ridge, .001f) + .25f * (tex - .5f)) *
                smooth(.35f, .75f, noise(a * 18, 1));
  float3 land = mix3(mul(v3(.045f, .070f, .042f), .6f + .8f * tex),
                     mul(v3(.30f, .28f, .23f), .55f + .7f * tex), cliff);
  land = mix3(land, v3(.6072f, .7176f, .828f), .48f);
  return mix3(col, land, smooth(ridge + .0009f, ridge - .0009f, e));
}
__device__ float floorDepth(float x, float z, float depth) {
  return depth + .12f * (noise(x * .22f, z * .22f) - .5f) +
         .06f * (noise(x * .9f + 7, z * .9f + 7) - .5f);
}
__device__ float3 stone(const float4 *peb, float x, float z, float footprint) {
  float k = noise(x * .85f, z * .85f) * 8, ia = floorf(k), f = frac(k), u = x / .78f * 1024,
        w = z / .78f * 1024;
  float4 a = sample4(peb, u + sinf(3 * ia) * 1024, w + sinf(7 * ia) * 1024, 1024, 0),
         b = sample4(peb, u + sinf(3 * (ia + 1)) * 1024, w + sinf(7 * (ia + 1)) * 1024, 1024, 0);
  float m = smooth(.2f, .8f, f - .1f * (a.x + a.y + a.z - b.x - b.y - b.z));
  float3 col = v3(lerp(a.x, b.x, m), lerp(a.y, b.y, m), lerp(a.z, b.z, m));
  return mix3(col, v3(.20f, .18f, .14f), smooth(.015f, .16f, footprint));
}
__global__ void render_water(const float4 *surface, const float4 *rip, const float4 *caustics,
                             const float4 *pebbles, float4 *hdr, int width, int height, float camX,
                             float camZ, float camY, float yaw, float pitch, float centerX,
                             float centerZ, float depth, float time, int view) {
  int ix = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      iy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (ix >= width || iy >= height)
    return;
  float sx = 2 * ((float)ix + .5f) / (float)width - 1,
        sy = 1 - 2 * ((float)iy + .5f) / (float)height;
  float3 rd = ray(sx, sy, (float)width / (float)height, yaw, pitch),
         sun = v3(.08959f, .51504f, -.85247f), SUN = v3(6, 5.4f, 4.44f), col = sky(rd);
  if (rd.y < .0015f) {
    float3 wd = norm(v3(rd.x, fminf(rd.y, -.0015f), rd.z));
    float t = -camY / wd.y;
    float4 a = make_float4(0, 0, 0, 0);
    for (int i = 0; i < 5; i++) {
      a = water(surface, rip, camX + wd.x * t, camZ + wd.z * t, centerX, centerZ, t);
      t = lerp(t, (a.x - camY) / wd.y, .7f);
    }
    float3 P = v3(camX + wd.x * t, camY + wd.y * t, camZ + wd.z * t);
    a = water(surface, rip, P.x, P.z, centerX, centerZ, t);
    float3 n = norm(v3(-a.y, 1, -a.z)), v = mul(wd, -1);
    float nv = dot3(n, v);
    if (nv < .02f) {
      n = norm(add(n, mul(v, .02f - nv)));
      nv = dot3(n, v);
    }
    float F = fresnel(nv);
    float3 rr = sub(wd, mul(n, 2 * dot3(wd, n)));
    rr.y = fabsf(rr.y);
    float3 reflection = mul(sky(rr), 1.25f), h = norm(add(v, sun));
    float nh = fmaxf(0, dot3(n, h)), nl = fmaxf(0, dot3(n, sun)),
          a2 = .00012f + 1.2f * a.w + .000025f * t, c2 = fmaxf(nh * nh, .0001f),
          tan2 = (1 - c2) / c2, D = expf(-tan2 / a2) / (3.14159265f * a2 * c2 * c2),
          Vis = .5f / (nl * sqrtf(nv * nv * (1 - a2) + a2) + nv * sqrtf(nl * nl * (1 - a2) + a2) +
                       .00001f);
    float3 spec = mul(SUN, fminf(12000, D * Vis * fresnel(dot3(h, v)) * nl));
    float3 tr = refract3(wd, n, 1 / 1.3335f);
    float dist = (-floorDepth(P.x, P.z, depth) - P.y) / tr.y;
    float3 FP = add(P, mul(tr, dist));
    for (int i = 0; i < 2; i++) {
      dist = (-floorDepth(FP.x, FP.z, depth) - P.y) / tr.y;
      FP = add(P, mul(tr, dist));
    }
    dist = fmaxf(0, dist);
    float dh = fmaxf(.05f, P.y - FP.y);
    float footprint = t / (float)height * 1.2f;
    float3 alb = stone(pebbles, FP.x, FP.z, footprint);
    float zone = fbm(FP.x * .16f + 3, FP.z * .16f + 3), sand = smooth(.64f, .8f, zone),
          marks = .5f +
                  .5f * sinf((FP.x * .93f + FP.z * .37f) * 16 + 3 * noise(FP.x * .8f, FP.z * .8f));
    alb = mix3(alb, mul(v3(.36f, .30f, .19f), .82f + .1f * marks), sand);
    float big = .65f * noise(FP.x * .45f, FP.z * .45f) +
                .35f * noise(FP.x * 1.3f + 3.1f, FP.z * 1.3f + 3.1f);
    alb = mul(alb, lerp(.62f, 1.22f, big));
    alb = mix3(v3(.30f, .29f, .27f), v3(powf(alb.x, 1.2f), powf(alb.y, 1.2f), powf(alb.z, 1.2f)),
               .72f);
    alb = prod(mul(alb, .6f), v3(1.1f, 1, .86f));
    float4 C = sample4(caustics, FP.x * 512 / 4.6f, FP.z * 512 / 4.6f, 512, 0);
    float3 caus = v3(C.x, C.y, C.z);
    float3 sunT = refract3(mul(sun, -1), v3(0, 1, 0), 1 / 1.3335f);
    float4 R = rippleAt(rip, FP.x - sunT.x * dh / (-sunT.y), FP.z - sunT.z * dh / (-sunT.y),
                        centerX, centerZ);
    caus = mul(caus, fminf(3, fmaxf(.45f, 1 / (1 + .12f * dh * R.w))));
    caus = mix3(caus, v3(1, 1, 1), smooth(.01f, .12f, footprint));
    float3 sig = v3(.428f, .126f, .156f);
    float Ts = 1 - fresnel(sun.y);
    float3 Esun = prod(prod(mul(SUN, Ts * (-sunT.y)), exp3(mul(sig, -dh / (-sunT.y)))), caus),
           Esky =
               prod(v3(.4285f, .4838f, .5391f), exp3(mul(v3(.4112f, .0948f, .1152f), -dh * 1.25f))),
           Lfloor = prod(mul(alb, 1 / 3.14159265f), add(Esun, Esky)), Tv = exp3(mul(sig, -dist));
    float cosS = dot3(sunT, mul(tr, -1)),
          ph = .36f / (12.5663706f * powf(1.64f - 1.6f * cosS, 1.5f));
    float3 Lmid =
        add(prod(mul(SUN, Ts * (ph + .02f)), exp3(mul(sig, -dh * .5f / (-sunT.y)))),
            prod(v3(.0341f, .0385f, .0429f), exp3(mul(v3(.4f, .074f, .088f), -dh * .6f))));
    float3 Lin =
        mul(prod(prod(v3(.028f / .428f, .052f / .126f, .068f / .156f), Lmid), sub(v3(1, 1, 1), Tv)),
            3.2f);
    float3 under = add(prod(Lfloor, Tv), Lin);
    col = add(add(mul(reflection, F), mul(under, 1 - F)), spec);
    float haze = (1 - expf(-t * .004f)) * .8f;
    col = mix3(col, v3(.57f, .6745f, .779f), haze);
    col = mix3(col, sky(rd), smooth(-.0005f, .0015f, rd.y));
    if (view == 1)
      col = mul(caus, .4f);
    if (view == 2)
      col = add(mul(n, .5f), v3(.5f, .5f, .5f));
  }
  float mu = dot3(rd, sun);
  col = add(col, mul(SUN, 18 * smooth(.99996f, .999985f, mu)));
  hdr[iy * width + ix] = make_float4(fmaxf(0, col.x), fmaxf(0, col.y), fmaxf(0, col.z), 1);
}
__global__ void bloom_pass(const float4 *input, float4 *output, int width, int height, int axis) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height)
    return;
  float3 sum = v3(0, 0, 0);
  float total = 0;
  for (int i = -12; i <= 12; i++) {
    int xx = min(width - 1, max(0, x + (axis == 0 ? i * 2 : 0))),
        yy = min(height - 1, max(0, y + (axis == 1 ? i * 2 : 0)));
    float4 a = input[yy * width + xx];
    float w = expf(-(float)(i * i) / 40);
    float3 c = v3(a.x, a.y, a.z);
    if (axis == 0)
      c = v3(fmaxf(0, c.x - 2.5f), fmaxf(0, c.y - 2.5f), fmaxf(0, c.z - 2.5f));
    sum = add(sum, mul(c, w));
    total += w;
  }
  sum = mul(sum, 1 / total);
  output[y * width + x] = make_float4(sum.x, sum.y, sum.z, 1);
}
// Lens aperture diffraction: RGB wavelengths, hexagonal aperture and scratches.
// FFT(aperture) -> |amplitude|^2 -> normalized point-spread function -> FFT.
__global__ void lens_aperture(float4 *output) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      y = (int)(blockIdx.y * blockDim.y + threadIdx.y), c = (int)blockIdx.z;
  if (x >= 256 || y >= 256)
    return;
  float radius = 28.16f * 550 / (c == 0 ? 620.0f : (c == 1 ? 530.0f : 460.0f)), sum = 0;
  for (int sy = 0; sy < 3; sy++)
    for (int sx = 0; sx < 3; sx++) {
      float dx = (float)x - 128 + ((float)sx + .5f) / 3 - .5f,
            dy = (float)y - 128 + ((float)sy + .5f) / 3 - .5f,
            ok = dx * dx + dy * dy < radius * radius ? 1.0f : 0.0f;
      for (int j = 0; j < 6; j++) {
        float ang = .261799f + (float)j * 1.04719755f;
        if (dx * cosf(ang) + dy * sinf(ang) > radius * .955f)
          ok = 0;
      }
      if (fabsf(dx * .93358f + dy * .35837f - .12f * radius) < .55f)
        ok = 0;
      if (fabsf(dx * .92388f + dy * .38268f + .38f * radius) < .40f)
        ok = 0;
      if (fabsf(dx * (-.88295f) + dy * .46947f - .25f * radius) < .25f)
        ok = 0;
      sum += ok;
    }
  output[c * 65536 + y * 256 + x] = make_float4(sum / 9, 0, 0, 0);
}
__global__ void lens_power(const float4 *amplitude, float4 *psf) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      y = (int)(blockIdx.y * blockDim.y + threadIdx.y), c = (int)blockIdx.z;
  if (x >= 256 || y >= 256)
    return;
  int id = c * 65536 + y * 256 + x;
  float4 a = amplitude[id];
  float dx = (float)(x < 128 ? x : x - 256), dy = (float)(y < 128 ? y : y - 256),
        r = sqrtf(dx * dx + dy * dy),
        val = (a.x * a.x + a.y * a.y) * (1 + 7 * sat((r - 1.5f) / 15));
  if (r > 30)
    val = 0;
  psf[id] = make_float4(val, 0, 0, 0);
}
__global__ void lens_rows(const float4 *psf, float *sums) {
  int z = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (z >= 768)
    return;
  float sum = 0;
  for (int x = 0; x < 256; x++)
    sum += psf[z * 256 + x].x;
  sums[z] = sum;
}
__global__ void lens_normalize(const float4 *psf, const float *sums, float4 *output) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      y = (int)(blockIdx.y * blockDim.y + threadIdx.y), c = (int)blockIdx.z;
  if (x >= 256 || y >= 256)
    return;
  float total = 0;
  for (int j = 0; j < 256; j++)
    total += sums[c * 256 + j];
  int id = c * 65536 + y * 256 + x;
  output[id] = make_float4(psf[id].x / fmaxf(total, .000001f) / 65536, 0, 0, 0);
}
// A 32-pixel empty border plus finite PSF support prevents circular wrap ghosts.
__global__ void glare_source(const float4 *hdr, float4 *output, int width, int height) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      y = (int)(blockIdx.y * blockDim.y + threadIdx.y), c = (int)blockIdx.z;
  if (x >= 256 || y >= 256)
    return;
  float fit = 192.0f / (float)max(width, height), gw = (float)width * fit, gh = (float)height * fit,
        px = ((float)x - 32) / gw, pz = ((float)y - 32) / gh;
  float val = 0;
  if (px >= 0 && px < 1 && pz >= 0 && pz < 1) {
    for (int j = 0; j < 3; j++)
      for (int i = 0; i < 3; i++) {
        int xx = min(width - 1, max(0, (int)(px * (float)width + ((float)i - 1) / fit * .4f))),
            yy = min(height - 1, max(0, (int)(pz * (float)height + ((float)j - 1) / fit * .4f)));
        float4 a = hdr[yy * width + xx];
        val += fminf(80000, fmaxf(0, (c == 0 ? a.x : (c == 1 ? a.y : a.z)) - 14)) / 9;
      }
  }
  output[c * 65536 + y * 256 + x] = make_float4(val, 0, 0, 0);
}
__global__ void glare_multiply(const float4 *input, const float4 *kernel, float4 *output) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      y = (int)(blockIdx.y * blockDim.y + threadIdx.y), c = (int)blockIdx.z;
  if (x >= 256 || y >= 256)
    return;
  int id = c * 65536 + y * 256 + x;
  float4 a = input[id], b = kernel[id];
  output[id] = make_float4(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x, 0, 0);
}
__device__ float tone(float x) {
  return powf(sat((x * (2.51f * x + .03f)) / (x * (2.43f * x + .59f) + .14f)), 1 / 2.2f);
}
__global__ void present(const float4 *hdr, const float4 *bloom, const float4 *diffraction,
                        unsigned *image, int width, int height, float exposure, int glare) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height)
    return;
  int id = y * width + x;
  float4 a = hdr[id], b = bloom[id];
  float3 c = add(v3(a.x, a.y, a.z), mul(v3(b.x, b.y, b.z), glare != 0 ? .08f : 0));
  if (glare != 0) {
    float fit = 192.0f / (float)max(width, height), gx = 32 + ((float)x + .5f) * fit,
          gy = 32 + ((float)y + .5f) * fit;
    float4 dr = sample4(diffraction, gx, gy, 256, 0), dg = sample4(diffraction, gx, gy, 256, 65536),
           db = sample4(diffraction, gx, gy, 256, 131072);
    c = add(c, mul(v3(fmaxf(0, dr.x), fmaxf(0, dg.x), fmaxf(0, db.x)), .14f));
  }
  c = mul(c, exposure);
  unsigned r = (unsigned)(tone(c.x) * 255 + .5f), g = (unsigned)(tone(c.y) * 255 + .5f),
           bl = (unsigned)(tone(c.z) * 255 + .5f);
  image[id] = r | (g << 8) | (bl << 16) | 4278190080u;
}
