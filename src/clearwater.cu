// SPDX-License-Identifier: MIT
// Clearwater CUDA reimplementation. Original optical design: Lumaris (2026).
// All spectrum generation, FFT, ripples, ray projection, optics and post are
// CUDA. The browser host only supplies inputs, resources and dispatches.
__device__ float sat(float x) { return fminf(1.0f, fmaxf(0.0f, x)); }
__device__ float square(float x) { return x * x; }
__device__ float frac(float x) { return x - floorf(x); }
__device__ float lerp(float a, float b, float t) { return a + (b - a) * t; }
__device__ float smooth(float a, float b, float x) {
  float t = sat((x - a) / (b - a));
  return t * t * (3.0f - 2.0f * t);
}
__device__ float3 v3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ float3 add(float3 a, float3 b) {
  return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 sub(float3 a, float3 b) {
  return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 mul(float3 a, float b) {
  return v3(a.x * b, a.y * b, a.z * b);
}
__device__ float3 prod(float3 a, float3 b) {
  return v3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ float dot3(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 norm(float3 a) {
  return mul(a, rsqrtf(fmaxf(dot3(a, a), 0.00000001f)));
}
__device__ float3 mix3(float3 a, float3 b, float t) {
  return add(mul(a, 1 - t), mul(b, t));
}
__device__ float3 exp3(float3 a) { return v3(expf(a.x), expf(a.y), expf(a.z)); }
__device__ float3 refract3(float3 d, float3 n, float eta) {
  float c = dot3(n, d);
  return sub(mul(d, eta),
             mul(n, eta * c + sqrtf(fmaxf(0, 1 - eta * eta * (1 - c * c)))));
}
__device__ float4 mix4(float4 a, float4 b, float t) {
  return make_float4(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t),
                     lerp(a.w, b.w, t));
}
__device__ unsigned hashU(unsigned x) {
  x ^= x >> 16;
  x *= 2146121005u;
  x ^= x >> 15;
  x *= 2221713035u;
  x ^= x >> 16;
  return x;
}
__device__ float random(unsigned x) {
  return ((float)(hashU(x) & 16777215u) + 1.0f) / 16777217.0f;
}
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
  return .55f * noise(x, z) +
         .28f * noise(x * 2.03f + 17.1f, z * 2.03f + 17.1f) +
         .12f * noise(x * 4.12f, z * 4.12f) +
         .05f * noise(x * 8.36f, z * 8.36f);
}
__device__ int wrap(int x, int n) { return (x % n + n) % n; }
__device__ float lengthL(int c) {
  return c == 0 ? 4.6f : (c == 1 ? 37.0f : 293.0f);
}
__device__ float4 sample4(const float4 *data, float x, float z, int n,
                          int offset) {
  int ix = (int)floorf(x), iz = (int)floorf(z);
  float fx = frac(x), fz = frac(z);
  return mix4(mix4(data[offset + wrap(iz, n) * n + wrap(ix, n)],
                   data[offset + wrap(iz, n) * n + wrap(ix + 1, n)], fx),
              mix4(data[offset + wrap(iz + 1, n) * n + wrap(ix, n)],
                   data[offset + wrap(iz + 1, n) * n + wrap(ix + 1, n)], fx),
              fz);
}

// A finite moving storm band in world space. Time is weather time in seconds;
// the showcase advances weather faster while wave propagation stays real-time.
__device__ float stormAt(const float4 *weather, float x, float z) {
  float4 a = weather[0], b = weather[1];
  if (b.z < 0)
    return 0;
  float q = (x * b.x + z * b.y - a.w) / 650.0f;
  float cells=.78f+.22f*noise(x*.0015f+b.z*.0007f,z*.0015f);
  return expf(-q * q) * b.w*cells;
}
__device__ float cloudAt(const float4 *weather, float x, float z) {
  float cloud = weather[0].z;
  float storm = stormAt(weather, x, z);
  float detail = fbm(x * .0018f + weather[1].z * .007f, z * .0018f);
  return sat(cloud + storm * .78f + .25f * (detail - .5f));
}
__global__ void weather_update(float4 *weather, float age, float direction,
                               float strength, float wind, float rain,
                               float clouds, float dt, float camX, float camZ) {
  if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0 ||
      threadIdx.y != 0)
    return;
  float4 old = weather[0];
  float windFront =
      (camX * cosf(direction) + camZ * sinf(direction) - (age - 900) * 1.7f) /
      1105.0f;
  float desired =
      wind + strength * 17.0f * (age < 0 ? 0 : expf(-square(windFront)));
  weather[0] =
      make_float4(lerp(old.x > 0 ? old.x : wind, desired, 1 - expf(-dt / 25)),
                  rain, clouds, (age - 900) * 1.7f);
  weather[1] = make_float4(cosf(direction), sinf(direction), age, strength);
}
// Directional JONSWAP frequency shape converted to a radial k-space density.
// Used as a bounded redistribution of the existing independently seeded bands.
__device__ float windDensity(float k, float projection, float wind) {
  float omega = sqrtf(9.81f * k), peak = 9.81f / fmaxf(3, wind) * .88f;
  float sigma = omega < peak ? .07f : .09f,
        delta = (omega - peak) / (sigma * peak);
  float peakBoost = powf(3.3f, expf(-.5f * delta * delta));
  float ratio = peak / fmaxf(omega, .0001f);
  return expf(-1.25f * ratio * ratio * ratio * ratio) * peakBoost *
         (.12f + .88f * projection * projection) / (k * k * k * k + .000001f);
}
__global__ void foam_step(const float4 *surface, const float4 *previous,
                          float4 *next, const float4 *weather, float dt) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 256 || z >= 256)
    return;
  float4 w = weather[0], d = weather[1];
  float4 a = surface[65536 + z * 256 + x];
  float4 detail =
      sample4(surface, (float)x * 37 / 4.6f, (float)z * 37 / 4.6f, 256, 0);
  float steep = sqrtf((a.y + detail.y) * (a.y + detail.y) +
                      (a.z + detail.z) * (a.z + detail.z));
  float production =
      smooth(.22f, .48f, steep) * smooth(.07f, .30f, a.x) * smooth(7, 20, w.x);
  float4 old = sample4(previous, (float)x - d.x * w.x * .035f * dt * 256 / 37,
                       (float)z - d.y * w.x * .035f * dt * 256 / 37, 256, 0);
  next[z * 256 + x] = make_float4(
      sat(old.x * expf(-dt * .18f) + production * dt * .65f), 0, 0, 0);
}
// Three independent narrow-band spectra: capillary detail, wind waves and
// swell. A GPU reduction normalizes each cascade by expected RMS slope (as
// upstream).
__global__ void seed_spectrum(float2 *seed, unsigned seedValue) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y), c = (int)blockIdx.z;
  if (x >= 256 || z >= 256)
    return;
  int id = c * 65536 + z * 256 + x;
  float L = lengthL(c), kx = 6.283185307f * (float)(x < 128 ? x : x - 256) / L,
        kz = 6.283185307f * (float)(z < 128 ? z : z - 256) / L,
        k = sqrtf(kx * kx + kz * kz);
  float P = 0;
  if (k > 0.00001f) {
    float peak = c == 0 ? .62f : (c == 1 ? 7.0f : 65.0f),
          kp = 6.283185307f / peak, lk = logf(k / kp),
          bump = expf(-.5f * lk * lk / (.36f * .36f));
    float tail =
        .035f * expf(-kp * kp / (k * k)) * expf(-k * k / (kp * kp * 180));
    float swell = .35f * expf(-.5f * square(logf(k / (kp * .3875f)) / .3f));
    float dir = (kx * .8f + kz * .6f) / k;
    P = (bump + tail + swell) * (.3f + .7f * dir * dir) *
        (dir < 0 ? .35f : 1.0f) / (k * k * k * k);
  }
  unsigned h = (unsigned)id + seedValue * 19391u;
  float radius = sqrtf(-2 * logf(random(h * 2u + 1u))),
        angle = 6.283185307f * random(h * 2u + 2u), amp = sqrtf(P * .5f);
  seed[id] =
      make_float2(radius * cosf(angle) * amp, radius * sinf(angle) * amp);
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
  scales[c] = (c == 0 ? .078f : (c == 1 ? .047f : .021f)) /
              sqrtf(fmaxf(sum, .0000000001f));
}
__global__ void evolve_spectrum(const float2 *seed, const float *scales,
                                float4 *output, float time, float sea,
                                float depth, const float4 *weather,
                                float *spectralEnergy, float weatherDt) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y), c = (int)blockIdx.z;
  if (x >= 256 || z >= 256)
    return;
  int id = c * 65536 + z * 256 + x,
      j = c * 65536 + wrap(-z, 256) * 256 + wrap(-x, 256);
  float2 a = seed[id], b = seed[j];
  float L = lengthL(c), kx = 6.283185307f * (float)(x < 128 ? x : x - 256) / L,
        kz = 6.283185307f * (float)(z < 128 ? z : z - 256) / L,
        k = sqrtf(kx * kx + kz * kz);
  float kd = fminf(20, k * depth),
        th = (1 - expf(-2 * kd)) / (1 + expf(-2 * kd));
  float omega = sqrtf((9.81f * k + .000074f * k * k * k) * th),
        co = cosf(omega * time), si = sinf(omega * time),
        scale = scales[c] * sea;
  float4 wind = weather[0], direction = weather[1];
  float oldEnergy = spectralEnergy[id];
  if (oldEnergy <= 0)
    oldEnergy = 1;
  float target = 1;
  if (k > .0001f && c < 2) {
    float projection = (kx * direction.x + kz * direction.y) / k;
    float initial = windDensity(k, (kx * .8f + kz * .6f) / k, 5);
    float changed = windDensity(k, projection, wind.x);
    float shape = fminf(6, fmaxf(.15f, changed / fmaxf(initial, .000001f)));
    target = lerp(1, shape, smooth(5, 16, wind.x)) *
             (1 + smooth(5, 24, wind.x) * (c == 0 ? 1.8f : 3.2f));
  }
  float response = c == 0 ? 35.0f : (c == 1 ? 180.0f : 900.0f);
  float remembered = lerp(
      oldEnergy, target,
      1 - expf(-weatherDt / (target < oldEnergy ? response * 3 : response)));
  spectralEnergy[id] = remembered;
  scale *= sqrtf(remembered);
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
__global__ void fft_pass(const float4 *input, float4 *output, int p, int axis,
                         float sign) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y), c = (int)blockIdx.z;
  if (x >= 256 || z >= 256)
    return;
  int j = axis == 0 ? x : z, k = j & (p - 1),
      i = ((j - (j & (2 * p - 1))) >> 1) + k,
      ia = c * 65536 + (axis == 0 ? z * 256 + i : i * 256 + x),
      ib = ia + (axis == 0 ? 128 : 32768);
  float4 a = input[ia], b = input[ib];
  float ang = sign * 3.14159265359f * (float)k / (float)p, co = cosf(ang),
        si = sinf(ang), sgn = (j & p) != 0 ? -1.0f : 1.0f;
  output[c * 65536 + z * 256 + x] = make_float4(
      a.x + sgn * (co * b.x - si * b.y), a.y + sgn * (si * b.x + co * b.y),
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
// Recover Hermitian height from packed H+i*dH/dx and pack horizontal displacement.
__global__ void chop_spectrum(const float4 *input,float4 *output){
 int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),z=(int)(blockIdx.y*blockDim.y+threadIdx.y),c=(int)blockIdx.z;
 if(x>=256||z>=256)return;
 int id=c*65536+z*256+x,op=c*65536+wrap(-z,256)*256+wrap(-x,256);
 float4 a=input[id],b=input[op];
 float re=(a.x+b.x)*.5f,im=(a.y-b.y)*.5f;
 float kx=(float)(x<128?x:x-256),kz=(float)(z<128?z:z-256);
 if(x==128)kx=0;if(z==128)kz=0;
 float inv=rsqrtf(fmaxf(kx*kx+kz*kz,.00001f))*(c==0?.18f:(c==1?1.15f:.75f));
 output[id]=make_float4((-kx*im-kz*re)*inv,(kx*re-kz*im)*inv,0,0);
}
// Resample the displaced parametric surface onto the ray solver's world grid.
// Limit inverse-map excursions and slope amplification before folding occurs.
__global__ void chop_surface(const float4 *input,const float4 *displacement,float4 *surface){
 int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),z=(int)(blockIdx.y*blockDim.y+threadIdx.y),c=(int)blockIdx.z;
 if(x>=256||z>=256)return;
 int offset=c*65536,id=offset+z*256+x;
 float scale=256/lengthL(c),qx=(float)x,qz=(float)z;
 for(int i=0;i<3;i++){
   float4 d=sample4(displacement,qx,qz,256,offset);
   qx=(float)x-fminf(6,fmaxf(-6,d.x*scale));qz=(float)z-fminf(6,fmaxf(-6,d.y*scale));
 }
 float4 h=sample4(input,qx,qz,256,offset);
 float4 xp=sample4(displacement,qx+1,qz,256,offset),xm=sample4(displacement,qx-1,qz,256,offset);
 float4 zp=sample4(displacement,qx,qz+1,256,offset),zm=sample4(displacement,qx,qz-1,256,offset);
 float xx=1+(xp.x-xm.x)*scale*.5f,xz=(zp.x-zm.x)*scale*.5f;
 float zx=(xp.y-xm.y)*scale*.5f,zz=1+(zp.y-zm.y)*scale*.5f;
 float inv=1/fmaxf(.45f,xx*zz-xz*zx);
 float nx=fminf(1.8f,fmaxf(-1.8f,(zz*h.y-zx*h.z)*inv));
 float nz=fminf(1.8f,fmaxf(-1.8f,(xx*h.z-xz*h.y)*inv));
 surface[id]=make_float4(h.x,nx,nz,nx*nx+nz*nz);
}
__global__ void ocean_foam(const float4 *surface,const float4 *displacement,const float4 *previous,
                            float4 *next,const float4 *weather,float dt){
 int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),z=(int)(blockIdx.y*blockDim.y+threadIdx.y);
 if(x>=256||z>=256)return;
 int id=z*256+x;
 float4 a=surface[65536+id];
 float4 xp=displacement[65536+z*256+wrap(x+1,256)],xm=displacement[65536+z*256+wrap(x-1,256)];
 float4 zp=displacement[65536+wrap(z+1,256)*256+x],zm=displacement[65536+wrap(z-1,256)*256+x];
 float xx=1+(xp.x-xm.x)*128/37,zz=1+(zp.y-zm.y)*128/37;
 float jac=xx*zz-(zp.x-zm.x)*(xp.y-xm.y)*square(128.0f/37);
 float steep=sqrtf(a.y*a.y+a.z*a.z);
 float breaking=smooth(.82f,.52f,jac)*smooth(.02f,.15f,a.x)*smooth(6,16,weather[0].x);
 float drift=weather[0].x*.035f;
 float4 old=sample4(previous,(float)x-(weather[1].x*drift+a.y*.4f)*dt*256/37,
                              (float)z-(weather[1].y*drift+a.z*.4f)*dt*256/37,256,0);
 float cover=sat(old.x*expf(-dt*.24f)+breaking*dt*2.4f);
 float fresh=sat(old.y*expf(-dt*2.8f)+breaking*dt*6);
 next[id]=make_float4(cover,fresh,jac,steep);
}

__device__ float3 ray(float sx, float sy, float aspect, float yaw,
                      float pitch) {
  float cy = cosf(yaw), syaw = sinf(yaw), cp = cosf(pitch), sp = sinf(pitch);
  return norm(
      v3(syaw * cp + sx * aspect * .62487f * cy - sy * .62487f * syaw * sp,
         sp + sy * .62487f * cp,
         -cy * cp + sx * aspect * .62487f * syaw + sy * .62487f * cy * sp));
}
// Fixed 120 Hz damped wave equation, camera-relative grid with integer
// recentering.
__global__ void ripple_step(const float4 *previous, float4 *next, int shiftX,
                            int shiftZ, float centerX, float centerZ,
                            float camX, float camZ, float camY, float yaw,
                            float pitch, float aspect, float tapX, float tapY,
                            int drop, const float4 *weather, float time) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 256 || z >= 256)
    return;
  int sx = x + shiftX, sz = z + shiftZ, id = z * 256 + x;
  float h = 0, vel = 0;
  if (sx > 0 && sx < 255 && sz > 0 && sz < 255) {
    int old = sz * 256 + sx;
    float4 a = previous[old];
    float lap = previous[old - 1].x + previous[old + 1].x +
                previous[old - 256].x + previous[old + 256].x - 4 * a.x;
    vel = (a.y + .21f * lap) * .994f;
    h = (a.x + vel) * .999f;
  }
  if (drop != 0) {
    float3 d = ray(tapX, tapY, aspect, yaw, pitch);
    if (d.y < -.01f) {
      float t = -camY / d.y, px = camX + d.x * t, pz = camZ + d.z * t,
            wx = centerX + ((float)x - 128) * .0625f,
            wz = centerZ + ((float)z - 128) * .0625f,
            dist = sqrtf((wx - px) * (wx - px) + (wz - pz) * (wz - pz));
      h -= .065f * expf(-dist * dist / 0.0225f);
    }
  }
  float wx = centerX + ((float)x - 128) * .0625f,
        wz = centerZ + ((float)z - 128) * .0625f;
  float rainfall = sat(weather[0].y + stormAt(weather, wx, wz) * .9f);
  unsigned tick = (unsigned)floorf(time * 120);
  float chance = random((unsigned)(x + z * 256) + tick * 65537u);
  if (chance < rainfall * .0024f)
    h -= .007f * (.4f + .6f * random(tick + (unsigned)id));
  float edge = smooth(0, 16, (float)min(min(x, 255 - x), min(z, 255 - z)));
  next[id] =
      make_float4(h * lerp(.85f, 1, edge), vel * lerp(.85f, 1, edge), 0, 0);
}
__global__ void ripple_normals(const float4 *input, float4 *output) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 256 || z >= 256)
    return;
  int id = z * 256 + x;
  float l = input[z * 256 + max(0, x - 1)].x,
        r = input[z * 256 + min(255, x + 1)].x,
        b = input[max(0, z - 1) * 256 + x].x,
        f = input[min(255, z + 1) * 256 + x].x, h = input[id].x;
  output[id] =
      make_float4(h, (r - l) * 8, (f - b) * 8, (r + l + b + f - 4 * h) * 256);
}
__device__ float4 rippleAt(const float4 *rip, float x, float z, float cx,
                           float cz) {
  float u = (x - cx) * 16 + 128, w = (z - cz) * 16 + 128;
  if (u < 1 || u > 254 || w < 1 || w > 254)
    return make_float4(0, 0, 0, 0);
  return sample4(rip, u, w, 256, 0);
}
__device__ float4 water(const float4 *surf, const float4 *rip, float x, float z,
                        float cx, float cz, float distance) {
  float4 a = sample4(surf, x * 256 / 4.6f, z * 256 / 4.6f, 256, 0),
         b = sample4(surf, x * 256 / 37, z * 256 / 37, 256, 65536),
         c = sample4(surf, x * 256 / 293, z * 256 / 293, 256, 131072),
         r = rippleAt(rip, x, z, cx, cz);
  float fade = 1 / (1 + distance * .018f);
  float windFade=1/(1+square(distance*.003f));
  return make_float4(a.x + b.x + c.x + r.x, (a.y * fade + b.y*windFade + c.y + r.y),
                     (a.z * fade + b.z*windFade + c.z + r.z),
                     fmaxf(0, a.w - a.y * a.y - a.z * a.z) +
                         .006f * (1 - fade)+.022f*(1-windFade));
}
// RGB photon splats. Fixed-point atomics avoid floating-point atomic
// requirements.
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
__global__ void trace_caustics(const float4 *surface, unsigned *photons,
                               float depth) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 1024 || z >= 1024)
    return;
  float wx = ((float)x + .5f) * 4.6f / 1024,
        wz = ((float)z + .5f) * 4.6f / 1024;
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
      sum = add(sum, v3((float)photons[id] * w, (float)photons[id + 1] * w,
                        (float)photons[id + 2] * w));
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
  float3 col =
      mix3(v3(.66f, .78f, .9f), v3(.11f, .27f, .62f), powf(sat(e), .42f));
  col =
      add(col, mul(v3(1, .86f, .66f), .22f * powf(mu, 6) + .3f * powf(mu, 64) +
                                          1.6f * powf(mu, 2400)));
  float a = atan2f(d.z, d.x),
        ridge = .04f + .016f * sinf(a * 2 + .7f) + .011f * sinf(a * 5 + 2.1f) +
                .006f * sinf(a * 11 + .3f) + .003f * sinf(a * 23 + 1.7f) +
                .002f * sinf(a * 41 + 2);
  float tex = fbm(a * 420, e * 420),
        cliff =
            smooth(.42f, .18f, e / fmaxf(ridge, .001f) + .25f * (tex - .5f)) *
            smooth(.35f, .75f, noise(a * 18, 1));
  float3 land = mix3(mul(v3(.045f, .070f, .042f), .6f + .8f * tex),
                     mul(v3(.30f, .28f, .23f), .55f + .7f * tex), cliff);
  land = mix3(land, v3(.6072f, .7176f, .828f), .48f);
  return mix3(col, land, smooth(ridge + .0009f, ridge - .0009f, e));
}

__device__ float lightning(const float4 *weather, float time) {
  float storm = stormAt(weather, 0, 0);
  float cycle = frac(time / 13.0f) * 13.0f;
  return storm * (expf(-square((cycle - 2.0f) / .035f)) +
                  .6f * expf(-square((cycle - 2.18f) / .055f)));
}
// Layered cloud volume approximation: integrate density through a bounded slab.
// The same sky function supplies reflected radiance and the visible
// environment.
__device__ float3 weatherSky(float3 d, const float4 *weather, float x, float z,
                             float time) {
  float overhead = stormAt(weather, x, z), altitude = fmaxf(.06f, d.y);
  float storm = stormAt(weather, x + d.x * 900, z + d.z * 900);
  float3 base = mul(sky(d), 1 - .76f * overhead);
  float3 horizon = mix3(v3(.63f, .75f, .83f), v3(.095f, .14f, .19f),
                        fmaxf(storm, overhead * .8f));
  base = mix3(base, horizon, (1 - smooth(.0f, .18f, d.y)) * .65f);
  if (d.y > .008f) {
    float trans = 1;
    float3 scatter = v3(0, 0, 0);
    float mu = fmaxf(0, dot3(d, v3(.08959f, .51504f, -.85247f)));
    for (int i = 0; i < 8; i++) {
      float h = 380 + (float)i * 65, t = fminf(6500, h / altitude),
            px = x + d.x * t + time * 3, pz = z + d.z * t;
      float local = stormAt(weather, px, pz);
      float broad = sat(weather[0].z + overhead * .48f + local * .50f);
      float shape = fbm(px * .0015f, pz * .0015f + (float)i * .09f);
      float detail = noise(px * .006f + (float)i * .3f, pz * .006f);
      float density = smooth(.80f - broad * .55f, .98f - broad * .55f,
                             shape * .82f + detail * .18f);
      float shade = lerp(.30f, .83f, (float)i / 7),
            rim = powf(mu, 12) * (1 - density) * .18f;
      float3 lit = mix3(v3(.19f, .25f, .33f), v3(.96f, .91f, .81f), shade);
      lit = mul(lit, 1 - .75f * fmaxf(overhead, local * .7f));
      lit = add(lit, mul(v3(1, .8f, .53f), rim * (1 - overhead * .85f)));
      float opacity = 1 - expf(-density * .85f / sqrtf(altitude));
      scatter = add(scatter, mul(lit, trans * opacity));
      trans *= 1 - opacity;
    }
    float3 clouded = add(mul(base, trans), scatter);
    base = mix3(base, clouded, smooth(.035f, .15f, d.y));
  }
  float curtain = storm * smooth(.20f, .035f, d.y) * smooth(-.02f, .06f, d.y);
  base = mix3(base, v3(.13f, .18f, .22f), curtain * .5f);
  float flash = lightning(weather, time);
  base = add(base, mul(v3(.7f, .82f, 1), flash * .9f));
  if (d.z < -.05f && flash > .002f) {
    float distance = (-1100 - z) / d.z, qy = d.y * distance,
          qx = x + d.x * distance;
    float zig = -220 + 27 * noise(qy * .027f, floorf(time / 13)) +
                14 * sinf(qy * .047f);
    float bolt = expf(-fabsf(qx - zig) / 1.5f) * smooth(40, 90, qy) *
                 smooth(660, 540, qy);
    float branchX = zig + (qy - 270) * .6f;
    bolt += expf(-fabsf(qx - branchX) / 1.1f) * smooth(140, 190, qy) *
            smooth(310, 260, qy) * .5f;
    base = add(base, mul(v3(.65f, .8f, 1), bolt * flash * 18));
  }
  return base;
}
// Main-demo cloud volume. Environment radiance and cloud shadows share density.
// Cached on the GPU once per frame instead of marching for every water pixel.
__device__ float noise3(float3 p) {
  float y = floorf(p.y), f = frac(p.y); f = f*f*(3-2*f);
  return lerp(noise(p.x + y*37, p.z + y*17),
              noise(p.x + (y+1)*37, p.z + (y+1)*17), f);
}
// Rounded cellular lobes break the smooth value-noise silhouettes.
__device__ float cloudLobes(float3 p) {
  float3 cell=v3(floorf(p.x),floorf(p.y),floorf(p.z));
  float nearest=4;
  for(int z=0;z<2;z++)for(int y=0;y<2;y++)for(int x=0;x<2;x++){
    float3 c=add(cell,v3((float)x,(float)y,(float)z));
    unsigned h=(unsigned)((int)c.x*1973+(int)c.y*7919+(int)c.z*9277);
    float3 delta=sub(add(c,v3((random(h)-.5f)*.5f,(random(h+13u)-.5f)*.5f,(random(h+37u)-.5f)*.5f)),p);
    nearest=fminf(nearest,dot3(delta,delta));
  }
  return sat(1-sqrtf(nearest));
}
__device__ float cloudDensity(float3 p, const float4 *weather, float time, int detail) {
  float qfront=(p.x*weather[1].x+p.z*weather[1].y-weather[0].w)/2200;
  float front=weather[1].z<0?0:expf(-qfront*qfront)*weather[1].w;
  float cover = sat(weather[0].z + front*.69f);
  float top = 1550 + front*1500;
  float height = (p.y-360)/(top-360);
  if(height < 0 || height > 1) return 0;
  float envelope = smooth(0,.13f,height)*(1-smooth(.52f,1,height));
  float3 q = v3((p.x-time*weather[1].x*3.5f)*.0011f,
                p.y*.0017f,(p.z-time*weather[1].y*3.5f)*.0011f);
  float shape = noise3(q)*.50f + cloudLobes(mul(q,2.1f))*.28f
              +noise3(add(mul(q,4.07f),v3(11,5,23)))*.15f+noise3(mul(q,8.2f))*.07f;
  float density = sat((shape-(.70f-cover*.42f))*7.0f)*envelope;
  if(detail != 0 && density > .01f) {
    float erosion = .65f*noise3(add(mul(q,5.4f),v3(time*.03f,0,0)))
                  +.35f*noise3(mul(q,13.7f));
    density = sat(density-(1-erosion)*.35f*(1-density));
  }
  return density*(1+front*.65f);
}
__device__ float cloudSun(float3 p, const float4 *weather, float time) {
  float3 sun = v3(.08959f,.51504f,-.85247f);
  float optical = cloudDensity(add(p,mul(sun,70)),weather,time,0)*100
                +cloudDensity(add(p,mul(sun,230)),weather,time,0)*240
                +cloudDensity(add(p,mul(sun,550)),weather,time,0)*440;
  return expf(-optical*.009f);
}
__device__ float3 volumeSky(float3 d, const float4 *weather, float3 origin, float time) {
  float overhead=stormAt(weather,origin.x,origin.z);
  float3 clear=mul(sky(d),1-overhead*.60f), light=mul(v3(1.8f,1.65f,1.42f),1-overhead*.45f);
  float elevation=fmaxf(d.y,.018f);
  float start=fmaxf(0,(360-origin.y)/elevation), end=fminf(11000,(3100-origin.y)/elevation);
  float step=fmaxf(0,end-start)/72, trans=1;
  float3 radiance=v3(0,0,0);
  float mu=dot3(d,v3(.08959f,.51504f,-.85247f));
  float phase=.38f+.65f*powf(fmaxf(0,mu),8);
  for(int i=0;i<72;i++) {
    if(trans < .012f) break;
    float t=start+((float)i+.5f)*step;
    float3 p=add(origin,mul(d,t));
    float density=cloudDensity(p,weather,time,1);
    if(density>.005f) {
      float direct=cloudSun(p,weather,time);
      float high=sat((p.y-360)/1300);
      float3 ambient=mix3(v3(.055f,.085f,.135f),v3(.32f,.41f,.55f),high);
      // Cheap multiple-scattering fill prevents pitch-black cloud interiors.
      float3 lit=add(mul(ambient,.65f+ .35f*expf(-density*2)),mul(light,phase*(direct+.10f*expf(-density))));
      float opacity=1-expf(-density*step*.009f);
      radiance=add(radiance,mul(lit,trans*opacity));
      trans*=1-opacity;
    }
  }
  float3 result=add(radiance,mul(clear,trans));
  float haze=(1-expf(-start*.000065f))*.75f;
  float3 horizon=mix3(v3(.58f,.70f,.82f),v3(.16f,.23f,.31f),overhead);
  result=mix3(result,horizon,haze);
  float curtain=stormAt(weather,origin.x+d.x*1300,origin.z+d.z*1300)
                  *smooth(.30f,.025f,d.y)*(.55f+.45f*noise(d.x*35+time*.08f,d.z*35));
  result=mix3(result,v3(.10f,.155f,.22f),curtain*.65f);
  return mix3(clear,result,smooth(.005f,.045f,d.y));
}
__global__ void sky_environment(float4 *environment, const float4 *weather,
                                 float camX,float camY,float camZ,float time) {
  int x=(int)(blockIdx.x*blockDim.x+threadIdx.x), y=(int)(blockIdx.y*blockDim.y+threadIdx.y);
  if(x>=1024||y>=256)return;
  float angle=((float)x+.5f)*6.283185307f/1024;
  float elevation=square(((float)y+.5f)/256)*1.570796327f;
  float3 d=v3(sinf(angle)*cosf(elevation),sinf(elevation),-cosf(angle)*cosf(elevation));
  float3 col=volumeSky(d,weather,v3(camX,camY,camZ),time);
  environment[y*1024+x]=make_float4(col.x,col.y,col.z,1);
}
__global__ void cloud_shadow(float4 *environment, const float4 *weather,
                              float camX,float camZ,float time) {
  int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),z=(int)(blockIdx.y*blockDim.y+threadIdx.y);
  if(x>=128||z>=128)return;
  float wx=floorf(camX/32)*32+((float)x-64)*32, wz=floorf(camZ/32)*32+((float)z-64)*32;
  float optical=0;
  for(int i=0;i<8;i++){
    float h=400+(float)i*180;
    optical+=cloudDensity(v3(wx+h*.17395f,h,wz-h*1.65518f),weather,time,0)*180/.51504f;
  }
  environment[262144+z*128+x]=make_float4(expf(-optical*.009f),0,0,0);
}
__device__ float3 environmentSky(const float4 *environment,float3 d) {
  float u=atan2f(d.x,-d.z)*1024/6.283185307f-.5f;
  float v=sqrtf(atan2f(sat(d.y),sqrtf(fmaxf(0,1-d.y*d.y)))/1.570796327f)*256-.5f;
  int x=(int)floorf(u),y=(int)floorf(fminf(254.999f,fmaxf(0,v)));
  float4 a=mix4(environment[y*1024+wrap(x,1024)],environment[y*1024+wrap(x+1,1024)],frac(u));
  float4 b=mix4(environment[(y+1)*1024+wrap(x,1024)],environment[(y+1)*1024+wrap(x+1,1024)],frac(u));
  float4 c=mix4(a,b,frac(fminf(254.999f,fmaxf(0,v))));
  return v3(c.x,c.y,c.z);
}
__device__ float environmentShadow(const float4 *environment,float x,float z,float camX,float camZ){
  float u=fminf(126.999f,fmaxf(0,(x-floorf(camX/32)*32)/32+64));
  float v=fminf(126.999f,fmaxf(0,(z-floorf(camZ/32)*32)/32+64));
  return sample4(environment,u,v,128,262144).x;
}

// Buoy signed-distance geometry, oriented to the sampled water normal.
__device__ float buoyDistance(float3 p) {
  float r = sqrtf(p.x * p.x + p.z * p.z);
  float body = fmaxf(r - .25f, fabsf(p.y - .12f) - .24f);
  float ring =
      sqrtf((r - .30f) * (r - .30f) + (p.y + .02f) * (p.y + .02f)) - .08f;
  float pole = fmaxf(r - .035f, fabsf(p.y - .65f) - .40f);
  float cap = fmaxf(r - .11f, fabsf(p.y - 1.06f) - .08f);
  return fminf(fminf(body, ring), fminf(pole, cap));
}
__device__ float floorDepth(float x, float z, float depth) {
  return depth + .12f * (noise(x * .22f, z * .22f) - .5f) +
         .06f * (noise(x * .9f + 7, z * .9f + 7) - .5f);
}
__device__ float3 stone(const float4 *peb, float x, float z, float footprint) {
  float k = noise(x * .85f, z * .85f) * 8, ia = floorf(k), f = frac(k),
        u = x / .78f * 1024, w = z / .78f * 1024;
  float4 a = sample4(peb, u + sinf(3 * ia) * 1024, w + sinf(7 * ia) * 1024,
                     1024, 0),
         b = sample4(peb, u + sinf(3 * (ia + 1)) * 1024,
                     w + sinf(7 * (ia + 1)) * 1024, 1024, 0);
  float m = smooth(.2f, .8f, f - .1f * (a.x + a.y + a.z - b.x - b.y - b.z));
  float3 col = v3(lerp(a.x, b.x, m), lerp(a.y, b.y, m), lerp(a.z, b.z, m));
  return mix3(col, v3(.20f, .18f, .14f), smooth(.015f, .16f, footprint));
}
__global__ void render_water(const float4 *surface, const float4 *rip,
                             const float4 *caustics, const float4 *pebbles,
                             float4 *hdr, int width, int height, float camX,
                             float camZ, float camY, float yaw, float pitch,
                             float centerX, float centerZ, float depth,
                             float time, int view, const float4 *weather,
                             const float4 *foam, int buoy, const float4 *environment) {
  int ix = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      iy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (ix >= width || iy >= height)
    return;
  float sx = 2 * ((float)ix + .5f) / (float)width - 1,
        sy = 1 - 2 * ((float)iy + .5f) / (float)height;
  float3 rd = ray(sx, sy, (float)width / (float)height, yaw, pitch),
         sun = v3(.08959f, .51504f, -.85247f), SUN = v3(6, 5.4f, 4.44f),
         col = environmentSky(environment, rd);
  float visibleDistance = 100000;
  SUN = mul(SUN, environmentShadow(environment,camX,camZ,camX,camZ));
  if (rd.y < .0015f) {
    float3 wd = norm(v3(rd.x, fminf(rd.y, -.0015f), rd.z));
    float t = -camY / wd.y;
    float4 a = make_float4(0, 0, 0, 0);
    for (int i = 0; i < 5; i++) {
      a = water(surface, rip, camX + wd.x * t, camZ + wd.z * t, centerX,
                centerZ, t);
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
    float localStorm = stormAt(weather, P.x, P.z);
    SUN = mul(v3(6, 5.4f, 4.44f),
              environmentShadow(environment,P.x,P.z,camX,camZ));
    float gust =
        (noise(P.x * .08f + time * .6f, P.z * .08f) - .5f) * localStorm * .055f;
    n = norm(add(n, v3(gust, 0, gust * .6f)));
    nv = fmaxf(.02f, dot3(n, v));
    float F = fresnel(nv);
    float3 rr = sub(wd, mul(n, 2 * dot3(wd, n)));
    rr.y = fabsf(rr.y);
    float3 reflection = mul(environmentSky(environment, rr), 1.25f),
           h = norm(add(v, sun));
    float nh = fmaxf(0, dot3(n, h)), nl = fmaxf(0, dot3(n, sun)),
          a2 = .00012f + 1.2f * a.w + .000025f * t, c2 = fmaxf(nh * nh, .0001f),
          tan2 = (1 - c2) / c2,
          D = expf(-tan2 / a2) / (3.14159265f * a2 * c2 * c2),
          Vis = .5f / (nl * sqrtf(nv * nv * (1 - a2) + a2) +
                       nv * sqrtf(nl * nl * (1 - a2) + a2) + .00001f);
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
    float zone = fbm(FP.x * .16f + 3, FP.z * .16f + 3),
          sand = smooth(.64f, .8f, zone),
          marks = .5f + .5f * sinf((FP.x * .93f + FP.z * .37f) * 16 +
                                   3 * noise(FP.x * .8f, FP.z * .8f));
    alb = mix3(alb, mul(v3(.36f, .30f, .19f), .82f + .1f * marks), sand);
    float big = .65f * noise(FP.x * .45f, FP.z * .45f) +
                .35f * noise(FP.x * 1.3f + 3.1f, FP.z * 1.3f + 3.1f);
    alb = mul(alb, lerp(.62f, 1.22f, big));
    alb =
        mix3(v3(.30f, .29f, .27f),
             v3(powf(alb.x, 1.2f), powf(alb.y, 1.2f), powf(alb.z, 1.2f)), .72f);
    alb = prod(mul(alb, .6f), v3(1.1f, 1, .86f));
    float4 C = sample4(caustics, FP.x * 512 / 4.6f, FP.z * 512 / 4.6f, 512, 0);
    float3 caus = v3(C.x, C.y, C.z);
    float3 sunT = refract3(mul(sun, -1), v3(0, 1, 0), 1 / 1.3335f);
    float4 R = rippleAt(rip, FP.x - sunT.x * dh / (-sunT.y),
                        FP.z - sunT.z * dh / (-sunT.y), centerX, centerZ);
    caus = mul(caus, fminf(3, fmaxf(.45f, 1 / (1 + .12f * dh * R.w))));
    caus = mix3(caus, v3(1, 1, 1), smooth(.01f, .12f, footprint));
    float3 sig = v3(.428f, .126f, .156f);
    float Ts = 1 - fresnel(sun.y);
    float3 Esun = prod(
               prod(mul(SUN, Ts * (-sunT.y)), exp3(mul(sig, -dh / (-sunT.y)))),
               caus),
           Esky = prod(v3(.4285f, .4838f, .5391f),
                       exp3(mul(v3(.4112f, .0948f, .1152f), -dh * 1.25f))),
           Lfloor = prod(mul(alb, 1 / 3.14159265f), add(Esun, Esky)),
           Tv = exp3(mul(sig, -dist));
    float cosS = dot3(sunT, mul(tr, -1)),
          ph = .36f / (12.5663706f * powf(1.64f - 1.6f * cosS, 1.5f));
    float3 Lmid = add(
        prod(mul(SUN, Ts * (ph + .02f)), exp3(mul(sig, -dh * .5f / (-sunT.y)))),
        prod(v3(.0341f, .0385f, .0429f),
             exp3(mul(v3(.4f, .074f, .088f), -dh * .6f))));
    float3 Lin =
        mul(prod(prod(v3(.028f / .428f, .052f / .126f, .068f / .156f), Lmid),
                 sub(v3(1, 1, 1), Tv)),
            3.2f);
    float3 under = add(prod(Lfloor, Tv), Lin);
    col = add(add(mul(reflection, F), mul(under, 1 - F)), spec);
    float4 foamValue = sample4(foam, P.x * 256 / 37, P.z * 256 / 37, 256, 0);
    float fx=P.x-time*weather[1].x*weather[0].x*.035f;
    float fz=P.z-time*weather[1].y*weather[0].x*.035f;
    float foamDetail=smooth(.24f,.67f,noise(fx*13,fz*13));
    float lace=smooth(.22f,.58f,noise(fx*2.7f,fz*2.7f));
    float foamMask=sat(smooth(.015f,.32f,foamValue.x)*(.3f+.7f*lace)*(.35f+.65f*foamDetail)+foamValue.y*.4f);
    col = mix3(col, mul(v3(.73f, .81f, .78f), 1 - .5f * localStorm),
               foamMask * .85f);
    float haze = (1 - expf(-t * (.004f + .009f * localStorm))) * .8f;
    col = mix3(col,
               mix3(v3(.57f, .6745f, .779f), v3(.09f, .14f, .18f), localStorm),
               haze);
    col = mix3(col, environmentSky(environment, rd),
               smooth(-.0005f, .0015f, rd.y));
    visibleDistance = t;
    if (view == 1)
      col = mul(caus, .4f);
    if (view == 2)
      col = add(mul(n, .5f), v3(.5f, .5f, .5f));
  }

  // Rain streaks are a near-camera optical layer; impacts are actual ripple
  // impulses.
  float rain = sat(weather[0].y + stormAt(weather, camX, camZ) * .9f);
  for (int layer = 0; layer < 2; layer++) {
    float scale = layer == 0 ? 80.0f : 133.0f;
    float slant =
        .05f + .16f * (weather[1].x * cosf(yaw) + weather[1].y * sinf(yaw));
    float rx = (sx + sy * slant) * scale;
    unsigned column = (unsigned)(int)floorf(rx);
    float ry = sy * scale * .25f + time * (layer == 0 ? 31.0f : 43.0f) +
               random(column) * 17;
    unsigned cellId = column * 1973u + (unsigned)(int)floorf(ry) * 9277u;
    float cell = random(cellId), center = .15f + .7f * random(cellId + 91u);
    float streak = smooth(.075f, 0, fabsf(frac(rx) - center)) *
                   smooth(.49f, .16f, fabsf(frac(ry) - .5f));
    col = add(col, mul(v3(.55f, .65f, .70f),
                       streak * rain * (cell > .83f ? .12f : 0)));
  }
  // An anchored buoy gives the waves a readable physical scale.
  if (buoy != 0) {
    float bx = 2.3f, bz = -8;
    float4 bw = water(surface, rip, bx, bz, centerX, centerZ, 0);
    float3 axis = norm(v3(-bw.y * .75f, 1, -bw.z * .75f)),
           right = norm(v3(axis.y, -axis.x, 0));
    float3 forward = v3(-axis.z * right.y, axis.z * right.x,
                        axis.x * right.y - axis.y * right.x);
    float3 origin = sub(v3(camX, camY, camZ), v3(bx, bw.x, bz));
    float along = -dot3(origin, rd),
          closest = dot3(origin, origin) - along * along;
    if (along > 0 && closest < 2.6f) {
      float distance = fmaxf(0, along - 1.7f);
      for (int j = 0; j < 40; j++) {
        float3 world = add(origin, mul(rd, distance));
        float3 local =
            v3(dot3(world, right), dot3(world, axis), dot3(world, forward));
        float sdf = buoyDistance(local);
        if (sdf < .003f) {
          if (distance < visibleDistance + .12f) {
            float e = .006f;
            float3 normal = norm(v3(buoyDistance(add(local, v3(e, 0, 0))) -
                                        buoyDistance(sub(local, v3(e, 0, 0))),
                                    buoyDistance(add(local, v3(0, e, 0))) -
                                        buoyDistance(sub(local, v3(0, e, 0))),
                                    buoyDistance(add(local, v3(0, 0, e))) -
                                        buoyDistance(sub(local, v3(0, 0, e)))));
            float3 worldN = add(add(mul(right, normal.x), mul(axis, normal.y)),
                                mul(forward, normal.z));
            float light = .35f + .65f * fmaxf(0, dot3(worldN, sun));
            float3 paint =
                local.y > .82f ? v3(.16f, .21f, .20f) : v3(.82f, .22f, .045f);
            if (local.y > .20f && local.y < .35f)
              paint = v3(.87f, .84f, .7f);
            col = mul(paint, light * (1 - .4f * stormAt(weather, bx, bz)));
            if (local.y > 1.0f)
              col = add(col, mul(v3(1, .32f, .025f),
                                 2.5f * powf(fmaxf(0, cosf(time * 2)), 20)));
          }
          break;
        }
        distance += fmaxf(.003f, sdf * .8f);
        if (distance > along + 1.7f)
          break;
      }
    }
  }
  float mu = dot3(rd, sun);
  col = add(col, mul(SUN, 18 * smooth(.99996f, .999985f, mu)));
  hdr[iy * width + ix] =
      make_float4(fmaxf(0, col.x), fmaxf(0, col.y), fmaxf(0, col.z), 1);
}
__global__ void bloom_pass(const float4 *input, float4 *output, int width,
                           int height, int axis) {
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
  float radius = 28.16f * 550 / (c == 0 ? 620.0f : (c == 1 ? 530.0f : 460.0f)),
        sum = 0;
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
  float dx = (float)(x < 128 ? x : x - 256),
        dy = (float)(y < 128 ? y : y - 256), r = sqrtf(dx * dx + dy * dy),
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
__global__ void lens_normalize(const float4 *psf, const float *sums,
                               float4 *output) {
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
// A 32-pixel empty border plus finite PSF support prevents circular wrap
// ghosts.
__global__ void glare_source(const float4 *hdr, float4 *output, int width,
                             int height) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      y = (int)(blockIdx.y * blockDim.y + threadIdx.y), c = (int)blockIdx.z;
  if (x >= 256 || y >= 256)
    return;
  float fit = 192.0f / (float)max(width, height), gw = (float)width * fit,
        gh = (float)height * fit, px = ((float)x - 32) / gw,
        pz = ((float)y - 32) / gh;
  float val = 0;
  if (px >= 0 && px < 1 && pz >= 0 && pz < 1) {
    for (int j = 0; j < 3; j++)
      for (int i = 0; i < 3; i++) {
        int xx = min(width - 1, max(0, (int)(px * (float)width +
                                             ((float)i - 1) / fit * .4f))),
            yy = min(height - 1, max(0, (int)(pz * (float)height +
                                              ((float)j - 1) / fit * .4f)));
        float4 a = hdr[yy * width + xx];
        val +=
            fminf(80000, fmaxf(0, (c == 0 ? a.x : (c == 1 ? a.y : a.z)) - 14)) /
            9;
      }
  }
  output[c * 65536 + y * 256 + x] = make_float4(val, 0, 0, 0);
}
__global__ void glare_multiply(const float4 *input, const float4 *kernel,
                               float4 *output) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      y = (int)(blockIdx.y * blockDim.y + threadIdx.y), c = (int)blockIdx.z;
  if (x >= 256 || y >= 256)
    return;
  int id = c * 65536 + y * 256 + x;
  float4 a = input[id], b = kernel[id];
  output[id] = make_float4(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x, 0, 0);
}
__device__ float tone(float x) {
  return powf(sat((x * (2.51f * x + .03f)) / (x * (2.43f * x + .59f) + .14f)),
              1 / 2.2f);
}
__global__ void present(const float4 *hdr, const float4 *bloom,
                        const float4 *diffraction, unsigned *image, int width,
                        int height, float exposure, int glare) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height)
    return;
  int id = y * width + x;
  float4 a = hdr[id], b = bloom[id];
  float3 c =
      add(v3(a.x, a.y, a.z), mul(v3(b.x, b.y, b.z), glare != 0 ? .08f : 0));
  if (glare != 0) {
    float fit = 192.0f / (float)max(width, height),
          gx = 32 + ((float)x + .5f) * fit, gy = 32 + ((float)y + .5f) * fit;
    float4 dr = sample4(diffraction, gx, gy, 256, 0),
           dg = sample4(diffraction, gx, gy, 256, 65536),
           db = sample4(diffraction, gx, gy, 256, 131072);
    c = add(c, mul(v3(fmaxf(0, dr.x), fmaxf(0, dg.x), fmaxf(0, db.x)), .14f));
  }
  c = mul(c, exposure);
  unsigned r = (unsigned)(tone(c.x) * 255 + .5f),
           g = (unsigned)(tone(c.y) * 255 + .5f),
           bl = (unsigned)(tone(c.z) * 255 + .5f);
  image[id] = r | (g << 8) | (bl << 16) | 4278190080u;
}

