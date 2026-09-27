// SPDX-License-Identifier: MIT
// Water component only. Reference pixels are never supplied to this renderer.
__device__ float3 studyStone(float x,float z){
  float rx=x*.94f+z*.342f,rz=z*.94f-x*.342f;
  float row=floorf(rz/.5f),u=rx/.82f+row*.31f,v=rz/.5f;
  float id=hash(floorf(u),floorf(v));
  float wobble=(noise(x*18,z*18)-.5f)*.018f;
  float edge=fminf(fminf(frac(u),1-frac(u))*.82f,fminf(frac(v),1-frac(v))*.5f)+wobble;
  float grain=fbm(x*12+id*21,z*12),vein=powf(1-fabsf(noise(x*4,z*11)*2-1),18);
  float3 stone=mix3(v3(.095f,.125f,.105f),v3(.30f,.315f,.26f),id*.55f+grain*.35f);
  stone=mul(stone,.78f+grain*.35f+vein*.10f);
  return mix3(v3(.055f,.07f,.035f),stone,smooth(.002f,.019f,edge));
}
// A small procedural lighting proxy supplies reflected foliage and plaster.
// It is not the assembled garden and contains no target-image samples.
__device__ float3 studyReflection(float3 p,float3 d){
  float t=(-4.5f-p.z)/fminf(-.001f,d.z),x=p.x+d.x*t,y=d.y*t;
  float leaves=fbm(x*3.8f,y*4.6f),twig=noise(x*24+y*5,y*32);
  float canopy=smooth(.39f,.62f,leaves);
  float3 light=mix3(v3(.025f,.038f,.014f),v3(.68f,.72f,.61f),canopy);
  light=mul(light,.65f+.35f*twig);
  float wall=(1-smooth(2.1f,2.3f,y))*smooth(.1f,.25f,y);
  float gate=1-smooth(.78f,.9f,sqrtf((x-1.4f)*(x-1.4f)+(y-1)*(y-1)));
  light=mix3(light,mul(v3(.48f,.47f,.39f),.5f+.5f*leaves),wall*(1-gate));
  float sun=powf(sat(dot3(d,norm(v3(.28f,.58f,-.765f)))),450);
  return add(light,mul(v3(1,.9f,.66f),sun*30));
}
__device__ float4 studySurface(const float4 *surface,float x,float z,float wave,float time){
  float4 a=sample4(surface,x*256/2.8f,z*256/2.8f,256,0);
  float h=a.x*wave*2.8f/4.6f,gx=a.y*wave,gz=a.z*wave;
  // Isolated drops: the same height derivatives refract eye and photon rays.
  for(int i=0;i<4;i++){
    float cx=i==0?-.2332f:(i==1?.1501f:(i==2?.8151f:.5056f));
    float cz=i==0?.6386f:(i==1?1.3879f:(i==2?1.5416f:1.0587f));
    float age=frac((time-5+.46f+(float)i*.09f)/4)*4;
    float dx=x-cx,dz=z-cz;float rad=sqrtf(dx*dx+dz*dz+.00001f),q=rad-age*.23f;
    float env=expf(-q*q/ .006f)*expf(-age*1.8f)*smooth(0,.1f,age);
    float amp=.0015f,phase=q*130;
    float slope=amp*env*(130*cosf(phase)-2*q/.006f*sinf(phase));
    h+=amp*env*sinf(phase);gx+=slope*dx/rad;gz+=slope*dz/rad;
  }
  return make_float4(h,gx,gz,0);
}
__global__ void study_render(const float4 *surface,const float4 *caustics,float4 *hdr,int width,int height,float cameraHeight,float pitch,float fov,float depth,float wave,float caustic,float tint,float time){
  int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),y=(int)(blockIdx.y*blockDim.y+threadIdx.y);if(x>=width||y>=height)return;
  float sx=(x+.5f)/width*2-1,sy=1-(y+.5f)/height*2;
  float scale=tanf(fov*.00872664626f),aspect=(float)width/height;
  float3 o=v3(0,cameraHeight,3.3f),d=norm(v3(sx*scale,sinf(pitch)+sy*scale/aspect*cosf(pitch),-cosf(pitch)+sy*scale/aspect*sinf(pitch)));
  float3 color=v3(.10f,.12f,.12f);
  // Context remains visibly unbuilt; it is not filled with the target photograph.
  if((float)y/height>.405f&&d.y<-.001f){
    float t=-o.y/d.y;float3 p=add(o,mul(d,t));
    float4 s=studySurface(surface,p.x,p.z,wave,time);
    float3 n=norm(v3(-s.y,1,-s.z)),r=refract3(d,n,1/1.3335f);
    float travel=-depth/r.y;float3 bed=add(p,mul(r,travel));
    float4 photons=sample4(caustics,(bed.x+2)*512/4,(bed.z+2)*512/4,512,0);
    float3 ground=studyStone(bed.x,bed.z);
    ground=prod(ground,v3(.16f+caustic*fminf(7,photons.x),.16f+caustic*fminf(7,photons.y),.16f+caustic*fminf(7,photons.z)));
    float3 absorb=exp3(mul(v3(.29f,.11f,.26f),-travel*tint));
    ground=add(prod(ground,absorb),prod(v3(.13f,.19f,.085f),sub(v3(1,1,1),absorb)));
    float3 reflection=studyReflection(p,sub(d,mul(n,2*dot3(d,n))));
    color=mix3(ground,reflection,fresnel(sat(-dot3(d,n))));
  }
  hdr[y*width+x]=make_float4(color.x,color.y,color.z,1);
}
__global__ void study_present(const float4 *hdr,unsigned *image,int width,int height,int stride,float exposure){
  int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),y=(int)(blockIdx.y*blockDim.y+threadIdx.y);if(x>=width||y>=height)return;
  float4 c=hdr[y*width+x];unsigned r=(unsigned)(tone(c.x*exposure)*255+.5f),g=(unsigned)(tone(c.y*exposure)*255+.5f),b=(unsigned)(tone(c.z*exposure)*255+.5f);
  image[y*stride+x]=r|(g<<8)|(b<<16)|4278190080u;
}

__global__ void study_caustics(const float4 *surface, unsigned *photons,
                              float depth, float wave, float time) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
      z = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= 1024 || z >= 1024)
    return;
  float wx = ((float)x + .5f) * 4 / 1024-2,
        wz = ((float)z + .5f) * 4 / 1024-2;
  float4 a = studySurface(surface,wx,wz,wave,time);
  float3 n = norm(v3(-a.y, 1, -a.z)),
         sun = norm(v3(.28f,.58f,-.765f));
  for (int c = 0; c < 3; c++) {
    float ior = c == 0 ? 1.3315f : (c == 1 ? 1.3335f : 1.3365f);
    float3 d = refract3(mul(sun, -1), n, 1 / ior);
    float travel = (-depth - a.x) / d.y,
          px = (wx + d.x * travel+2) * 512 / 4,
          pz = (wz + d.z * travel+2) * 512 / 4;
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
