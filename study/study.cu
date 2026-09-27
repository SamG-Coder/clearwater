// SPDX-License-Identifier: MIT
// Water and stonework components. Reference pixels are never supplied to this renderer.
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
// Section 02: actual world-space stonework, shared by view, reflection and shadows.
__device__ float studyBox(float3 p,float3 b){
  float3 q=v3(fabsf(p.x)-b.x,fabsf(p.y)-b.y,fabsf(p.z)-b.z);
  float3 outside=v3(fmaxf(q.x,0),fmaxf(q.y,0),fmaxf(q.z,0));
  return sqrtf(dot3(outside,outside))+fminf(fmaxf(q.x,fmaxf(q.y,q.z)),0);
}
__device__ float studyMasonryDistance(float3 p){
  float x=(p.x-.05f)/.924f,z=p.z+1.0f;
  float arch=sat(1-x*x/(.54f*.54f));
  float top=.115f+.12f*arch,bottom=-.045f+.15f*arch;
  // Two curved parapets and the lower arched walking slab.
  float rail=fmaxf(fabsf(x)-.54f,fmaxf(fabsf(fabsf(z)-.13f)-.038f,fmaxf(p.y-top,bottom-p.y)));
  float deck=fmaxf(fabsf(x)-.54f,fmaxf(fabsf(z)-.15f,fmaxf(p.y-(bottom+.045f),bottom-p.y)));
  float post=studyBox(v3(fabsf(x)-.565f,p.y-.08f,fabsf(z)-.13f),v3(.051f,.155f,.051f));
  float cap=studyBox(v3(fabsf(x)-.565f,p.y-.207f,fabsf(z)-.13f),v3(.057f,.032f,.057f));
  float bridge=fminf(fminf(rail,deck),fminf(post,cap));
  // Tapered pond margins meet the bridge landings; the channel stays open.
  float right=1.0f+.7f*(p.z+1),left=1.2f+.85f*(p.z+1);
  float banks=fmaxf(fminf(right-p.x,p.x+left),fmaxf(fabsf(p.y+.04f)-.13f,fabsf(p.z-.25f)-1.55f));
  float landings=fmaxf(.61f-fabsf(x),fmaxf(fabsf(p.y+.015f)-.105f,fabsf(z+.20f)-.26f));
  return fminf(bridge,fminf(banks,landings));
}
__device__ float studyMasonryTrace(float3 o,float3 d,float maxT){
  // Restrict traversal to the stonework's vertical slab.
  float t=0,end=maxT;
  if(fabsf(d.y)>.00001f){float a=(-.19f-o.y)/d.y,b=(.30f-o.y)/d.y;t=fmaxf(0,fminf(a,b));end=fminf(end,fmaxf(a,b));}
  else if(o.y<-.19f||o.y>.30f)return -1;
  for(int i=0;i<100;i++){
    if(t>end)return -1;
    float3 p=add(o,mul(d,t));float dist=studyMasonryDistance(p);
    if(dist<.0007f)return t;
    t+=fmaxf(.0004f,dist*.65f);
  }
  return -1;
}
__device__ float studyMasonryShadow(float3 p){
  float3 sun=norm(v3(.28f,.58f,-.765f));float t=.008f,visibility=1;
  for(int i=0;i<48;i++){
    float3 q=add(p,mul(sun,t));if(q.y>.30f)break;
    float h=studyMasonryDistance(q);if(h<.0007f)return .36f;
    visibility=fminf(visibility,10*h/fmaxf(.03f,t));t+=fminf(.18f,fmaxf(.008f,h*.7f));
  }
  return .36f+.64f*sat(visibility);
}
__device__ float3 studyMasonryNormal(float3 p){
  float e=.001f;
  return norm(v3(studyMasonryDistance(add(p,v3(e,0,0)))-studyMasonryDistance(sub(p,v3(e,0,0))),studyMasonryDistance(add(p,v3(0,e,0)))-studyMasonryDistance(sub(p,v3(0,e,0))),studyMasonryDistance(add(p,v3(0,0,e)))-studyMasonryDistance(sub(p,v3(0,0,e)))));
}
__device__ float3 studyMasonryShade(float3 p,float3 d){
  float3 n=studyMasonryNormal(p),sun=norm(v3(.28f,.58f,-.765f));
  float x=(p.x-.05f)/.924f,arch=sat(1-x*x/(.54f*.54f));
  float u=p.x/.23f,v=(p.y-.12f*arch)/.072f;
  if(fabsf(n.y)>.6f){u=p.x/.21f;v=p.z/.17f;}
  else if(fabsf(n.x)>.6f)u=p.z/.14f;
  u+=floorf(v)*.48f;
  float id=hash(floorf(u),floorf(v)),grit=fbm(p.x*110+p.y*35,p.z*110-p.y*80);
  float edge=fminf(fminf(frac(u),1-frac(u)),fminf(frac(v),1-frac(v)));
  float joint=smooth(.015f,.058f,edge+(grit-.5f)*.035f);
  float3 stone=mix3(v3(.22f,.24f,.21f),v3(.52f,.50f,.42f),id*.65f+grit*.35f);
  stone=mul(stone,.7f+grit*.6f);stone=mix3(v3(.055f,.063f,.045f),stone,joint);
  float moss=smooth(.43f,.66f,fbm(p.x*18+p.y*4,p.z*22+p.y*13))*(1-smooth(.05f,.3f,p.y));
  stone=mix3(stone,v3(.055f,.078f,.018f),moss*.85f);
  float wet=1-smooth(-.01f,.07f,p.y);stone=mul(stone,1-.37f*wet);
  float shadow=studyMasonryTrace(add(p,mul(n,.004f)),sun,4)>0?.18f:1;
  float light=.24f+1.3f*sat(dot3(n,sun))*shadow;
  float glint=powf(sat(dot3(norm(sub(sun,d)),n)),70)*wet*.22f;
  return add(mul(stone,light),v3(glint,glint*.94f,glint*.8f));
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
__global__ void study_render(const float4 *surface,const float4 *caustics,float4 *hdr,int width,int height,float cameraHeight,float pitch,float fov,float depth,float wave,float caustic,float tint,float time,int stonework){
  int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),y=(int)(blockIdx.y*blockDim.y+threadIdx.y);if(x>=width||y>=height)return;
  float sx=(x+.5f)/width*2-1,sy=1-(y+.5f)/height*2;
  float scale=tanf(fov*.00872664626f),aspect=(float)width/height;
  float3 o=v3(0,cameraHeight,3.3f),d=norm(v3(sx*scale,sinf(pitch)+sy*scale/aspect*cosf(pitch),-cosf(pitch)+sy*scale/aspect*sinf(pitch)));
  float3 color=v3(.10f,.12f,.12f);
  // Context remains visibly unbuilt; it is not filled with the target photograph.
  float waterT=d.y<-.001f?-o.y/d.y:1000;
  float stoneT=stonework?studyMasonryTrace(o,d,waterT): -1;
  if(stoneT>=0){color=studyMasonryShade(add(o,mul(d,stoneT)),d);}
  else if((float)y/height>.375f&&d.y<-.001f){
    float t=-o.y/d.y;float3 p=add(o,mul(d,t));
    float4 s=studySurface(surface,p.x,p.z,wave,time);
    float3 n=norm(v3(-s.y,1,-s.z)),r=refract3(d,n,1/1.3335f);
    float travel=-depth/r.y;float3 bed=add(p,mul(r,travel));
    float4 photons=sample4(caustics,(bed.x+2)*512/4,(bed.z+2)*512/4,512,0);
    float3 ground=studyStone(bed.x,bed.z);
    ground=prod(ground,v3(.16f+caustic*fminf(7,photons.x),.16f+caustic*fminf(7,photons.y),.16f+caustic*fminf(7,photons.z)));
    float3 absorb=exp3(mul(v3(.29f,.11f,.26f),-travel*tint));
    ground=add(prod(ground,absorb),prod(v3(.13f,.19f,.085f),sub(v3(1,1,1),absorb)));
    if(stonework&&bed.z<.7f){ground=mul(ground,studyMasonryShadow(add(bed,v3(0,.003f,0))));}
    float3 rd=sub(d,mul(n,2*dot3(d,n)));
    float3 reflection=studyReflection(p,rd);
    float reflectedT=stonework?studyMasonryTrace(add(p,v3(0,.003f,0)),rd,8):-1;
    if(reflectedT>=0)reflection=studyMasonryShade(add(add(p,v3(0,.003f,0)),mul(rd,reflectedT)),rd);
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
