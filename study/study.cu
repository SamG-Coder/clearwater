// SPDX-License-Identifier: MIT
// Water, stonework, moon-gate wall and pavilion components. Reference pixels are never supplied to this renderer.
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
  float x=(p.x-.05f)/.965f,z=p.z+1.0f;
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
  float x=(p.x-.05f)/.965f,arch=sat(1-x*x/(.54f*.54f));
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
// Section 03: plaster wall with a cut-through circular gate and a tiled coping.
__device__ float studyRoofHeight(float x){float q=(x-.97f)/.61f;return .90f+.25f*expf(-q*q)+.055f*x;}
__device__ float studyGateRadius(float3 p){float x=p.x-.97f,y=p.y-.40f;return sqrtf(x*x+y*y);}
__device__ float studyWallDistance(float3 p){
  float roof=studyRoofHeight(p.x),rad=studyGateRadius(p),z=p.z+2;
  float body=fmaxf(fabsf(p.x)-3.0f,fmaxf(fabsf(z)-.085f,fmaxf(-.10f-p.y,p.y-roof)));
  body=fmaxf(body,.400f-rad);
  float ring=fmaxf(fabsf(rad-.434f)-.034f,fabsf(z)-.098f);
  ring=fmaxf(ring,-.10f-p.y);
  float tileRib=.009f*cosf(p.x*96.6644f);
  float tiles=fmaxf(fabsf(p.x)-3.03f,fmaxf(fabsf(z)-.18f,fabsf(p.y-(roof+.025f-fabsf(z)*.27f+tileRib))-.032f));
  float ridge=fmaxf(fabsf(p.x)-3.03f,fmaxf(fabsf(z)-.023f,fabsf(p.y-roof-.068f)-.022f));
  return fminf(fminf(body,ring),fminf(tiles,ridge));
}
__device__ float studyWallTrace(float3 o,float3 d,float maxT){
  if(fabsf(d.z)<.00001f)return -1;
  float a=(-2.19f-o.z)/d.z,b=(-1.81f-o.z)/d.z,t=fmaxf(0,fminf(a,b)),end=fminf(maxT,fmaxf(a,b));
  for(int i=0;i<80;i++){
    if(t>end)return -1;float dist=studyWallDistance(add(o,mul(d,t)));
    if(dist<.0005f)return t;t+=fmaxf(.0003f,dist*.6f);
  }
  return -1;
}
__device__ float3 studyWallShade(float3 p){
  float e=.001f;float3 n=norm(v3(studyWallDistance(add(p,v3(e,0,0)))-studyWallDistance(sub(p,v3(e,0,0))),studyWallDistance(add(p,v3(0,e,0)))-studyWallDistance(sub(p,v3(0,e,0))),studyWallDistance(add(p,v3(0,0,e)))-studyWallDistance(sub(p,v3(0,0,e)))));
  float grain=fbm(p.x*65,p.y*65),patch=fbm(p.x*9+2,p.y*13),rad=studyGateRadius(p),roof=studyRoofHeight(p.x);
  float3 material=v3(.56f,.55f,.50f);
  float stain=smooth(.46f,.72f,patch)*(1-smooth(-.05f,.48f,p.y));
  float runoff=smooth(.53f,.76f,fbm(p.x*48,p.y*3))*expf(-fmaxf(0,roof-p.y)*5);
  float chipped=smooth(.55f,.72f,fbm(p.x*30,p.y*34))*(1-smooth(.02f,.30f,p.y));
  material=mix3(material,v3(.28f,.29f,.22f),sat(stain*.65f+runoff*.30f+chipped*.45f));
  float cracks=1-smooth(.007f,.018f,fabsf(noise(p.x*14,p.y*27)-.5f));
  material=mul(material,.87f+grain*.22f-cracks*.045f);
  if(rad<.472f){
    float angle=atan2f(p.y-.40f,p.x-.97f),segment=frac(angle*7.639437f);
    float joint=smooth(.012f,.07f,fminf(segment,1-segment));
    float id=hash(floorf(angle*7.639437f),1);
    material=mul(mix3(v3(.31f,.32f,.29f),v3(.57f,.55f,.48f),id),(.68f+.5f*grain)*(.4f+.6f*joint));
  }
  if(p.y>roof-.035f){
    float u=frac(p.x/.065f),v=frac((p.z+2)/.07f),id=hash(floorf(p.x/.065f),floorf((p.z+2)/.07f));
    float joint=smooth(.015f,.10f,fminf(u,1-u))*smooth(.015f,.12f,fminf(v,1-v));
    material=mul(mix3(v3(.06f,.075f,.075f),v3(.19f,.22f,.21f),id),(.7f+.3f*grain)*(.35f+.65f*joint));
  }
  // Ambient courtyard bounce keeps plaster readable before foliage is built.
  float light=.72f+.45f*sat(dot3(n,norm(v3(-.45f,.7f,.55f))));
  float reveal=rad<.405f?.58f:1;
  return mul(material,light*reveal);
}
// Section 04: cropped foreground pavilion. All parts are world-space geometry.
// Map.y identifies plaster (1), foundation stone (2), timber (3), or roof tile (4).
__device__ float studyCapsule(float3 p,float3 a,float3 b,float radius){
  float3 pa=sub(p,a),ba=sub(b,a);float h=sat(dot3(pa,ba)/dot3(ba,ba));
  float3 q=sub(pa,mul(ba,h));return sqrtf(dot3(q,q))-radius;
}
__device__ float studyPavilionRoof(float x,float z){
  float corner=sat((x+1.45f)/.4f);
  return 1.215f-.38f*(x+1.75f)+.16f*(z+1.1f)+.06f*corner*corner;
}
__device__ float2 studyPavilionMap(float3 p){
  float best=studyBox(sub(p,v3(-2.58f,1.05f,-.49f)),v3(.77f,.635f,.055f)),material=1;
  // Three nosed steps, laid foundation courses, and the raised porch.
  for(int i=0;i<3;i++){
    float front=-.67f-(float)i*.15f,top=.17f+(float)i*.04f;
    float right=-1.505f-(float)i*.057f;
    float d=studyBox(sub(p,v3((right-3.3f)*.5f,top*.5f,(front-1.65f)*.5f)),v3((right+3.3f)*.5f,top*.5f,(front+1.65f)*.5f))-.002f;
    d=fmaxf(d,(p.x+.3795f*(3.3f-p.z))*.935f);
    if(d<best){best=d;material=2;}
  }
  float plinth=studyBox(sub(p,v3(-2.58f,.325f,-.50f)),v3(.78f,.095f,.065f));
  if(plinth<best){best=plinth;material=2;}
  // The two columns share the porch frontage in the reference.
  for(int i=0;i<2;i++){
    float x=i==0?-2.08f:-1.745f,z=-1.10f,r=i==0?.026f:.034f;
    float dx=p.x-x,dz=p.z-z;
    float postTop=studyPavilionRoof(x,z)-.058f;
    float shaft=fmaxf(sqrtf(dx*dx+dz*dz)-r,fmaxf(.25f-p.y,p.y-postTop));
    if(shaft<best){best=shaft;material=3;}
    float foot=studyBox(sub(p,v3(x,.248f,z)),v3(r+.014f,.016f,r+.014f))-.003f;
    if(foot<best){best=foot;material=2;}
    for(int j=0;j<3;j++){
      float capital=studyBox(sub(p,v3(x,postTop-.085f+(float)j*.035f,z)),v3(r+.02f+(float)j*.025f,.017f,r+.019f+(float)j*.014f));
      if(capital<best){best=capital;material=3;}
    }
  }
  float beam=fmaxf(fabsf(p.x+1.545f)-.535f,fmaxf(fabsf(p.z+.69f)-.036f,fabsf(p.y-studyPavilionRoof(p.x,p.z)+.092f)-.037f));
  float sideBeam=fmaxf(fabsf(p.x+1.745f)-.045f,fmaxf(fabsf(p.z+1.10f)-.47f,fabsf(p.y-studyPavilionRoof(p.x,p.z)+.098f)-.034f));
  float lower=studyBox(sub(p,v3(-1.91f,.335f,-1.22f)),v3(.19f,.013f,.019f));
  float brace=studyCapsule(p,v3(-1.745f,1.01f,-1.1f),v3(-1.96f,1.22f,-1.1f),.017f);
  float ceiling=studyBox(sub(p,v3(-2.4f,1.065f,-1.2f)),v3(.9f,.014f,.49f));
  float lintel=studyBox(sub(p,v3(-2.4f,1.13f,-.80f)),v3(.9f,.055f,.025f));
  float wood=fminf(fminf(beam,sideBeam),fminf(lower,brace));
  wood=fminf(wood,fminf(ceiling,lintel));
  wood=fmaxf(wood,(p.x-(.25f*p.z-.9225f))*.970f);
  if(wood<best){best=wood;material=3;}
  // Repeated exposed rafters under the curved roof; finite bounds prevent repeats leaking out.
  float rx=frac((p.x+3.3f)/.094f)*.094f-.047f,h=studyPavilionRoof(p.x,p.z);
  float rafter=fmaxf(fabsf(p.x+2.16f)-1.14f,fmaxf(fabsf(rx)-.018f,fmaxf(fabsf(p.y-(h-.056f))-.022f,fabsf(p.z+1.10f)-.59f)));
  rafter=fmaxf(rafter,(p.x-(.25f*p.z-.9225f))*.970f);
  if(rafter<best){best=rafter;material=3;}
  // Hollow half-round cover tiles over the lower pan course, with overlapping laps.
  float tx=frac((p.x+3.3f)/.094f)*.094f-.047f,ty=p.y-h-.003f*frac((p.z+1.69f)/.11f);
  float tube=fmaxf(fabsf(sqrtf(tx*tx+ty*ty)-.031f)-.006f,-ty);
  float pan=fabsf(ty+.016f)-.010f;
  float roof=fmaxf(fabsf(p.x+2.16f)-1.17f,fmaxf(fabsf(p.z+1.1f)-.59f,fminf(tube,pan)));
  roof=fmaxf(roof,(p.x-(.25f*p.z-.9225f))*.970f);
  if(roof<best){best=roof;material=4;}
  return make_float2(best,material);
}
__device__ float studyPavilionTrace(float3 o,float3 d,float maxT){
  float begin=0,end=maxT;
  for(int axis=0;axis<3;axis++){
    float a=axis==0?o.x:(axis==1?o.y:o.z),v=axis==0?d.x:(axis==1?d.y:d.z);
    float lo=axis==0?-3.365f:(axis==1?-.004f:-1.71f),hi=axis==0?-.98f:(axis==1?2.0f:-.43f);
    if(fabsf(v)<.000001f){if(a<lo||a>hi)return -1;}
    else {float nearT=(lo-a)/v,farT=(hi-a)/v;begin=fmaxf(begin,fminf(nearT,farT));end=fminf(end,fmaxf(nearT,farT));}
  }
  for(int i=0;i<112;i++){
    if(begin>end)return -1;
    float2 sample=studyPavilionMap(add(o,mul(d,begin)));
    if(sample.x<.00045f)return begin;
    begin+=fmaxf(.00025f,sample.x*.62f);
  }
  return -1;
}
__device__ float3 studyPavilionNormal(float3 p){
  float e=.0007f;
  float2 xp=studyPavilionMap(add(p,v3(e,0,0))),xm=studyPavilionMap(sub(p,v3(e,0,0)));
  float2 yp=studyPavilionMap(add(p,v3(0,e,0))),ym=studyPavilionMap(sub(p,v3(0,e,0)));
  float2 zp=studyPavilionMap(add(p,v3(0,0,e))),zm=studyPavilionMap(sub(p,v3(0,0,e)));
  return norm(v3(xp.x-xm.x,yp.x-ym.x,zp.x-zm.x));
}
__device__ float3 studyPavilionShade(float3 p,float3 d){
  float2 hit=studyPavilionMap(p);float3 n=studyPavilionNormal(p);
  float grain=fbm(p.x*80+p.z*13,p.y*85+p.z*40);
  float3 color=v3(.59f,.56f,.48f);
  if(hit.y<1.5f){
    float damp=(1-smooth(.18f,.42f,p.y))*smooth(.4f,.73f,fbm(p.x*17,p.y*24));
    float runoff=smooth(.56f,.76f,fbm(p.x*58,p.y*5));
    color=mul(color,.91f+grain*.15f-damp*.40f-runoff*.09f);
  }else if(hit.y<2.5f){
    float u=p.x/.20f+floorf(p.y/.080f)*.5f,v=p.y/.080f;
    if(n.y>.6f){u=p.x/.26f;v=p.z/.17f;}
    float edge=fminf(fminf(frac(u),1-frac(u)),fminf(frac(v),1-frac(v)));
    float joint=smooth(.012f,.065f,edge+(grain-.5f)*.035f);
    color=mix3(v3(.13f,.15f,.12f),mix3(v3(.32f,.33f,.28f),v3(.58f,.55f,.45f),hash(floorf(u),floorf(v))),joint);
    color=mul(color,.72f+grain*.52f);
  }else if(hit.y<3.5f){
    float streak=noise(p.x*600+p.z*170,p.y*8),fiber=powf(sat(.5f+.5f*sinf(p.x*1450+p.z*950+noise(p.x*40,p.y*14)*12)),12);
    color=mix3(v3(.050f,.025f,.011f),v3(.21f,.105f,.038f),streak*.35f+grain*.25f+.2f);
    color=mul(color,(1-fiber*.22f)*(1-.53f*smooth(.8f,1.05f,p.y)));
  }else{
    float u=frac((p.x+3.3f)/.094f),v=frac((p.z+1.69f)/.11f);
    float seams=smooth(.01f,.075f,fminf(u,1-u))*smooth(.01f,.10f,fminf(v,1-v));
    float id=hash(floorf((p.x+3.3f)/.094f),floorf((p.z+1.69f)/.11f));
    color=mul(mix3(v3(.065f,.071f,.065f),v3(.24f,.235f,.20f),id),(.55f+.45f*seams)*(.8f+.3f*grain));
  }
  float3 sun=norm(v3(.28f,.58f,-.765f));
  float shadow=studyPavilionTrace(add(p,mul(n,.003f)),sun,5)>=0?.15f:1;
  float ambient=.30f+.43f*sat(dot3(n,norm(v3(-.45f,.7f,.55f))));
  if(n.y<-.3f)ambient*=.45f;
  float light=ambient+1.05f*sat(dot3(n,sun))*shadow;
  float glint=hit.y>2.5f&&hit.y<3.5f?powf(sat(dot3(n,norm(sub(sun,d)))),45)*.045f:0;
  return add(mul(color,light),v3(glint,glint*.75f,glint*.4f));
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
__global__ void study_render(const float4 *surface,const float4 *caustics,float4 *hdr,int width,int height,float cameraHeight,float pitch,float fov,float depth,float wave,float caustic,float tint,float time,int stonework,int wall,int pavilion){
  int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),y=(int)(blockIdx.y*blockDim.y+threadIdx.y);if(x>=width||y>=height)return;
  float sx=(x+.5f)/width*2-1,sy=1-(y+.5f)/height*2;
  float scale=tanf(fov*.00872664626f),aspect=(float)width/height;
  // Off-axis architectural projection: vertical columns remain vertical.
  // Pitch controls lens shift; center-ray direction and local angular scale are preserved.
  float cp=cosf(pitch);
  float3 o=v3(0,cameraHeight,3.3f),d=norm(v3(sx*scale/cp,tanf(pitch)+sy*scale/(aspect*cp*cp),-1));
  float3 color=v3(.10f,.12f,.12f);
  // Context remains visibly unbuilt; it is not filled with the target photograph.
  float waterT=d.y<-.001f?-o.y/d.y:1000;
  float stoneT=stonework?studyMasonryTrace(o,d,waterT): -1;
  float wallT=wall?studyWallTrace(o,d,fminf(waterT,stoneT>=0?stoneT:1000)):-1;
  float pavilionT=pavilion?studyPavilionTrace(o,d,fminf(waterT,fminf(stoneT>=0?stoneT:1000,wallT>=0?wallT:1000))):-1;
  if(pavilionT>=0){color=studyPavilionShade(add(o,mul(d,pavilionT)),d);}
  else if(wallT>=0){color=studyWallShade(add(o,mul(d,wallT)));}
  else if(stoneT>=0){color=studyMasonryShade(add(o,mul(d,stoneT)),d);}
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
    if(pavilion&&studyPavilionTrace(add(bed,v3(0,.003f,0)),norm(v3(.28f,.58f,-.765f)),6)>=0)ground=mul(ground,.55f);
    float3 rd=sub(d,mul(n,2*dot3(d,n)));
    float3 reflection=studyReflection(p,rd);
    float wallReflection=wall?studyWallTrace(add(p,v3(0,.003f,0)),rd,12):-1;
    if(wallReflection>=0)reflection=studyWallShade(add(add(p,v3(0,.003f,0)),mul(rd,wallReflection)));
    float reflectedT=stonework?studyMasonryTrace(add(p,v3(0,.003f,0)),rd,wallReflection>=0?wallReflection:8):-1;
    if(reflectedT>=0)reflection=studyMasonryShade(add(add(p,v3(0,.003f,0)),mul(rd,reflectedT)),rd);
    float pavilionReflection=pavilion?studyPavilionTrace(add(p,v3(0,.003f,0)),rd,fminf(wallReflection>=0?wallReflection:12,reflectedT>=0?reflectedT:12)):-1;
    if(pavilionReflection>=0)reflection=studyPavilionShade(add(add(p,v3(0,.003f,0)),mul(rd,pavilionReflection)),rd);
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
