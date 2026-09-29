// SPDX-License-Identifier: MIT
// Included after clearwater.cu. Metres, seconds, kilograms; no CPU body updates.
// Five float4 records/body: position+mass, velocity+wet fraction, quaternion,
// world angular velocity, previous sampled water height+diagnostics.
__device__ float3 duckCross(float3 a,float3 b) {
  return v3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
}
__device__ float3 duckRotate(float4 q,float3 p) {
  float3 v=v3(q.x,q.y,q.z),t=mul(duckCross(v,p),2);
  return add(p,add(mul(t,q.w),duckCross(v,t)));
}
__device__ float3 duckLocal(float4 q,float3 p) {
  return duckRotate(make_float4(-q.x,-q.y,-q.z,q.w),p);
}
__device__ unsigned duckCell(int x,int z) {
  return hashU((unsigned)x*73856093u^(unsigned)z*19349663u)&65535u;
}
__device__ float4 duckWater(const float4 *surface,const float4 *sea,float x,float z,int tornadoOn,float tx,float tz) {
  float4 a=sample4(surface,x*256/4.6f,z*256/4.6f,256,0);
  float4 b=sample4(surface,x*256/37,z*256/37,256,65536);
  float4 c=sample4(surface,x*256/293,z*256/293,256,131072);
  float4 d=make_float4(0,0,0,0);
  if(tornadoOn!=0)d=tornadoWaterAt(sea,x,z,tx,tz);
  return make_float4(a.x+b.x+c.x+d.x,a.y+b.y+c.y+d.y,a.z+b.z+c.z+d.z,0);
}
__global__ void ducks_init(float4 *bodies,const float4 *surface,const float4 *sea,float originX,float originZ,int tornadoOn,float tx,float tz) {
  int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),z=(int)(blockIdx.y*blockDim.y+threadIdx.y),i=z*128+x;
  if(x>=128||i>=10000)return;
  float px=originX+((float)(i%100)-49.5f)*1.05f+(random((unsigned)i*7u)-.5f)*.35f;
  float pz=originZ+((float)(i/100)-49.5f)*1.05f+(random((unsigned)i*7u+1u)-.5f)*.35f;
  float h=duckWater(surface,sea,px,pz,tornadoOn,tx,tz).x;
  float angle=random((unsigned)i+199u)*6.2831853f;
  bodies[i*5]=make_float4(px,h+.074f,pz,.14f+.10f*random((unsigned)i*13u+73u));
  bodies[i*5+1]=make_float4(0,0,0,.04f);
  bodies[i*5+2]=make_float4(0,sinf(angle*.5f),0,cosf(angle*.5f));
  bodies[i*5+3]=make_float4(0,0,0,0);
  bodies[i*5+4]=make_float4(h,0,0,0);
}
__global__ void ducks_hash_clear(unsigned *heads) {
  int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),y=(int)(blockIdx.y*blockDim.y+threadIdx.y);
  if(x<256&&y<256)heads[y*256+x]=0;
}
__global__ void ducks_hash(const float4 *bodies,unsigned *heads,unsigned *links) {
  int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),y=(int)(blockIdx.y*blockDim.y+threadIdx.y),i=y*128+x;
  if(x>=128||i>=10000)return;
  float4 p=bodies[i*5];unsigned cell=duckCell((int)floorf(p.x/.6f),(int)floorf(p.z/.6f));
  links[i]=atomicExch(&heads[cell],(unsigned)i+1u);
}
__global__ void ducks_step(const float4 *previous,float4 *next,const unsigned *heads,const unsigned *links,const float4 *surface,const float4 *sea,const float4 *air,const float4 *weather,float dt,int tornadoOn,float tx,float tz,float time) {
  int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),y=(int)(blockIdx.y*blockDim.y+threadIdx.y),i=y*128+x;
  if(x>=128||i>=10000)return;
  float4 p=previous[i*5],vel=previous[i*5+1],q=previous[i*5+2],ang=previous[i*5+3],history=previous[i*5+4];
  float3 v=v3(vel.x,vel.y,vel.z),omega=v3(ang.x,ang.y,ang.z),force=v3(0,-9.81f*p.w,0),torque=v3(0,0,0);
  float4 wave=duckWater(surface,sea,p.x,p.z,tornadoOn,tx,tz);
  float waterV=fminf(3,fmaxf(-3,(wave.x-history.x)/dt));
  float wet=0;
  // Four submerged ellipsoid quadrants approximate displaced volume. Their
  // force arms generate pitch/roll torque rather than snapping to the normal.
  for(int j=0;j<4;j++) {
    float3 r=duckRotate(q,v3((j%2==0?-.095f:.095f),0,(j<2?-.13f:.13f)));
    float4 w=duckWater(surface,sea,p.x+r.x,p.z+r.z,tornadoOn,tx,tz);
    float s=fminf(1,fmaxf(-1,(w.x-p.y-r.y)/.10f));
    float fraction=.5f+.75f*s-.25f*s*s*s;
    wet+=fraction*.25f;
    float3 pointV=add(v,duckCross(omega,r));
    float3 waterVelocity=v3(-w.y*.24f,waterV,-w.z*.24f);
    if(tornadoOn!=0&&fabsf(p.x-tx)<135&&fabsf(p.z-tz)<135) {
      float4 current=sample4(sea,(p.x-tx)/3+47.5f,(p.z-tz)/3+47.5f,96,0);
      waterVelocity.x+=current.y;waterVelocity.z+=current.z;
    }
    float drag=smooth(0,.08f,fraction)*.65f;
    float3 f=mul(sub(waterVelocity,pointV),drag);
    f.y+=1000*9.81f*.00352f*fraction;
    force=add(force,f);torque=add(torque,duckCross(r,f));
  }
  float3 wind=v3(weather[1].x*weather[0].x,0,weather[1].y*weather[0].x);
  if(tornadoOn!=0&&fabsf(p.x-tx)<138&&fabsf(p.z-tz)<138&&p.y<570) {
    float4 a=tornadoSample(air,v3((p.x-tx)/6+23.5f,p.y/6-.5f,(p.z-tz)/6+23.5f));
    wind=v3(a.x,a.y,a.z);
  }
  float3 relative=sub(wind,v);
  float speed=sqrtf(dot3(relative,relative));
  // Distributed drag acts at four exposed parts, using their velocity from
  // translation AND rotation. Local air samples generate tumbling torque and
  // aerodynamic damping instead of prescribing a spiral or an angular speed.
  float3 aerodynamic=v3(0,0,0);
  float gustStrength=tornadoOn!=0?3.5f*sat(sqrtf(dot3(wind,wind))/25)*expf(-(square(p.x-tx)+square(p.z-tz))/18000):0;
  for(int j=0;j<4;j++) {
    float3 localArm=j==0?v3(-.13f,.02f,0):(j==1?v3(.13f,.02f,0):(j==2?v3(0,.16f,-.12f):v3(0,.03f,.17f)));
    float3 arm=duckRotate(q,localArm),pointWind=wind;
    if(tornadoOn!=0&&fabsf(p.x-tx)<137&&fabsf(p.z-tz)<137&&p.y<569) {
      float4 a=tornadoSample(air,v3((p.x+arm.x-tx)/6+23.5f,(p.y+arm.y)/6-.5f,(p.z+arm.z-tz)/6+23.5f));
      pointWind=v3(a.x,a.y,a.z);
    }
    // Unresolved metre-scale eddies sampled at each exposed part. Nearby ducks
    // share the same field, but different positions/orientations feel shear.
    float3 gustPoint=mul(v3(p.x+arm.x-tx,p.y+arm.y,p.z+arm.z-tz),2.3f);
    pointWind=add(pointWind,mul(tornadoGust(gustPoint,time*2.1f),gustStrength));
    float3 relativePoint=sub(pointWind,add(v,duckCross(omega,arm)));
    float pointSpeed=sqrtf(dot3(relativePoint,relativePoint));
    float3 face=duckRotate(q,j<2?v3(1,0,0):v3(0,0,1));
    float exposure=.55f+.9f*fabsf(dot3(face,relativePoint))/fmaxf(.01f,pointSpeed);
    float3 drag=mul(relativePoint,.5f*1.225f*1.05f*.016f*exposure*pointSpeed*(1-wet));
    aerodynamic=add(aerodynamic,drag);torque=add(torque,duckCross(arm,drag));
  }
  // Coarse bluff-body lift coefficient for the asymmetric hollow toy. Lift is
  // perpendicular to relative airflow and follows body orientation; the water
  // support disappears naturally as displaced volume falls to zero.
  float3 up=duckRotate(q,v3(0,1,0)),flow=mul(relative,1/fmaxf(.001f,speed));
  float3 liftAxis=sub(up,mul(flow,dot3(up,flow)));
  float liftLength=sqrtf(dot3(liftAxis,liftAxis));
  float liftCoefficient=.55f*sat(up.y*.6f+.4f);
  float3 lift=mul(liftAxis,.5f*1.225f*.055f*speed*speed*liftCoefficient*(1-wet)/fmaxf(.05f,liftLength));
  // The 6 m air grid cannot resolve flow separation around a half-metre toy
  // at the air/water interface. A bounded surface-pressure lift closure bridges
  // that layer; resolved air drag takes over aloft. It is not a position force.
  float altitude=fmaxf(0,p.y-wave.x),radial2=square(p.x-tx)+square(p.z-tz);
  float surfaceLift=tornadoOn!=0?.5f*1.225f*.065f*speed*speed*1.15f*expf(-altitude/8)*expf(-radial2/4900):0;
  force.y+=surfaceLift*(.3f+.7f*fabsf(up.y));
  force=add(force,add(aerodynamic,lift));
  torque=add(torque,duckCross(duckRotate(q,v3(0,.015f,-.012f)),lift));
  int gx=(int)floorf(p.x/.6f),gz=(int)floorf(p.z/.6f);float contacts=0;
  for(int dz=-1;dz<=1;dz++)for(int dx=-1;dx<=1;dx++) {
    unsigned entry=heads[duckCell(gx+dx,gz+dz)];
    // Chained buckets have no occupancy cap and cannot silently drop bodies.
    for(int visit=0;visit<10000&&entry!=0u;visit++) {
      int other=(int)entry-1;entry=links[other];if(other==i)continue;
      float4 b=previous[other*5],bv=previous[other*5+1];
      float3 delta=v3(p.x-b.x,p.y-b.y,p.z-b.z);float d2=dot3(delta,delta);
      if(d2<.1936f) {
        float distance=sqrtf(fmaxf(d2,.000001f));
        float3 n=mul(delta,1/distance);
        if(d2<.000001f)n=v3(i<other?-1:1,0,0);
        float3 dv=sub(v,v3(bv.x,bv.y,bv.z));float vn=dot3(dv,n);
        float normalForce=fmaxf(0,180*(.44f-distance)-5*vn);
        float3 friction=mul(sub(dv,mul(n,vn)),-.45f);
        float frictionLength=sqrtf(dot3(friction,friction));
        friction=mul(friction,fminf(1,.25f*normalForce/fmaxf(.0001f,frictionLength)));
        force=add(force,add(mul(n,normalForce),friction));
        torque=add(torque,duckCross(mul(n,-.22f),friction));contacts+=1;
      }
    }
  }
  v=add(v,mul(force,dt/p.w));
  // Semi-implicit integration; body principal inertia transformed to world.
  float3 localTorque=duckLocal(q,torque),localOmega=duckLocal(q,omega);
  localOmega.x=(localOmega.x+localTorque.x*dt/(.012f*p.w/.55f))/(1+dt*(.35f+wet*8));
  localOmega.y=(localOmega.y+localTorque.y*dt/(.018f*p.w/.55f))/(1+dt*(.25f+wet*5));
  localOmega.z=(localOmega.z+localTorque.z*dt/(.008f*p.w/.55f))/(1+dt*(.35f+wet*8));
  omega=duckRotate(q,localOmega);
  float3 qv=v3(q.x,q.y,q.z),dq=add(mul(omega,q.w),duckCross(omega,qv));
  q=make_float4(q.x+dq.x*dt*.5f,q.y+dq.y*dt*.5f,q.z+dq.z*dt*.5f,q.w-dot3(omega,qv)*dt*.5f);
  float inv=rsqrtf(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);q=make_float4(q.x*inv,q.y*inv,q.z*inv,q.w*inv);
  next[i*5]=make_float4(p.x+v.x*dt,p.y+v.y*dt,p.z+v.z*dt,p.w);
  next[i*5+1]=make_float4(v.x,v.y,v.z,wet);
  next[i*5+2]=q;next[i*5+3]=make_float4(omega.x,omega.y,omega.z,0);
  next[i*5+4]=make_float4(wave.x,contacts,speed,0);
}
// Moving submerged bodies inject a signed dipole into the existing near-camera
// wave equation. Gather by hash avoids float atomics and conserves mean height.
__global__ void ducks_wakes(const float4 *bodies,const unsigned *heads,const unsigned *links,float4 *ripples,float centerX,float centerZ,float dt) {
  int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),z=(int)(blockIdx.y*blockDim.y+threadIdx.y);
  if(x>=256||z>=256)return;
  float wx=centerX+((float)x-128)*.0625f,wz=centerZ+((float)z-128)*.0625f;
  int gx=(int)floorf(wx/.6f),gz=(int)floorf(wz/.6f);float impulse=0;
  for(int dz=-1;dz<=1;dz++)for(int dx=-1;dx<=1;dx++) {
    unsigned entry=heads[duckCell(gx+dx,gz+dz)];
    for(int visit=0;visit<10000&&entry!=0u;visit++) {
      int i=(int)entry-1;entry=links[i];float4 p=bodies[i*5],v=bodies[i*5+1];
      float rx=wx-p.x,rz=wz-p.z,r2=rx*rx+rz*rz;
      if(r2<.16f&&v.w>.003f)impulse+=(rx*v.x+rz*v.z)*expf(-r2/.025f)*smooth(.003f,.035f,v.w);
    }
  }
  int i=z*256+x;float4 a=ripples[i];
  a.y+=fminf(.0005f,fmaxf(-.0005f,impulse*dt*.35f));ripples[i]=a;
}
// Solid analytical geometry: body, head, beak, tail, wings and two eyes.
__device__ float3 duckPartCenter(int part) {
  if(part==1)return v3(0,.165f,-.145f);
  if(part==2)return v3(0,.145f,-.265f);
  if(part==3)return v3(0,.05f,.205f);
  if(part==4)return v3(-.135f,.018f,.025f);
  if(part==5)return v3(.135f,.018f,.025f);
  if(part==6)return v3(-.079f,.192f,-.21f);
  if(part==7)return v3(.079f,.192f,-.21f);
  if(part==8)return v3(0,.074f,-.115f);
  if(part==9)return v3(0,.124f,-.261f);
  return v3(0,0,0);
}
__device__ float3 duckPartRadius(int part) {
  if(part==1)return v3(.106f,.106f,.108f);
  if(part==2)return v3(.079f,.022f,.082f);
  if(part==3)return v3(.095f,.042f,.11f);
  if(part==4||part==5)return v3(.043f,.062f,.135f);
  if(part==6||part==7)return v3(.013f,.015f,.013f);
  if(part==8)return v3(.081f,.11f,.087f);
  if(part==9)return v3(.075f,.012f,.073f);
  return v3(.16f,.10f,.21f);
}
__device__ float duckHitPart(float3 origin,float3 direction,int part) {
  float3 c=duckPartCenter(part),r=duckPartRadius(part);
  float3 o=prod(sub(origin,c),v3(1/r.x,1/r.y,1/r.z)),d=prod(direction,v3(1/r.x,1/r.y,1/r.z));
  float a=dot3(d,d),b=dot3(o,d),c0=dot3(o,o)-1,disc=b*b-a*c0;
  if(disc<0)return 100000;
  float t=(-b-sqrtf(disc))/a;return t>.01f?t:100000;
}
__device__ float duckMouldDistance(float3 p) {
  float distance=1;
  for(int j=0;j<6;j++) {
    int part=j==0?0:(j==1?1:(j==2?8:(j==3?3:(j==4?4:5))));
    float3 r=duckPartRadius(part),q=sub(p,duckPartCenter(part));
    float d=(sqrtf(square(q.x/r.x)+square(q.y/r.y)+square(q.z/r.z))-1)*fminf(r.x,fminf(r.y,r.z));
    float k=part==4||part==5?.018f:.045f;
    float h=sat(.5f+.5f*(d-distance)/k);
    distance=lerp(d,distance,h)-k*h*(1-h);
  }
  return distance;
}
__device__ float2 duckHit(float3 origin,float3 direction) {
  float t=100000;int material=0;
  bool close=dot3(origin,origin)<324;
  if(close) {
    float along=-dot3(origin,direction),disc=along*along-dot3(origin,origin)+.16f;
    if(disc>=0){
      float travel=fmaxf(.01f,along-sqrtf(disc)),end=along+sqrtf(disc);
      for(int step=0;step<56&&travel<end;step++) {
        float d=duckMouldDistance(add(origin,mul(direction,travel)));
        if(d<.00065f){t=travel;material=10;break;}
        travel+=fmaxf(.0004f,d*.85f);
      }
    }
  }
  for(int part=0;part<10;part++){
    if(close&&part!=2&&part!=9&&part!=6&&part!=7)continue;
    float h=duckHitPart(origin,direction,part);if(h<t){t=h;material=part;}
  }
  return make_float2(t,(float)material);
}
__global__ void ducks_pixels_clear(unsigned *depth,int width,int height) {
  int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),y=(int)(blockIdx.y*blockDim.y+threadIdx.y);
  int count=(width/8)*(height/8),i=y*width+x;
  if(x<width&&y<height&&i<count+2)depth[i]=0;
}
// Linked tile bins have no per-tile cap. A global overflow flag triggers an
// exact full scan fallback, so even pathological views never omit bodies.
__global__ void ducks_project(const float4 *bodies,unsigned *depth,unsigned *nodes,int width,int height,float camX,float camY,float camZ,float yaw,float pitch) {
  int ix=(int)(blockIdx.x*blockDim.x+threadIdx.x),iy=(int)(blockIdx.y*blockDim.y+threadIdx.y),i=iy*128+ix;
  if(ix>=128||i>=10000)return;
  float4 p=bodies[i*5];
  float3 delta=v3(p.x-camX,p.y-camY,p.z-camZ);
  float3 forward=v3(sinf(yaw)*cosf(pitch),sinf(pitch),-cosf(yaw)*cosf(pitch)),right=v3(cosf(yaw),0,sinf(yaw)),up=duckCross(right,forward);
  float z=dot3(delta,forward);if(z<-.45f)return;
  float focal=(float)height/1.24974f,px=dot3(delta,right),py=dot3(delta,up),r=.40f;
  float halfX=(float)width/focal*.5f,halfY=.62487f;
  if(fabsf(px)>z*halfX+r*sqrtf(1+halfX*halfX)||fabsf(py)>z*halfY+r*sqrtf(1+halfY*halfY))return;
  float minX=0,maxX=(float)width-1,minY=0,maxY=(float)height-1;
  if(z>r+.001f) {
    float den=z*z-r*r;
    float ex=r*sqrtf(fmaxf(0,px*px+z*z-r*r)),ey=r*sqrtf(fmaxf(0,py*py+z*z-r*r));
    minX=(float)width*.5f+(px*z-ex)/den*focal;maxX=(float)width*.5f+(px*z+ex)/den*focal;
    minY=(float)height*.5f-(py*z+ey)/den*focal;maxY=(float)height*.5f-(py*z-ey)/den*focal;
  }
  int left=max(0,(int)floorf(minX/8)),rightEdge=min(width/8-1,(int)floorf(maxX/8)),top=max(0,(int)floorf(minY/8)),bottom=min(height/8-1,(int)floorf(maxY/8));
  int count=(width/8)*(height/8);
  for(int y=top;y<=bottom;y++)for(int x=left;x<=rightEdge;x++) {
    unsigned slot=atomicAdd(&depth[count],1u);
    if(slot<1048576u){nodes[slot*2]=(unsigned)i;nodes[slot*2+1]=atomicExch(&depth[y*(width/8)+x],slot+1u);}
    else atomicExch(&depth[count+1],1u);
  }
}
__global__ void ducks_shade(const float4 *bodies,const unsigned *depth,const unsigned *nodes,float4 *hdr,const float4 *surface,const float4 *sea,const float4 *environment,int width,int height,float camX,float camY,float camZ,float yaw,float pitch,int tornadoOn,float tx,float tz) {
  int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),y=(int)(blockIdx.y*blockDim.y+threadIdx.y);if(x>=width||y>=height)return;
  int pixel=y*width+x,count=(width/8)*(height/8);
  unsigned entry=depth[(y/8)*(width/8)+x/8];bool overflow=depth[count+1]!=0u;
  if(entry==0u&&!overflow)return;
  float4 background=hdr[pixel];
  float3 rd=ray(2*((float)x+.5f)/width-1,1-2*((float)y+.5f)/height,(float)width/height,yaw,pitch);
  float2 hit=make_float2(100000,0);int i=-1;
  for(int visit=0;visit<10000&&(overflow||entry!=0u);visit++) {
    int candidate=visit;
    if(!overflow){candidate=(int)nodes[(entry-1u)*2u];entry=nodes[(entry-1u)*2u+1u];}
    float4 bp=bodies[candidate*5];float3 delta=v3(bp.x-camX,bp.y-camY,bp.z-camZ);
    float along=dot3(delta,rd);
    if(along<-.45f||along>hit.x+.45f||dot3(delta,delta)-along*along>.2025f)continue;
    float4 bq=bodies[candidate*5+2];
    float2 h=duckHit(duckLocal(bq,mul(delta,-1)),duckLocal(bq,rd));
    if(h.x<hit.x){hit=h;i=candidate;}
  }
  if(i<0)return;
  float4 p=bodies[i*5],q=bodies[i*5+2];
  float3 origin=duckLocal(q,v3(camX-p.x,camY-p.y,camZ-p.z)),direction=duckLocal(q,rd);
  float3 world=v3(camX+rd.x*hit.x,camY+rd.y*hit.x,camZ+rd.z*hit.x);
  float4 w=duckWater(surface,sea,world.x,world.z,tornadoOn,tx,tz);
  if(world.y<w.x-.012f||hit.x>background.w+.15f)return;
  int part=(int)hit.y;float3 r=duckPartRadius(part),local=sub(add(origin,mul(direction,hit.x)),duckPartCenter(part));
  float3 localN=norm(v3(local.x/(r.x*r.x),local.y/(r.y*r.y),local.z/(r.z*r.z)));
  float3 point=add(origin,mul(direction,hit.x));
  // Soften joins between the moulded body, neck and wing forms. Eyes and bill
  // remain crisp, while the body reads as one rubber casting.
  if(part==10) {
    float e=.001f;
    localN=norm(v3(duckMouldDistance(add(point,v3(e,0,0)))-duckMouldDistance(sub(point,v3(e,0,0))),duckMouldDistance(add(point,v3(0,e,0)))-duckMouldDistance(sub(point,v3(0,e,0))),duckMouldDistance(add(point,v3(0,0,e)))-duckMouldDistance(sub(point,v3(0,0,e)))));
  } else if(part!=2&&part!=9&&part!=6&&part!=7) {
    float3 blended=mul(localN,1.5f);float weight=1.5f;
    for(int j=0;j<9;j++) {
      if(j==part||j==2||j==6||j==7)continue;
      float3 rr=duckPartRadius(j),pp=sub(point,duckPartCenter(j));
      float radius=fminf(rr.x,fminf(rr.y,rr.z));
      float d=(sqrtf(square(pp.x/rr.x)+square(pp.y/rr.y)+square(pp.z/rr.z))-1)*radius;
      float w=expf(-fabsf(d)*110)*.65f;
      blended=add(blended,mul(norm(v3(pp.x/(rr.x*rr.x),pp.y/(rr.y*rr.y),pp.z/(rr.z*rr.z))),w));weight+=w;
    }
    localN=norm(mul(blended,1/weight));
  }
  float3 n=duckRotate(q,localN);
  float variant=random((unsigned)i*37u)*.10f;
  float3 paint=v3(1,.66f+variant,.018f);
  if(part==2)paint=v3(.96f,.235f,.023f);
  if(part==9)paint=v3(.76f,.115f,.012f);
  if(part==6||part==7)paint=v3(.008f,.012f,.014f);
  if(part==4||part==5) {
    // Shallow embossed feather arcs in the wing, without separate dark discs.
    float arc=sqrtf(square(local.y/.055f)+square((local.z-.015f)/.13f));
    float groove=expf(-square((arc-.70f)/.045f))*.08f;
    paint=mul(paint,.97f-groove);
  }
  float3 sun=v3(.08959f,.51504f,-.85247f),halfVector=norm(sub(sun,rd));
  float shadow=environmentShadow(environment,world.x,world.z,camX,camZ);
  float diffuse=.24f+.20f*sat(n.y*.5f+.5f)+.74f*fmaxf(0,dot3(n,sun))*shadow;
  float roughness=part==6||part==7?180:46;
  float spec=powf(fmaxf(0,dot3(n,halfVector)),roughness)*.48f*shadow;
  float rim=powf(1-fmaxf(0,dot3(n,mul(rd,-1))),5)*.16f;
  float occlusion=1-.15f*expf(-square((point.y-.055f)/.055f))*expf(-square((point.z+.12f)/.09f));
  float3 color=add(mul(paint,diffuse*occlusion),v3(spec+rim*.6f,spec+rim*.8f,spec+rim));
  color=mix3(color,v3(.54f,.66f,.74f),1-expf(-hit.x*.0008f));
  hdr[pixel]=make_float4(color.x,color.y,color.z,hit.x);
}
