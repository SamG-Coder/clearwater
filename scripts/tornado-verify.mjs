import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
const output='previews/tornado';
await mkdir(output,{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
try {
  const page=await browser.newPage({viewport:{width:1280,height:800}}),errors=[];
  page.on('pageerror',e=>errors.push(String(e)));
  page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.goto('http://127.0.0.1:5173/?t=8');
  await page.waitForFunction(()=>window.clearwaterDiagnostics?.ready||window.clearwaterDiagnostics?.errors.length,{}, {timeout:180000});
  assert.deepEqual(await page.evaluate(()=>window.clearwaterDiagnostics.errors),[]);
  await page.selectOption('#quality','768');
  await page.waitForFunction(()=>window.clearwaterDiagnostics.width===768);
  const baseline=await page.evaluate(()=>window.clearwaterLab.benchmark(16,true));
  await page.click('#tornado');
  await page.evaluate(()=>window.clearwaterLab.pause());
  await page.evaluate(()=>window.clearwaterLab.tornadoAdvance(1));
  await page.click('#toggle');
  await page.screenshot({path:`${output}/initial.png`});
  await page.evaluate(()=>window.clearwaterLab.tornadoAdvance(299));
  const inspect=()=>page.evaluate(async()=>{
    const a=await window.clearwaterLab.tornadoInspect();
    function divergence(v) {
      let sum=0,n=0;
      for(let z=2;z<46;z++)for(let y=2;y<94;y++)for(let x=2;x<46;x++){
        let i=((z*96+y)*48+x)*4;
        let d=(v[i]-v[i-4]+v[i+1]-v[i-48*4+1]+v[i+2]-v[i-96*48*4+2])/6;
        sum+=d*d;n++;
      }
      return Math.sqrt(sum/n);
    }
    let maxSpeed=0,liquid=0,spray=0,seaHeight=0,foam=0;
    for(let i=0;i<a.velocity.length;i+=4){maxSpeed=Math.max(maxSpeed,Math.hypot(...a.velocity.slice(i,i+3)));liquid+=a.moisture[i+1];spray+=a.moisture[i+3];}
    for(let i=0;i<9216*4;i+=4){seaHeight=Math.max(seaHeight,Math.abs(a.sea[i]));foam=Math.max(foam,a.sea[i+3]);}
    let spectralPeak=0,spectralEnergy=0,edgePeak=0,sprayCount=0,sprayTop=0;
    for(let z=0;z<256;z++)for(let x=0;x<256;x++){
      let i=(9216+z*256+x)*4,h=a.sea[i];spectralPeak=Math.max(spectralPeak,Math.abs(h));spectralEnergy+=h*h;
      if(Math.hypot(x-128,z-128)*1.125>=139)edgePeak=Math.max(edgePeak,Math.abs(h));
    }
    for(let i=0;i<a.sprayParticles.length;i+=8)if(a.sprayParticles[i+3]>0){sprayCount++;sprayTop=Math.max(sprayTop,a.sprayParticles[i+1]);}
    const profile=[0,8,24,48,72,90].map(y=>({height:y*6,moisture:a.moisture.slice(((24*96+y)*48+24)*4,((24*96+y)*48+24)*4+4),velocity:a.velocity.slice(((24*96+y)*48+24)*4,((24*96+y)*48+24)*4+4)}));
    let cloudMotion=0;
    for(let z=2;z<46;z+=4)for(let y=60;y<90;y+=4)for(let x=2;x<46;x+=4){let i=((z*96+y)*48+x)*4;cloudMotion=Math.max(cloudMotion,Math.hypot(a.cloudMap[i]-(x-23.5)*6,a.cloudMap[i+2]-(z-23.5)*6));}
    return {finite:[a.velocity,a.moisture,a.sea,a.cloudMap,a.sprayParticles,a.waveModes].every(v=>v.every(Number.isFinite)),maxSpeed,liquid,spray,seaHeight,foam,spectralPeak,spectralEnergy,edgePeak,sprayCount,sprayTop,zeroMode:a.waveModes.slice(0,4),before:divergence(a.unprojected),after:divergence(a.velocity),steps:a.steps,profile,cloudMotion};
  });
  const data=await inspect();
  const coupling=await page.evaluate(()=>window.clearwaterLab.tornadoCouplingTest());
  await page.screenshot({path:`${output}/evolved.png`});
  const enabled=await page.evaluate(()=>window.clearwaterLab.benchmark(16,true));
  const renderOnly=await page.evaluate(()=>window.clearwaterLab.benchmark(16,false));
  await page.evaluate(()=>{document.getElementById('quality').value='1152';document.getElementById('quality').dispatchEvent(new Event('change'));});
  await page.waitForFunction(()=>window.clearwaterDiagnostics.width===1152);
  const balanced=await page.evaluate(()=>window.clearwaterLab.benchmark(24,true));
  await page.evaluate(()=>window.clearwaterLab.tornadoAdvance(900));
  const sustained=await inspect();
  await page.screenshot({path:`${output}/sustained.png`});
  await page.evaluate(()=>{Object.assign(window.clearwaterLab.state,{y:8,z:-535,pitch:.3});});
  await page.screenshot({path:`${output}/close.png`});
  await page.evaluate(()=>{window.clearwaterLab.state.tornadoStrength=1.5;document.getElementById('depth').value='.5';});
  await page.evaluate(()=>window.clearwaterLab.tornadoAdvance(300));
  const stress=await inspect();
  const inspectedFrame=await page.evaluate(()=>window.clearwaterDiagnostics.frames);
  await page.waitForFunction(n=>window.clearwaterDiagnostics.frames>n+2,inspectedFrame);
  const readbacks=await page.evaluate(()=>window.clearwaterDiagnostics.readbackBytes);
  const frame=await page.evaluate(()=>window.clearwaterDiagnostics.frames);
  await page.waitForFunction(n=>window.clearwaterDiagnostics.frames>n+5,frame);
  assert.equal(await page.evaluate(()=>window.clearwaterDiagnostics.readbackBytes),readbacks,'live rendering must not read fluid buffers back');
  await page.evaluate(()=>document.getElementById('clearWeather').click());
  assert.equal(await page.evaluate(()=>window.clearwaterLab.state.tornado),false);
  await writeFile(`${output}/verification.json`,JSON.stringify({data,sustained,stress,coupling,baseline,enabled,renderOnly,balanced,errors},null,2));
  console.log(JSON.stringify({data,sustained,stress,coupling,baseline,enabled,renderOnly,balanced,errors},null,2));
  assert(data.finite,'fields must remain finite');
  assert(data.after<data.before,'pressure projection must reduce divergence');
  assert(data.seaHeight>0&&data.foam>0,'simulated wind must drive water and foam');
  assert(data.liquid>0&&data.spray>0,'moisture and spray must survive transport');
  assert(data.cloudMotion>1,'the flow must transport surrounding clouds');
  assert(data.spectralPeak>.001&&data.spectralEnergy>0,'tornado forcing must produce actual FFT wave displacement');
  assert.equal(data.edgePeak,0,'the spectral patch must fade out before its periodic boundary');
  assert.deepEqual(data.zeroMode,[0,0,0,0],'forcing must not change the ocean mean level');
  assert(data.sprayCount>0&&data.sprayTop>1,'spray parcels must leave the water surface');
  assert(coupling.windWorkRms>0,'existing FFT waves must contribute to tornado wind forcing');
  assert.equal(coupling.controlWindWorkRms,0,'zeroing the FFT surface must remove that contribution');
  assert.equal(coupling.pressureDifference,0,'the control must preserve atmospheric pressure forcing');
  assert.equal(coupling.farForce,0,'forcing must stay local to the vortex');
  for(const result of [sustained,stress]){
    assert(result.finite&&result.maxSpeed<150&&result.seaHeight<2&&result.spectralPeak<2,'long-running fluid and water must stay bounded');
    assert(result.after<result.before,'projection must remain effective');
    assert(result.liquid>0&&result.spray>0,'the driven vortex must remain moist');
  }
  assert.deepEqual(errors,[]);
} finally {await browser.close();}
