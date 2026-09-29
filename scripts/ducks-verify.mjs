import {chromium} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const out='previews/ducks';await mkdir(out,{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
function stats(a){let maxSpeed=0,maxOmega=0,contacts=0,wet=0,qerr=0,airborne=0,lofted=0,risingSpinning=0,maxAltitude=-1e9;for(let i=0;i<10000;i++){let k=i*20;maxSpeed=Math.max(maxSpeed,Math.hypot(...a.slice(k+4,k+7)));maxOmega=Math.max(maxOmega,Math.hypot(...a.slice(k+12,k+15)));contacts+=a[k+17];wet+=a[k+7];qerr=Math.max(qerr,Math.abs(Math.hypot(...a.slice(k+8,k+12))-1));let altitude=a[k+1]-a[k+16];if(altitude>.5)airborne++;if(altitude>5)lofted++;if(altitude>5&&a[k+5]>.5&&Math.hypot(...a.slice(k+12,k+15))>1)risingSpinning++;maxAltitude=Math.max(maxAltitude,altitude);}return {count:a.length/20,finite:a.every(Number.isFinite),maxSpeed,maxOmega,contacts,wet:wet/10000,qerr,airborne,lofted,risingSpinning,maxAltitude};}
try{
const page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[];
page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
await page.goto('http://127.0.0.1:5173/?t=8');
await page.waitForFunction(()=>window.clearwaterDiagnostics?.ready||window.clearwaterDiagnostics?.errors.length,null,{timeout:180000});
assert.deepEqual(await page.evaluate(()=>clearwaterDiagnostics.errors),[]);
await page.selectOption('#quality','1152');await page.click('#ducks');await page.evaluate(()=>clearwaterLab.pause());
await page.evaluate(()=>clearwaterLab.ducksAdvance(240));
const initial=await page.evaluate(()=>clearwaterLab.ducksInspect()),floating=stats(initial.bodies);
assert.equal(floating.count,10000);assert(floating.finite&&floating.qerr<.00001&&floating.maxSpeed<5);
await page.click('#toggle');await page.screenshot({path:out+'/overview.png'});
await page.evaluate(()=>document.getElementById('duckClose').click());await page.waitForTimeout(250);await page.screenshot({path:out+'/close.png'});
const closeBins=await page.evaluate(()=>clearwaterLab.ducksBins());assert.equal(closeBins.overflow,0);
const closeGpu=await page.evaluate(()=>clearwaterLab.benchmark(40,true));
// Pause holds every state record while the render loop and camera stay usable.
const paused=await page.evaluate(()=>clearwaterLab.ducksInspect());await page.waitForTimeout(250);
assert.deepEqual((await page.evaluate(()=>clearwaterLab.ducksInspect())).bodies,paused.bodies);
// Isolated overlapping pair in air: compare with the same separated bodies.
const pair=[...paused.bodies];
for(let i=0;i<10000;i++){let k=i*20;pair[k]=1000+i*2;pair[k+1]=10;pair[k+2]=1000;pair[k+4]=pair[k+5]=pair[k+6]=0;pair[k+12]=pair[k+13]=pair[k+14]=0;}
pair[0]=0;pair[2]=0;pair[20]=.30;pair[22]=0;
await page.evaluate(a=>clearwaterLab.ducksWrite(a),pair);await page.evaluate(()=>clearwaterLab.ducksAdvance(1));
const collided=(await page.evaluate(()=>clearwaterLab.ducksInspect())).bodies;
assert(collided[4]<0&&collided[24]>0,'contact must produce opposite impulses');assert(collided[5]<0&&collided[25]<0,'airborne bodies must fall');
pair[20]=2;await page.evaluate(a=>clearwaterLab.ducksWrite(a),pair);await page.evaluate(()=>clearwaterLab.ducksAdvance(1));
const separated=(await page.evaluate(()=>clearwaterLab.ducksInspect())).bodies;
assert.equal(separated[17],0);assert(collided[17]>0);assert(collided[24]>separated[24]+.1,'collision control must remove the separating impulse');
await page.evaluate(a=>clearwaterLab.ducksWrite(a),paused.bodies);
await page.evaluate(()=>clearwaterLab.ducksAdvance(1200));const sustained=stats((await page.evaluate(()=>clearwaterLab.ducksInspect())).bodies);
assert(sustained.finite&&sustained.qerr<.00001&&sustained.maxSpeed<10,'floating bodies remain stable');
// Put the vortex in the flock through the real control, then advance 12 seconds.
await page.evaluate(()=>{document.getElementById('tornado').click();clearwaterLab.pause();});
await page.evaluate(()=>clearwaterLab.ducksAdvance(1440));
const storm=stats((await page.evaluate(()=>clearwaterLab.ducksInspect())).bodies);
assert(storm.finite&&storm.maxSpeed<150&&storm.maxOmega<500&&storm.qerr<.00001,'vortex bodies remain finite');
assert(storm.maxSpeed>sustained.maxSpeed+1,'simulated air must accelerate ducks');
assert(storm.maxAltitude>20&&storm.lofted>1000,'light ducks must be entrained well above the surface');
assert(storm.risingSpinning>100,'airborne ducks must tumble while rising, not settle into a uniformly aligned ring');
await page.evaluate(()=>clearwaterLab.ducksAdvance(2400));
const lofted=stats((await page.evaluate(()=>clearwaterLab.ducksInspect())).bodies);
assert(lofted.lofted>100&&lofted.maxAltitude>10,'the vortex must sustain airborne ducks, not only an initial hop');
assert(lofted.finite&&lofted.qerr<.00001&&lofted.maxSpeed<150,'sustained lofting must remain bounded');
await page.evaluate(()=>{document.getElementById('duckOverview').click();Object.assign(clearwaterLab.state,{y:12,pitch:.12,z:clearwaterLab.state.duckZ+120});});
await page.waitForTimeout(250);await page.screenshot({path:out+'/tornado.png'});
const tornadoGpu=await page.evaluate(()=>clearwaterLab.benchmark(40,true));
const stormBins=await page.evaluate(()=>clearwaterLab.ducksBins());assert.equal(stormBins.overflow,0);
// Normal live frames never download body positions to the host.
const frame=await page.evaluate(()=>clearwaterDiagnostics.frames);
await page.waitForFunction(f=>clearwaterDiagnostics.frames>f+2,frame);
const bytes=await page.evaluate(()=>clearwaterDiagnostics.readbackBytes);
await page.evaluate(()=>{clearwaterLab.state.playing=true;});await page.waitForTimeout(500);await page.evaluate(()=>clearwaterLab.pause());
assert.equal(await page.evaluate(()=>clearwaterDiagnostics.readbackBytes),bytes);
assert.deepEqual(errors,[]);
const result={floating,sustained,storm,lofted,collision:{pairContacts:collided[17],leftVelocity:collided[4],rightVelocity:collided[24],controlVelocity:separated[24]},closeBins,stormBins,closeGpu,tornadoGpu,errors};
await writeFile(out+'/verification.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
