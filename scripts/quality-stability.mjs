import {chromium} from 'playwright';
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const browser=await chromium.launch({channel:'msedge',headless:false,args:['--enable-unsafe-webgpu']});
const errors=[];
try {
 const page=await browser.newPage({viewport:{width:1440,height:900}});
 page.on('pageerror',e=>errors.push(String(e)));
 page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.goto('http://127.0.0.1:5186/?t=5');
 await page.waitForFunction(()=>clearwaterDiagnostics?.ready||clearwaterDiagnostics?.errors.length,null,{timeout:120000});
 assert.deepEqual(await page.evaluate(()=>clearwaterDiagnostics.errors),[]);
 assert.equal(await page.evaluate(()=>clearwaterDiagnostics.readbackBytes),0);
 await page.locator('#toggle').click();
 await page.evaluate(()=>{clearwaterLab.state.weatherAge=0;clearwaterLab.state.pitch=.04;});
 await page.evaluate(()=>clearwaterLab.weatherAdvance(900));
 // Freeze truly freezes the final image, not just the wave clock.
 const stillA=await page.locator('#water').screenshot();
 await page.waitForTimeout(200);
 const stillB=await page.locator('#water').screenshot();
 assert.ok(stillA.equals(stillB),'paused image must remain bit-identical');
 // Match the GPU event hash to capture an actual lightning pulse.
 const random=x=>{x=(x^(x>>>16))>>>0;x=Math.imul(x,2146121005)>>>0;x=(x^(x>>>15))>>>0;x=Math.imul(x,2221713035)>>>0;x=(x^(x>>>16))>>>0;return ((x&16777215)+1)/16777217;};
 const flashTime=27+1+random(3+193)*6;
 await page.evaluate(t=>clearwaterLab.seek(t),flashTime);
 await page.screenshot({path:'previews/quality/lightning.png'});
 await page.evaluate(()=>{clearwaterLab.state.pitch=-.55;});
 await page.evaluate(()=>clearwaterLab.seek(35));
 await page.screenshot({path:'previews/quality/rain-impacts.png'});
 const sizes={};
 for(const size of [768,1152,1536]){
  await page.evaluate(size=>{const el=document.getElementById('quality');el.value=String(size);el.dispatchEvent(new Event('change'));},size);
  await page.waitForFunction(size=>clearwaterDiagnostics.width===size,size);
  await page.evaluate(()=>clearwaterLab.seek(35));
  sizes[size]=await page.evaluate(()=>clearwaterLab.benchmark(40,true));
 }
 // Stress the extremes of the actual UI, then look up, down, around and fly far away.
 await page.evaluate(()=>{document.getElementById('energy').value=3;clearwaterLab.state.wind=24;});
 await page.evaluate(()=>clearwaterLab.weatherAdvance(200));
 const stress=[];
 for(const [time,yaw,pitch,x,z] of [[0,0,0,0,0],[17,1.5,.8,0,0],[55,3,-1.2,0,0],[95,-1,-.3,10000,-10000],[140,0,.1,0,0]]){
  await page.evaluate(([yaw,pitch,x,z])=>Object.assign(clearwaterLab.state,{yaw,pitch,x,z}),[yaw,pitch,x,z]);
  await page.evaluate(t=>clearwaterLab.seek(t),time);
  const weather=await page.evaluate(()=>clearwaterLab.weatherInspect());
  const optics=await page.evaluate(()=>clearwaterLab.inspectOptics());
  assert.ok(weather.compressionMin>0,'horizontal surface must not fold');
  assert.ok(optics.hdrFinite&&optics.glareFinite&&weather.spectrum.finite);
  stress.push({time,yaw,pitch,x,z,compressionMin:weather.compressionMin});
 }
 assert.deepEqual(errors,[]);
 const report={passed:true,pausedPixelsIdentical:true,flashTime,sizes,stress,errors};
 await writeFile('previews/quality/stability.json',JSON.stringify(report,null,2));
 console.log(JSON.stringify(report,null,2));
} finally {await browser.close();}
