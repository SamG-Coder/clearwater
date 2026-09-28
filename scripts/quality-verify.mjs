import {chromium} from 'playwright';
import {mkdir, writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const tag=process.argv[2] || 'current';
await mkdir('previews/quality',{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:false,args:['--enable-unsafe-webgpu']});
const errors=[];
try {
 const page=await browser.newPage({viewport:{width:1440,height:900}});
 page.on('pageerror',e=>errors.push(String(e)));
 page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.goto('http://127.0.0.1:5186/?t=5');
 await page.waitForFunction(()=>clearwaterDiagnostics?.ready||clearwaterDiagnostics?.errors.length,null,{timeout:120000});
 assert.deepEqual(await page.evaluate(()=>clearwaterDiagnostics.errors),[]);
 await page.evaluate(()=>{clearwaterLab.pause();clearwaterLab.state.pitch=.04;});
 await page.evaluate(()=>clearwaterLab.seek(5));
 await page.locator('#toggle').click();
 const result={tag,scenes:{}};
 for (const scene of ['calm','front','storm']) {
  if(scene==='front') {await page.evaluate(()=>{clearwaterLab.state.weatherAge=0;});await page.evaluate(()=>clearwaterLab.weatherAdvance(450));}
  if(scene==='storm') await page.evaluate(()=>clearwaterLab.weatherAdvance(450));
  await page.screenshot({path:`previews/quality/${tag}-${scene}.png`});
  result.scenes[scene]=await page.evaluate(()=>clearwaterLab.benchmark());
 }
 result.optics=await page.evaluate(()=>clearwaterLab.inspectOptics());
 result.water=await page.evaluate(()=>clearwaterLab.inspect());
 result.errors=errors;
 assert.deepEqual(errors,[]);
 await writeFile(`previews/quality/${tag}.json`,JSON.stringify(result,null,2));
 console.log(JSON.stringify(result.scenes,null,2));
} finally {await browser.close();}
