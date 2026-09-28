import {chromium} from 'playwright';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const tag=process.argv[2]||'after';
const out='previews/foam';
await mkdir(out,{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:false,args:['--enable-unsafe-webgpu']});
const errors=[];
try {
 const page=await browser.newPage({viewport:{width:1920,height:1080}});
 page.on('pageerror',e=>errors.push(String(e)));
 const source=(await readFile('app.js','utf8')).replace('window.clearwaterLab = {',`window.clearwaterLab = {
 async foamStats(){return exclusive(async()=>{
  const data=await rt.read(foam[foamIndex]);let total=0,fresh=0,covered=0,peak=0;
  for(let i=0;i<data.length;i+=4){total+=data[i];fresh+=data[i+1];covered+=data[i]>.1?1:0;peak=Math.max(peak,data[i]);}
  return {mean:total/65536,fresh:fresh/65536,covered:covered/65536,peak,finite:Array.from(data).every(Number.isFinite)};
 });},`);
 await page.route('**/app.js',r=>r.fulfill({contentType:'text/javascript',body:source}));
 await page.goto('http://127.0.0.1:5186/?t=5');
 await page.waitForFunction(()=>clearwaterDiagnostics?.ready||clearwaterDiagnostics?.errors.length,null,{timeout:120000});
 assert.deepEqual(await page.evaluate(()=>clearwaterDiagnostics.errors),[]);
 await page.evaluate(()=>{
  clearwaterLab.pause();Object.assign(clearwaterLab.state,{time:5,weatherAge:0,x:.5,z:-7.5,y:2.3,yaw:.05,pitch:-.3});
  document.getElementById('energy').value=1.25;document.getElementById('depth').value=5;
  const q=document.getElementById('quality');q.add(new Option('Review','1920'));q.value='1920';q.dispatchEvent(new Event('change'));
 });
 await page.addStyleTag({content:'body > :not(canvas):not(style){display:none!important}'});
 const result={};
 result.calm=await page.evaluate(()=>clearwaterLab.foamStats());
 assert.equal(result.calm.peak,0,'calm water must not spawn whitecaps');
 await page.evaluate(()=>clearwaterLab.weatherAdvance(900));
 await page.evaluate(()=>clearwaterLab.seek(36));
 result.storm=await page.evaluate(()=>clearwaterLab.foamStats());
 await page.screenshot({path:`${out}/${tag}-storm.png`});
 result.performance=await page.evaluate(()=>clearwaterLab.benchmark(40,true));
 await page.evaluate(()=>{Object.assign(clearwaterLab.state,{x:0,z:-15,y:2,pitch:-.65});});
 await page.evaluate(()=>clearwaterLab.seek(36));
 await page.screenshot({path:`${out}/${tag}-close.png`});
 await page.evaluate(()=>clearwaterLab.weatherAdvance(30));
 await page.screenshot({path:`${out}/${tag}-close-next.png`});
 await page.evaluate(()=>{Object.assign(clearwaterLab.state,{weatherAge:-1,wind:5,rain:0});});
 await page.evaluate(()=>clearwaterLab.seek(36));
 await page.screenshot({path:`${out}/${tag}-isolated.png`});
 await page.evaluate(()=>clearwaterLab.weatherAdvance(300));
 result.decay=await page.evaluate(()=>clearwaterLab.foamStats());
 assert.ok(result.storm.finite&&result.decay.finite);
 assert.ok(result.storm.peak>.001&&result.storm.peak<=1);
 assert.ok(result.decay.mean<result.storm.mean*.15,'foam must dissipate after breaking stops');
 assert.deepEqual(errors,[]);
 await writeFile(`${out}/${tag}.json`,JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
