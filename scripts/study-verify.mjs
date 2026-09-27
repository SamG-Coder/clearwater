import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
const browser=await chromium.launch({channel:'msedge',headless:false,args:['--enable-unsafe-webgpu']});
try {
 const page=await browser.newPage({viewport:{width:1720,height:1080}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 page.on('response',r=>{if(r.status()>=400)errors.push(r.url());});
 await page.goto(process.env.STUDY_URL||'http://127.0.0.1:5186/study/');
 await page.waitForFunction(()=>window.studyDiagnostics?.ready||window.studyDiagnostics?.errors.length,{},{timeout:120000});
 assert.deepEqual(await page.evaluate(()=>studyDiagnostics.errors),[]);
 const frames=async()=>{const n=await page.evaluate(()=>studyDiagnostics.frames);await page.waitForFunction(n=>studyDiagnostics.frames>n+1,n);};
 await frames();assert.equal(await page.evaluate(()=>studyDiagnostics.readbackBytes),0);
 const geometry=await page.evaluate(()=>{const c=document.querySelector('canvas'),r=document.querySelector('#reference');return{canvas:[c.width,c.height],image:[r.naturalWidth,r.naturalHeight],c:c.getBoundingClientRect().toJSON(),r:r.getBoundingClientRect().toJSON()};});
 assert.deepEqual(geometry.canvas,[1672,941]);assert.deepEqual(geometry.image,geometry.canvas);assert.deepEqual(geometry.c,geometry.r);
 const first=await page.evaluate(()=>studyLab.measure());await frames();const repeated=await page.evaluate(()=>studyLab.measure());
 assert(first.rowEdgesOpaque,'Padded rows must preserve opaque pixels at both image edges');
 assert(Number.isFinite(first.mae)&&first.mae>0&&first.mae<255);assert(Math.abs(first.mae-repeated.mae)<.01,'Paused frames must be repeatable');
 await page.screenshot({path:'previews/study-overlay.png'});
 await page.selectOption('#mode','render');assert.equal(await page.locator('#reference').evaluate(e=>e.style.opacity),'0');
 await page.screenshot({path:'previews/study-render.png'});
 await page.selectOption('#mode','difference');assert.equal(await page.locator('#reference').evaluate(e=>e.style.mixBlendMode),'difference');await page.screenshot({path:'previews/study-difference.png'});
 await page.selectOption('#mode','wipe');await page.locator('#wipe').fill('37');assert.equal(await page.locator('#reference').evaluate(e=>e.style.clipPath),'inset(0px 63% 0px 0px)');
 await page.click('#zoom');assert.equal(await page.locator('canvas').evaluate(e=>e.getBoundingClientRect().width),1672);await page.click('#zoom');
 await page.locator('#exposure').fill('0.25');await frames();const changed=await page.evaluate(()=>studyLab.measure());assert(Math.abs(changed.mae-first.mae)>5,'Exposure control must affect the GPU output');
 await page.click('#reset');await page.click('#pause');await frames();assert(await page.evaluate(()=>studyLab.state.time>5));await page.click('#pause');const time=await page.evaluate(()=>studyLab.state.time);await frames();assert.equal(await page.evaluate(()=>studyLab.state.time),time);
 await page.click('#reset');await frames();assert.deepEqual(errors,[]);assert.deepEqual(await page.evaluate(()=>studyDiagnostics.errors),[]);
 const report={url:page.url(),geometry,first,repeated,changed,errors};await writeFile('previews/study-verification.json',JSON.stringify(report,null,2));console.log(JSON.stringify({mae:first.mae,rmse:first.rmse,errors,normalReadbackBytes:0}));
} finally {await browser.close();}
