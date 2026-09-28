import {chromium} from 'playwright';
import {mkdir} from 'node:fs/promises';
const out='previews/quality/motion-source';
await mkdir(out,{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:false,args:['--enable-unsafe-webgpu']});
try {
 const context=await browser.newContext({viewport:{width:1440,height:900},recordVideo:{dir:out,size:{width:1440,height:900}}});
 const page=await context.newPage();
 await page.goto('http://127.0.0.1:5186/?t=5');
 await page.waitForFunction(()=>clearwaterDiagnostics?.ready,null,{timeout:120000});
 await page.evaluate(()=>{clearwaterLab.state.weatherAge=0;clearwaterLab.state.pitch=.04;});
 await page.evaluate(()=>clearwaterLab.weatherAdvance(900));
 await page.locator('#toggle').click();
 await page.evaluate(()=>{clearwaterLab.state.weatherRate=0;clearwaterLab.resume();});
 await page.waitForTimeout(4400);
 await context.close();
 console.log(await page.video().path());
}finally{await browser.close();}
