import {chromium} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
const label=process.argv[2]||'latest';
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
try {
  const page=await browser.newPage({viewport:{width:1280,height:800}});
  await page.goto('http://127.0.0.1:5173/?t=8');
  await page.waitForFunction(()=>window.clearwaterDiagnostics?.ready,{}, {timeout:180000});
  await page.click('#tornado');
  await page.evaluate(()=>window.clearwaterLab.tornadoAdvance(90));
  const simulation=await page.evaluate(()=>window.clearwaterLab.benchmark(60,true));
  const rendering=await page.evaluate(()=>window.clearwaterLab.benchmark(60,false));
  const errors=await page.evaluate(()=>window.clearwaterDiagnostics.errors);
  await mkdir('previews/tornado',{recursive:true});
  await writeFile(`previews/tornado/profile-${label}.json`,JSON.stringify({simulation,rendering,errors},null,2));
  console.log(JSON.stringify({simulation,rendering,errors},null,2));
} finally {await browser.close();}
