import {chromium} from 'playwright';
import {readFile,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const browser=await chromium.launch({channel:'msedge',headless:false,args:['--enable-unsafe-webgpu']});
try {
 const page=await browser.newPage();
 const hook=`window.clearwaterLab = {
 async spectralOracle(){return exclusive(async()=>{
  const modes=[[3,7],[5,0],[0,9]],lengths=[4.6,37,293],chops=[.18,1.15,.75];
  const seeds=new Float32Array(3*65536*2);modes.forEach(([x,z],c)=>seeds[(c*65536+z*256+x)*2]=.05);
  const bs=rt.createBuffer(seeds),sc=rt.createBuffer(new Float32Array([1,1,1]));
  const we=rt.createBuffer(new Float32Array([5,0,0,0,.8,.6,0,1]));
  const en=rt.createBuffer(new Float32Array(3*65536).fill(1));
  const f=[rt.createBuffer(3*65536*16),rt.createBuffer(3*65536*16)];
  const d=[rt.createBuffer(3*65536*16),rt.createBuffer(3*65536*16)];
  try{
   const b=rt.batch();b.dispatch(k.evolve_spectrum.bind({seed:bs,scales:sc,output:f[0],weather:we,spectralEnergy:en},{time:.37,sea:1,depth:5,weatherDt:0}),WG);
   b.dispatch(k.chop_spectrum.bind({input:f[0],output:d[0]},{sea:1}),WG);
   transform(b,f,1);transform(b,d,1);b.submit();
   const h=await rt.read(f[0]),disp=await rt.read(d[0]);let maxError=0;
   for(let c=0;c<3;c++){
    const kx=2*Math.PI*modes[c][0]/lengths[c],kz=2*Math.PI*modes[c][1]/lengths[c],km=Math.hypot(kx,kz);
    const omega=Math.sqrt((9.81*km+.000074*km**3)*Math.tanh(Math.min(20,km*5)));
    for(let z=0;z<256;z++)for(let x=0;x<256;x++){
     const phase=2*Math.PI*(modes[c][0]*x+modes[c][1]*z)/256+omega*.37;
     const co=.1*Math.cos(phase),si=.1*Math.sin(phase),ch=chops[c],id=(c*65536+z*256+x)*4;
     const expected=[co,-kx*si,-kz*si,-ch*kx*kz/km*co,-ch*kx/km*si,-ch*kz/km*si,-ch*kx*kx/km*co,-ch*kz*kz/km*co];
     const actual=[...h.slice(id,id+4),...disp.slice(id,id+4)];
     for(let j=0;j<8;j++)maxError=Math.max(maxError,Math.abs(actual[j]-expected[j]));
    }
   }
   return {maxError,fields:8,modes,cells:196608};
  }finally{for(const b of [bs,sc,we,en,...f,...d])rt.destroyBuffer(b);}
 });},`;
 const source=(await readFile('app.js','utf8')).replace('window.clearwaterLab = {',hook);
 await page.route('**/app.js',r=>r.fulfill({contentType:'text/javascript',body:source}));
 await page.goto('http://127.0.0.1:5186/?t=5');
 await page.waitForFunction(()=>clearwaterDiagnostics?.ready,null,{timeout:120000});
 await page.evaluate(()=>clearwaterLab.pause());
 const result=await page.evaluate(()=>clearwaterLab.spectralOracle());
 assert.ok(result.maxError<.00002,JSON.stringify(result));
 await writeFile('previews/foam/spectral-oracle.json',JSON.stringify(result,null,2));console.log(result);
}finally{await browser.close();}
