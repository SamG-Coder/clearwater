import {chromium} from 'playwright';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import path from 'node:path';
const preview=process.argv.includes('--preview'),width=1920,height=1080,fps=30;
const out=path.resolve('recordings/ClearWater-ducks-showcase-2026-09-29');await mkdir(out,{recursive:true});
const shots=[
 {name:'ducks',duration:7,title:'added 10,000 rubber ducks lol',detail:'',from:{x:-1,y:1.2,z:48},to:{x:1.2,y:1.4,z:43},look:{x:0,y:.2,z:37}},
 {name:'rain',duration:7,title:'then made it rain',detail:'',from:{x:-32,y:24,z:90},to:{x:-20,y:18,z:82},look:{x:0,y:1,z:0}},
 {name:'weather',duration:7,title:'they were doing alright...',detail:'',from:{x:12,y:1.5,z:23},to:{x:18,y:3,z:30},look:{x:0,y:.3,z:0}},
 {name:'tornado',duration:9,title:'so naturally i added a tornado',detail:'',tornado:true,from:{x:-180,y:30,z:440},to:{x:-145,y:20,z:405},look:{x:0,y:210,z:0}},
 {name:'chaos',duration:9,title:'yeah... that got out of hand',detail:'',from:{x:70,y:9,z:95},to:{x:58,y:14,z:80},look:{x:0,y:20,z:0}},
 {name:'links',duration:6,title:'clearwater',detail:'demo: samg-coder.github.io/clearwater/\nproject: github.com/SamG-Coder/clearwater',from:{x:125,y:16,z:400},to:{x:110,y:18,z:385},look:{x:0,y:190,z:0}}
];
let source=await readFile('app.js','utf8');
if(!source.includes('if (!locked && !document.hidden)'))throw Error('Capture hook changed');
source=source.replace('let rt,','let captureMode=false;\nlet rt,').replace('if (!locked && !document.hidden)','if (!locked && !captureMode && !document.hidden)').replace('window.clearwaterLab = {',`window.clearwaterLab = {
 async captureFrame(config) {
  captureMode=true;
  return exclusive(async()=>{
   play(false);await resize();Object.assign(state,config.camera||{});
   const count=config.steps||1;
   for(let i=0;i<count;i++){
    frameDt=config.dt||0;weatherDt=frameDt;state.time+=frameDt;
    const b=rt.batch();waves(b);ripples(b,frameDt);tornadoStep(b,frameDt);ducksStep(b,frameDt);b.submit();
   }
   if(config.render!==false)render();await rt.idle();
   return {width,height,time:state.time,ducks:duckSteps,tornado:tornadoSteps};
  });
 },`);
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu','--disable-background-timer-throttling','--disable-renderer-backgrounding']});
let encoder,encodeDone,encoderErrors='',frames=0;const errors=[];
try {
 const page=await browser.newPage({viewport:{width,height},deviceScaleFactor:1});
 page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.route('**/app.js',route=>route.fulfill({status:200,contentType:'text/javascript',body:source}));
 await page.goto('http://127.0.0.1:5173/?t=8');
 await page.waitForFunction(()=>window.clearwaterDiagnostics?.ready||window.clearwaterDiagnostics?.errors.length,null,{timeout:180000});
 if((await page.evaluate(()=>clearwaterDiagnostics.errors)).length)throw Error('Startup failed');
 await page.evaluate(()=>{
  document.getElementById('ducks').click();clearwaterLab.pause();
  Object.assign(clearwaterLab.state,{duckX:0,duckZ:0,tornadoX:0,tornadoZ:0,wind:13,rain:.58,clouds:.64,weatherAge:850,weatherRate:1,stormStrength:.55});
  document.getElementById('quality').value='1920';document.getElementById('quality').dispatchEvent(new Event('change'));
  document.getElementById('depth').value='5';document.getElementById('energy').value='1.15';
  const style=document.createElement('style');style.textContent='body > :not(canvas):not(#film-caption):not(style){display:none!important} body::after{content:"";position:fixed;left:0;right:0;bottom:0;height:240px;background:linear-gradient(transparent,rgba(0,12,22,.52));pointer-events:none;z-index:1} #film-caption{position:fixed;left:66px;bottom:56px;color:#f5f8f7;font-family:Segoe UI,Arial,sans-serif;text-shadow:0 2px 12px #071b27;pointer-events:none;z-index:2} #film-caption .line{height:2px;width:40px;background:#e9d269;margin-bottom:16px} #film-caption .title{font-size:34px;font-weight:400;letter-spacing:.3px} #film-caption .detail{white-space:pre-line;font-size:23px;line-height:1.5;letter-spacing:.3px;margin-top:9px;color:#e1e9e7}';document.head.append(style);
  const caption=document.createElement('div');caption.id='film-caption';caption.innerHTML='<div class="line"></div><div class="title"></div><div class="detail"></div>';document.body.append(caption);
 });
 await page.evaluate(()=>clearwaterLab.captureFrame({dt:0}));
 for(let n=0;n<15;n++)await page.evaluate(()=>clearwaterLab.captureFrame({dt:1/30,steps:10,render:false}));
 if(!preview){
  encoder=spawn('ffmpeg',['-hide_banner','-loglevel','warning','-y','-f','image2pipe','-framerate',String(fps),'-vcodec','mjpeg','-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','18','-maxrate','16M','-bufsize','32M','-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-movflags','+faststart',path.join(out,'ClearWater-ducks-storm-tornado-turbulent.mp4')],{stdio:['pipe','ignore','pipe'],windowsHide:true});
  encoder.stderr.on('data',b=>encoderErrors+=b.toString());encoder.stdin.on('error',()=>{});encodeDone=once(encoder,'close');
 }
 let focus={x:0,z:0};
 for(const [index,shot] of shots.entries()){
  if(index===2||index===3){
   const bodies=(await page.evaluate(()=>clearwaterLab.ducksInspect())).bodies;
   focus={x:0,z:0};for(let i=0;i<bodies.length;i+=20){focus.x+=bodies[i]/10000;focus.z+=bodies[i+2]/10000;}
  }
  await page.evaluate(s=>{
   document.querySelector('#film-caption .title').textContent=s.title;document.querySelector('#film-caption .detail').textContent=s.detail;
  },shot);
  for(let f=0;f<shot.duration*fps;f++){
   if(shot.tornado&&f===60)await page.evaluate(center=>{document.getElementById('tornado').click();clearwaterLab.pause();Object.assign(clearwaterLab.state,{tornadoX:center.x,tornadoZ:center.z,tornadoStrength:1,clouds:.64});},focus);
   let t=f/(shot.duration*fps-1),u=t*t*(3-2*t),camera={};
   for(const key of ['x','y','z'])camera[key]=shot.from[key]+(shot.to[key]-shot.from[key])*u;
   let dx=shot.look.x-camera.x,dy=shot.look.y-camera.y,dz=shot.look.z-camera.z;
   camera.yaw=Math.atan2(dx,-dz);camera.pitch=Math.atan2(dy,Math.hypot(dx,dz));
   camera.x+=focus.x;camera.z+=focus.z;
   const save= f===Math.floor(shot.duration*fps/2),render=!preview||save;
   const result=await page.evaluate(config=>clearwaterLab.captureFrame(config),{camera,dt:1/fps,render});
   if(result.width!==width||result.height!==height)throw Error('Capture dimensions changed');
   if(render){
    const png=await page.screenshot({type:preview?'png':'jpeg',...(preview?{}:{quality:95}),animations:'allow',timeout:60000});
    if(!preview){await new Promise((resolve,reject)=>encoder.stdin.write(png,e=>e?reject(e):resolve()));frames++;}
    if(save)await writeFile(path.join(out,`${index+1}-${shot.name}${preview?'-preview':''}.${preview?'png':'jpg'}`),png);
   }
   if(f%90===0)console.log(JSON.stringify({shot:shot.name,frame:f,total:shot.duration*fps,frames}));
  }
 }
 if(encoder){encoder.stdin.end();const [code]=await encodeDone;if(code!==0)throw Error('FFmpeg '+code+' '+encoderErrors);}
 if(errors.length)throw Error(errors.join('\n'));
 const stats=await page.evaluate(()=>clearwaterLab.ducksInspect());
 if(!stats.bodies.every(Number.isFinite))throw Error('Non-finite duck state');
 await writeFile(path.join(out,preview?'preview.json':'capture.json'),JSON.stringify({width,height,fps,frames,duration:preview?null:frames/fps,source:'Actual demo capture with scripted camera, fixed 1/30 second simulation advances and four 120 Hz duck substeps per video frame. Not an end-to-end performance recording.',shots,ducks:stats.bodies.length/20,duckSteps:stats.steps,errors},null,2));
 const tweet='added 10,000 rubber ducks to clearwater lol\n\nthen made it rain and threw a tornado at them. they handled it about as well as you\'d expect\n\ntry it: https://samg-coder.github.io/clearwater/\nproject: https://github.com/SamG-Coder/clearwater';
 await writeFile(path.join(out,'tweet.txt'),tweet+'\n');console.log('Saved '+out);
} finally {if(encoder&&!encoder.stdin.destroyed)encoder.stdin.end();await browser.close();}
