import {chromium} from 'playwright';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import path from 'node:path';

const preview=process.argv.includes('--preview');
const width=1920,height=1080,fps=30;
const out=path.resolve('recordings/ClearWater-tornado-2026-09-29');
await mkdir(out,{recursive:true});
const shots=[
  {name:'tornado',duration:8,title:'added a tornado to clearwater',detail:'simulated wind, clouds and water',
   from:{x:-100,y:4,z:65},to:{x:-45,y:7,z:5},look:{x:0,y:175,z:-650}},
  {name:'surface',duration:7,title:'FFT waves + surface spray',detail:'wind and pressure drive the water around it',
   from:{x:72,y:5,z:-540},to:{x:48,y:4,z:-554},look:{x:0,y:3,z:-650}},
  {name:'clouds',duration:8,title:'the clouds move with it too',detail:'simulation + rendering in .cu / CUDA WebShader',
   from:{x:160,y:22,z:-170},to:{x:195,y:30,z:-205},look:{x:0,y:210,z:-650}},
];
let source=await readFile('app.js','utf8');
if(!source.includes('if (!locked && !document.hidden)'))throw Error('Frame-loop capture hook changed');
source=source.replace('let rt,','let captureMode=false;\nlet rt,')
  .replace('if (!locked && !document.hidden)','if (!locked && !captureMode && !document.hidden)')
  .replace('window.clearwaterLab = {',`window.clearwaterLab = {
    async captureFrame(config) {
      captureMode=true;
      return exclusive(async()=>{
        play(false);await resize();Object.assign(state,config.camera||{});
        frameDt=config.dt||0;weatherDt=0;state.time+=frameDt;
        const batch=rt.batch();waves(batch);ripples(batch,frameDt);tornadoStep(batch,frameDt);
        batch.submit();render();await rt.idle();
        return {width,height,time:state.time,steps:tornadoSteps};
      });
    },`);
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu','--disable-background-timer-throttling','--disable-renderer-backgrounding']});
let encoder,encodeDone,encoderErrors='',frames=0;
const errors=[];
try {
  const page=await browser.newPage({viewport:{width,height},deviceScaleFactor:1});
  page.on('pageerror',e=>errors.push(String(e)));
  page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.route('**/app.js',route=>route.fulfill({status:200,contentType:'text/javascript',body:source}));
  await page.goto('http://127.0.0.1:5173/?t=8');
  await page.waitForFunction(()=>clearwaterDiagnostics?.ready||clearwaterDiagnostics?.errors.length,null,{timeout:180000});
  const startupErrors=await page.evaluate(()=>clearwaterDiagnostics.errors);
  if(startupErrors.length)throw Error(startupErrors.join('\n'));
  await page.evaluate(()=>{
    document.getElementById('tornado').click();clearwaterLab.pause();
    Object.assign(clearwaterLab.state,{tornadoX:0,tornadoZ:-650,tornadoStrength:1.15,clouds:.52});
    document.getElementById('quality').value='1920';document.getElementById('quality').dispatchEvent(new Event('change'));
    document.getElementById('depth').value='5';document.getElementById('energy').value='1.25';
    const style=document.createElement('style');
    style.textContent='body > :not(canvas):not(#film-caption):not(style){display:none!important} #film-caption{position:fixed;left:64px;bottom:58px;color:#f3f7f7;font-family:Segoe UI,Arial,sans-serif;text-shadow:0 2px 14px #071b27;pointer-events:none} #film-caption .line{height:2px;width:42px;background:#b2d6cf;margin-bottom:17px} #film-caption .title{font-size:32px;font-weight:400;letter-spacing:.3px} #film-caption .detail{font-size:19px;letter-spacing:.6px;margin-top:9px;color:#d7e7e8}';
    document.head.append(style);
    const contrast=document.createElement('style');
    contrast.textContent='body::after{content:"";position:fixed;left:0;right:0;bottom:0;height:210px;background:linear-gradient(transparent,rgba(0,12,22,.42));pointer-events:none;z-index:1} #film-caption{z-index:2}';
    document.head.append(contrast);
    const caption=document.createElement('div');caption.id='film-caption';caption.innerHTML='<div class="line"></div><div class="title"></div><div class="detail"></div>';document.body.append(caption);
  });
  await page.evaluate(()=>clearwaterLab.captureFrame({dt:0}));
  for(let n=0;n<150;n++)await page.evaluate(()=>clearwaterLab.captureFrame({dt:1/30}));
  if(!preview) {
    encoder=spawn('ffmpeg',['-hide_banner','-loglevel','warning','-y','-f','image2pipe','-framerate',String(fps),'-vcodec','png','-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','18','-maxrate','12M','-bufsize','24M','-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-movflags','+faststart',path.join(out,'ClearWater-tornado-tweet.mp4')],{stdio:['pipe','ignore','pipe'],windowsHide:true});
    encoder.stderr.on('data',b=>{encoderErrors+=b.toString();});
    encoder.stdin.on('error',()=>{});encodeDone=once(encoder,'close');
  }
  for(const [index,shot] of shots.entries()) {
    await page.evaluate(s=>{document.querySelector('#film-caption .title').textContent=s.title;document.querySelector('#film-caption .detail').textContent=s.detail;},shot);
    const count=preview?1:shot.duration*fps;
    for(let f=0;f<count;f++) {
      const t=preview?.5:f/(count-1),u=t*t*(3-2*t),camera={};
      for(const key of ['x','y','z'])camera[key]=shot.from[key]+(shot.to[key]-shot.from[key])*u;
      const dx=shot.look.x-camera.x,dy=shot.look.y-camera.y,dz=shot.look.z-camera.z;
      camera.yaw=Math.atan2(dx,-dz);camera.pitch=Math.atan2(dy,Math.hypot(dx,dz));
      const state=await page.evaluate(config=>clearwaterLab.captureFrame(config),{camera,dt:preview?0:1/fps});
      if(state.width!==width||state.height!==height)throw Error('Wrong capture dimensions');
      const png=await page.screenshot({type:'png',animations:'allow',timeout:60000});
      if(preview)await writeFile(path.join(out,`${index+1}-${shot.name}-preview.png`),png);
      else {
        await new Promise((resolve,reject)=>encoder.stdin.write(png,e=>e?reject(e):resolve()));
        if(f===Math.floor(count/2))await writeFile(path.join(out,`${index+1}-${shot.name}.png`),png);
      }
      frames++;
      if(f%60===0)console.log(JSON.stringify({shot:shot.name,frame:f,total:count,outputFrames:frames}));
    }
    console.log('Completed '+shot.name);
  }
  if(encoder){encoder.stdin.end();const [code]=await encodeDone;if(code!==0)throw Error('FFmpeg '+code+' '+encoderErrors);}
  if(errors.length)throw Error(errors.join('\n'));
  await writeFile(path.join(out,preview?'preview.json':'capture.json'),JSON.stringify({width,height,fps,frames,duration:preview?null:frames/fps,source:'Actual WebGPU demo output, advanced at a fixed 1/30 second simulation timestep per captured frame. Scripted camera; this is not a real-time performance recording.',shots,errors},null,2));
  console.log('Saved '+out);
} finally {
  if(encoder&&!encoder.stdin.destroyed)encoder.stdin.end();
  await browser.close();
}
