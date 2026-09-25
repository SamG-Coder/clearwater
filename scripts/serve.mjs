import http from 'node:http';import {readFile,stat} from 'node:fs/promises';import {fileURLToPath} from 'node:url';import path from 'node:path';
export function createStaticServer(root=fileURLToPath(new URL('../',import.meta.url))){
  root=path.resolve(root);const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json; charset=utf-8','.cu':'text/plain; charset=utf-8','.wgsl':'text/plain; charset=utf-8','.svg':'image/svg+xml','.png':'image/png','.ico':'image/x-icon'};
  return http.createServer(async(req,res)=>{
    try{
      const url=new URL(req.url,'http://localhost'),decoded=decodeURIComponent(url.pathname);
      if(decoded.includes('\0'))throw new Error('Invalid path');let file=path.resolve(root,'.'+decoded);
      if(file!==root&&!file.startsWith(root+path.sep)){res.writeHead(403);res.end('Forbidden');return;}
      if((await stat(file)).isDirectory())file=path.join(file,'index.html');
      const data=await readFile(file);res.writeHead(200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream','Cache-Control':'no-cache','Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp','X-Content-Type-Options':'nosniff'});res.end(data);
    }catch(error){res.writeHead(error.code==='ENOENT'?404:400,{'Content-Type':'text/plain'});res.end(error.code==='ENOENT'?'Not found. Run npm install to install Three.js.':'Bad request.');}
  });
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  const host=process.env.HOST||'127.0.0.1',port=Number(process.env.PORT||5173),server=createStaticServer(process.argv[2]);
  server.listen(port,host,()=>console.log(`CUDA WebShader: http://localhost:${port}\nCtrl+C to stop. WebGPU requires localhost or HTTPS.`));
}
