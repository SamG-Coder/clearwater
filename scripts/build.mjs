import {cp, mkdir, readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
const root=path.resolve(import.meta.dirname,'..'),out=path.join(root,'dist');
await mkdir(out,{recursive:true});
const modules=new Set();
async function moduleGraph(file){
 file=path.resolve(file);if(modules.has(file))return;modules.add(file);
 const source=await readFile(file,'utf8');
 for(const match of source.matchAll(/(?:from\s*|import\s*\()\s*['"]([^'"]+)['"]/g)){
  if(!match[1].startsWith('.'))throw Error(`External browser import: ${match[1]}`);
  await moduleGraph(path.resolve(path.dirname(file),match[1]));
 }
}
await moduleGraph(path.join(root,'app.js'));
for(const file of [...modules,...['index.html','style.css','src/clearwater.cu','assets/seabed.jpg','LICENSE','THIRD_PARTY_NOTICES.md','vendor/cuda-webshader/LICENSE'].map(f=>path.join(root,f))]){
 const relative=path.relative(root,file);if(relative.startsWith('..'))throw Error('Asset outside project');
 await mkdir(path.dirname(path.join(out,relative)),{recursive:true});await cp(file,path.join(out,relative));
}
await writeFile(path.join(out,'.nojekyll'),'');
console.log(`Built Pages with ${modules.size} browser modules, shared CUDA source and licensed assets.`);
