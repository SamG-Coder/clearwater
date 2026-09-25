// Infer a single sampling format for each connected texture-parameter chain.
export function inferTextureTypes(functions,kernel,walk,fail){
 for(const f of functions)if(['texture3d','surface2d'].includes(f.result))fail('Helpers cannot return texture or surface handles.',f);
 const definitions=new Map(functions.map(f=>[f.name,f])),parents=new Map(),uses=[],edges=[];
 const parameters=new Map(functions.map(f=>[f,new Map(f.params.filter(p=>p.type==='texture3d').map(p=>{parents.set(p,p);return [p.name,p];}))]));
 const root=p=>{if(parents.get(p)!==p)parents.set(p,root(parents.get(p)));return parents.get(p);};
 for(const f of functions.filter(f=>f.qualifier!=='__global__'||f===kernel))walk(f.body,n=>{
  if(n.kind!=='call'||n.callee.kind!=='id')return;const name=n.callee.name,params=parameters.get(f);
  if(['tex1D','tex1Dfetch','tex2D','tex3D','tex2DLayered','texCubemap'].includes(name)){const p=n.args[0]?.kind==='id'?params.get(n.args[0].name):null;if(p)uses.push([p,name==='tex2DLayered'&&n.callee.templateArgument==='float'?'tex2DLayeredScalar':name==='tex3D'&&n.callee.templateArgument==='float4'?'tex3Dfloat4':name==='tex1Dfetch'?'fetch_'+n.callee.templateArgument:name==='tex2D'&&n.callee.templateArgument==='uchar2'?'tex2Duchar2':name==='tex2D'&&['float2','float4'].includes(n.callee.templateArgument)?'tex2D'+n.callee.templateArgument:name==='tex2D'&&['uint','unsigned','unsigned int'].includes(n.callee.templateArgument)?'tex2Duint':name==='tex2D'&&['uchar','unsigned char'].includes(n.callee.templateArgument)?'tex2Duchar':name,n]);return;}
  const target=definitions.get(name);if(!target){if(n.args.some(a=>a.kind==='id'&&params.has(a.name)))fail('Texture helper calls require a statically resolved function; supply explicit template arguments.',n);return;}
  target.params.forEach((p,i)=>{if(p.type!=='texture3d')return;const arg=n.args[i],from=arg?.kind==='id'?params.get(arg.name):null;if(from)edges.push([from,p]);});
 });
 for(const [a,b]of edges)parents.set(root(a),root(b));const sampling=new Map();
 for(const [p,kind,node]of uses){const key=root(p);if(sampling.has(key)&&sampling.get(key)!==kind)fail('A texture parameter cannot mix sampling dimensions or formats through helper calls.',node);sampling.set(key,kind);}
 for(const p of parents.keys())p.textureSampling=sampling.get(root(p))||'tex3D';
}
export function textureShape(sampling){return {dimension:['tex2DLayered','tex2DLayeredScalar','texCubemap'].includes(sampling)?'2d-array':['tex3D','tex3Dfloat4'].includes(sampling)?'3d':'2d',format:['tex2Dfloat2','fetch_float2'].includes(sampling)?'rg32float':sampling==='tex2Duchar2'?'rg8uint':sampling==='tex2Duchar'?'r8uint':['fetch_uint','tex2Duint'].includes(sampling)?'r32uint':(sampling==='tex3Dfloat4'||sampling==='tex1D'||sampling==='tex2Dfloat4'||sampling==='tex2DLayered'||sampling==='fetch_float4')?'rgba32float':['tex2D','tex2DLayeredScalar','texCubemap'].includes(sampling)?'r32float':'r8unorm'};}
