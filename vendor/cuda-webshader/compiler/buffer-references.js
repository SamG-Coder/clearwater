// Captured scalar storage buffers use arena-owned resource IDs and element offsets.
export function addBufferImports(e){
 const types=e.ast.bufferReferenceTypes||[];if(!types.length)return;
 if(!e.persistentObjects)e.fail('Captured scalar buffers require objectHeap: persistent.',e.kernel);
 for(const fn of e.ast.functions.filter(f=>f.qualifier==='__global__'))for(const p of fn.params)if(p.pointer&&types.some(t=>t.element===p.type)){
   if(p.constant)e.fail('Captured scalar buffers currently require mutable pointer parameters.',p);
   const existing=e.objectImports.find(i=>i.name===p.name);if(existing&&existing.type!==p.type)e.fail('Captured buffer names must retain a single type.',p);
   if(!existing)e.objectImports.push({name:p.name,type:p.type,id:e.objectImports.length+1,binding:e.objectHeaps.size+e.objectImports.length,targets:[],scalarBuffer:true});
 }
 if(e.objectImports.length>32)e.fail('At most 32 captured buffers are supported.',e.kernel);
}
export function bufferReferenceArgument(e,a,to,n){
 const element=(e.ast.bufferReferenceTypes||[]).find(t=>t.type===to)?.element,imported=a.rootSymbol?.objectImport;
 if(!element||!imported?.scalarBuffer||imported.type!==element)e.fail('Captured scalar pointer requires a matching arena buffer.',n);
 if(a.rootSymbol.constant)e.fail('Cannot capture a const buffer as mutable.',n);
 const offset=a.pointerCode||'0i',name='cw_capture_'+e.temp++;
 return {code:name,pre:[...a.pre,`let ${name}_offset=i32(${offset});`,`var ${name}: u32 = 0u;`,`if(${name}_offset>=0i && u32(${name}_offset)<1048576u && u32(${name}_offset)<=arrayLength(&cw_import_${imported.id})) { ${name}=${imported.id*1048576}u+u32(${name}_offset); }`]};
}
export function bufferReferenceIndex(e,n,base,index,raw){
 const element=(e.ast.bufferReferenceTypes||[]).find(t=>t.type===base.type)?.element;
 if(!element||!['i32','u32'].includes(index.type))e.fail('Captured buffer indexing requires a scalar element and integer index.',n);
 const name='cw_captured_'+e.temp++,imports=e.objectImports.filter(i=>i.scalarBuffer&&i.type===element);
 const pre=[...base.pre,...index.pre,`let ${name}_handle=${base.code};`,`let ${name}_offset=${name}_handle & 1048575u;`,`let ${name}_raw=${index.code};`,`let ${name}_index=i32(${name}_raw);`,`var ${name}_value: ${element};`];
 const targets=imports.map(i=>({guard:`${index.type==='u32'?`${name}_raw<=2147483647u && `:''}(${name}_handle >> 20u)==${i.id}u && ${name}_index>= -i32(${name}_offset) && ${name}_index < i32(arrayLength(&cw_import_${i.id}))-i32(${name}_offset)`,code:`cw_import_${i.id}[u32(i32(${name}_offset)+${name}_index)]`}));
 if(!imports.length)e.fail('Captured buffer has no registered resource.',n);
 for(const t of targets)pre.push(`if(${t.guard}) { ${name}_value=${t.code}; }`);
 return e.result(n,element,name+'_value',pre,{rootSymbol:{kind:'local',name,constant:false,deviceAllocation:true},...(raw?{bufferReferenceTargets:targets}:{})});
}
