// Bounded device allocations. Identities are pool slots, never native addresses.
export function deviceHeaps(emitter) {
 const option=emitter.options.deviceHeap;if(option===undefined)return new Map();
 if(!option||typeof option!=='object'||Array.isArray(option)||Object.keys(option).some(k=>!['maxAllocations','maxElements'].includes(k)))emitter.fail('deviceHeap requires maxAllocations and maxElements.',emitter.kernel);
 const {maxAllocations,maxElements}=option;
 if(!Number.isInteger(maxAllocations)||maxAllocations<1||maxAllocations>65536||!Number.isInteger(maxElements)||maxElements<1||maxElements>65536||maxAllocations*maxElements>16777216)emitter.fail('Device heap dimensions exceed bounded capacity.',emitter.kernel);
 if(!emitter.persistentObjects)emitter.fail('Device heap requires objectHeap: persistent.',emitter.kernel);
 const result=new Map();for(const type of emitter.ast.devicePointerTypes||[]){
  const field=emitter.ast.structs.flatMap(s=>s.fields).find(f=>f.type===type),element=field.pointerElement,layout=emitter.storageLayout(element),stride=Math.ceil(layout.size/layout.align)*layout.align;
  const offset=Math.ceil(maxAllocations*4/layout.align)*layout.align,name=type.replace('cw_deviceptr_','device_');
  result.set(type,{name,type,element,capacity:maxAllocations,maxElements,stride,byteLength:offset+maxAllocations*maxElements*stride,binding:emitter.objectHeaps.size+emitter.objectImports.length+result.size,variable:'cw_'+name,recordLayout:JSON.stringify({element,maxAllocations,maxElements,stride,offset})});
 }return result;
}
export function deviceHeapDeclarations(heaps){return [...heaps.values()].flatMap(h=>[
 `struct CW_${h.name} { lengths: array<atomic<u32>, ${h.capacity}>, values: array<${h.element}, ${h.capacity*h.maxElements}>, }`,
 `@group(1) @binding(${h.binding}) var<storage, read_write> ${h.variable}: CW_${h.name};`,
 `fn cw_allocate_${h.name}(count: u32) -> u32 {
 if (count == 0u || count > ${h.maxElements}u) { return 0u; }
 for (var slot=0u; slot<${h.capacity}u; slot++) {
   loop { let claim=atomicCompareExchangeWeak(&${h.variable}.lengths[slot],0u,count);
     if(claim.exchanged) { return slot+1u; }
     if(claim.old_value!=0u) { break; }
   }
 }
 return 0u;
 }`,
 `fn cw_free_${h.name}(handle: u32) -> i32 {
 if(handle==0u) { return 0i; }
 if(handle>${h.capacity}u) { return 1i; }
 let previous=atomicExchange(&${h.variable}.lengths[handle-1u],0u);
 return select(1i,0i,previous!=0u);
 }`
]);}
export function deviceHeapCall(e,n){
 if(n.callee.kind!=='id'||!['cudaMalloc','cudaFree'].includes(n.callee.name))return;
 if(n.callee.name==='cudaFree'){
  if(n.args.length!==1)e.fail('cudaFree requires one device pointer field.',n);
  const value=e.expr(n.args[0]),heap=e.deviceHeaps.get(value.type);if(!heap)e.fail('cudaFree requires an enabled typed device heap.',n);
  return e.result(n,'i32',`cw_free_${heap.name}(${value.code})`,value.pre);
 }
 if(n.args.length!==2||n.args[0].kind!=='allocation-output'||n.args[0].value.kind!=='unary'||n.args[0].value.op!=='&')e.fail('Device cudaMalloc requires (void **)&pointerField and a byte count.',n);
 const target=e.expr(n.args[0].value.value,true);e.writable(target,n.args[0].value.value);
 const heap=e.deviceHeaps.get(target.type);if(!heap)e.fail('cudaMalloc requires an enabled typed device heap.',n);
 const bytes=e.expr(n.args[1]);if(!['i32','u32','cw_size64'].includes(bytes.type))e.fail('Allocation size must be an integer byte count.',n);
 const id='cw_malloc_'+e.temp++,size=heap.element.startsWith('vec')?Number(heap.element[3])*4:4;
 const low=bytes.type==='cw_size64'?id+'_bytes.x':`u32(${id}_bytes)`,valid=bytes.type==='cw_size64'?`${id}_bytes.y==0u`:bytes.type==='i32'?`${id}_bytes>=0i`:'true';
 return e.result(n,'i32',`select(2i,0i,${id}!=0u || (${valid} && ${low}==0u))`,[...bytes.pre,`let ${id}_bytes=${bytes.code};`,...target.pre,`var ${id}=0u;`,`if(${valid} && ${low}%${size}u==0u) { ${id}=cw_allocate_${heap.name}(${low}/${size}u); }`,`${target.code}=${id};`]);
}
export function deviceHeapIndex(e,n,base,index,raw){
 const heap=e.deviceHeaps.get(base.type);if(!heap)e.fail('Device pointer field dereference requires device heap allocation support.',n);
 if(!['i32','u32'].includes(index.type))e.fail('Device heap indices must be 32-bit integers.',n);
 const id='cw_heap_access_'+e.temp++,pre=[...base.pre,...index.pre,`let ${id}_handle=${base.code};`,`let ${id}_index=u32(${index.code});`,`let ${id}_slot=min(${id}_handle-1u,${heap.capacity-1}u);`,`let ${id}_valid=${id}_handle>0u && ${id}_handle<=${heap.capacity}u && ${id}_index<atomicLoad(&${heap.variable}.lengths[${id}_slot]);`];
 const code=`${heap.variable}.values[${id}_slot*${heap.maxElements}u+min(${id}_index,${heap.maxElements-1}u)]`,rootSymbol={name:heap.name,code:heap.variable+'.values',kind:'local',type:heap.element,constant:false,referenceSpace:'storage',deviceAllocation:true};
 if(raw)return e.result(n,heap.element,code,pre,{rootSymbol,devicePointerGuard:id+'_valid'});
 const value=id+'_value';pre.push(`var ${value}: ${heap.element};`,`if(${id}_valid) { ${value}=${code}; }`);return e.result(n,heap.element,value,pre,{rootSymbol});
}
