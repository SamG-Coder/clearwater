// Explicit ownership for object storage shared by multiple compiled kernels.
export class ObjectArena {
  constructor(runtime){this.runtime=runtime;this.buffers=[];this.signature=null;this.imports=new Map();this.disposed=false;}
  bind(kernel,resources={}){
    this.assertAlive();if(kernel.runtime!==this.runtime)throw Error('Object arena belongs to another runtime.');
    const types=kernel.artifact.metadata.objectHeap?.types;if(!kernel.objectLayout||!types)throw Error('Kernel does not use a persistent object arena.');
    const definitions=kernel.artifact.metadata.objectHeap.imports||[],imports=new Map(this.imports);
    for(const definition of definitions){const resource=resources[definition.name]||imports.get(definition.name);if(!resource)throw Error('Missing captured object buffer '+definition.name);this.runtime.checkResource(resource);if(!resource.gpuBuffer||resource.size%4||resource.size>4194304)throw Error('Captured object buffers require at most 1,048,576 aligned tokens.');if(resource.objectArena&&resource.objectArena!==this)throw Error('Object pointer buffer belongs to another arena.');if(imports.has(definition.name)&&imports.get(definition.name)!==resource)throw Error('Captured object buffers cannot be rebound while the arena is alive.');imports.set(definition.name,resource);}
    const allBuffers=[...imports.values()].map(r=>r.gpuBuffer);if(new Set(allBuffers).size!==allBuffers.length||kernel.artifact.metadata.bindings.some(b=>allBuffers.includes(resources[b.name]?.gpuBuffer)))throw Error('Captured object buffer aliases another binding.');
    const signature=JSON.stringify({types,definitions});if(this.signature&&this.signature!==signature)throw Error('Object arena layout or type tags do not match this kernel.');
    if(!this.signature){
      for(const t of types)if(t.byteLength>this.runtime.device.limits.maxStorageBufferBindingSize)throw Error('Object pool exceeds the storage binding limit.');
      try{for(const t of types)this.buffers.push(this.runtime.createBuffer(t.byteLength,{label:'Object pool '+t.name}));this.signature=signature;}catch(error){for(const b of this.buffers)this.runtime.destroyBuffer(b);this.buffers=[];throw error;}
    }
    this.imports=imports;for(const resource of imports.values())resource.objectArena=this;
    const group=this.runtime.device.createBindGroup({layout:kernel.objectLayout,entries:[...types.map((t,i)=>({binding:t.binding,resource:{buffer:this.buffers[i].gpuBuffer}})),...definitions.map(d=>({binding:d.binding,resource:{buffer:imports.get(d.name).gpuBuffer}}))]});this.runtime.stats.bindGroupsCreated++;return group;
  }
  assertAlive(){this.runtime.assertAlive();if(this.disposed)throw Error('Object arena is disposed.');for(const b of [...this.buffers,...this.imports.values()])this.runtime.checkResource(b);}
  dispose(){if(this.disposed)return;for(const b of this.buffers)if(!b.destroyed)this.runtime.destroyBuffer(b);this.buffers=[];this.disposed=true;}
}
