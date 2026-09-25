// Each generation reads an immutable frontier and appends to a distinct queue.
// All scheduling stays on the GPU; headers are inspected only after completion.
export async function runRecursiveLaunches(parent,buffers,scalars,workgroups,objectArena){
 const runtime=parent.runtime,artifact=parent.artifact,q=artifact.metadata.deviceLaunchQueue.queues.find(q=>q.caller===artifact.name&&q.frontier);
 if(!q||artifact.children.length!==1)throw Error('Recursive scheduling requires one compiled self-launch child.');
 const child=await runtime.kernel(artifact.children[0].artifact),root=parent.bind(buffers,scalars,{objectArena,queueOnly:true}),types=artifact.metadata.objectHeap.types;
 const outgoing=objectArena.buffers[types.findIndex(t=>t.name===q.name)],frontier=objectArena.buffers[types.findIndex(t=>t.name===q.frontier.name)],scratch=[];
 try{
  const indirect=runtime.createBuffer(q.byteLength,{usage:GPUBufferUsage.INDIRECT,label:'Recursive CUDA launch dimensions'}),status=runtime.createBuffer((q.maxGenerations+1)*8);scratch.push(indirect,status);
  runtime.batch().clear(outgoing).clear(frontier).dispatch(root,workgroups).submit();
  const childBuffers=Object.fromEntries(q.buffers.map(b=>[b.name,buffers[b.parent]]));
  for(let generation=0;generation<q.maxGenerations;generation++){
   runtime.batch().copy(outgoing,status,{byteLength:8,targetOffset:generation*8}).copy(outgoing,frontier).copy(frontier,indirect).clear(outgoing).submit();
   let batch=runtime.batch();
   for(let slot=0;slot<q.capacity;slot++){
    if(batch.cursor+runtime.uniformAlignment>runtime.uniformCapacity){batch.submit();batch=runtime.batch();}
    const invocation=child.bind(childBuffers,{cw_launch_slot:slot},{objectArena,queueOnly:true});
    batch.dispatch(invocation,[1],{resource:indirect,offset:16+slot*q.stride*4});
   }
   batch.submit();
  }
  runtime.batch().copy(outgoing,status,{byteLength:8,targetOffset:q.maxGenerations*8}).submit();await runtime.idle();
  const flags=await runtime.read(status,Uint32Array);
  for(let i=0;i<=q.maxGenerations;i++)if(flags[2*i+1])throw Error('Recursive CUDA queue overflow or invalid launch in generation '+i+'.');
  if(flags[q.maxGenerations*2])throw Error('Recursive CUDA launch generation limit exceeded with pending children.');
  return {maxGenerations:q.maxGenerations,launchCounts:Array.from({length:q.maxGenerations+1},(_,i)=>flags[i*2]),controlReadbackBytes:status.byteLength,gpuScheduled:true};
 }finally{for(const resource of scratch)runtime.destroyBuffer(resource);}
}
