/** Reusable, GPU-resident plans. Allocate/compile once, then encode into caller-owned batches. */
import {kernelOptions} from '../kernels.js';
function dimension(value,name){if(!Number.isSafeInteger(value)||value<0||value>0xffffffff)throw new RangeError(`${name} must be a u32.`);return value;}
function fit(runtime,resource,elements,name){runtime.checkResource(resource);if(!Number.isSafeInteger(elements)||elements*4>resource.byteLength)throw new RangeError(`${name} does not contain ${elements} float32 elements.`);}
function launchFits(runtime,groups){if(groups.some(n=>n>runtime.device.limits.maxComputeWorkgroupsPerDimension))throw new RangeError('Plan exceeds the device dispatch limit. Split the workload.');}
export async function prepareSaxpy(runtime,sources,{x,y,n,a=1,choice}={}){
  dimension(n,'n');fit(runtime,x,n,'x');fit(runtime,y,n,'y');
  let id=choice?.id || 'saxpy';if(!['saxpy','saxpy_vec4'].includes(id))throw new Error('Invalid SAXPY variant.');
  // The vec4 ABI requires complete aligned records. A measured winner is a hint, not permission to overrun tails.
  if(id==='saxpy_vec4'&&(n%4||x.size%16||y.size%16))id='saxpy';
  const block=choice?.workgroupSize?.[0] || 128;
  if(!Number.isInteger(block)||block<1)throw new RangeError('Invalid workgroup size.');
  const records=id==='saxpy_vec4'?n/4:n,groups=[Math.ceil(records/block),1,1];launchFits(runtime,groups);
  const kernel=await runtime.kernel(sources[id],{entry:id,workgroupSize:[block,1,1]});
  const invocation=kernel.bind({x,y},id==='saxpy_vec4'?{a,n4:records}:{a,n});
  return {id,groups,invocation,encode(batch,nextA){if(nextA!==undefined)invocation.setScalars({a:nextA});batch.dispatch(invocation,groups);return batch;}};
}
export async function prepareMatmul(runtime,sources,{A,B,C,M,N,K,choice}={}){
  for(const [name,value]of Object.entries({M,N,K}))dimension(value,name);
  fit(runtime,A,M*K,'A');fit(runtime,B,K*N,'B');fit(runtime,C,M*N,'C');
  const id=choice?.id || 'matmul_tiled';if(!['matmul_naive','matmul_tiled','matmul_register'].includes(id))throw new Error('Invalid matrix variant.');
  const tile=id==='matmul_naive'?8:16,groups=[Math.ceil(N/tile),Math.ceil(M/tile),1];launchFits(runtime,groups);
  const kernel=await runtime.kernel(sources[id],kernelOptions(id)),invocation=kernel.bind({A,B,C},{M,N,K});
  return {id,groups,invocation,encode(batch){return batch.dispatch(invocation,groups);}};
}
export async function prepareReduction(runtime,sources,{input,n}={}){
  dimension(n,'n');fit(runtime,input,n,'input');
  const kernel=await runtime.kernel(sources.reduce_sum,kernelOptions('reduce_sum'));
  const scratch=[],levels=[];let current=input,length=n,disposed=false;
  try{
    // Even n=0/1 produces a separate, owned scalar output with deterministic zero/singleton semantics.
    do{
      const groups=Math.max(1,Math.ceil(length/256));launchFits(runtime,[groups]);
      const output=runtime.createBuffer(groups*4,{label:`reduction level ${levels.length}`});scratch.push(output);
      levels.push({invocation:kernel.bind({input:current,output},{n:length}),groups:[groups,1,1]});current=output;length=groups;
    }while(length>1);
  }catch(error){scratch.forEach(r=>runtime.destroyBuffer(r));throw error;}
  return {output:current,levels:levels.length,
    encode(batch){if(disposed)throw new Error('Reduction plan is disposed.');for(const level of levels)batch.dispatch(level.invocation,level.groups);return batch;},
    dispose(){if(disposed)return;disposed=true;scratch.forEach(r=>runtime.destroyBuffer(r));}
  };
}
