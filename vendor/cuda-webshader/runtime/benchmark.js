/** Honest timings: GPU timestamp queries when available; otherwise labeled submit-to-completion wall time. */
import {kernelOptions} from '../kernels.js';
import {compareArrays,randomFloats} from '../../tests/cases.js';
const median=a=>{const s=[...a].sort((a,b)=>a-b);return s.length%2?s[(s.length-1)/2]:(s[s.length/2-1]+s[s.length/2])/2;};
export async function measure(runtime,invocation,groups,{samples=5,targetBatchMs=6,maxIterations=2048,warmup=3}={}){
  if(!Number.isInteger(samples)||samples<3||!Number.isInteger(maxIterations)||maxIterations<1)throw new RangeError('Invalid benchmark sample configuration.');
  const device=runtime.device,hasTimestamp=device.features.has('timestamp-query');
  let querySet,resolve,readback;
  if(hasTimestamp){querySet=device.createQuerySet({type:'timestamp',count:2});resolve=device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});readback=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});}
  try{
    const warm=runtime.batch();for(let i=0;i<warmup;i++)warm.dispatch(invocation,groups);warm.submit();await runtime.idle();
    // Calibrate with GPU time: a single submit's wall latency can dwarf kernel
    // execution and previously selected batches below timestamp resolution.
    const probeIterations=Math.min(128,maxIterations),probeStart=performance.now();
    const probe=runtime.batch(hasTimestamp?{timestampWrites:{querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});
    for(let i=0;i<probeIterations;i++)probe.dispatch(invocation,groups);
    probe.endPass();
    if(hasTimestamp){probe.encoder.resolveQuerySet(querySet,0,2,resolve,0);probe.encoder.copyBufferToBuffer(resolve,0,readback,0,16);}
    probe.submit();await runtime.idle();let probeMs=(performance.now()-probeStart)/probeIterations;
    if(hasTimestamp){await readback.mapAsync(GPUMapMode.READ);const values=new BigUint64Array(readback.getMappedRange());probeMs=Number(values[1]-values[0])/1e6/probeIterations;readback.unmap();}
    const iterations=Math.max(1,Math.min(maxIterations,Math.ceil(targetBatchMs/Math.max(probeMs,0.000001))));const timings=[];
    for(let s=0;s<samples;s++){
      const start=performance.now(),batch=runtime.batch(hasTimestamp?{timestampWrites:{querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});
      for(let i=0;i<iterations;i++)batch.dispatch(invocation,groups);
      batch.endPass();
      if(hasTimestamp){batch.encoder.resolveQuerySet(querySet,0,2,resolve,0);batch.encoder.copyBufferToBuffer(resolve,0,readback,0,16);}
      batch.submit();await runtime.idle();const wallBatchMs=performance.now()-start;let gpuBatchMs=null;
      if(hasTimestamp){await readback.mapAsync(GPUMapMode.READ);const values=new BigUint64Array(readback.getMappedRange());const elapsed=values[1]-values[0];gpuBatchMs=elapsed>0n?Number(elapsed)/1e6:null;readback.unmap();}
      timings.push({wallBatchMs,gpuBatchMs,wallPerDispatchMs:wallBatchMs/iterations,gpuPerDispatchMs:gpuBatchMs===null?null:gpuBatchMs/iterations});
    }
    const validGpu=timings.every(t=>t.gpuPerDispatchMs!==null),gpuMedianMs=validGpu?median(timings.map(t=>t.gpuPerDispatchMs)):null,wallMedianMs=median(timings.map(t=>t.wallPerDispatchMs));
    return {iterations,samples,warmup,timestampQuery:hasTimestamp,gpuMedianMs,wallMedianMs,metric:validGpu?'GPU timestamp (amortized per dispatch)':'submit-to-completion wall time (amortized per dispatch)',medianMs:gpuMedianMs??wallMedianMs,timings,notes:['Compilation, allocation and initial data upload are outside the timed section.','Wall time includes JS encoding, uniform uploads, submission and completion waiting.','GPU timestamps include compute-pass execution; privacy quantization and caches affect short workloads.','Repeated dispatches reuse buffers. Effective bandwidth is a logical byte-rate, not measured DRAM traffic.']};
  }finally{querySet?.destroy();resolve?.destroy();readback?.destroy();}
}
export async function runBenchmarks(runtime,sources,{n=1<<20,matrixSize=256,onResult=()=>{},samples=5}={}){
  if(!Number.isSafeInteger(n)||n<4||n%4||n>4*1024*1024)throw new RangeError('n must be a multiple of four in [4,4194304].');
  if(!Number.isInteger(matrixSize)||matrixSize<1||matrixSize>512)throw new RangeError('Matrix size must be in [1,512].');
  const results=[],resources=[],skipped=[];const keep=x=>(resources.push(x),x);
  const emit=r=>{results.push(r);onResult(r);};
  try{
    const xData=randomFloats(n,31),yData=randomFloats(n,17),x=keep(runtime.createBuffer(xData)),y=keep(runtime.createBuffer(yData)),a=0.125;
    const expected=Float32Array.from(yData,(v,i)=>v+a*xData[i]);
    for(const id of ['saxpy','saxpy_vec4'])for(const block of [64,128,256]){
      if(block>runtime.device.limits.maxComputeInvocationsPerWorkgroup){skipped.push({id,block,reason:'Workgroup invocation limit'});continue;}
      const recordCount=id==='saxpy'?n:n/4;
      if(Math.ceil(recordCount/block)>runtime.device.limits.maxComputeWorkgroupsPerDimension){skipped.push({id,block,reason:'Workgroup dispatch count limit'});continue;}
      runtime.write(y,yData);const kernel=await runtime.kernel(sources[id],{entry:id,workgroupSize:[block,1,1]});const records=id==='saxpy'?n:n/4;
      const inv=kernel.bind({x,y},id==='saxpy'?{a,n}:{a,n4:records}),groups=[Math.ceil(records/block)];
      runtime.batch().dispatch(inv,groups).submit();const verified=compareArrays(await runtime.read(y),expected);
      if(!verified.pass)throw new Error(`${id}/${block} failed validation before benchmarking.`);
      runtime.write(y,yData);const timing=await measure(runtime,inv,groups,{samples});
      emit({name:`${id} / ${block}`,family:'saxpy',id,workgroupSize:[block,1,1],elements:n,verified:true,...timing,effectiveGBs:(n*12)/(timing.medianMs*1e6),gflops:(2*n)/(timing.medianMs*1e6)});
    }
    const M=matrixSize,N=matrixSize,K=matrixSize,Adata=randomFloats(M*K,234),Bdata=randomFloats(K*N,879),A=keep(runtime.createBuffer(Adata)),B=keep(runtime.createBuffer(Bdata)),C=keep(runtime.createBuffer(M*N*4));
    const reference=new Float32Array(M*N);
    for(let r=0;r<M;r++)for(let c=0;c<N;c++){let sum=0;for(let k=0;k<K;k++)sum+=Adata[r*K+k]*Bdata[k*N+c];reference[r*N+c]=sum;}
    for(const id of ['matmul_naive','matmul_tiled','matmul_register']){
      const kernel=await runtime.kernel(sources[id],kernelOptions(id)),tile=id==='matmul_naive'?8:16,groups=[Math.ceil(N/tile),Math.ceil(M/tile)],inv=kernel.bind({A,B,C},{M,N,K});
      runtime.batch().dispatch(inv,groups).submit();const verified=compareArrays(await runtime.read(C),reference,{absolute:0.001,relative:0.001});
      if(!verified.pass)throw new Error(`${id} failed validation before benchmarking: ${JSON.stringify(verified)}`);
      const timing=await measure(runtime,inv,groups,{samples,maxIterations:512});
      emit({name:id,family:'matmul',id,workgroupSize:kernel.artifact.metadata.workgroupSize,shape:[M,N,K],verified:true,...timing,gflops:2*M*N*K/(timing.medianMs*1e6)});
    }
    const choices={};
    for(const family of ['saxpy','matmul']){
      const variants=results.filter(r=>r.family===family),allGpu=variants.every(r=>r.gpuMedianMs!==null),field=allGpu?'gpuMedianMs':'wallMedianMs';variants.sort((a,b)=>a[field]-b[field]);
      choices[family]={id:variants[0].id,workgroupSize:variants[0].workgroupSize,rankingMetric:field,medianMs:variants[0][field],baselineSpeedup:family==='matmul'?results.find(r=>r.id==='matmul_naive')[field]/variants[0][field]:null};
    }
    return {schema:'cuda-webshader.benchmark.v1',mode:'REAL measured WebGPU; not estimated CUDA speed',date:new Date().toISOString(),device:runtime.describe(),n,matrixSize,results,choices,skipped};
  }finally{resources.forEach(r=>runtime.destroyBuffer(r));}
}
