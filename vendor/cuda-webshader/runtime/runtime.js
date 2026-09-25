import {ObjectArena} from './object-arena.js';
import {runRecursiveLaunches} from './recursive-launches.js';
import {SORT_SOURCE,SORT_FLOAT_SOURCE} from './sort-kernels.js';
import {FFT_SOURCE,FORWARD_FFT_SOURCE,REAL_FFT_SOURCE} from './fft-kernels.js';
import {SCAN_SOURCE} from './scan-kernels.js';
/** WebGPU runtime: cached pipelines/bindings, batched dispatch and a per-batch uniform snapshot arena. */
import {compile} from '../compiler/compiler.js';
const roundUp = (n, alignment) => Math.ceil(n / alignment) * alignment;
let resourceId = 0;
export function packScalars(metadata, values, target = new ArrayBuffer(metadata.uniformSize)) {
  if (target.byteLength < metadata.uniformSize) throw new RangeError('Uniform destination is too small.');
  const known = new Set(metadata.scalars.map(s => s.name));
  for (const name of Object.keys(values)) if (!known.has(name)) throw new Error(`Unknown scalar parameter '${name}'.`);
  const writes = [];
  for (const p of metadata.scalars) {
    let v = Object.hasOwn(values,p.name)?values[p.name]:p.defaultValue;
    if(p.sourceType==='bool'){if(![true,false,0,1].includes(v))throw new TypeError(`Scalar '${p.name}' must be a boolean or 0/1.`);v=Number(v);}
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new TypeError(`Scalar '${p.name}' must be finite.`);
    if(['cw_short','cw_ushort'].includes(p.sourceType)&&(!Number.isInteger(v)||v<(p.sourceType==='cw_short'?-32768:0)||v>(p.sourceType==='cw_short'?32767:65535)))throw new RangeError(`${p.name} is outside its 16-bit range.`);
    if (p.type === 'u32') { if (!Number.isInteger(v) || v < 0 || v > 0xffffffff) throw new RangeError(`${p.name} is not a u32.`); }
    else if (p.type === 'i32') { if (!Number.isInteger(v) || v < -2147483648 || v > 2147483647) throw new RangeError(`${p.name} is not an i32.`); }
    else if (!Number.isFinite(Math.fround(v))) throw new RangeError(`${p.name} overflows f32.`);
    writes.push([p,v]);
  }
  // Validate the complete update before touching the destination: failed updates are transactional.
  for(const c of metadata.scalarConstraints||[]){const v=values[c.name];if(!Number.isInteger(v)||v<c.minimum||v%c.multipleOf!==0)throw new RangeError(`${c.name} must be a nonnegative multiple of ${c.multipleOf} for full-workgroup execution.`);}
  const view = new DataView(target);
  for(const [p,v] of writes) {
    if(p.type === 'u32') view.setUint32(p.offset,v,true);
    else if(p.type === 'i32') view.setInt32(p.offset,v,true);
    else view.setFloat32(p.offset,v,true);
  }
  for(const scale of metadata.textureScales||[]){view.setFloat32(scale.offset,1,true);view.setFloat32(scale.offset+4,1,true);if(scale.dimension==='3d')view.setFloat32(scale.offset+8,1,true);if(scale.pointOffset!==undefined)view.setFloat32(scale.pointOffset,0,true);}
  return target;
}
export function validateWorkgroup(metadata, limits) {
  const size = metadata.workgroupSize;
  for (let i = 0; i < 3; i++) if (size[i] > limits[['maxComputeWorkgroupSizeX','maxComputeWorkgroupSizeY','maxComputeWorkgroupSizeZ'][i]]) throw new Error(`Workgroup ${size} exceeds a device dimension limit.`);
  if (size.reduce((a,b)=>a*b,1)>limits.maxComputeInvocationsPerWorkgroup) throw new Error(`Workgroup ${size} exceeds maxComputeInvocationsPerWorkgroup.`);
  if (metadata.workgroupStorageBytes > limits.maxComputeWorkgroupStorageSize) throw new Error(`Kernel requires ${metadata.workgroupStorageBytes} workgroup bytes; device supports ${limits.maxComputeWorkgroupStorageSize}.`);
  if (metadata.uniformSize > limits.maxUniformBufferBindingSize) throw new Error('Uniform parameters exceed the device binding limit.');
  if((metadata.textures?.length||0)>limits.maxSampledTexturesPerShaderStage||(metadata.textures?.length||0)>limits.maxSamplersPerShaderStage)throw Error('Too many sampled textures for this device.');
  if (metadata.bindings.length + (metadata.objectHeap?.persistent?metadata.objectHeap.types.length+(metadata.objectHeap.imports?.length||0):0) > limits.maxStorageBuffersPerShaderStage) throw new Error('Too many storage buffers for this device.');
}
export class GpuRuntime {
  static async create(options = {}) {
    if(options.useAdapterWorkgroupLimits !== undefined && typeof options.useAdapterWorkgroupLimits !== 'boolean') throw new TypeError('useAdapterWorkgroupLimits must be a boolean.');
    if(options.useAdapterBufferLimits !== undefined && typeof options.useAdapterBufferLimits !== 'boolean') throw new TypeError('useAdapterBufferLimits must be a boolean.');
    if (!globalThis.navigator?.gpu && !options.device) throw new Error('WebGPU is required. Open this project on localhost or HTTPS in a WebGPU-capable browser. WebGL cannot run these kernels.');
    const adapter = options.adapter || (!options.device ? await navigator.gpu.requestAdapter({powerPreference:'high-performance'}) : null);
    if (!adapter && !options.device) throw new Error('No WebGPU adapter is available. Check the browser GPU settings and graphics driver.');
    const features = ['shader-f16','timestamp-query','core-features-and-limits','float32-filterable','subgroups','subgroup-size-control'].filter(f => adapter?.features.has(f));
    const requiredLimits = adapter ? {
      ...(options.useAdapterWorkgroupLimits?{maxComputeWorkgroupStorageSize:adapter.limits.maxComputeWorkgroupStorageSize}:{}),
      maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage,16),
      maxComputeInvocationsPerWorkgroup: Math.min(adapter.limits.maxComputeInvocationsPerWorkgroup, 1024),
      maxComputeWorkgroupSizeX: Math.min(adapter.limits.maxComputeWorkgroupSizeX,1024),
      maxStorageBufferBindingSize: options.useAdapterBufferLimits ? adapter.limits.maxStorageBufferBindingSize : Math.min(adapter.limits.maxStorageBufferBindingSize, 256 * 1024 * 1024),
      maxBufferSize: options.useAdapterBufferLimits ? adapter.limits.maxBufferSize : Math.min(adapter.limits.maxBufferSize, 256 * 1024 * 1024)
    } : undefined;
    const device = options.device || await adapter.requestDevice({requiredFeatures:features, requiredLimits});
    return new GpuRuntime(device, {...options,adapter,ownsDevice:!options.device});
  }
  constructor(device, options = {}) {
    this.device=device; this.adapter=options.adapter; this.ownsDevice=options.ownsDevice ?? false;
    this.disposed=false; this.lost=null; this.onError=options.onError || (error=>console.error(error));
    this.buffers=new Set();this.textures=new Set(); this.pipelineCache=new Map(); this.pipelineQueue=Promise.resolve();
    this.uniformAlignment=device.limits.minUniformBufferOffsetAlignment;
    this.uniformCapacity=roundUp(options.uniformCapacity || 65536,this.uniformAlignment);
    this.uniformBuffer=device.createBuffer({label:'CUDA WebShader uniform snapshot arena',size:this.uniformCapacity,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.batchMemory=[];
    this.stats={pipelineCompiles:0,pipelineCacheHits:0,bindGroupsCreated:0,submissions:0,dispatches:0,uniformBytesUploaded:0,dataBytesUploaded:0,readbackBytes:0};
    this.errorListener=e=>this.onError(e.error || e);
    device.addEventListener('uncapturederror',this.errorListener);
    device.lost.then(info=>{this.lost=info;if(info.reason!=='destroyed'&&!this.disposed)this.onError(new Error(`GPU device lost: ${info.message}. Reload to recreate GPU resources.`));});
  }
  assertAlive() { if(this.disposed)throw new Error('Runtime is disposed.');if(this.lost)throw new Error(`GPU device lost: ${this.lost.message}`); }
  describe() {
    const info=this.adapter?.info;
    return {vendor:info?.vendor || 'not exposed',architecture:info?.architecture || '',device:info?.device || '',description:info?.description || '',features:[...this.device.features],timestampQuery:this.device.features.has('timestamp-query'),limits:{maxComputeInvocationsPerWorkgroup:this.device.limits.maxComputeInvocationsPerWorkgroup,maxComputeWorkgroupStorageSize:this.device.limits.maxComputeWorkgroupStorageSize,maxStorageBufferBindingSize:this.device.limits.maxStorageBufferBindingSize}};
  }
  createTexture2D(data,{width,height,filter='linear',addressMode='repeat',label='CUDA float texture',storage=false,normalizedCoords=true,format='r32float'}={}){
    this.assertAlive();if(!['r32float','rg32float','rgba32float'].includes(format)||format==='rg32float'&&storage)throw Error('Unsupported 2D texture format or storage format.');const components=format==='rgba32float'?4:format==='rg32float'?2:1;if(![width,height].every(n=>Number.isInteger(n)&&n>0&&n<=this.device.limits.maxTextureDimension2D)||!(data===null&&storage)&&(!(data instanceof Float32Array)||data.length!==width*height*components||data.some(v=>!Number.isFinite(v))))throw new RangeError('2D texture requires finite floats matching width and height within device limits.');
    if(typeof normalizedCoords!=='boolean'||(!normalizedCoords&&addressMode!=='clamp-to-edge'))throw Error('Unnormalized 2D textures require clamp-to-edge addressing.');
    if(!this.device.features.has('float32-filterable'))throw Error('Float textures require float32-filterable on this device.');if(!['linear','nearest'].includes(filter)||!['repeat','clamp-to-edge','mirror-repeat'].includes(addressMode))throw Error('Unsupported texture sampler settings.');
    const gpuTexture=this.device.createTexture({label,size:[width,height,1],dimension:'2d',format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC|(storage?GPUTextureUsage.STORAGE_BINDING:0)});if(data)this.device.queue.writeTexture({texture:gpuTexture},data,{bytesPerRow:width*4*components,rowsPerImage:height},[width,height,1]);const resource={id:++resourceId,runtime:this,owned:true,destroyed:false,gpuTexture,view:gpuTexture.createView(),sampler:this.device.createSampler({minFilter:filter,magFilter:filter,addressModeU:addressMode,addressModeV:addressMode}),format,dimension:'2d',width,height,depth:1,storage,normalizedCoords,filter};this.textures.add(resource);this.stats.dataBytesUploaded+=data?.byteLength||0;return resource;
  }
  createCubemapTexture(data,{width,filter='linear',addressMode='clamp-to-edge',seamless=false,label='CUDA scalar cubemap'}={}) {
    this.assertAlive();
    if(!Number.isInteger(width)||width<1||width>this.device.limits.maxTextureDimension2D||!(data instanceof Float32Array)||data.length!==width*width*6||data.some(v=>!Number.isFinite(v)))throw Error('Cubemap requires six square finite float faces.');
    if(seamless!==false||!['repeat','clamp-to-edge','mirror-repeat'].includes(addressMode)||!['linear','nearest'].includes(filter)||!this.device.features.has('float32-filterable'))throw Error('Cubemap supports nonseamless linear or nearest float filtering.');
    const gpuTexture=this.device.createTexture({label,size:[width,width,6],dimension:'2d',format:'r32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC});
    this.device.queue.writeTexture({texture:gpuTexture},data,{bytesPerRow:width*4,rowsPerImage:width},[width,width,6]);
    const resource={id:++resourceId,runtime:this,owned:true,destroyed:false,gpuTexture,view:gpuTexture.createView({dimension:'2d-array'}),sampler:this.device.createSampler({minFilter:filter,magFilter:filter,addressModeU:addressMode,addressModeV:addressMode}),format:'r32float',dimension:'2d-array',width,height:width,depth:6,cubemap:true,filter,normalizedCoords:true};
    this.textures.add(resource);this.stats.dataBytesUploaded+=data.byteLength;return resource;
  }
  createLayeredTexture2D(data,{width,height,layers,filter='linear',addressMode='clamp-to-edge',storage=false,format='rgba32float',label='CUDA float4 layered texture'}={}) {
    this.assertAlive();if(!['r32float','rgba32float'].includes(format))throw Error('Layered texture requires scalar or float4 format.');const components=format==='r32float'?1:4;
    if(![width,height].every(n=>Number.isInteger(n)&&n>0&&n<=this.device.limits.maxTextureDimension2D)||!Number.isInteger(layers)||layers<1||layers>this.device.limits.maxTextureArrayLayers)throw Error('Layered texture dimensions exceed device limits.');
    if(!(data===null&&storage)&&(!(data instanceof Float32Array)||data.length!==width*height*layers*components||data.some(v=>!Number.isFinite(v))))throw Error('Layered texture requires matching finite scalar or float4 data.');
    if(!this.device.features.has('float32-filterable')||!['linear','nearest'].includes(filter)||!['repeat','clamp-to-edge','mirror-repeat'].includes(addressMode))throw Error('Unsupported layered texture sampling settings.');
    const gpuTexture=this.device.createTexture({label,size:[width,height,layers],dimension:'2d',format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC|(storage?GPUTextureUsage.STORAGE_BINDING:0)});
    if(data)this.device.queue.writeTexture({texture:gpuTexture},data,{bytesPerRow:width*components*4,rowsPerImage:height},[width,height,layers]);
    const resource={id:++resourceId,runtime:this,owned:true,destroyed:false,gpuTexture,view:gpuTexture.createView({dimension:'2d-array'}),sampler:this.device.createSampler({minFilter:filter,magFilter:filter,addressModeU:addressMode,addressModeV:addressMode}),format,dimension:'2d-array',width,height,depth:layers,storage,normalizedCoords:true,filter};
    this.textures.add(resource);this.stats.dataBytesUploaded+=data?.byteLength||0;return resource;
  }
  createByteTexture2D(data,{width,height,components=1,label='CUDA byte element texture'}={}) {
    this.assertAlive();
    if(![1,2].includes(components)||![width,height].every(n=>Number.isInteger(n)&&n>0&&n<=this.device.limits.maxTextureDimension2D)||!(data instanceof Uint8Array)||data.length!==width*height*components)throw new RangeError('Byte texture requires Uint8Array matching valid width and height.');
    const format=components===2?'rg8uint':'r8uint';
    const gpuTexture=this.device.createTexture({label,size:[width,height,1],dimension:'2d',format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    this.device.queue.writeTexture({texture:gpuTexture},data,{bytesPerRow:width*components,rowsPerImage:height},[width,height,1]);
    const resource={id:++resourceId,runtime:this,owned:true,destroyed:false,gpuTexture,view:gpuTexture.createView(),sampler:this.device.createSampler({minFilter:'nearest',magFilter:'nearest'}),format,dimension:'2d',width,height,depth:1,normalizedCoords:false,filter:'nearest',addressMode:'clamp-to-edge'};
    this.textures.add(resource);this.stats.dataBytesUploaded+=data.byteLength;return resource;
  }
  createUintTexture2D(data,{width,height,label='CUDA unsigned element texture'}={}) {
    this.assertAlive();
    if(![width,height].every(n=>Number.isInteger(n)&&n>0&&n<=this.device.limits.maxTextureDimension2D)||!(data instanceof Uint32Array)||data.length!==width*height)throw new RangeError('Uint texture requires Uint32Array matching valid width and height.');
    const gpuTexture=this.device.createTexture({label,size:[width,height,1],dimension:'2d',format:'r32uint',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    this.device.queue.writeTexture({texture:gpuTexture},data,{bytesPerRow:width*4,rowsPerImage:height},[width,height,1]);
    const resource={id:++resourceId,runtime:this,owned:true,destroyed:false,gpuTexture,view:gpuTexture.createView(),sampler:this.device.createSampler({minFilter:'nearest',magFilter:'nearest'}),format:'r32uint',dimension:'2d',width,height,depth:1,normalizedCoords:false,filter:'nearest',addressMode:'clamp-to-edge'};
    this.textures.add(resource);this.stats.dataBytesUploaded+=data.byteLength;return resource;
  }
  createTexture1D(data,{filter='linear',addressMode='clamp-to-edge',label='CUDA float4 transfer texture'}={}){
    this.assertAlive();if(!(data instanceof Float32Array)||!data.length||data.length%4||data.length/4>this.device.limits.maxTextureDimension2D||data.some(v=>!Number.isFinite(v)))throw new RangeError('1D texture requires finite float4 records within device width limits.');if(!this.device.features.has('float32-filterable'))throw Error('Float4 transfer textures require float32-filterable on this device.');if(!['linear','nearest'].includes(filter)||!['repeat','clamp-to-edge','mirror-repeat'].includes(addressMode))throw Error('Unsupported texture sampler settings.');const width=data.length/4,gpuTexture=this.device.createTexture({label,size:[width,1,1],dimension:'2d',format:'rgba32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});this.device.queue.writeTexture({texture:gpuTexture},data,{bytesPerRow:width*16,rowsPerImage:1},[width,1,1]);const resource={id:++resourceId,runtime:this,owned:true,destroyed:false,gpuTexture,view:gpuTexture.createView(),sampler:this.device.createSampler({minFilter:filter,magFilter:filter,addressModeU:addressMode,addressModeV:'clamp-to-edge',addressModeW:'clamp-to-edge'}),format:'rgba32float',dimension:'2d',width,height:1,depth:1};this.textures.add(resource);this.stats.dataBytesUploaded+=data.byteLength;return resource;
  }
  createLinearTexture(data,{label='CUDA linear texture',components=1}={}) {
    this.assertAlive();const floating=data instanceof Float32Array;if(!floating&&!(data instanceof Uint8Array)&&!(data instanceof Uint32Array)||!Number.isInteger(components)||!(floating?[1,2,4]:[1]).includes(components)||data.length%components||floating&&data.some(v=>!Number.isFinite(v)))throw Error('Linear textures require bytes, uint elements or finite float records with 1, 2 or 4 components.');
    const limit=this.device.limits.maxTextureDimension2D,length=data.length/components;if(!length||length>Math.min(limit*limit,0x7fffffff))throw new RangeError('Linear texture length exceeds device dimensions.');
    const width=Math.min(length,limit),height=Math.ceil(length/width),format=floating?({1:'r32float',2:'rg32float',4:'rgba32float'}[components]):data instanceof Uint32Array?'r32uint':'r8unorm',upload=new data.constructor(width*height*components);upload.set(data);
    const gpuTexture=this.device.createTexture({label,size:[width,height,1],dimension:'2d',format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});this.device.queue.writeTexture({texture:gpuTexture},upload,{bytesPerRow:width*components*data.BYTES_PER_ELEMENT,rowsPerImage:height},[width,height,1]);
    const resource={id:++resourceId,runtime:this,owned:true,destroyed:false,gpuTexture,view:gpuTexture.createView(),sampler:this.device.createSampler({minFilter:'nearest',magFilter:'nearest'}),dimension:'2d',format,width,height,depth:1,linearLength:length};this.textures.add(resource);this.stats.dataBytesUploaded+=upload.byteLength;return resource;
  }
  createTexture3D(data,{width,height,depth,filter='linear',addressMode='repeat',label='CUDA 3D texture',format='r8unorm',storage=false,normalizedCoords=true}={}){
    this.assertAlive();if(!['r8unorm','r32float','rgba8unorm','rgba32float'].includes(format)||storage&&!['r32float','rgba8unorm'].includes(format))throw Error('Unsupported 3D texture format or storage format.');
    if(typeof normalizedCoords!=='boolean'||!normalizedCoords&&addressMode!=='clamp-to-edge')throw Error('Unnormalized 3D texture coordinates require clamp-to-edge addressing.');
    const components=format==='rgba32float'?4:1,bytes=format==='rgba32float'?16:format==='r8unorm'?1:4,Type=['r32float','rgba32float'].includes(format)?Float32Array:Uint8Array;
    if(![width,height,depth].every(n=>Number.isInteger(n)&&n>0&&n<=this.device.limits.maxTextureDimension3D)||!(data===null&&storage)&&(!(data instanceof Type)||data.length!==width*height*depth*components||data.some(v=>!Number.isFinite(v))))throw new RangeError('3D texture requires matching byte or finite float data and valid dimensions.');
    if(!['linear','nearest'].includes(filter)||!['repeat','clamp-to-edge','mirror-repeat'].includes(addressMode))throw Error('Unsupported texture sampler settings.');
    if(['r32float','rgba32float'].includes(format)&&filter==='linear'&&!this.device.features.has('float32-filterable'))throw Error('Float32 linear filtering requires float32-filterable.');
    let upload=data;if(data&&format==='rgba8unorm'){upload=new Uint8Array(data.length*4);for(let i=0;i<data.length;i++){upload[i*4]=data[i];upload[i*4+3]=255;}}
    const gpuTexture=this.device.createTexture({label,size:[width,height,depth],dimension:'3d',format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC|(storage?GPUTextureUsage.STORAGE_BINDING:0)});
    if(upload)this.device.queue.writeTexture({texture:gpuTexture},upload,{bytesPerRow:width*bytes,rowsPerImage:height},[width,height,depth]);
    const resource={id:++resourceId,runtime:this,owned:true,destroyed:false,gpuTexture,view:gpuTexture.createView(),sampler:this.device.createSampler({minFilter:filter,magFilter:filter,addressModeU:addressMode,addressModeV:addressMode,addressModeW:addressMode}),format,dimension:'3d',width,height,depth,storage,filter,normalizedCoords,...(format==='rgba8unorm'?{scalarByteVolume:true}:{})};this.textures.add(resource);this.stats.dataBytesUploaded+=upload?.byteLength||0;return resource;
  }
  destroyTexture(resource){this.checkResource(resource);if(!resource.gpuTexture)throw Error('Expected a texture resource.');resource.gpuTexture.destroy();resource.destroyed=true;this.textures.delete(resource);}
  createBuffer(dataOrBytes, {label='compute buffer',usage=0} = {}) {
    this.assertAlive(); const data=ArrayBuffer.isView(dataOrBytes)?dataOrBytes:null, bytes=data?data.byteLength:dataOrBytes;
    if(!Number.isSafeInteger(bytes)||bytes<0)throw new RangeError('Buffer size must be a nonnegative integer.');
    const size=Math.max(4,roundUp(bytes,4));
    if(size>this.device.limits.maxStorageBufferBindingSize)throw new RangeError('Buffer exceeds maxStorageBufferBindingSize.');
    const gpuBuffer=this.device.createBuffer({label,size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST|usage,mappedAtCreation:!!data});
    if(data){new Uint8Array(gpuBuffer.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));gpuBuffer.unmap();this.stats.dataBytesUploaded+=bytes;}
    const r={id:++resourceId,gpuBuffer,byteLength:bytes,size,label,runtime:this,owned:true,destroyed:false};this.buffers.add(r);return r;
  }
  importBuffer(gpuBuffer, byteLength=gpuBuffer.size, label='imported storage buffer') {
    this.assertAlive();
    if(!(gpuBuffer.usage&GPUBufferUsage.STORAGE))throw new Error('Imported GPUBuffer must have STORAGE usage.');
    if(byteLength<0||byteLength>gpuBuffer.size)throw new RangeError('Invalid imported buffer length.');
    return {id:++resourceId,gpuBuffer,byteLength,size:gpuBuffer.size,label,runtime:this,owned:false,destroyed:false};
  }
  checkResource(resource) { if(!resource||resource.runtime!==this||resource.destroyed)throw new Error('Buffer is destroyed or belongs to a different runtime.'); }
  destroyBuffer(resource) { this.checkResource(resource);if(!resource.gpuBuffer)throw Error('Expected a buffer resource.'); if(resource.owned){resource.gpuBuffer.destroy();this.buffers.delete(resource);}resource.destroyed=true; }
  write(resource,data,offset=0) {
    this.assertAlive();this.checkResource(resource);if(!resource.gpuBuffer)throw Error('Expected a buffer resource.');
    if(!ArrayBuffer.isView(data)||offset%4||data.byteLength%4||offset<0||offset+data.byteLength>resource.size)throw new RangeError('Write must be aligned and fit in the destination.');
    this.device.queue.writeBuffer(resource.gpuBuffer,offset,data.buffer,data.byteOffset,data.byteLength);this.stats.dataBytesUploaded+=data.byteLength;
  }
  async read(resource, Type=Float32Array, byteLength=resource.byteLength, offset=0) {
    this.assertAlive();this.checkResource(resource);if(!resource.gpuBuffer)throw Error('Expected a buffer resource.');
    if(![Float32Array,Uint32Array,Int32Array].includes(Type))throw new TypeError('Readback supports 32-bit float/integer arrays.');
    if(!Number.isInteger(byteLength)||byteLength<0||byteLength%4||offset%4||offset<0||offset+byteLength>resource.size)throw new RangeError('Invalid readback range.');
    if(byteLength===0)return new Type(0);
    const staging=this.device.createBuffer({label:'explicit readback (not render path)',size:byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{const encoder=this.device.createCommandEncoder();encoder.copyBufferToBuffer(resource.gpuBuffer,offset,staging,0,byteLength);this.device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);const result=new Type(staging.getMappedRange().slice(0));staging.unmap();this.stats.readbackBytes+=byteLength;return result;}finally{staging.destroy();}
  }
  async readTextureSlice(resource,{slice=0}={}) {
    this.assertAlive();this.checkResource(resource);
    if(!resource.gpuTexture||resource.linearLength||!['2d','3d'].includes(resource.dimension)||!['r32float','r8unorm','rgba8unorm'].includes(resource.format)||resource.format==='rgba8unorm'&&!resource.scalarByteVolume)throw Error('Slice inspection requires a scalar texture.');
    if(!Number.isInteger(slice)||slice<0||slice>=resource.depth)throw new RangeError('Slice must be within the texture depth.');
    const {width,height,format}=resource,bytes=format==='r8unorm'?1:4,bytesPerRow=roundUp(width*bytes,256),size=bytesPerRow*height;
    const staging=this.device.createBuffer({label:'texture slice inspection',size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{const encoder=this.device.createCommandEncoder();encoder.copyTextureToBuffer({texture:resource.gpuTexture,origin:[0,0,slice]},{buffer:staging,bytesPerRow,rowsPerImage:height},[width,height,1]);this.device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
      const view=new DataView(staging.getMappedRange()),data=new Float32Array(width*height);
      for(let y=0;y<height;y++)for(let x=0;x<width;x++){const offset=y*bytesPerRow+x*bytes;data[y*width+x]=format==='r32float'?view.getFloat32(offset,true):view.getUint8(offset)/255;}
      staging.unmap();this.stats.readbackBytes+=size;return {data,width,height,slice};
    }finally{staging.destroy();}
  }
  async sortPairs(keys,values,{count,keyType='u32',waitForCompletion=true}={}) {
    this.assertAlive();if(!['u32','f32'].includes(keyType))throw Error('Pair sort keyType must be u32 or f32.');const source=keyType==='f32'?SORT_FLOAT_SOURCE:SORT_SOURCE;for(const r of [keys,values]){this.checkResource(r);if(!r.gpuBuffer)throw Error('Pair sort requires storage buffers.');}
    if(keys.gpuBuffer===values.gpuBuffer)throw Error('Pair sort key and value buffers must be distinct.');
    if(!Number.isInteger(count)||count<1||count>1048576||count*4>keys.byteLength||count*4>values.byteLength)throw new RangeError('Pair sort count must fit both buffers and be in [1,1048576].');
    const padded=2**Math.ceil(Math.log2(count)),[prepare,stage,finish]=await this.fixedKernels('sort-'+keyType,async()=>{const result=[];for(const entry of ['sortPrepare','sortStage','sortFinish'])result.push(await this.kernel(source,{entry,workgroupSize:[128,1,1]}));return result;}),pairs=this.createBuffer(padded*8),order=this.createBuffer(padded*4),batch=this.batch({label:'stable '+keyType+' pair sort'});
    try{const groups=[Math.ceil(padded/128),1,1];batch.dispatch(prepare.bind({keys,values,pairs,order},{count,padded}),groups);
      // Every sorting stage uses the same buffers. Snapshot changing scalars per
      // dispatch while reusing one binding instead of creating O(log² n) groups.
      const stageInvocation=stage.bind({pairs,order},{count:padded,size:2,stride:1});
      for(let size=2;size<=padded;size*=2)for(let stride=size/2;stride>=1;stride/=2)batch.dispatch(stageInvocation.setScalars({size,stride}),groups);
      batch.dispatch(finish.bind({pairs,keys,values},{count}),[Math.ceil(count/128),1,1]);batch.submit();if(waitForCompletion)await this.idle();}
    finally{if(!batch.ended)batch.discard();this.destroyBuffer(pairs);this.destroyBuffer(order);}
  }
  async inverseFFT2D(input,output,{width,height}={}) {return this.complexFFT2D(input,output,{width,height,inverse:true});}
  async complexFFT2D(input,output,{width,height,inverse=false}={}) {
    this.assertAlive();for(const r of [input,output]){this.checkResource(r);if(!r.gpuBuffer)throw Error('Inverse FFT requires storage buffers.');}
    const valid=n=>Number.isInteger(n)&&n>=1&&n<=2048&&(n&(n-1))===0;
    if(!valid(width)||!valid(height)||width*height*8>input.byteLength||width*height*8>output.byteLength)throw new RangeError('Inverse FFT needs power-of-two dimensions up to 2048 and complete float2 buffers.');
    if(typeof inverse!=='boolean')throw Error('FFT inverse must be boolean.');const source=inverse?FFT_SOURCE:FORWARD_FFT_SOURCE,entry=inverse?'inverseFftAxis':'forwardFftAxis';
    const rows=await this.kernel(source,{entry,workgroupSize:[Math.min(width,1024),1,1]}),columns=await this.kernel(source,{entry,workgroupSize:[Math.min(height,1024),1,1]}),scratch=this.createBuffer(width*height*8),batch=this.batch({label:(inverse?'inverse':'forward')+' complex 2D FFT'});
    try{batch.dispatch(rows.bind({input,output:scratch},{width,height,axis:0}),[height,1,1]);batch.dispatch(columns.bind({input:scratch,output},{width,height,axis:1}),[width,1,1]);batch.submit();await this.idle();}
    finally{if(!batch.ended)batch.discard();this.destroyBuffer(scratch);}
  }
  async realFFT2D(input,output,{width,height,inverse=false,realStride=width}={}) {
    this.assertAlive();for(const r of [input,output]){this.checkResource(r);if(!r.gpuBuffer)throw Error('Real FFT requires storage buffers.');}
    const valid=n=>Number.isInteger(n)&&n>=1&&n<=2048&&(n&(n-1))===0;
    if(!valid(width)||!valid(height)||typeof inverse!=='boolean'||!Number.isInteger(realStride)||realStride<width||realStride>65536)throw Error('Real FFT needs power-of-two dimensions up to 2048 and a valid real row stride.');
    const packed=Math.floor(width/2)+1,realBytes=realStride*height*4,complexBytes=packed*height*8;
    if(input.byteLength<(inverse?complexBytes:realBytes)||output.byteLength<(inverse?realBytes:complexBytes))throw Error('Real FFT buffers are too small for their row layouts.');
    const prepare=await this.kernel(REAL_FFT_SOURCE,{entry:inverse?'unpackSpectrum':'realToComplex',workgroupSize:[128,1,1]}),finish=await this.kernel(REAL_FFT_SOURCE,{entry:inverse?'complexToReal':'packSpectrum',workgroupSize:[128,1,1]}),a=this.createBuffer(width*height*8),b=this.createBuffer(width*height*8);
    try {
      const scalars={width,height,...(inverse?{}:{stride:realStride})};
      this.batch().dispatch(prepare.bind({input,output:a},scalars),[Math.ceil(width*height/128),1,1]).submit();
      await this.complexFFT2D(a,b,{width,height,inverse});
      this.batch().dispatch(finish.bind({input:b,output},{width,height,...(inverse?{stride:realStride}:{})}),[Math.ceil((inverse?width:packed)*height/128),1,1]).submit();await this.idle();
    }finally{this.destroyBuffer(a);this.destroyBuffer(b);}
  }
  async exclusiveScan(input,output,{count,total,waitForCompletion=true}={}) {
    this.assertAlive();for(const resource of [input,output,...(total?[total]:[])]){this.checkResource(resource);if(!resource.gpuBuffer)throw Error('Exclusive scan requires storage buffers.');}
    if(!Number.isSafeInteger(count)||count<1||count>1048576||count*4>input.byteLength||count*4>output.byteLength||total&&total.byteLength<4)throw new RangeError('Exclusive scan count must fit the buffers and be in [1,1048576].');
    const buffers=[input,output,...(total?[total]:[])];if(new Set(buffers.map(r=>r.gpuBuffer)).size!==buffers.length)throw Error('Exclusive scan input, output and total must not alias.');
    const [blocks,add]=await this.fixedKernels('scan',async()=>{const result=[];for(const entry of ['scanBlocks','addScanOffsets'])result.push(await this.kernel(SCAN_SOURCE,{entry,workgroupSize:[256,1,1]}));return result;}),scratch=[],batch=this.batch({label:'exclusive uint scan'});
    try{
      const record=(source,destination,n)=>{const groups=Math.ceil(n/512),totals=this.createBuffer(groups*4);scratch.push(totals);batch.dispatch(blocks.bind({input:source,output:destination,totals},{count:n}),[groups,1,1]);
        if(groups===1)return totals;const offsets=this.createBuffer(groups*4);scratch.push(offsets);const sum=record(totals,offsets,groups);batch.dispatch(add.bind({output:destination,offsets},{count:n}),[Math.ceil(n/256),1,1]);return sum;};
      const sum=record(input,output,count);if(total)batch.copy(sum,total,{byteLength:4});batch.submit();if(waitForCompletion)await this.idle();
    }finally{if(!batch.ended)batch.discard();for(const r of scratch)this.destroyBuffer(r);}
  }
  createObjectArena(){this.assertAlive();return new ObjectArena(this);}
  async fixedKernels(key,build){
    // Cache only the runtime's fixed helpers; arbitrary editor source still compiles normally.
    this._fixedKernels??=new Map();
    if(!this._fixedKernels.has(key)){const pending=build();this._fixedKernels.set(key,pending);pending.catch(()=>this._fixedKernels.delete(key));}
    return this._fixedKernels.get(key);
  }
  async kernel(sourceOrArtifact,options={}) {
    this.assertAlive();const artifact=typeof sourceOrArtifact==='string'?compile(sourceOrArtifact,options):sourceOrArtifact;
    validateWorkgroup(artifact.metadata,this.device.limits);
    for(const feature of artifact.metadata.requiredFeatures||[])if(!this.device.features.has(feature))throw Error('Kernel requires WebGPU feature '+feature);
    for(const feature of artifact.metadata.requiredWgslFeatures||[])if(!navigator.gpu?.wgslLanguageFeatures?.has(feature))throw Error('Kernel requires WGSL feature '+feature);
    // Use complete source/ABI, not an unchecked short hash, as the cache key.
    const key=artifact.wgsl+'\n'+JSON.stringify(artifact.metadata)+'\n'+JSON.stringify(artifact.children?.map(c=>({queueId:c.queueId,wgsl:c.artifact.wgsl,metadata:c.artifact.metadata})));
    if(this.pipelineCache.has(key)){this.stats.pipelineCacheHits++;return this.pipelineCache.get(key);}
    const build=async()=>{
      this.device.pushErrorScope('validation');
      let thrown=null,result;
      try{
        const module=this.device.createShaderModule({label:`CUDA -> WGSL: ${artifact.name}`,code:artifact.wgsl});
        const info=await module.getCompilationInfo();
        const errors=info.messages.filter(m=>m.type==='error');
        if(errors.length)throw new Error(`${artifact.name}: WGSL validation failed\n`+errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
        const entries=artifact.metadata.bindings.map(b=>({binding:b.binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:b.readOnly?'read-only-storage':'storage',minBindingSize:Math.max(4,b.minBindingSize||b.stride)}}));
        for(const t of artifact.metadata.textures||[])entries.push({binding:t.binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:['r32uint','r8uint','rg8uint'].includes(t.format)?'uint':t.coordinates==='linear'?'unfilterable-float':'float',viewDimension:t.dimension,multisampled:false}},{binding:t.samplerBinding,visibility:GPUShaderStage.COMPUTE,sampler:{type:['linear','pixel-byte','pixel-uint'].includes(t.coordinates)?'non-filtering':'filtering'}});
        for(const surface of artifact.metadata.surfaces||[])entries.push({binding:surface.binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only',format:surface.format,viewDimension:surface.dimension}});
        if(artifact.metadata.uniformSize)entries.push({binding:artifact.metadata.uniformBinding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:artifact.metadata.uniformSize}});
        const layout=this.device.createBindGroupLayout({label:artifact.name,entries});
        const objectLayout=artifact.metadata.objectHeap?.persistent?this.device.createBindGroupLayout({entries:[...artifact.metadata.objectHeap.types.map(t=>({binding:t.binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage',minBindingSize:t.byteLength}})),...(artifact.metadata.objectHeap.imports||[]).map(i=>({binding:i.binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage',minBindingSize:4}}))]}):null;
        const pipeline=await this.device.createComputePipelineAsync({label:artifact.name,layout:this.device.createPipelineLayout({bindGroupLayouts:objectLayout?[layout,objectLayout]:[layout]}),compute:{module,entryPoint:artifact.entryPoint || 'main'}});
        this.stats.pipelineCompiles++; result=new Kernel(this,artifact,pipeline,layout,info.messages,objectLayout);
      }catch(error){thrown=error;}
      const validation=await this.device.popErrorScope();
      if(thrown)throw thrown;if(validation)throw new Error(validation.message);return result;
    };
    // Error scopes are stack-based. Serialize pipeline creation to prevent interleaved scopes.
    const promise=this.pipelineQueue.then(build);this.pipelineQueue=promise.catch(()=>{});this.pipelineCache.set(key,promise);
    promise.catch(()=>this.pipelineCache.delete(key));return promise;
  }
  batch(options={}) {this.assertAlive();return new ComputeBatch(this,options);}
  async idle() {this.assertAlive();await this.device.queue.onSubmittedWorkDone();}
  dispose() {if(this.disposed)return;this.disposed=true;for(const t of this.textures){t.gpuTexture.destroy();t.destroyed=true;}this.textures.clear();for(const b of this.buffers){b.gpuBuffer.destroy();b.destroyed=true;}this.buffers.clear();this.uniformBuffer.destroy();this.pipelineCache.clear();this._fixedKernels?.clear();this.batchMemory=[];this.device.removeEventListener('uncapturederror',this.errorListener);if(this.ownsDevice)this.device.destroy();}
}
export class Kernel {
  constructor(runtime,artifact,pipeline,layout,messages,objectLayout=null){this.objectLayout=objectLayout;this.runtime=runtime;this.artifact=artifact;this.pipeline=pipeline;this.layout=layout;this.messages=messages;}
  bind(buffers,scalars={},options={}) {return new Invocation(this,buffers,scalars,options);}
  async runQueued(buffers,scalars,workgroups,{objectArena}={}) {
    const children=this.artifact.children;if(!children?.length)throw Error('Compile with scheduleDeviceLaunches: true before running queued children.');
    if(this.artifact.metadata.deviceLaunchQueue?.queues.some(q=>q.caller===this.artifact.name&&q.frontier))return runRecursiveLaunches(this,buffers,scalars,workgroups,objectArena);
    const runtime=this.runtime,prepared=[];
    for(const child of children)prepared.push({queue:this.artifact.metadata.deviceLaunchQueue.queues.find(q=>q.id===child.queueId),kernel:await runtime.kernel(child.artifact)});
    const parent=this.bind(buffers,scalars,{objectArena,queueOnly:true}),types=this.artifact.metadata.objectHeap.types,scratch=[];
    try {
      const batch=runtime.batch();for(const {queue} of prepared)batch.clear(objectArena.buffers[types.findIndex(t=>t.name===queue.name)]);
      batch.dispatch(parent,workgroups).submit();
      for(const {queue,kernel} of prepared){
        const queueBuffer=objectArena.buffers[types.findIndex(t=>t.name===queue.name)],indirect=runtime.createBuffer(queue.byteLength,{usage:GPUBufferUsage.INDIRECT,label:'GPU child dispatch dimensions'});scratch.push(indirect);
        runtime.batch().copy(queueBuffer,indirect).submit();
        const childBuffers=Object.fromEntries(queue.buffers.map(b=>[b.name,buffers[b.parent]]));
        let childrenBatch=runtime.batch();
        for(let slot=0;slot<queue.capacity;slot++){
          if(childrenBatch.cursor+runtime.uniformAlignment>runtime.uniformCapacity){childrenBatch.submit();childrenBatch=runtime.batch();}
          const invocation=kernel.bind(childBuffers,{cw_launch_slot:slot},{objectArena});
          childrenBatch.dispatch(invocation,[1],{resource:indirect,offset:16+slot*queue.stride*4});
        }
        childrenBatch.submit();
      }
      await runtime.idle();
      for(const {queue} of prepared){const flags=await runtime.read(objectArena.buffers[types.findIndex(t=>t.name===queue.name)],Uint32Array,8);if(flags[1])throw Error('GPU child-launch queue overflow or invalid launch dimensions.');}
    } finally {for(const resource of scratch)runtime.destroyBuffer(resource);}
  }
}
export class Invocation {
  constructor(kernel,buffers,scalars,{objectArena,queueOnly=false,scalarBuffers={}}={}) {
    this.kernel=kernel;this.runtime=kernel.runtime;this.runtime.assertAlive();this.version=0;this.values={};
    this.uniformData=new ArrayBuffer(kernel.artifact.metadata.uniformSize);this.buffers={...buffers};
    if(kernel.artifact.metadata.deviceLaunchQueue?.producerOnly&&kernel.artifact.metadata.deviceLaunchQueue.queues.some(q=>q.caller===kernel.artifact.name)&&!queueOnly)throw Error('This kernel produces child-launch queues only; explicitly bind queueOnly: true until automatic scheduling is supported.');
    const meta=kernel.artifact.metadata,entries=[],seen=new Map(),known=new Set([...meta.bindings,...(meta.textures||[]),...(meta.surfaces||[]),...(meta.objectHeap?.imports||[])].map(b=>b.name));
    this.scalarBuffers=Object.entries(scalarBuffers).map(([name,{resource,offset=0}])=>{
      const scalar=meta.scalars.find(s=>s.name===name);
      if(!scalar||!['u32','i32'].includes(scalar.type)||['bool','cw_short','cw_ushort'].includes(scalar.sourceType)||meta.scalarConstraints?.some(c=>c.name===name)||meta.surfaces?.length)throw Error('GPU scalar requires an unconstrained 32-bit integer parameter without surfaces: '+name);
      this.runtime.checkResource(resource);
      if(!resource.gpuBuffer||!(resource.gpuBuffer.usage&GPUBufferUsage.COPY_SRC)||!Number.isSafeInteger(offset)||offset<0||offset%4||offset+4>resource.byteLength)throw Error('Invalid GPU scalar source: '+name);
      return {resource,sourceOffset:offset,targetOffset:scalar.offset};
    });
    for(const [alias,target]of Object.entries(meta.bufferAliases||{})){known.add(alias);if(Object.hasOwn(buffers,alias)&&buffers[alias]!==buffers[target])throw new Error('Declared buffer alias '+alias+' must use the same resource as '+target);}
    for(const name of Object.keys(buffers))if(!known.has(name))throw new Error(`Unknown buffer '${name}'.`);
    for(const b of meta.bindings){
      const resource=buffers[b.name];this.runtime.checkResource(resource);if(!resource.gpuBuffer)throw Error('Binding '+b.name+' requires a buffer.');
      if(resource.size<(b.minBindingSize||b.stride))throw new RangeError(`Buffer ${b.name} is smaller than one ${b.elementType} record.`);
      if(resource.size%b.stride)throw new RangeError(`Buffer ${b.name} is not aligned to ${b.stride}-byte records.`);
      if(seen.has(resource.gpuBuffer)&&(!b.readOnly||!seen.get(resource.gpuBuffer)))throw new Error('Writable buffer aliasing across bindings is rejected; use separate buffers or a single in-place parameter.');
      seen.set(resource.gpuBuffer,b.readOnly);entries.push({binding:b.binding,resource:{buffer:resource.gpuBuffer,offset:0,size:resource.size}});
    }
    for(const t of meta.textures||[]){const r=buffers[t.name];this.runtime.checkResource(r);if(!r.gpuTexture||r.dimension!==t.dimension||(t.coordinates==='cube-direction'&&!r.cubemap)||(t.coordinates==='linear'&&!r.linearLength)||(['pixel-byte','pixel-uint'].includes(t.coordinates)&&(r.normalizedCoords!==false||r.filter!=='nearest'||r.addressMode!=='clamp-to-edge'))||(r.format!==t.format&&!(t.coordinates==='linear'&&t.format==='r8unorm'&&r.format==='r32float')&&!(t.dimension==='3d'&&t.format==='r8unorm'&&r.format==='rgba8unorm'&&r.scalarByteVolume)))throw Error('Texture '+t.name+' requires a matching '+t.dimension+' '+t.format+' resource.');entries.push({binding:t.binding,resource:r.view},{binding:t.samplerBinding,resource:r.sampler});}
    for(const surface of meta.surfaces||[]){const r=buffers[surface.name];this.runtime.checkResource(r);if(!r.gpuTexture||!r.storage||r.dimension!==surface.dimension||r.format!==surface.format)throw Error('Surface '+surface.name+' requires a matching writable texture.');if((meta.textures||[]).some(t=>buffers[t.name]?.gpuTexture===r.gpuTexture)||seen.has(r.gpuTexture))throw Error('Writable surface aliasing within a dispatch is unsupported.');seen.set(r.gpuTexture,true);entries.push({binding:surface.binding,resource:r.view});}
    if(meta.uniformSize)entries.push({binding:meta.uniformBinding,resource:{buffer:this.runtime.uniformBuffer,offset:0,size:meta.uniformSize}});
    this.bindGroup=this.runtime.device.createBindGroup({label:`${kernel.artifact.name}: persistent bindings`,layout:kernel.layout,entries});this.runtime.stats.bindGroupsCreated++;
    if(kernel.objectLayout){if(!(objectArena instanceof ObjectArena))throw Error('Persistent object kernels require an explicit objectArena binding.');for(const binding of meta.bindings.filter(b=>String(b.elementType).startsWith('cw_objectptr_')||meta.objectHeap?.pointerBuffers?.includes(b.name))){const resource=buffers[binding.name];if(resource.objectArena&&resource.objectArena!==objectArena)throw Error('Object pointer buffer belongs to another arena.');}
      this.objectArena=objectArena;this.objectBindGroup=objectArena.bind(kernel,buffers);for(const binding of meta.bindings.filter(b=>String(b.elementType).startsWith('cw_objectptr_')||meta.objectHeap?.pointerBuffers?.includes(b.name)))buffers[binding.name].objectArena=objectArena;}
    this.setScalars(scalars);
  }
  setScalars(values) {const merged={...this.values,...values};packScalars(this.kernel.artifact.metadata,merged,this.uniformData);const view=new DataView(this.uniformData);for(const scale of this.kernel.artifact.metadata.textureScales||[]){const texture=this.buffers[scale.name];view.setFloat32(scale.offset,texture.normalizedCoords===false?1/texture.width:1,true);view.setFloat32(scale.offset+4,texture.normalizedCoords===false?1/texture.height:1,true);if(scale.dimension==='3d')view.setFloat32(scale.offset+8,texture.normalizedCoords===false?1/texture.depth:1,true);if(scale.pointOffset!==undefined)view.setFloat32(scale.pointOffset,texture.normalizedCoords===false&&texture.filter==='nearest'?1:0,true);}for(const length of this.kernel.artifact.metadata.textureLengths||[])view.setUint32(length.offset,this.buffers[length.name].linearLength,true);this.values=merged;this.version++;return this;}
}
export class ComputeBatch {
  constructor(runtime,{label='compute batch',timestampWrites}={}) {
    this.runtime=runtime;this.encoder=runtime.device.createCommandEncoder({label});this.pass=null;this.ended=false;
    this.data=runtime.batchMemory.pop() || new Uint8Array(runtime.uniformCapacity);this.cursor=0;this.snapshots=new Map();
    this.timestampWrites=timestampWrites;this.passCount=0;this.lastPipeline=null;this.lastBindGroup=null;this.lastOffset=-1;this.dispatchCount=0;
  }
  assertOpen(){if(this.ended)throw new Error('Batch has already been submitted or discarded.');this.runtime.assertAlive();}
  beginPass(){if(!this.pass){if(this.timestampWrites&&this.passCount)throw new Error('Timestamped batch supports a single compute pass.');this.pass=this.encoder.beginComputePass(this.timestampWrites?{timestampWrites:this.timestampWrites}:{});this.passCount++;this.lastPipeline=null;this.lastBindGroup=null;this.lastOffset=-1;}return this.pass;}
  endPass(){if(this.pass){this.pass.end();this.pass=null;}}
  dispatch(invocation,workgroups,indirect=null) {
    this.assertOpen();if(invocation.runtime!==this.runtime)throw new Error('Invocation belongs to another runtime.');
    if(indirect){this.runtime.checkResource(indirect.resource);if(!indirect.resource.gpuBuffer||!(indirect.resource.gpuBuffer.usage&GPUBufferUsage.INDIRECT)||!Number.isSafeInteger(indirect.offset)||indirect.offset<0||indirect.offset%4||indirect.offset+12>indirect.resource.byteLength)throw Error('Invalid indirect dispatch buffer or offset.');if(invocation.kernel.artifact.metadata.surfaces?.length)throw Error('Indirect surface dispatch needs explicit extent validation.');}
    const groups=Array.isArray(workgroups)?[...workgroups]:[workgroups];while(groups.length<3)groups.push(1);
    for(const surface of invocation.kernel.artifact.metadata.surfaces||[]){const r=invocation.buffers[surface.name],block=invocation.kernel.artifact.metadata.workgroupSize;if(surface.coordinates==='global-x'&&(r.height!==1||groups[1]*block[1]!==1||groups[2]*block[2]!==1))throw Error('Surface dispatch requires a one-row texture and a one-dimensional launch.');if(surface.coordinates==='global-xy-layer'){const layer=surface.layer?.scalar?(Object.hasOwn(invocation.values,surface.layer.scalar)?invocation.values[surface.layer.scalar]:invocation.kernel.artifact.metadata.scalars.find(p=>p.name===surface.layer.scalar)?.defaultValue):surface.layer?.value;if(!Number.isInteger(layer)||layer<0||layer>=r.depth)throw Error('Surface layer is outside the allocated array.');}if(!['global-x','global-xy','global-xyz','global-xy-layer'].includes(surface.coordinates)||groups[0]*block[0]>r.width||groups[1]*block[1]>r.height||groups[2]*block[2]>(surface.coordinates==='global-xyz'?r.depth:1))throw Error('Surface dispatch exceeds the checked global '+(surface.coordinates==='global-xyz'?'XYZ':'XY')+' extent.');}
    if(groups.length!==3||groups.some(x=>!Number.isSafeInteger(x)||x<0||x>this.runtime.device.limits.maxComputeWorkgroupsPerDimension))throw new RangeError('Invalid workgroup counts. These are block counts, not thread counts.');
    if(groups.some(x=>x===0))return this;
    for(const r of Object.values(invocation.buffers))this.runtime.checkResource(r);
    const meta=invocation.kernel.artifact.metadata;let offset=0;
    if(meta.uniformSize){
      const previous=this.snapshots.get(invocation);
      if(previous?.version===invocation.version)offset=previous.offset;
      else{
        offset=roundUp(this.cursor,this.runtime.uniformAlignment);
        if(offset+meta.uniformSize>this.data.byteLength)throw new RangeError('Batch uniform arena is full. Submit smaller batches or raise uniformCapacity.');
        this.data.set(new Uint8Array(invocation.uniformData),offset);this.cursor=offset+meta.uniformSize;this.snapshots.set(invocation,{version:invocation.version,offset});
      }
    }
    // Queue-ordered copies supply GPU-produced scalar values without CPU readback.
    // End the previous pass so each dispatch observes its own counter snapshot.
    if(invocation.scalarBuffers.length){this.endPass();for(const scalar of invocation.scalarBuffers){this.runtime.checkResource(scalar.resource);this.encoder.copyBufferToBuffer(scalar.resource.gpuBuffer,scalar.sourceOffset,this.runtime.uniformBuffer,offset+scalar.targetOffset,4);}}
    const pass=this.beginPass();
    if(this.lastPipeline!==invocation.kernel.pipeline){pass.setPipeline(invocation.kernel.pipeline);this.lastPipeline=invocation.kernel.pipeline;}
    if(this.lastBindGroup!==invocation.bindGroup||this.lastOffset!==offset){pass.setBindGroup(0,invocation.bindGroup,meta.uniformSize?[offset]:[]);this.lastBindGroup=invocation.bindGroup;this.lastOffset=offset;}
    if(invocation.objectArena){invocation.objectArena.assertAlive();pass.setBindGroup(1,invocation.objectBindGroup);}
    if(indirect)pass.dispatchWorkgroupsIndirect(indirect.resource.gpuBuffer,indirect.offset);else pass.dispatchWorkgroups(...groups);this.dispatchCount++;return this;
  }
  copyToLinearTexture(source,target,{sourceOffset=0}={}){
    this.assertOpen();this.runtime.checkResource(source);this.runtime.checkResource(target);
    const bytes={r32float:4,rg32float:8,rgba32float:16,r32uint:4}[target.format];
    if(!source.gpuBuffer||!target.gpuTexture||!target.linearLength||!bytes)throw Error('Linear texture copy requires a buffer and float or uint linear texture.');
    if(!Number.isSafeInteger(sourceOffset)||sourceOffset<0||sourceOffset%bytes||sourceOffset+target.linearLength*bytes>source.byteLength)throw Error('Linear texture copy must fit aligned source records.');
    this.endPass();for(let row=0,remaining=target.linearLength;remaining>0;row++){
      const width=Math.min(remaining,target.width);this.encoder.copyBufferToTexture({buffer:source.gpuBuffer,offset:sourceOffset+row*target.width*bytes},{texture:target.gpuTexture,origin:[0,row,0]},[width,1,1]);remaining-=width;
    }
    return this;
  }
  copyToTexture(source,target,{sourceOffset=0,bytesPerRow}={}){
    this.assertOpen();this.runtime.checkResource(source);this.runtime.checkResource(target);
    if(!source.gpuBuffer||!target.gpuTexture||target.dimension!=='2d'||!['r32float','rg32float'].includes(target.format))throw Error('Texture copy requires a buffer and a 2D r32float or rg32float texture.');
    const texelBytes=target.format==='rg32float'?8:4,rowBytes=target.width*texelBytes,stride=bytesPerRow===undefined?rowBytes:bytesPerRow,bytes=(target.height-1)*stride+rowBytes;
    if(!Number.isSafeInteger(stride)||stride<rowBytes||stride%texelBytes)throw Error('Texture copy row pitch must contain whole texels and fit a row.');
    if(!Number.isSafeInteger(sourceOffset)||sourceOffset<0||sourceOffset%texelBytes||sourceOffset+bytes>source.byteLength)throw Error('Texture copy source range must be aligned and contain the complete image.');
    this.endPass();
    if(stride%256===0)this.encoder.copyBufferToTexture({buffer:source.gpuBuffer,offset:sourceOffset,bytesPerRow:stride,rowsPerImage:target.height},{texture:target.gpuTexture},[target.width,target.height,1]);
    else for(let y=0;y<target.height;y++)this.encoder.copyBufferToTexture({buffer:source.gpuBuffer,offset:sourceOffset+y*stride},{texture:target.gpuTexture,origin:[0,y,0]},[target.width,1,1]);
    return this;
  }
  clear(resource){this.assertOpen();this.runtime.checkResource(resource);this.endPass();this.encoder.clearBuffer(resource.gpuBuffer,0,resource.size);return this;}
  copy(source,target,range){this.assertOpen();this.runtime.checkResource(source);this.runtime.checkResource(target);if(!source.gpuBuffer||!target.gpuBuffer)throw Error('Copy requires buffer resources.');if(source.gpuBuffer===target.gpuBuffer)throw new RangeError('Copy requires distinct buffers.');if(range===undefined&&(source.byteLength!==target.byteLength||source.byteLength%4))throw new RangeError('Feedback copy requires equal aligned byte length.');const {sourceOffset=0,targetOffset=0,byteLength=source.byteLength}=range??{};if([sourceOffset,targetOffset,byteLength].some(n=>!Number.isSafeInteger(n)||n<0||n%4)||sourceOffset+byteLength>source.byteLength||targetOffset+byteLength>target.byteLength)throw new RangeError('Copy range must be aligned and within both buffers.');this.endPass();if(byteLength)this.encoder.copyBufferToBuffer(source.gpuBuffer,sourceOffset,target.gpuBuffer,targetOffset,byteLength);return this;}
  submit(){
    this.assertOpen();this.endPass();this.ended=true;
    if(this.cursor){this.runtime.device.queue.writeBuffer(this.runtime.uniformBuffer,0,this.data,0,this.cursor);this.runtime.stats.uniformBytesUploaded+=this.cursor;}
    this.runtime.device.queue.submit([this.encoder.finish()]);this.runtime.stats.submissions++;this.runtime.stats.dispatches+=this.dispatchCount;
    this.runtime.batchMemory.push(this.data);this.data=null;this.snapshots.clear();
  }
  discard(){if(this.ended)return;this.endPass();this.ended=true;this.runtime.batchMemory.push(this.data);this.data=null;this.snapshots.clear();}
}
