import * as THREE from 'three/webgpu';
import {storage} from 'three/tsl';
/**
 * The ONLY module coupled to Three r186 backend internals.
 * Three owns the allocation; the compute runtime borrows the exact GPUBuffer on the same GPUDevice.
 * Never set attribute.needsUpdate after compute writes: that would upload the stale CPU seed array.
 */
export function createSharedFloat4(renderer,runtime,array,{label='shared float4 records'}={}){
  return createSharedStorage(renderer,runtime,array,4,'vec4',label);
}
export function createSharedScalar(renderer,runtime,array,{label='shared scalar field'}={}){
  return createSharedStorage(renderer,runtime,array,1,'float',label);
}
function createSharedStorage(renderer,runtime,array,width,type,label){
  if(THREE.REVISION!=='186')throw new Error(`Interop is pinned to Three r186, found r${THREE.REVISION}. Re-run interop tests before upgrading.`);
  if(!renderer.backend?.isWebGPUBackend||renderer.backend.device!==runtime.device)throw new Error('Compute and Three.js must use the exact same initialized WebGPU device.');
  if(!(array instanceof Float32Array)||array.length===0||array.length%width)throw new TypeError(`Provide a nonempty Float32Array of ${width===4?'float4 records':'scalar values'}.`);
  const attribute=new THREE.StorageInstancedBufferAttribute(array,width);attribute.name=label;
  renderer.backend.createStorageAttribute(attribute);
  const gpuBuffer=renderer.backend.get(attribute).buffer;
  if(!gpuBuffer)throw new Error('Three.js did not allocate a storage GPUBuffer.');
  const resource=runtime.importBuffer(gpuBuffer,array.byteLength,label),node=storage(attribute,type,array.length/width).toReadOnly();
  let disposed=false;
  return {attribute,node,resource,gpuBuffer,dispose(){if(disposed)return;disposed=true;runtime.destroyBuffer(resource);renderer.backend.destroyAttribute(attribute);}};
}
