import {recordLeaves} from './record-parameters.js';
import {integerExpression} from './integer-expression.js';
// Resolve launch settings from immutable values in their lexical scope.
function launchConstants(fn,constantValue){
 const result=new Map();
 const evaluate=(node,scope)=>{
  const replace=n=>{if(!n||typeof n!=='object')return n;if(n.kind==='id'){const value=scope.get(n.name);if(!Number.isSafeInteger(value))throw Error('Nonconstant launch setting');return {kind:'literal',value:String(value),token:n.token};}if(Array.isArray(n))return n.map(replace);return Object.fromEntries(Object.entries(n).map(([k,v])=>[k,k==='token'?v:replace(v)]));};
  return constantValue(replace(node));
 };
 const visit=(node,scope)=>{
  if(!node||typeof node!=='object')return;
  if(Array.isArray(node)){for(const n of node)visit(n,scope);return;}
  if(node.kind==='block'){const inner=new Map(scope);for(const n of node.body)visit(n,inner);return;}
  if(node.kind==='for'){const inner=new Map(scope);visit(node.init,inner);visit(node.body,inner);return;}
  if(node.kind==='decl'){let value;try{if(node.constant&&!node.pointer&&!node.reference&&node.init)value=evaluate(node.init,scope);}catch{}scope.set(node.name,value);return;}
  if(node.kind==='device-launch'){const captured=new Map(scope);result.set(node,setting=>evaluate(setting,captured));return;}
  for(const [key,value] of Object.entries(node))if(!['token','type'].includes(key))visit(value,new Map(scope));
 };
 const scope=new Map([['warpSize',32]]);for(const p of fn.params)scope.set(p.name,undefined);visit(fn.body,scope);return result;
}
// Explicit producer stage for a bounded GPU child-launch queue.
// Queue production alone does not execute the queued child work.
export function launchQueues(e,walk,constantValue,pointerParts){
 const options=e.options.deviceLaunchQueue;if(options===undefined)return [];
 if(!options||typeof options!=='object'||Array.isArray(options)||Object.keys(options).some(k=>!['maxLaunches','maxGenerations'].includes(k))||!Number.isInteger(options.maxLaunches)||options.maxLaunches<1||options.maxLaunches>65535)e.fail('deviceLaunchQueue requires maxLaunches in 1..65535.',e.kernel);
 if(options.maxGenerations!==undefined&&(!Number.isInteger(options.maxGenerations)||options.maxGenerations<1||options.maxGenerations>64))e.fail('Recursive queues require maxGenerations in 1..64.',e.kernel);
 if(!e.persistentObjects)e.fail('Device launch queues require objectHeap: persistent.',e.kernel);
 const queues=[];
 for(const caller of e.ast.functions){const constants=launchConstants(caller,constantValue),aliases=new Map();walk(caller.body,n=>{if(n.kind==='decl'){if(aliases.has(n.name))aliases.set(n.name,null);else aliases.set(n.name,pointerParts(n.init)?.base?.name||null);}});
 const rootOf=(name,seen=new Set())=>{if(seen.has(name))return null;if(caller.params.some(p=>p.name===name))return name;seen.add(name);return aliases.get(name)?rootOf(aliases.get(name),seen):null;};walk(caller.body,n=>{
  if(n.kind!=='device-launch')return;
  if(caller.qualifier!=='__global__')e.fail('Device launches in helpers are not supported yet.',n);
  const child=e.ast.functions.find(f=>f.name===n.callee.name&&f.qualifier==='__global__');if(!child)e.fail('Child launch must name a declared global kernel.',n);
  let childEntry=child.name;if(child.templateParameter){if(child.templateKind==='type'||n.callee.templateArgument===undefined)e.fail('Child template launches require explicit integer arguments.',n);let values;try{values=n.callee.templateArgument.split(',').map(integerExpression);}catch{e.fail('Child template arguments must be constant integers.',n);}if(values.length!==(child.templateParameters?.length||1)||values.some(v=>v<0))e.fail('Child template arguments must match nonnegative integer parameters.',n);childEntry+='<' + values.join(',') + '>';}else if(n.callee.templateArgument!==undefined)e.fail('Child template arguments require a templated kernel.',n);
  if(n.configuration.length<2||n.configuration.length>3||child.params.length!==n.args.length)e.fail('Child queues require grid and block dimensions, optional shared bytes and every argument.',n);
  let sharedMemoryBytes=0;if(n.configuration.length===3){try{sharedMemoryBytes=constants.get(n)(n.configuration[2]);}catch{sharedMemoryBytes=NaN;}if(!Number.isInteger(sharedMemoryBytes)||sharedMemoryBytes<0||sharedMemoryBytes>16384)e.fail('Child shared bytes must be a constant in 0..16384.',n);}
  let block;try{block=constants.get(n)(n.configuration[1]);}catch{}if(!Number.isInteger(block)||block<1||block>1024)e.fail('Child block size must be a constant in 1..1024.',n);
  const scalars=[],buffers=[];for(let i=0;i<child.params.length;i++){
   const p=child.params[i],arg=n.args[i];
   if(p.pointer){const root=rootOf(pointerParts(arg)?.base?.name),parent=caller.params.find(x=>x.name===root);if(!parent?.pointer||p.type!==parent.type)e.fail('Child buffer arguments must derive from matching named parent buffers.',arg);if(parent.constant&&!p.constant)e.fail('Child buffer arguments cannot discard const.',arg);buffers.push({name:p.name,parent:parent.name,type:p.type,argument:i});}
   else if(!p.reference&&e.structs.has(p.type)){for(const leaf of recordLeaves(e,p.type,p))scalars.push({name:p.name+'.'+leaf.path.join('.'),type:leaf.type,recordType:p.type,path:leaf.path,argument:i,word:3+scalars.length});}
   else{if(p.reference||!['i32','u32','f32'].includes(p.type))e.fail('Child queue scalar arguments require 32-bit integer or float values.',p);scalars.push({name:p.name,type:p.type,argument:i,word:3+scalars.length});}
  }
  buffers.forEach((b,i)=>b.offsetWord=3+scalars.length+i);
  const id=queues.length,stride=3+scalars.length+buffers.length,queue={id,name:'launch_queue_'+id,caller:caller.name,child:child.name,childEntry,capacity:options.maxLaunches,block:[block,1,1],sharedMemoryBytes,stride,scalars,buffers,binding:e.objectHeaps.size+e.objectImports.length+e.deviceHeaps.size+id,byteLength:16+options.maxLaunches*stride*4,variable:'cw_launch_queue_'+id};
  queue.recordLayout=JSON.stringify({caller:queue.caller,child:queue.child,childEntry,capacity:queue.capacity,stride,scalars,buffers,block:queue.block,sharedMemoryBytes});n.queueId=id;queues.push(queue);
 });}
 if(options.maxGenerations!==undefined){
  if(queues.length!==1||queues[0].caller!==queues[0].child||queues[0].buffers.some(b=>b.name!==b.parent))e.fail('Recursive queues require one self-launch site retaining named buffer allocations.',e.kernel);
  const q=queues[0];q.maxGenerations=options.maxGenerations;
  q.frontier={name:q.name+'_frontier',capacity:q.capacity,binding:q.binding+1,byteLength:q.byteLength,recordLayout:q.recordLayout,variable:q.variable+'_frontier'};
 }
 return queues;
}
export function launchQueueStorageTypes(queues){return queues.flatMap(q=>[{name:q.name,capacity:q.capacity,binding:q.binding,byteLength:q.byteLength,recordLayout:q.recordLayout},...(q.frontier?[(({variable,...type})=>type)(q.frontier)]:[])]);}
export function launchQueueDeclarations(queues){return queues.flatMap(q=>[
 `struct CWLaunchQueue_${q.id} { count: atomic<u32>, overflow: atomic<u32>, pad0: u32, pad1: u32, words: array<u32, ${q.capacity*q.stride}>, }`,
 `@group(1) @binding(${q.binding}) var<storage,read_write> ${q.variable}: CWLaunchQueue_${q.id};`,
 ...(q.frontier?[`@group(1) @binding(${q.frontier.binding}) var<storage,read_write> ${q.frontier.variable}: CWLaunchQueue_${q.id};`]:[])
]);}
export function emitLaunch(e,n){
 const q=e.launchQueues.find(q=>q.id===n.queueId);if(!q)e.fail('Device child-kernel launches require GPU scheduling support.',n);
 const grid=e.expr(n.configuration[0]);if(!['i32','u32','f32'].includes(grid.type))e.fail('Child grid must be a one-dimensional scalar.',n);
 const name='cw_launch_'+e.temp++,pre=[...grid.pre,`let ${name}_grid=${grid.code};`],values=[];
 const pointerGuards=[];
 for(const buffer of q.buffers){const value=e.argument(n.args[buffer.argument]),root=value.rootSymbol;if(!root||!['buffer','buffer-alias'].includes(root.kind)||root.rootBufferName!==buffer.parent||value.type.element!==buffer.type)e.fail('Child pointer must retain its original typed storage allocation.',n.args[buffer.argument]);const local=name+'_offset_'+buffer.offsetWord;pre.push(...value.pre,`let ${local}: i32 = ${value.pointerCode||'0i'};`);values.push({word:buffer.offsetWord,code:`bitcast<u32>(${local})`});pointerGuards.push(`${local}>=0i && u32(${local})<=arrayLength(&${value.code})`);}
 const records=new Map();
 for(const scalar of q.scalars){let value;if(scalar.path){let snapshot=records.get(scalar.argument);if(!snapshot){const record=e.expr(n.args[scalar.argument]);if(record.type!==scalar.recordType)e.fail('Child record argument type must match.',n.args[scalar.argument]);snapshot=name+'_record_'+scalar.argument;pre.push(...record.pre,`let ${snapshot}=${record.code};`);records.set(scalar.argument,snapshot);}value={type:scalar.type,code:snapshot+scalar.path.map(p=>'.cw_field_'+p).join(''),pre:[]};}else value=e.expr(n.args[scalar.argument]);const code=e.convert(value.code,value.type,scalar.type,n.args[scalar.argument]),local=name+'_'+scalar.word;pre.push(...value.pre,`let ${local}=${code};`);values.push({word:scalar.word,code:scalar.type==='u32'?local:scalar.type==='bool'?`select(0u,1u,${local})`:`bitcast<u32>(${local})`});}
 const max=grid.type==='f32'?'65535.0f':grid.type==='u32'?'65535u':'65535i',minimum=grid.type==='f32'?'1.0f':grid.type==='u32'?'1u':'1i';
 pre.push(`if(${name}_grid>=${minimum} && ${name}_grid<=${max}${pointerGuards.length?' && '+pointerGuards.join(' && '):''}) {`,`let ${name}_slot=atomicAdd(&${q.variable}.count,1u);`,`if(${name}_slot<${q.capacity}u) {`,`let ${name}_base=${name}_slot*${q.stride}u;`,`${q.variable}.words[${name}_base]=u32(${name}_grid);`,`${q.variable}.words[${name}_base+1u]=1u;`,`${q.variable}.words[${name}_base+2u]=1u;`,...values.map(v=>`${q.variable}.words[${name}_base+${v.word}u]=${v.code};`),`} else { atomicStore(&${q.variable}.overflow,1u); }`,`} else { atomicStore(&${q.variable}.overflow,1u); }`);
 return e.result(n,'void','',pre);
}
