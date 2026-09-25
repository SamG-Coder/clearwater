import {vectorConstructor} from './vector-constructor.js';
import {unrollCount} from './loop-unroll.js';
import {trimWgslDependencies} from './wgsl-dependencies.js';
import {specializeDependencies} from './specialize-dependencies.js';
import {cloneAst} from './clone-ast.js';
import {pruneConstexpr} from './constexpr.js';
import {lowerPrintf} from './diagnostics.js';
import {containsNativeBool,nativeRecordLayout,decodeNativeRecord} from './native-records.js';
import {lowerNativeTiles,emitNativeTile} from './native-tiles.js';
import {lowerReturnPhases} from './return-phases.js';
import {markUniformRecordSnapshots} from './uniform-records.js';
import {recordLeaves,recordConstructor} from './record-parameters.js';
import {addBufferImports,bufferReferenceArgument,bufferReferenceIndex} from './buffer-references.js';
import {launchQueues,launchQueueDeclarations,launchQueueStorageTypes,emitLaunch} from './device-launch.js';
import {deviceHeaps,deviceHeapDeclarations,deviceHeapCall,deviceHeapIndex} from './device-heap.js';
import {lowerConstantRows} from './constant-rows.js';
import {lowerDeferredPointers} from './deferred-pointers.js';
import {lowerTiledGroups} from './tiled-groups.js';
import {FLOAT64_WGSL} from './float64.js';
import {inferTextureTypes,textureShape} from './texture-types.js';
import {integerExpression} from './integer-expression.js';
import {parse, CompileError,builtinType} from './parser.js';
import {uniformBlockGuards} from './full-workgroups.js';
export {CompileError, parse};
export const COMPILER_VERSION = '0.1.1';
export const isArray = t => !!t && typeof t === 'object' && t.kind === 'array';
export const arrayOf = (element, length = null) => ({kind: 'array', element, length});
export const typeName = t => isArray(t) ? `array<${typeName(t.element)}${t.length == null ? '' : `, ${t.length}`}>` : t;
export const vectorLength = t => typeof t === 'string' && /^vec[234]</.test(t) ? Number(t[3]) : 0;
export const vectorElement = t => vectorLength(t)?t.slice(5,-1):t;
export const typeStride = t => isArray(t) ? typeStride(t.element) * t.length : (vectorLength(t) === 3 ? 4 : (vectorLength(t) || 1)) * (vectorElement(t)==='f16'?2:4);
const numeric = t => ['f16','f32', 'i32', 'u32','cw_uchar','cw_short','cw_ushort'].includes(t);
const uniformScalar=t=>['cw_short','cw_ushort'].includes(t)?{type:t==='cw_short'?'i32':'u32',sourceType:t}:{type:t};
const narrow = t => ['cw_uchar','cw_short','cw_ushort'].includes(t);
const indent = lines => lines.map(l => `  ${l}`);
const rootName = n => n?.kind === 'id' ? n.name : ['index', 'member'].includes(n?.kind) ? rootName(n.base) : n?.kind==='object-deref'?rootName(n.value):null;
// Separate a named pointer root from left-associated element offsets.
const pointerParts=n=>{
  if(n?.kind==='id')return {base:n,offset:null};
  if(n?.kind==='unary'&&n.op==='&'&&n.value.kind==='index'&&n.value.base.kind==='id')return {base:n.value.base,offset:n.value.index};
  if(n?.kind==='binary'&&['+','-'].includes(n.op)){const p=pointerParts(n.left);if(p)return {base:p.base,offset:p.offset?{...n,left:p.offset}:n.op==='+'?n.right:{kind:'unary',op:'-',value:n.right,token:n.token}};}
  return null;
};
const shiftedPointers=fn=>{const names=new Set();walk(fn.body,n=>{if(n.kind==='assign'&&['=','+=','-='].includes(n.op)&&n.left.kind==='id')names.add(n.left.name);});return names;};
export function walk(node, visit) {
  if (!node || typeof node !== 'object') return;
  if (node.kind) visit(node);
  for (const [key, val] of Object.entries(node)) {
    if (['token', 'symbol', 'type', 'source', 'resolved','pointerBaseSymbol'].includes(key)) continue;
    if (Array.isArray(val)) val.forEach(n => walk(n, visit));
    else if (val && typeof val === 'object') walk(val, visit);
  }
}
const cudaValueSize=type=>vectorElement(type)==='f16'?(vectorLength(type)||1)*2:['cw_short','cw_ushort','cw_uchar2'].includes(type)?2:['cw_uchar','bool'].includes(type)?1:type==='cw_uchar4'?4:type==='cw_extent'?24:type==='cw_size64'?8:['f32','i32','u32'].includes(type)?4:vectorLength(type)?vectorLength(type)*4:null;
function constantValue(n) {
  if(n.kind==='sizeof'){const size=cudaValueSize(n.target);if(size===null)throw new CompileError('sizeof requires a supported built-in value type.',n.token);return size;}
  if (n.kind === 'literal') return Number(n.value.replace(/^0[xX]/.test(n.value) ? /[uU]$/ : /[fFuU]$/, ''));
  if (n.kind === 'unary' && ['+', '-'].includes(n.op)) return (n.op === '-' ? -1 : 1) * constantValue(n.value);
  if (n.kind === 'binary') {
    const a = constantValue(n.left), b = constantValue(n.right);
    const ops = {'+': () => a + b, '-': () => a - b, '*': () => a * b, '/': () => Math.trunc(a / b), '%': () => a % b, '<<': () => a << b, '>>': () => a >> b};
    if (ops[n.op]) return ops[n.op]();
  }
  throw new CompileError('Array dimensions must be positive compile-time integer expressions.', n.token);
}
function analyse(functions, params) {
  const atomic = new Set(),atomicBindings=new Set(), reads = new Set(), writes = new Set(), bufferNames = new Set(params.filter(p => p.pointer).map(p => p.name));
  let aliases=new Map();const resolve=name=>aliases.has(name)?aliases.get(name):name;
  const scan = (n, mode = 'read') => {
    if (!n || !n.kind) return;
    if(n.kind==='block'||n.kind==='for'){const saved=aliases;aliases=new Map(aliases);if(n.kind==='block')n.body.forEach(s=>scan(s));else{scan(n.init);scan(n.condition);scan(n.body);scan(n.step);}aliases=saved;return;}
    if(['if','while','do'].includes(n.kind)){scan(n.condition);const saved=aliases;aliases=new Map(saved);scan(n.kind==='if'?n.yes:n.body);aliases=new Map(saved);if(n.kind==='if')scan(n.no);aliases=saved;return;}
    if(n.kind==='decl'){scan(n.init);const base=pointerParts(n.init)?.base;aliases.set(n.name,n.pointer&&base?.kind==='id'?resolve(base.name):null);return;}
    if(n.kind==='call'&&n.callee.kind==='id'&&n.callee.name==='cudaMalloc'){scan(n.args[0]?.value?.value,'write');scan(n.args[1]);return;}
    if (n.kind === 'assign') { scan(n.left, n.op === '=' ? 'write' : 'both'); scan(n.right); return; }
    if (n.kind === 'unary' && ['++', '--'].includes(n.op)) { scan(n.value, 'both'); return; }
    if (n.kind === 'call' && n.callee.kind === 'id' && ['atomicAdd', 'atomicMin', 'atomicMax', 'atomicExch','atomicCAS'].includes(n.callee.name)) {
      const target = n.args[0];
      if (target?.kind === 'unary' && target.op === '&') { const name = rootName(target.value); if (name){atomic.add(resolve(name)??name);if(bufferNames.has(resolve(name)))atomicBindings.add(resolve(name));} scan(target.value, 'both'); }
      n.args.slice(1).forEach(a => scan(a)); return;
    }
    if(n.kind==='object-deref'){const name=resolve(rootName(n.value));if(bufferNames.has(name)){if(mode!=='write')reads.add(name);if(mode!=='read')writes.add(name);}scan(n.value);return;}
    if (n.kind === 'index') {
      const castParts=n.base.kind==='pointer-cast'?pointerParts(n.base.value):null;
      const name = resolve(castParts?.base?.name||rootName(n)); if (bufferNames.has(name)) { if (mode !== 'write') reads.add(name); if (mode !== 'read') writes.add(name); }
      if(castParts?.offset)scan(castParts.offset);
      scan(n.index); if (n.base.kind === 'index') scan(n.base, mode); return;
    }
    if (n.kind === 'member') {const name=resolve(rootName(n.base));if(mode!=='read'&&n.base.kind==='index'&&params.some(p=>p.pointer&&p.name===name&&p.type==='cw_uchar4')){atomic.add(name);atomicBindings.add(name);reads.add(name);}scan(n.base, mode); return; }
    for (const [key, val] of Object.entries(n)) {
      if (['token', 'source'].includes(key)) continue;
      if (Array.isArray(val)) val.forEach(x => scan(x)); else if (val?.kind) scan(val);
    }
  };
  functions.forEach(f => {aliases=new Map(f.params.filter(p=>!p.pointer).map(p=>[p.name,null]));scan(f.body);});
  return {atomic,atomicBindings, reads, writes, storageBarrier: [...writes].some(x => reads.has(x))};
}
class Emitter {
  constructor(ast, kernel, options, templates,bufferUsage) {
    this.ast = ast; this.kernel = kernel; this.options = options; this.scopes = [new Map()]; this.temp = 0; this.loopDepth = 0; this.integerIntrinsics=new Set();
    this.structs=new Map((ast.structs||[]).map(s=>[s.type,s]));for(const s of this.structs.values())for(const field of s.fields){let type=field.type;for(const dim of [...field.dimensions].reverse()){const length=constantValue(dim);if(!Number.isSafeInteger(length)||length<1||length>256)this.fail('Struct field array dimensions must be 1..256.',field);type=arrayOf(type,length);}field.resolvedType=type;}
    inferTextureTypes(ast.functions,kernel,walk,(message,node)=>this.fail(message,node));
    this.objectHeaps=new Map();walk(ast,node=>{if(node.kind==='object-new'&&!this.objectHeaps.has(node.name))this.objectHeaps.set(node.name,{name:node.name,type:'cw_struct_'+node.name,code:'cw_heap_'+node.name,capacity:1024,tag:this.objectHeaps.size+1});});
    this.persistentObjects=options.objectHeap==='persistent';if(options.objectHeap!==undefined&&!['persistent','invocation'].includes(options.objectHeap))this.fail('objectHeap must be persistent or invocation.',kernel);
    this.objectImports=[];
    if(ast.objectListTypes?.length){if(!this.persistentObjects)this.fail('Captured object pointer lists require persistent object arenas.',kernel);for(const fn of ast.functions.filter(f=>f.qualifier==='__global__'))for(const p of fn.params)if(p.pointer&&String(p.type).startsWith('cw_objectptr_')){const existing=this.objectImports.find(i=>i.name===p.name);if(existing&&existing.type!==p.type)this.fail('Shared object buffer names must retain one type.',p);if(!existing)this.objectImports.push({name:p.name,type:p.type,id:this.objectImports.length+1,binding:this.objectHeaps.size+this.objectImports.length,targets:[]});}}
    addBufferImports(this);
    for(const fn of ast.functions.filter(f=>f.qualifier==='__global__'))walk(fn.body,n=>{if(n.kind==='assign'){const name=rootName(n.left),imported=this.objectImports.find(i=>i.name===name&&fn.params.some(p=>p.name===name));if(imported){if(n.right?.kind==='object-new'){if(!imported.targets.includes(n.right.name))imported.targets.push(n.right.name);}else if(!(n.right?.kind==='id'&&['NULL','nullptr'].includes(n.right.name))&&!(n.right?.kind==='literal'&&Number(n.right.value)===0))imported.unknownTargets=true;}}});
    for(const record of this.structs.values()){
      record.listOrigins={};
      for(const ctor of ast.functions.filter(f=>f.classConstructor&&f.classOwner===record.name)){
        const fields=[];walk(ctor.body,n=>{if(n.kind==='assign'&&n.left?.kind==='member'&&n.right?.kind==='id'){const index=ctor.params.findIndex(p=>p.name===n.right.name&&String(p.type).startsWith('cw_objectlist_'));if(index>=0)fields.push({field:n.left.member,index});}});
        for(const fn of ast.functions.filter(f=>f.qualifier==='__global__'))walk(fn.body,n=>{if(n.kind==='object-new'&&n.name===record.name)for(const f of fields){const arg=n.args[f.index],imported=this.objectImports.find(i=>arg?.kind==='id'&&i.name===arg.name&&fn.params.some(p=>p.name===arg.name));if(imported){const origins=record.listOrigins[f.field]??=[];if(!origins.includes(imported.id))origins.push(imported.id);}}});
      }
    }
    const storageLayout=type=>{if(isArray(type)){const e=storageLayout(type.element),stride=Math.ceil(e.size/e.align)*e.align;return {align:e.align,size:stride*type.length};}if(this.structs.has(type)){let size=0,align=4;for(const f of this.structs.get(type).fields){const v=storageLayout(f.resolvedType);align=Math.max(align,v.align);size=Math.ceil(size/v.align)*v.align+v.size;}return {align,size:Math.ceil(size/align)*align};}const width=vectorLength(type);if(width)return {align:width===2?8:16,size:width*4};if(['f32','i32','u32','cw_uchar','cw_uchar2','cw_uchar4','cw_short','cw_ushort'].includes(type)||(String(type).startsWith('cw_objectptr_')||String(type).startsWith('cw_objectlist_')||(String(type).startsWith('cw_deviceptr_')||String(type).startsWith('cw_bufferref_'))))return {align:4,size:4};this.fail('Persistent object fields require host-shareable scalar, vector or record values.',kernel);};
    this.storageLayout=storageLayout;
    this.containsDevicePointer=type=>(String(type).startsWith('cw_deviceptr_')||String(type).startsWith('cw_bufferref_'))||isArray(type)&&this.containsDevicePointer(type.element)||this.structs.has(type)&&this.structs.get(type).fields.some(f=>this.containsDevicePointer(f.resolvedType));
    for(const heap of this.objectHeaps.values()){heap.aliveCode=heap.code+'_alive';if(this.persistentObjects){const layout=storageLayout(heap.type);heap.binding=this.objectHeaps.size?heap.tag-1:0;heap.byteLength=Math.ceil((heap.capacity*4)/layout.align)*layout.align+layout.size*heap.capacity;heap.recordLayout=JSON.stringify([...this.structs.values()].map(s=>({type:s.type,fields:s.fields.map(f=>({name:f.name,type:f.resolvedType}))})));heap.variable=heap.code+'_storage';heap.code=heap.variable+'.objects';heap.aliveCode=heap.variable+'.alive';}}
    this.deviceHeaps=deviceHeaps(this);
    this.launchQueues=launchQueues(this,walk,constantValue,pointerParts);
    if(options.deviceLaunchConsumer!==undefined){
      this.launchConsumer=this.launchQueues.find(q=>q.id===options.deviceLaunchConsumer&&q.child===kernel.name);
      if(!this.launchConsumer||!this.launchConsumer.frontier&&this.launchQueues.some(q=>q.caller===kernel.name))this.fail('Queue consumers require a matching leaf child kernel.',kernel);

    }
    this.overloads=new Map();for(const f of ast.functions)if(f.overloadName){const list=this.overloads.get(f.overloadName)||[];list.push(f);this.overloads.set(f.overloadName,list);}
    this.functions = new Map(); this.shared = [];this.pointerConstraints=[]; this.templates=templates;this.helperCalls=new Map();this.globalSymbols=new Map();this.constantScalars=[];
    for (const f of ast.functions) {
      if (this.functions.has(f.name)) this.fail(`Duplicate function '${f.name}'.`, f);
      this.functions.set(f.name, f);
    }
    this.helpers = ast.functions.filter(f => f.qualifier === '__device__'&&!f.params.some(p=>p.pointer));
    this.pointerHelpers=new Map();this.referenceHelpers=new Map();this.bufferSymbols=new Map();
    this.bufferAliases=options.bufferAliases??{};
    if(typeof this.bufferAliases!=='object'||Array.isArray(this.bufferAliases))this.fail('bufferAliases must map pointer names to canonical pointer names.',kernel);
    for(const [alias,target]of Object.entries(this.bufferAliases)){const a=kernel.params.find(p=>p.name===alias),b=kernel.params.find(p=>p.name===target);if(typeof target!=='string'||alias===target||Object.hasOwn(this.bufferAliases,target)||!a?.pointer||!b?.pointer||a.type!==b.type)this.fail('Buffer aliases require distinct same-type pointer parameters and no chains.',kernel);}
    this.deviceParams=(ast.deviceGlobals||[]).map(g=>{const count=constantValue(g.length);if(!Number.isSafeInteger(count)||count<1||count>16777216)this.fail('Device global arrays require 1..16777216 elements.',g);if(kernel.params.some(p=>p.name===g.name))this.fail('Kernel parameters cannot shadow device global bindings.',g);return {...g,pointer:true,constant:false,origin:'device-global',count};});
    for(const g of ast.constantGlobals)if(this.structs.has(g.type)&&g.dimensions.length){
      const fields=this.structs.get(g.type).fields;if(!fields.length||g.dimensions.length!==1||g.init||fields.some(f=>f.dimensions.length||!['f32','i32','u32'].includes(f.type)))this.fail('Constant struct storage requires an uninitialized one-dimensional array of flat 32-bit scalar fields.',g);
      const count=constantValue(g.dimensions[0]),stride=fields.length*4;if(!Number.isSafeInteger(count)||count<1||count*stride>67108864)this.fail('Constant struct storage must fit 64 MiB.',g);
      if(kernel.params.some(p=>p.name===g.name)||this.deviceParams.some(p=>p.name===g.name))this.fail('Constant storage binding names must be distinct.',g);
      this.deviceParams.push({...g,pointer:true,constant:true,origin:'constant-struct-storage',count,storageStride:stride,fields:fields.map((f,i)=>({name:f.name,type:f.type,offset:i*4}))});
    }
    this.usage = analyse([kernel], [...kernel.params,...this.deviceParams]);this.usage.atomic=this.usage.atomicBindings;
    const globalUsage=analyse(ast.functions.map(f=>({...f,params:f.params.map(p=>({...p,pointer:false}))})),this.deviceParams);for(const kind of ['reads','writes'])for(const name of globalUsage[kind])this.usage[kind].add(name);for(const name of globalUsage.atomicBindings)this.usage.atomic.add(name);
    for(const kind of ['reads','writes','atomic'])for(const name of bufferUsage?.[kind]||[])this.usage[kind].add(name);
    for(const kind of ['reads','writes','atomic'])for(const [alias,target]of Object.entries(this.bufferAliases))if(this.usage[kind].has(alias))this.usage[kind].add(target);
    for(const p of kernel.params)if(p.pointer&&(['cw_uchar','bool'].includes(p.type)&&this.usage.writes.has(p.name)||p.volatileParameter))this.usage.atomic.add(p.name);
    this.usage.storageBarrier=[...this.usage.writes].some(n=>this.usage.reads.has(n));
    this.initialBufferUsage=Object.fromEntries(['reads','writes','atomic'].map(k=>[k,new Set(this.usage[k])]));
    this.workgroupSize = [...(options.workgroupSize || [128, 1, 1])];
    while (this.workgroupSize.length < 3) this.workgroupSize.push(1);
    if (this.workgroupSize.length !== 3 || this.workgroupSize.some(v => !Number.isSafeInteger(v) || v < 1) || this.workgroupSize.reduce((a, b) => a * b, 1) > 1024) this.fail('Workgroup dimensions must be positive integers with at most 1024 total invocations.', kernel);
    if(kernel.launchThreads&&this.workgroupSize.reduce((a,b)=>a*b,1)>kernel.launchThreads)this.fail(`Launch exceeds __launch_bounds__(${kernel.launchThreads}).`,kernel);
    this.currentFunction = kernel;
    this.dynamicSharedBytes=options.sharedMemoryBytes??0;this.dynamicSharedUsed=false;
    if(!Number.isSafeInteger(this.dynamicSharedBytes)||this.dynamicSharedBytes<0||this.dynamicSharedBytes>65536)this.fail('sharedMemoryBytes must be an integer in [0,65536].',kernel);
  }
  fail(message, n) { throw new CompileError(message, n?.token, this.ast.source); }
  lookup(name, n) {
    for (let i = this.scopes.length - 1; i >= 0; --i) { const s = this.scopes[i].get(name); if (s) return s; }
    if(this.globalSymbols.has(name)&&this.globalSymbols.get(name).kind==='buffer')return this.globalSymbols.get(name);
    const global=this.ast.constantGlobals.find(g=>g.name===name);
    if(global){
      if(global.constexprValue!==undefined){
        if(!global.symbol){const value=global.constexprValue;let code;
          if(global.type==='cw_f64'){const bits=new DataView(new ArrayBuffer(8));bits.setFloat64(0,value,true);this.float64Used=true;code=`vec2<u32>(${bits.getUint32(0,true)}u, ${bits.getUint32(4,true)}u)`;}
          else if(global.type==='f32'){const bits=new DataView(new ArrayBuffer(4));bits.setFloat32(0,value,true);code=`bitcast<f32>(${bits.getUint32(0,true)}u)`;}
          else code=global.type==='u32'?`${value}u`:`i32(${value})`;
          global.symbol={name,type:global.type,code,constant:true,kind:'constant-global',constexprValue:value};
        }
        return global.symbol;
      }
      if(this.structs.has(global.type)||vectorLength(global.type)||global.dimensions.length===2){
        if(!this.globalSymbols.has(name)){const initial=[];if(global.init){if(!vectorLength(global.type)||global.dimensions.length||global.init.kind!=='initializer'||global.init.items.length>vectorLength(global.type))this.fail('Constant vector initialization requires a flat scalar list; initialized aggregate arrays and structs are unsupported.',global);const element=vectorElement(global.type);for(const item of global.init.items){const literal=item.kind==='unary'&&['+','-'].includes(item.op)?item.value:item;if(literal.kind!=='literal')this.fail('Constant vector components require numeric literals.',item);const value=constantValue(item);if(!Number.isFinite(value)||element==='f32'&&!Number.isFinite(Math.fround(value))||element==='i32'&&(!Number.isInteger(value)||value<-2147483648||value>2147483647)||element==='u32'&&(!Number.isInteger(value)||value<0||value>4294967295))this.fail('Constant vector component is outside its scalar range.',item);initial.push(element==='f32'?Math.fround(value):value);}}if(this.structs.has(global.type)&&global.dimensions.length)this.fail('Constant struct arrays are unsupported.',global);let aggregateType=global.type;if(global.dimensions.length){let total=1;for(const dim of [...global.dimensions].reverse()){const length=constantValue(dim);total*=length;if(!Number.isInteger(length)||length<1||total>256)this.fail('Constant aggregate arrays require 1..256 total elements.',global);aggregateType=arrayOf(aggregateType,length);}}const leaves=[];
          const build=(type,path)=>{if(type==='cw_f64'||type==='cw_size64'){if(type==='cw_f64')this.float64Used=true;else this.extentUsed=true;const low=build('u32',path+'.lo'),high=build('u32',path+'.hi');return {code:`vec2<u32>(${low.code}, ${high.code})`,shape:{kind:'wide',type,items:[low.shape,high.shape]}};}if(this.structs.has(type)){const fields=this.structs.get(type).fields.map(f=>[f.name,build(f.resolvedType,path+'.'+f.name)]);return {code:`${type}(${fields.map(([,v])=>v.code).join(', ')})`,shape:{kind:'struct',fields:fields.map(([name,v])=>[name,v.shape])}};}if(isArray(type)||vectorLength(type)){const count=isArray(type)?type.length:vectorLength(type),element=isArray(type)?type.element:vectorElement(type),items=Array.from({length:count},(_,i)=>build(element,path+(isArray(type)?'['+i+']':'.'+'xyzw'[i])));return {code:`${typeName(type)}(${items.map(v=>v.code).join(', ')})`,shape:{kind:'array',items:items.map(v=>v.shape)}};}if(!numeric(type)&&type!=='bool'||leaves.length>=(vectorLength(global.type)?1024:256))this.fail('Constant aggregates exceed the supported component limit.',global);const field='cw_struct_constant_'+this.ast.constantGlobals.indexOf(global)+'_'+leaves.length;leaves.push({name:path,...(type==='bool'?{type:'u32',sourceType:'bool'}:uniformScalar(type)),origin:'constant',field,defaultValue:initial[leaves.length]??0});return {code:type==='bool'?'(cw_params.'+field+' != 0u)':'cw_params.'+field,shape:{kind:'scalar',name:path,type}};};
          const value=build(aggregateType,'constant.'+name),symbol={name:'constant.'+name,type:aggregateType,code:value.code,constant:true,atomic:false,kind:'constant-global',aggregate:value.shape};this.constantScalars.push(...leaves);this.globalSymbols.set(name,symbol);global.symbol=symbol;
        }return this.globalSymbols.get(name);
      }
      if(!numeric(global.type))this.fail('Referenced constant globals require float, int or unsigned int scalar values.',n);
      if(!this.globalSymbols.has(name)){
        if(global.dimensions?.length){
          const length=constantValue(global.dimensions[0]);if(!Number.isInteger(length)||length<1||length>256)this.fail('Constant arrays require 1..256 scalar elements.',global);
          if(global.init&&(global.init.kind!=='initializer'||global.init.items.length>length))this.fail('Constant array initializers require a scalar list no longer than the array.',global);
          const fields=[],values=[];
          for(let i=0;i<length;i++){
            const init=global.init?.items[i];let value=0;
            if(init){const literal=init.kind==='unary'&&['+','-'].includes(init.op)?init.value:init;if(literal.kind!=='literal')this.fail('Constant array initializers must be numeric literals with an optional sign.',global);value=constantValue(init);}
            if(['cw_short','cw_ushort'].includes(global.type)&&(!Number.isInteger(value)||value<(global.type==='cw_short'?-32768:0)||value>(global.type==='cw_short'?32767:65535)))this.fail('Short constant initializer is outside its 16-bit range.',global);
            if(!Number.isFinite(value)||(global.type==='f32'&&!Number.isFinite(Math.fround(value)))||(global.type==='u32'&&(!Number.isInteger(value)||value<0||value>4294967295))||(global.type==='i32'&&(!Number.isInteger(value)||value<-2147483648||value>2147483647)))this.fail('Constant array initializer is outside its scalar range.',global);
            const field=`cw_array_${this.ast.constantGlobals.indexOf(global)}_${i}`,scalarName=`constant.${name}[${i}]`;
            fields.push('cw_params.'+field);values.push(scalarName);this.constantScalars.push({name:scalarName,...uniformScalar(global.type),origin:'constant',field,defaultValue:global.type==='f32'?Math.fround(value):value});
          }
          const symbol={name:'constant.'+name,type:arrayOf(global.type,length),code:`array<${global.type}, ${length}>(${fields.join(', ')})`,constant:true,atomic:false,kind:'constant-global',elements:values};this.globalSymbols.set(name,symbol);global.symbol=symbol;return symbol;
        }
        let value=0;
        if(global.init){const literal=global.init.kind==='unary'&&['+','-'].includes(global.init.op)?global.init.value:global.init;if(literal.kind!=='literal')this.fail('Constant global initializers must be numeric literals with an optional sign.',global);value=constantValue(global.init);}
        if(['cw_short','cw_ushort'].includes(global.type)&&(!Number.isInteger(value)||value<(global.type==='cw_short'?-32768:0)||value>(global.type==='cw_short'?32767:65535)))this.fail('Short constant initializer is outside its 16-bit range.',global);
            if(!Number.isFinite(value)||(global.type==='f32'&&!Number.isFinite(Math.fround(value)))||(global.type==='u32'&&(!Number.isInteger(value)||value<0||value>4294967295))||(global.type==='i32'&&(!Number.isInteger(value)||value<-2147483648||value>2147483647)))this.fail('Constant global initializer is outside its supported scalar range.',global);
        const scalarName='constant.'+name,symbol={name:scalarName,type:global.type,code:'cw_params.c_'+name,constant:true,atomic:false,kind:'constant-global'};
        this.globalSymbols.set(name,symbol);global.symbol=symbol;
        this.constantScalars.push({name:scalarName,...uniformScalar(global.type),origin:'constant',field:'c_'+name,defaultValue:global.type==='f32'?Math.fround(value):value});
      }
      return this.globalSymbols.get(name);
    }
    this.fail(`Unknown identifier '${name}'.`, n);
  }
  add(name, symbol, n, global = false) { const scope = global ? this.scopes[0] : this.scopes.at(-1); if (scope.has(name)) this.fail(`Duplicate identifier '${name}'.`, n); scope.set(name, symbol); return symbol; }
  common(a, b, n) {
    if (typeName(a) === typeName(b) && !isArray(a) && a !== 'void') return a;
    if((a==='cw_f64'||b==='cw_f64')&&[a,b].every(t=>t==='cw_f64'||numeric(t)||t==='bool'))return 'cw_f64';
    // C++ promotes a bool to int before the usual scalar arithmetic conversions.
    if(a==='bool'&&numeric(b))a='i32';
    if(b==='bool'&&numeric(a))b='i32';
    if (numeric(a) && numeric(b)) return ['f32','f16'].includes(a) || ['f32','f16'].includes(b) ? 'f32' : a === 'u32' || b === 'u32' ? 'u32' : 'i32';
    this.fail(`Incompatible operand types: ${typeName(a)} and ${typeName(b)}. Use explicit scalar/vector components.`, n);
  }
  convert(code, from, to, n) {
    if(String(from).startsWith('cw_objectptr_')&&String(to).startsWith('cw_objectptr_')&&this.structs.get('cw_struct_'+from.slice(13))?.base===to.slice(13))return code;
    if(to==='bool'&&(String(from).startsWith('cw_objectptr_')||(String(from).startsWith('cw_deviceptr_')||String(from).startsWith('cw_bufferref_'))))return `(${code} != 0u)`;
    if((String(to).startsWith('cw_objectptr_')||(String(to).startsWith('cw_deviceptr_')||String(to).startsWith('cw_bufferref_')))&&['0i','0u','0'].includes(code)&&['i32','u32'].includes(from))return '0u';
    if(to==='bool'&&isArray(from)&&from.length===null&&n){return 'true';} // All runtime storage bindings are required and non-null.
    if(from==='cw_size64'||to==='cw_size64')this.extentUsed=true;
    if(to==='cw_size64'&&['i32','u32','bool'].includes(from))return from==='i32'?`vec2<u32>(u32(${code}), select(0u, 4294967295u, ${code} < 0i))`:from==='bool'?`vec2<u32>(select(0u,1u,${code}),0u)`:`vec2<u32>(${code},0u)`;
    if (typeName(from) === typeName(to) && !isArray(to)) return code;
    if(from==='f16'&&to==='cw_f64'){this.float64Used=true;return `cw_d_from_f32(f32(${code}))`;}
    if(to==='cw_f64'&&(numeric(from)||from==='bool')){this.float64Used=true;return from==='bool'?`cw_d_from_u32(select(0u,1u,${code}))`:from==='f32'?`cw_d_from_f32(${code})`:['i32','cw_short'].includes(from)?`cw_d_from_i32(i32(${code}))`:`cw_d_from_u32(u32(${code}))`;}
    if(from==='cw_f64'&&to==='f32'){this.float64Used=true;return `cw_d_to_f32(${code})`;}
    if(from==='cw_f64'&&to==='bool'){this.float64Used=true;return `!cw_d_zero(${code})`;}
    if(from==='cw_size64'&&to==='f32'){this.float64Used=true;return `cw_d_u64_to_f32(${code})`;}
    if(to==='cw_short'||to==='cw_ushort'){if(from==='bool')return to==='cw_short'?`select(0i,1i,${code})`:`select(0u,1u,${code})`;if(numeric(from)){const word=`i32(${code})`;return to==='cw_short'?`(bitcast<i32>((bitcast<u32>(${word}) & 65535u) << 16u) >> 16u)`:`(u32(${word}) & 65535u)`;}}
    if(from==='cw_short'||from==='cw_ushort'){if(to==='bool')return `(${code} != ${from==='cw_short'?'0i':'0u'})`;if(numeric(to)&&to!=='cw_uchar')return `${to}(${code})`;}
    if(to==='cw_uchar'){if(from==='bool')return `select(0u,1u,${code})`;if(numeric(from))return `(${from==='f32'?`u32(i32(${code}))`:`u32(${code})`} & 255u)`;}
    if(from==='cw_uchar'){if(to==='bool')return `(${code} != 0u)`;if(numeric(to))return `${to}(${code})`;}
    if(from==='cw_size64'&&['i32','u32'].includes(to))return `${to}((${code}).x)`;
    if (numeric(from) && numeric(to)) return `${to}(${code})`;
    if (to === 'bool' && numeric(from)) return `(${code} != ${from === 'f16' ? '0.0h' : from === 'f32' ? '0.0f' : from === 'u32' ? '0u' : '0i'})`;
    if (from === 'bool' && numeric(to)) return `select(${to}(0), ${to}(1), ${code})`;
    this.fail(`Cannot convert ${typeName(from)} to ${typeName(to)}.`, n);
  }
  result(n, type, code, pre = [], extra = {}) { n.type = type; return {type, code, pre, ...extra}; }
  expr(n, raw = false, captureIndex = false) {
    if(this.expressionCache?.has(n))return this.expressionCache.get(n);
    if(!raw&&n?.kind==='binary'&&!['&&','||'].includes(n.op)){
      const chain=[];let leaf=n;
      while(leaf?.kind==='binary'&&!['&&','||'].includes(leaf.op)){chain.push(leaf);leaf=leaf.left;}
      if(chain.length>32){
        const cache=this.expressionCache??=new Map(),owned=[];const save=(node,value)=>{owned.push([node,cache.has(node),cache.get(node)]);cache.set(node,value);};
        let value=this.expr(leaf);save(leaf,value);
        try{for(let i=chain.length-1;i>=0;i--){const node=chain[i];value=this.exprNode(node,false,false);
          if(['f32','i32','u32'].includes(value.type)){const name='cw_expression_'+this.temp++;value={...value,code:name,pre:[...value.pre,`let ${name}: ${value.type} = ${value.code};`]};}
          save(node,value);
        }return value;}finally{for(const [node,present,previous]of owned.reverse()){if(present)cache.set(node,previous);else cache.delete(node);}}
      }
    }
    return this.exprNode(n,raw,captureIndex);
  }
  exprNode(n, raw = false, captureIndex = false) {

    if (!n) this.fail('Missing expression.', this.kernel);
    switch (n.kind) {
      case 'initializer': {
        if(isArray(n.target)){
          const {element,length}=n.target;if(!Number.isInteger(length)||n.items.length>length)this.fail('Array initializer has too many elements.',n);
          if(!isArray(element)&&!['i32','u32','f32','bool'].includes(element))this.fail('Local array initializers require scalar elements or explicitly nested scalar arrays.',n);
          const code='cw_array_init_'+this.temp++,pre=[`var ${code}: ${typeName(n.target)};`];
          for(let i=0;i<n.items.length;i++){const item=n.items[i];if(isArray(element)&&item.kind!=='initializer')this.fail('Nested arrays require explicit brace initializers.',item);if(item.kind==='initializer')item.target=element;const value=this.expr(item);let compatible=typeName(value.type)===typeName(element);
            if(!compatible&&['i32','u32'].includes(value.type)&&['i32','u32','f32'].includes(element)){try{const v=constantValue(item);compatible=Number.isInteger(v)&&(element==='f32'?Math.fround(v)===v:element==='u32'?v>=0&&v<=4294967295:v>=-2147483648&&v<=2147483647);}catch{}}
            if(!compatible)this.fail('Array initializer narrows or changes its element type; use an explicit cast.',item);
            pre.push(...value.pre,`${code}[${i}u] = ${isArray(element)?value.code:this.convert(value.code,value.type,element,item)};`);
          }
          return this.result(n,n.target,code,pre);
        }
        const width=vectorLength(n.target),element=vectorElement(n.target);
        if(!width||n.items.length>width)this.fail('Vector initializers require at most one scalar per component.',n);
        const values=n.items.map(item=>this.expr(item));
        if(values.some((v,i)=>{if(v.type===element)return false;if(element==='f32'&&['i32','u32'].includes(v.type)){try{const value=constantValue(n.items[i]);return !Number.isInteger(value)||Math.fround(value)!==value;}catch{}}return true;}))this.fail('Vector initializer components must match the element type; use explicit casts for conversions.',n);
        const codes=values.map(v=>this.convert(v.code,v.type,element,n));while(codes.length<width)codes.push(`${element}(0)`);
        return this.result(n,n.target,`${n.target}(${codes.join(', ')})`,values.flatMap(v=>v.pre));
      }
      case 'byte-index': {
        const base=this.lookup(n.baseName,n),stride=cudaValueSize(n.target);if(!['buffer','buffer-alias','shared'].includes(base.kind)||base.type.element!==n.target||!stride)this.fail('Byte pointer casts must retain the storage pointee type.',n);
        const multiple=node=>{if(node?.kind==='literal')return Number(node.value.replace(/[uU]$/,''))%stride===0;if(node?.kind==='sizeof')return cudaValueSize(node.target)%stride===0;if(node?.kind==='binary'&&node.op==='*')return multiple(node.left)||multiple(node.right);return false;};
        if(stride>1&&!multiple(n.offset)){const factor=n.offset?.kind==='binary'&&n.offset.op==='*'?[n.offset.left,n.offset.right].find(a=>a.kind==='id'&&this.lookup(a.name,a).kind==='uniform'&&['i32','u32','cw_short','cw_ushort','cw_size64'].includes(this.lookup(a.name,a).type)):null;if(!factor)this.fail('Byte-address casts to wider types need provable alignment or an integer launch pitch factor.',n);this.pointerConstraints.push({name:factor.name,minimum:0,multipleOf:stride});}
        if(base.constant&&!n.constant)this.fail('Cannot discard const through a byte pointer cast.',n);const offset=this.expr(n.offset);if(!['i32','u32','cw_size64'].includes(offset.type))this.fail('Byte offsets require integer values.',n);n.resolvedStride=stride;
        if(offset.type==='cw_size64'){const shift=Math.log2(stride);if(!Number.isInteger(shift))this.fail('Wide byte addressing requires power-of-two pointee size.',n);const name='cw_byte_offset_'+this.temp++,low=shift?`((${name}.x >> ${shift}u) | (${name}.y << ${32-shift}u))`:`${name}.x`,maximumHigh=shift?2**(shift-1)-1:0;const tooLarge=shift?`${name}.y > ${maximumHigh}u`:`(${name}.y != 0u || ${name}.x > 2147483647u)`;return this.result(n,'i32',`select(i32(${low}),2147483647i,${tooLarge})`,[...offset.pre,`let ${name} = ${offset.code};`]);}
        return this.result(n,offset.type,`(${offset.code} / ${stride}${offset.type==='u32'?'u':'i'})`,offset.pre);
      }
      case 'sizeof': {const size=cudaValueSize(n.target);if(size===null)this.fail('sizeof requires a supported built-in value type.',n);this.extentUsed=true;n.numericValue=size;return this.result(n,'cw_size64',`vec2<u32>(${size}u, 0u)`);}
      case 'literal': {
        const isHex = /^0[xX]/.test(n.value), isFloat = !isHex && /[fF]$/.test(n.value), isUnsigned = /[uU]$/.test(n.value);
        if (!isFloat && /[.eE]/.test(n.value) && !/^0[xX]/.test(n.value)) {const value=Number(n.value);if(!Number.isFinite(value))this.fail('Invalid double literal.',n);const bits=new DataView(new ArrayBuffer(8));bits.setFloat64(0,value,true);this.float64Used=true;n.numericValue=value;return this.result(n,'cw_f64',`vec2<u32>(${bits.getUint32(0,true)}u, ${bits.getUint32(4,true)}u)`);}
        let value = Number(n.value.replace(isHex ? /[uU]$/ : /[fFuU]$/, ''));
        if (!Number.isFinite(value)) this.fail('Invalid numeric literal.', n);
        let type = isFloat ? 'f32' : isUnsigned || (/^0[xX]/.test(n.value) && value > 2147483647) ? 'u32' : 'i32';
        if (!isFloat && (value < (type === 'u32' ? 0 : -2147483648) || value > (type === 'u32' ? 4294967295 : 2147483647))) this.fail('Integer literal is outside the supported 32-bit range.', n);
        if (isFloat && !Number.isFinite(Math.fround(value))) this.fail('Floating literal overflows f32.', n);
        n.numericValue = value;
        if(type==='i32'&&value===-2147483648)return this.result(n,type,'i32(2147483648u)');
        return this.result(n, type, `${isFloat && Number.isInteger(value) && !/[eE]/.test(String(value)) ? value + '.0' : value}${type === 'f32' ? 'f' : type === 'u32' ? 'u' : 'i'}`);
      }
      case 'id': {
        if(n.name==='warpSize'&&!this.scopes.some(scope=>scope.has('warpSize'))){n.kind='literal';n.value='32';delete n.name;return this.expr(n,raw);}
        if(n.name==='NULL'||n.name==='nullptr'){n.kind='literal';n.value='0';delete n.name;return this.expr(n,raw);}
        if (['true', 'false'].includes(n.name)) return this.result(n, 'bool', n.name);
        const s = this.lookup(n.name, n); n.symbol = s;
        if(s.kind==='thread-block')this.fail('A thread_block handle can only be used for block synchronization.',n);
        const atomic=s.atomic&&!isArray(s.type);return this.result(n, s.type, atomic&&!raw?(s.volatileShared&&s.type==='f32'?`bitcast<f32>(atomicLoad(&${s.code}))`:`atomicLoad(&${s.code})`):s.code, [], {rootSymbol: s, atomicRoot: s.atomic,atomic});
      }
      case 'index': {
        if(n.base.kind==='pointer-cast'&&n.base.volatilePointer)this.fail('Volatile pointer casts require shared pointer slot assignment.',n);
        if(n.base.kind==='pointer-cast'&&n.base.target==='cw_uchar2') {
          const cast=n.base,pointer=this.argument(cast.value),index=this.expr(n.index);
          if(pointer.type?.element!=='cw_uchar'||!pointer.rootSymbol?.rootBufferName||!['i32','u32'].includes(index.type))this.fail('uchar2 pointer views require byte storage and an integer index.',n);
          if(raw&&(cast.constant||pointer.rootSymbol.constant))this.fail('Cannot modify a const uchar2 pointer view.',n);
          if(raw&&!pointer.rootSymbol.atomic)this.fail('uchar2 stores require atomic byte storage.',n);
          const offset='cw_pair_offset_'+this.temp++,components=Array.from({length:2},(_,i)=>{const at=`(${offset} + ${i}u)`;return {word:`${pointer.code}[${at} >> 2u]`,shift:`((${at} & 3u) * 8u)`};});
          const values=components.map(({word,shift},i)=>`(((${pointer.rootSymbol.atomic?`atomicLoad(&${word})`:word} >> ${shift}) & 255u) << ${i*8}u)`);
          n.packedPairView=cast.value;
          return this.result(n,'cw_uchar2',`(${values.join(' | ')})`,[...pointer.pre,...index.pre,`let ${offset} = u32(${pointer.pointerCode}) + u32(${index.code}) * 2u;`],{rootSymbol:pointer.rootSymbol,...(raw?{packedPairComponents:components}:{})});
        }
        if(n.base.kind==='pointer-cast'&&[2,4].includes(vectorLength(n.base.target))) {
          const cast=n.base,pointer=this.argument(cast.value),index=this.expr(n.index),element=vectorElement(cast.target),count=vectorLength(cast.target);
          if(!['f32','i32','u32'].includes(element)||pointer.type?.element!==element||!pointer.rootSymbol?.rootBufferName||!['i32','u32'].includes(index.type))this.fail('Vector pointer views require matching 32-bit scalar storage and an integer index.',n);
          if(raw&&(cast.constant||pointer.rootSymbol.constant))this.fail('Cannot modify a const vector pointer view.',n);
          const offset='cw_vector_offset_'+this.temp++,pre=[...pointer.pre,...index.pre,`let ${offset} = ${pointer.pointerCode} + i32(${index.code}) * ${count}i;`];
          const components=Array.from({length:count},(_,i)=>`${pointer.code}[${offset} + ${i}i]`);
          if(pointer.rootSymbol.atomic)this.fail('Vector pointer views of atomic storage are unsupported.',n);
          n.scalarVectorView=cast.value;n.scalarVectorCount=count;
          return this.result(n,cast.target,`${cast.target}(${components.join(', ')})`,pre,{rootSymbol:pointer.rootSymbol,scalarVectorComponents:components});
        }
        if(n.base.kind==='pointer-cast'&&n.base.target==='u32') {
          const cast=n.base,address=cast.value;
          if(address.kind==='unary'&&address.op==='&'&&address.value.kind==='id') {
            const value=this.expr(address.value,true);
            if(value.type!=='cw_uchar4'||value.rootSymbol?.kind!=='local'||constantValue(n.index)!==0)this.fail('Packed word local views require one named uchar4 and index zero.',n);
            if(raw&&cast.constant)this.fail('Cannot modify a const packed word view.',n);
            n.packedWordLocal=address.value;
            return this.result(n,'u32',value.code,value.pre,{rootSymbol:value.rootSymbol});
          }
          if(raw)this.fail('Packed word stores to byte buffers are not yet supported.',n);
          const pointer=this.argument(address),index=this.expr(n.index);
          if(pointer.type?.element!=='cw_uchar'||!pointer.rootSymbol?.rootBufferName||!['i32','u32'].includes(index.type))this.fail('Packed word loads require byte storage and an integer word index.',n);
          const offset='cw_word_byte_'+this.temp++;
          const bytes=Array.from({length:4},(_,i)=>{
            const at=`(${offset} + ${i}u)`,word=`${pointer.code}[${at} >> 2u]`,read=pointer.rootSymbol.atomic?`atomicLoad(&${word})`:word;
            return `(((${read} >> ((${at} & 3u) * 8u)) & 255u) << ${i*8}u)`;
          });
          n.packedWordBytes=address;
          return this.result(n,'u32',`(${bytes.join(' | ')})`,[...pointer.pre,...index.pre,`let ${offset} = u32(${pointer.pointerCode}) + u32(${index.code}) * 4u;`],{rootSymbol:pointer.rootSymbol});
        }
        if(n.pointerTarget&&this.lookup(n.base.name,n).type.element!==n.pointerTarget)this.fail('Byte pointer dereference must retain its pointee type.',n);if(raw&&n.pointerConstant)this.fail('Cannot modify a const byte pointer.',n);
        const base = this.expr(n.base, raw);
        if(String(base.type).startsWith('cw_objectlist_')){
          if(raw)this.fail('Captured object lists currently support indexed reads only.',n);const index=this.expr(n.index),snapshot='cw_list_'+this.temp++,result=snapshot+'_value',imports=this.objectImports.filter(i=>i.type===base.type.replace('cw_objectlist_','cw_objectptr_')&&(!base.objectListOrigins||base.objectListOrigins.includes(i.id)));if(!imports.length)this.fail('Captured object list has no resolved buffer origin.',n);
          const pre=[...base.pre,...index.pre,`let ${snapshot} = ${base.code};`,`var ${result}: ${base.type.replace('cw_objectlist_','cw_objectptr_')};`,...imports.map(i=>`if ((${snapshot} >> 20u) == ${i.id}u) { ${result} = cw_import_${i.id}[(${snapshot} & 1048575u) + u32(${index.code})]; }`)];n.objectListRead={imports:imports.map(i=>i.name)};return this.result(n,base.type.replace('cw_objectlist_','cw_objectptr_'),result,pre,{pointerTargets:imports.some(i=>i.unknownTargets)?undefined:[...new Set(imports.flatMap(i=>i.targets))]});
        }
        if(this.structs.get(base.type)?.methods?.some(m=>m.name==='operator[]')){
          const reference=this.structs.get(base.type).methods.find(m=>m.name==='operator[]'&&m.indexedReference);
          if(reference&&(raw||!base.rootSymbol?.constant)){if(reference.access==='private'&&this.currentFunction.classOwner!==this.structs.get(base.type).name)this.fail('Private class index accessor is inaccessible here.',n);n.base={kind:'member',base:n.base,member:reference.field,token:n.token,accessorOwner:this.structs.get(base.type).name};return this.expr(n,raw);}
          if(raw)this.fail('Read-only class indexing cannot be used as a writable reference.',n);
          const receiver=n.base,index=n.index;delete n.base;delete n.index;Object.assign(n,{kind:'call',callee:{kind:'member',base:receiver,member:'operator[]',token:n.token},args:[index]});return this.call(n);
        }
        const index = this.expr(n.index);
        if(base.rootSymbol?.kind==='pointer-array'){
          const slots=base.rootSymbol,root=slots.pointerRoot;if(!root)this.fail('Pointer array must be assigned a shared-array address before use.',n);
          if(!['i32','u32'].includes(index.type))this.fail('Pointer array indices must be 32-bit integers.',n);
          const temp='cw_shared_pointer_'+this.temp++;n.pointerArrayElement=true;n.pointerBaseSymbol=root;
          const pointerCode=temp,symbol={...root,kind:'buffer-alias',constant:slots.constant||root.constant,sharedPointer:root.code,offsetCode:pointerCode};
          return this.result(n,arrayOf(slots.elementType),root.code,[...base.pre,...index.pre,`let ${temp}: i32 = ${slots.code}[${index.code}];`],{rootSymbol:symbol,pointerCode,atomicRoot:root.atomic});
        }
        if(String(base.type).startsWith('cw_bufferref_'))return bufferReferenceIndex(this,n,base,this.expr(n.index),raw);
        if(String(base.type).startsWith('cw_deviceptr_'))return deviceHeapIndex(this,n,base,index,raw);
        if(n.dereference&&!['buffer','buffer-alias'].includes(base.rootSymbol?.kind))this.fail('Dereference requires a storage-buffer pointer.',n);
        if (!isArray(base.type) || !['i32', 'u32'].includes(index.type)) this.fail('Indexing requires an array and a 32-bit integer index.', n);
        const offset=base.code===base.rootSymbol?.code?base.rootSymbol.offsetCode:undefined;let indexCode=offset?`(${offset} + ${this.convert(index.code,index.type,'i32',n)})`:index.code;
        const capturePre=[];if(captureIndex&&['local','shared','buffer','buffer-alias'].includes(base.rootSymbol?.kind)){const temp='cw_argument_index_'+this.temp++;capturePre.push(`let ${temp} = ${indexCode};`);indexCode=temp;}
        let code = `${base.code}[${indexCode}]`;const type = base.type.element;
        if(base.rootSymbol?.nativeLayout){
          const layout=base.rootSymbol.nativeLayout,indexName='cw_native_index_'+this.temp++,codeName='cw_native_value_'+this.temp++;
          const pre=[...base.pre,...index.pre,...capturePre,`let ${indexName} = u32(${indexCode}) * ${layout.size}u;`,`var ${codeName}: ${type} = ${decodeNativeRecord(layout,base.code,indexName)};`];
          return this.result(n,type,codeName,pre,{rootSymbol:{name:codeName,type,code:codeName,constant:true,kind:'local'}});
        }
        if(['cw_uchar','bool'].includes(type)&&base.rootSymbol?.rootBufferName){const temp='cw_byte_index_'+this.temp++,word=`${base.code}[${temp} >> 2u]`,shift=`((${temp} & 3u) * 8u)`,read=base.atomicRoot?`atomicLoad(&${word})`:word;return this.result(n,type,type==='bool'?`(((${read} >> ${shift}) & 255u) != 0u)`:`((${read} >> ${shift}) & 255u)`,[...base.pre,...index.pre,...capturePre,`let ${temp} = u32(${indexCode});`],{rootSymbol:base.rootSymbol,...(raw&&base.atomicRoot?{packedBase:word,packedShiftCode:shift,packedAtomic:true}:{})});}
        const atomic = base.atomicRoot && !isArray(type);
        const addressPre=[];if(atomic&&raw&&type==='cw_uchar4'){const temp='cw_pixel_index_'+this.temp++;addressPre.push(`let ${temp} = ${indexCode};`);code=`${base.code}[${temp}]`;}
        return this.result(n, type, atomic && !raw ? (base.rootSymbol?.volatileShared&&type==='f32'?`bitcast<f32>(atomicLoad(&${code}))`:`atomicLoad(&${code})`) : code, [...base.pre, ...index.pre,...capturePre,...addressPre], {rootSymbol: base.rootSymbol, atomicRoot: base.atomicRoot, atomic,...((this.structs.has(type)||['f32','i32','u32'].includes(type)&&!atomic)&&['buffer','buffer-alias'].includes(base.rootSymbol?.kind)?{storageReferenceRoot:base.code,storageReferenceIndexCode:indexCode}:{}),...(base.rootSymbol?.kind==='shared'&&n.base.kind==='id'&&!isArray(type)?{sharedReferenceIndexCode:indexCode}:{})});
      }
      case 'member': {
        if (n.base.kind === 'id' && ['threadIdx', 'blockIdx', 'blockDim', 'gridDim'].includes(n.base.name)) {
          if (!['x', 'y', 'z'].includes(n.member)) this.fail('CUDA dimensions have x, y and z components only.', n);
          const code = {threadIdx: 'cw_thread', blockIdx: 'cw_block', blockDim: 'cw_block_size', gridDim: 'cw_grid'}[n.base.name];
          return this.result(n, 'u32', `${code}.${n.member}`);
        }
        const base = this.expr(n.base, raw), size = vectorLength(base.type);
        if(base.type==='cw_extent'){if(!['width','height','depth'].includes(n.member))this.fail('cudaExtent has width, height and depth fields.',n);return this.result(n,'cw_size64',`${base.code}.${n.member}`,base.pre,{rootSymbol:base.rootSymbol});}
        if(['cw_uchar2','cw_uchar4'].includes(base.type)){if(n.member.length!==1||!(base.type==='cw_uchar2'?'xy':'xyzw').includes(n.member))this.fail('Packed byte vector has only its declared byte components.',n);const shift='xyzw'.indexOf(n.member)*8,read=base.atomic&&raw?`atomicLoad(&${base.code})`:base.code;return this.result(n,'cw_uchar',`((${read} >> ${shift}u) & 255u)`,base.pre,{rootSymbol:base.rootSymbol,packedBase:base.code,packedShift:shift,packedAtomic:!!base.atomic});}
        if(this.structs.has(base.type)){const storageRoot=base.storageReferenceRoot||(base.code===base.rootSymbol?.code?base.rootSymbol?.storageReferenceRoot:null),storageIndex=base.storageReferenceIndexCode??base.rootSymbol?.storageReferenceIndexCode,storagePath=base.storageReferencePath??base.rootSymbol?.storageReferencePath??'';const field=this.structs.get(base.type).fields.find(f=>f.name===n.member);if(!field)this.fail('Unknown struct field '+n.member,n);if(field.access==='private'&&this.currentFunction.classOwner!==this.structs.get(base.type).name&&n.accessorOwner!==this.structs.get(base.type).name)this.fail('Private class field '+n.member+' is inaccessible here.',n);return this.result(n,field.resolvedType,`${base.code}.cw_field_${n.member}`,base.pre,{...(storageRoot&&(this.structs.has(field.resolvedType)||['f32','i32','u32'].includes(field.resolvedType))?{storageReferenceRoot:storageRoot,storageReferenceIndexCode:storageIndex,storageReferencePath:storagePath+'.cw_field_'+n.member}:{}),rootSymbol:field.constant&&!(n.initializingField&&this.currentFunction.classConstructor&&this.currentFunction.classOwner===this.structs.get(base.type).name)?{...base.rootSymbol,constant:true}:base.rootSymbol,...(String(field.resolvedType).startsWith('cw_objectlist_')?{objectListOrigins:this.structs.get(base.type).listOrigins[n.member]}:{})});}
        if (!size || n.member.length !== 1 || 'xyzw'.indexOf(n.member) < 0 || 'xyzw'.indexOf(n.member) >= size) this.fail('Only valid single vector components (.x/.y/.z/.w) are supported.', n);
        if(raw&&base.scalarVectorComponents)return this.result(n,vectorElement(base.type),base.scalarVectorComponents['xyzw'.indexOf(n.member)],base.pre,{rootSymbol:base.rootSymbol});
        return this.result(n, vectorElement(base.type), `${base.code}.${n.member}`, base.pre, {rootSymbol: base.rootSymbol,...(base.devicePointerGuard?{devicePointerGuard:base.devicePointerGuard}:{})});
      }
      case 'ptx-sad4': {
        const output=this.expr({kind:'id',name:n.outputName,token:n.token},true);if(!['u32','i32'].includes(output.type))this.fail('PTX output requires a 32-bit integer register.',n);
        const values=n.args.map(a=>this.expr(a)),pre=[],names=[];
        for(const v of values){if(!['u32','i32'].includes(v.type))this.fail('PTX r operands require 32-bit integers.',n);const name='cw_sad_'+this.temp++;pre.push(...v.pre,`let ${name}: u32 = u32(${v.code});`);names.push(name);}
        const [a,b,c]=names,terms=[0,8,16,24].map(shift=>`u32(abs(i32((${a} >> ${shift}u) & 255u) - i32((${b} >> ${shift}u) & 255u)))`);
        return this.result(n,'u32',`(${c} + ${terms.join(' + ')})`,pre);
      }
      case 'cast': {
        if(String(n.target).startsWith('cw_objectptr_')){const value=this.expr(n.value),target=this.structs.get('cw_struct_'+n.target.slice(13));if(String(value.type).startsWith('cw_objectptr_')&&target?.base===value.type.slice(13))return this.result(n,n.target,value.code,value.pre);return this.result(n,n.target,this.convert(value.code,value.type,n.target,n),value.pre);}

        if(n.target==='cw_uchar'&&n.value.kind==='binary'&&n.value.op==='*'){
          const doubleInteger=a=>a.kind==='literal'&&/[.eE]/.test(a.value)&&!/^0[xX]/.test(a.value)&&!/[fFuU]$/.test(a.value)&&Number.isInteger(Number(a.value))&&Number(a.value)>=1&&Number(a.value)<=65535;
          const literal=doubleInteger(n.value.right)?n.value.right:doubleInteger(n.value.left)?n.value.left:null;
          if(literal){const operand=literal===n.value.right?n.value.left:n.value.right,value=this.expr(operand);if(value.type!=='f32')this.fail('Exact byte scaling requires a float32 operand.',n);this.exactByteScaleUsed=true;n.byteScale=Number(literal.value);n.byteScaleValue=operand;return this.result(n,'cw_uchar',`cw_exact_byte_scale(${value.code}, ${n.byteScale}u)`,value.pre);}
        }
        const literal=n.value.kind==='unary'&&['+','-'].includes(n.value.op)?n.value.value:n.value;if(n.target==='f32'&&literal.kind==='literal'&&/[.eE]/.test(literal.value)&&!/^0[xX]/.test(literal.value)&&!/[fFuU]$/.test(literal.value)){const rounded=Math.fround(Number(literal.value));if(!Number.isFinite(rounded))this.fail('Explicit float literal conversion overflows f32.',literal);literal.value=String(rounded)+'f';}const value = this.expr(n.value); return this.result(n, n.target, this.convert(value.code, value.type, n.target, n), value.pre); }
      case 'unary': {
        if(['+','-'].includes(n.op)){
          const value=this.expr(n.value),method=this.structs.get(value.type)?.methods?.find(m=>m.name==='operator'+n.op);
          if(method){
            if(raw)this.fail('Read-only class unary operators cannot be used as writable references.',n);
            if(method.selfReference){n.classIdentity=n.value;return this.result(n,value.type,value.code,value.pre,{rootSymbol:value.rootSymbol});}
            const base=n.value,op=n.op;delete n.value;delete n.op;Object.assign(n,{kind:'call',callee:{kind:'member',base,member:'operator'+op,token:n.token},args:[]});return this.call(n);
          }
        }
        if(['+','-'].includes(n.op)){
          const value=this.expr(n.value),name='cw_unary_'+(n.op==='+'?'plus':'minus');
          if(this.structs.has(value.type)&&(this.functions.has(name)||this.overloads.has(name))){
            if(raw)this.fail('Free unary operator results are not writable references.',n);
            const argument=n.value;delete n.value;delete n.op;Object.assign(n,{kind:'call',callee:{kind:'id',name,token:n.token},args:[argument]});return this.call(n);
          }
        }
        if (['++', '--'].includes(n.op)){if(n.value.kind!=='id')this.fail('Expression increments require named local scalars or references.',n);const value=this.expr(n.value,true);if(!['local','reference'].includes(value.rootSymbol?.kind))this.fail('Expression increments require named local scalars or references.',n);const update=this.effect(n),tmp='cw_update_'+this.temp++,snapshot=`let ${tmp}: ${value.type} = ${value.code};`;return this.result(n,value.type,tmp,n.prefix?[...update,snapshot]:[snapshot,...update]);}
        if (n.op === '&' || n.op === '*') this.fail('Pointers are supported only as kernel buffer parameters and &buffer[index] atomic targets.', n);
        let value = this.expr(n.value);if(narrow(value.type))value={...value,type:'i32',code:`i32(${value.code})`};
        if (n.op === '!') return this.result(n, 'bool', `(!${this.convert(value.code, value.type, 'bool', n)})`, value.pre);
        if(value.type==='cw_f64'&&['+','-'].includes(n.op))return this.result(n,'cw_f64',n.op==='+'?value.code:`cw_d_neg(${value.code})`,value.pre);
        if (!numeric(value.type) || (n.op === '~' && value.type === 'f32')) this.fail('Invalid unary operator/type.', n);
        if (n.op === '-' && value.type === 'u32') return this.result(n, 'u32', `(0u - ${value.code})`, value.pre);
        return this.result(n, value.type, n.op === '+' ? value.code : `(${n.op}${value.code})`, value.pre);
      }
      case 'object-new': {
        const heap=this.objectHeaps.get(n.name);n.heapTag=heap.tag;const slot='cw_alloc_'+this.temp++,call=n.constructorCall??={kind:'call',token:n.token,callee:{kind:'id',name:n.name,token:n.token},args:n.args},value=this.call(call);
        const search=this.persistentObjects?`loop { if (${slot} >= ${heap.capacity}u) { break; } let claim = atomicCompareExchangeWeak(&${heap.aliveCode}[${slot}], 0u, 1u); if (claim.exchanged) { break; } if (claim.old_value != 0u) { ${slot} += 1u; } }`:`loop { if (${slot} >= ${heap.capacity}u) { break; } if (!${heap.aliveCode}[${slot}]) { break; } ${slot} += 1u; }`;
        const pre=[`var ${slot}: u32 = 0u;`,search,`if (${slot} < ${heap.capacity}u) {`,...indent([...(this.persistentObjects?[]:[`${heap.aliveCode}[${slot}] = true;`]),...value.pre,`${heap.code}[${slot}] = ${value.code};`]),'}'];
        return this.result(n,n.pointerType,`select(0u, ${heap.tag*1048576}u + ${slot} + 1u, ${slot} < ${heap.capacity}u)`,pre);
      }
      case 'object-deref': {
        if(!n.virtualHandleCode){const value=this.expr(n.value);if(isArray(value.type)&&this.structs.has(value.type.element)&&['buffer','buffer-alias'].includes(value.rootSymbol?.kind)){const base=n.value;delete n.value;Object.assign(n,{kind:'index',base,index:{kind:'literal',value:'0',token:n.token},dereference:true});return this.expr(n,raw);}}
        const value=n.virtualHandleCode?{type:'cw_objectptr_'+n.concreteHeap,code:n.virtualHandleCode,pre:[]}:this.expr(n.value),name=String(value.type).replace('cw_objectptr_',''),heap=this.objectHeaps.get(name);if(!String(value.type).startsWith('cw_objectptr_')||!heap)this.fail('Object dereference requires a concrete allocated class.',n);
        n.heapName=name;return this.result(n,heap.type,`${heap.code}[(${value.code} & 1048575u) - 1u]`,value.pre,{...(this.persistentObjects?{storageReferenceIndexCode:`((${value.code} & 1048575u) - 1u)`}:{}),rootSymbol:(heap.symbol??={kind:'local',name:heap.code,code:heap.code,type:heap.type,constant:false,referenceSpace:this.persistentObjects?'storage':'private'})});
      }
      case 'object-delete': {
        const value=this.expr(n.value),name=String(value.type).replace('cw_objectptr_',''),heaps=[...this.objectHeaps.values()].filter(h=>h.name===name||this.structs.get(h.type)?.base===name);if(!heaps.length)this.fail('delete requires an allocated class or interface.',n);n.deleteHeapTags=heaps.map(h=>({name:h.name,tag:h.tag}));
        const snapshot='cw_delete_'+this.temp++;return this.result(n,'void','',[...value.pre,`let ${snapshot} = ${value.code};`,...heaps.map(heap=>`if ((${snapshot} >> 20u) == ${heap.tag}u && (${snapshot} & 1048575u) != 0u) { ${this.persistentObjects?`atomicStore(&${heap.aliveCode}[(${snapshot} & 1048575u) - 1u], 0u);`:`${heap.aliveCode}[(${snapshot} & 1048575u) - 1u] = false;`} }`)]);
      }
      case 'device-launch': return emitLaunch(this,n);
      case 'binary': {
        let a = this.expr(n.left), b = this.expr(n.right);if(narrow(a.type))a={...a,type:'i32',code:`i32(${a.code})`};if(narrow(b.type))b={...b,type:'i32',code:`i32(${b.code})`};
        if([a.type,b.type].some(t=>(String(t).startsWith('cw_deviceptr_')||String(t).startsWith('cw_bufferref_')))){
          if(!['==','!='].includes(n.op))this.fail('Device pointer fields currently support identity comparison and copying only.',n);
          const type=(String(a.type).startsWith('cw_deviceptr_')||String(a.type).startsWith('cw_bufferref_'))?a.type:b.type;n.operandType=type;return this.result(n,'bool',`(${this.convert(a.code,a.type,type,n)} ${n.op} ${this.convert(b.code,b.type,type,n)})`,[...a.pre,...b.pre]);
        }
        if([a.type,b.type].some(t=>String(t).startsWith('cw_objectptr_'))){
          if(!['==','!='].includes(n.op))this.fail('Object references support only identity equality; allocation and dereference are not yet supported.',n);
          const type=String(a.type).startsWith('cw_objectptr_')?a.type:b.type,ac=this.convert(a.code,a.type,type,n),bc=this.convert(b.code,b.type,type,n);n.operandType=type;return this.result(n,'bool',`(${ac} ${n.op} ${bc})`,[...a.pre,...b.pre]);
        }
        const operatorName='cw_binary_'+{'+':'add','-':'subtract','*':'multiply','/':'divide'}[n.op],operatorCandidates=this.overloads.get(operatorName)||[this.functions.get(operatorName)].filter(Boolean);
        const vectorOperator=[a.type,b.type].some(t=>vectorLength(t))&&operatorCandidates.some(f=>f.freeOperator===n.op&&f.params.length===2&&f.params[0].type===a.type&&f.params[1].type===b.type);
        if(this.structs.has(a.type)||this.structs.has(b.type)||vectorOperator){
          const name=operatorName;
          if(!this.functions.has(name)&&!this.overloads.has(name))this.fail('No supported free class operator is declared for '+n.op,n);
          const left=n.left,right=n.right;delete n.left;delete n.right;delete n.op;Object.assign(n,{kind:'call',callee:{kind:'id',name,token:n.token},args:[left,right]});return this.call(n);
        }
        if (['&&', '||'].includes(n.op)) {
          const ac = this.convert(a.code, a.type, 'bool', n), bc = this.convert(b.code, b.type, 'bool', n);
          if (!b.pre.length) return this.result(n, 'bool', `(${ac} ${n.op} ${bc})`, [...a.pre]);
          const tmp = `cw_tmp_${this.temp++}`;
          const pre = [...a.pre, `var ${tmp}: bool = ${ac};`, `if (${n.op === '&&' ? tmp : `!${tmp}`}) {`, ...indent([...b.pre, `${tmp} = ${bc};`]), '}'];
          return this.result(n, 'bool', tmp, pre);
        }
        if(a.type==='cw_size64'||b.type==='cw_size64'){
          if(['<<','>>'].includes(n.op)&&a.type==='cw_size64'){let shift;try{shift=constantValue(n.right);}catch{}if(!Number.isInteger(shift)||shift<0||shift>63)this.fail('Wide shifts require a constant count in 0..63.',n);this.float64Used=true;this.extentUsed=true;n.operandType='cw_size64';return this.result(n,'cw_size64',`${n.op==='<<'?'cw_d_shl':'cw_d_shr'}(${a.code}, ${shift}u)`,[...a.pre,...b.pre]);}
          if(!['==','!=','<','>','<=','>=','*','+','-'].includes(n.op)||![a.type,b.type].every(t=>['cw_size64','i32','u32','bool'].includes(t)))this.fail('Size values support integer comparisons and multiplication only.',n);
          const left='cw_size_left_'+this.temp++,right='cw_size_right_'+this.temp++,promote=(v,code)=>v.type==='cw_size64'?code:v.type==='bool'?`vec2<u32>(select(0u, 1u, ${code}), 0u)`:v.type==='i32'?`vec2<u32>(u32(${code}), select(0u, 4294967295u, ${code} < 0i))`:`vec2<u32>(u32(${code}), 0u)`;
          const pre=[...a.pre,`let ${left} = ${a.code};`,...b.pre,`let ${right} = ${b.code};`],ac=promote(a,left),bc=promote(b,right);this.extentUsed=true;n.operandType='cw_size64';
          if(['+','-'].includes(n.op)){this.float64Used=true;return this.result(n,'cw_size64',`${n.op==='+'?'cw_d_uadd':'cw_d_usub'}(${ac}, ${bc})`,pre);}
          if(n.op==='*'){this.sizeMultiplyUsed=true;return this.result(n,'cw_size64',`cw_size_multiply(${ac}, ${bc})`,pre);}
          const code=n.op==='=='?`all(${ac} == ${bc})`:n.op==='!='?`any(${ac} != ${bc})`:n.op==='<'?`cw_size_less(${ac}, ${bc})`:n.op==='>'?`cw_size_less(${bc}, ${ac})`:n.op==='<='?`!cw_size_less(${bc}, ${ac})`:`!cw_size_less(${ac}, ${bc})`;
          return this.result(n,'bool',code,pre);
        }
        if(['cw_uchar2','cw_uchar4'].includes(a.type)||['cw_uchar2','cw_uchar4'].includes(b.type))this.fail('uchar4 arithmetic requires explicit byte components.',n);
        if(vectorLength(a.type)||vectorLength(b.type)){const type=vectorLength(a.type)?a.type:b.type;const element=vectorElement(type),ops=element==='f32'?['+','-','*','/']:['+','-','*'];const compatible=v=>v.type===type||v.type===element||element==='f32'&&numeric(v.type);if(!ops.includes(n.op)||!compatible(a)||!compatible(b))this.fail('Vector arithmetic requires matching vector/scalar element types; integer vectors support +, -, * only.',n);const code=v=>v.type===type?v.code:`${type}(${this.convert(v.code,v.type,element,n)})`;n.operandType=type;return this.result(n,type,`(${code(a)} ${n.op} ${code(b)})`,[...a.pre,...b.pre]);}
        let common = ['<<', '>>'].includes(n.op) ? a.type : this.common(a.type, b.type, n);
        if (vectorLength(common)) this.fail('CUDA vector arithmetic requires explicit components; operator overloads are outside this subset.', n);
        if (['&', '|', '^', '<<', '>>', '%'].includes(n.op) && !['i32', 'u32'].includes(common)) this.fail('Bitwise, shift and remainder operators require integers.', n);
        if (common === 'bool' && !['==', '!='].includes(n.op)) this.fail('Boolean values support only logical/equality operators.', n);
        const ac = this.convert(a.code, a.type, common, n), bc = this.convert(b.code, b.type, ['<<', '>>'].includes(n.op) ? 'u32' : common, n);
        const out = ['==', '!=', '<', '>', '<=', '>='].includes(n.op) ? 'bool' : common;
        n.operandType = common;
        if(common==='cw_f64'){
          this.float64Used=true;const functions={'+':'cw_d_add','-':'cw_d_sub','*':'cw_d_mul','/':'cw_d_div','==':'cw_d_eq','<':'cw_d_lt'};
          const code=functions[n.op]?`${functions[n.op]}(${ac}, ${bc})`:n.op==='!='?`!cw_d_eq(${ac}, ${bc})`:n.op==='>'?`cw_d_lt(${bc}, ${ac})`:n.op==='<='?`cw_d_le(${ac}, ${bc})`:`cw_d_le(${bc}, ${ac})`;
          return this.result(n,out,code,[...a.pre,...b.pre]);
        }
        if(n.op==='/'&&common==='f32'){this.compensatedDivisionUsed=true;return this.result(n,out,`cw_divide_f32(${ac}, ${bc})`,[...a.pre,...b.pre]);}
        return this.result(n, out, `(${ac} ${n.op} ${bc})`, [...a.pre, ...b.pre]);
      }
      case 'conditional': {
        const cond = this.expr(n.condition), yes = this.expr(n.yes), no = this.expr(n.no), type = this.common(yes.type, no.type, n);
        const tmp = `cw_tmp_${this.temp++}`;
        // select() is eager. A branch is mandatory to preserve guarded loads, divisions and atomic side effects.
        const pre = [...cond.pre, `var ${tmp}: ${typeName(type)};`, `if (${this.convert(cond.code, cond.type, 'bool', n)}) {`, ...indent([...yes.pre, `${tmp} = ${this.convert(yes.code, yes.type, type, n)};`]), '} else {', ...indent([...no.pre, `${tmp} = ${this.convert(no.code, no.type, type, n)};`]), '}'];
        return this.result(n, type, tmp, pre);
      }
      case 'call': return this.call(n);
      case 'assign': this.fail('Assignments with a used return value are unsupported. Put assignments in their own statements.', n); break;
      default: this.fail(`Unsupported expression '${n.kind}'.`, n);
    }
  }
  surfaceGridCoordinate(node,axis,scale,seen=new Set()){
    // Restrict trap-mode stores to coordinates whose full range the runtime can
    // validate before submission. Never silently turn a CUDA trap into a drop.
    const literal=(n,value)=>n?.kind==='sizeof'?cudaValueSize(n.target)===value:n?.kind==='literal'&&Number(n.value.replace(/[uUlL]+$/,''))===value;
    if(scale===1&&node?.kind==='binary'&&node.op==='*'&&(literal(node.left,1)||literal(node.right,1)))return this.surfaceGridCoordinate(literal(node.left,1)?node.right:node.left,axis,1,seen);
    if(scale!==1)return node?.kind==='binary'&&node.op==='*'&&((literal(node.right,scale)&&this.surfaceGridCoordinate(node.left,axis,1,seen))||(literal(node.left,scale)&&this.surfaceGridCoordinate(node.right,axis,1,seen)));
    if(node?.kind==='id'){
      if(seen.has(node.name))return false;const declarations=[];let changed=false;
      walk(this.kernel.body,n=>{if(n.kind==='decl'&&n.name===node.name)declarations.push(n);if(n.kind==='call'&&!['surf1Dwrite','surf2Dwrite','surf3Dwrite','surf2DLayeredwrite'].includes(n.callee?.name)&&!(/^(?:make_(?:float|int|uint)[234]|float|int|uint|unsigned)$/.test(n.callee?.name||'')&&!this.ast.functions.some(f=>f.name===n.callee.name)))for(const arg of n.args)walk(arg,a=>{if(a.kind==='id'&&a.name===node.name)changed=true;});if((n.kind==='assign'&&n.left?.kind==='id'&&n.left.name===node.name)||(n.kind==='unary'&&['++','--'].includes(n.op)&&n.value?.name===node.name))changed=true;});
      if(changed||declarations.length!==1)return false;return this.surfaceGridCoordinate(declarations[0].init,axis,1,new Set([...seen,node.name]));
    }
    const member=(n,name)=>n?.kind==='member'&&n.base?.kind==='id'&&n.base.name===name&&n.member===axis;
    const product=n=>n?.kind==='binary'&&n.op==='*'&&((member(n.left,'blockIdx')&&member(n.right,'blockDim'))||(member(n.right,'blockIdx')&&member(n.left,'blockDim')));
    return node?.kind==='binary'&&node.op==='+'&&((product(node.left)&&member(node.right,'threadIdx'))||(product(node.right)&&member(node.left,'threadIdx')));
  }
  argument(n){
    if(n.kind==='id'&&n.name==='warpSize'&&!this.scopes.some(scope=>scope.has('warpSize')))return this.expr(n);
    if(n.kind==='unary'&&n.op==='&'&&n.value.kind==='id'){const value=this.expr(n.value),symbol=value.rootSymbol;if(!symbol||!['local','reference'].includes(symbol.kind)||symbol.constant||!(numeric(value.type)||this.structs.has(value.type)))this.fail('Local pointer arguments require a mutable named scalar or record.',n);return this.result(n,arrayOf(value.type),value.code,value.pre,{rootSymbol:symbol,localPointer:true,pointerCode:symbol.kind==='reference'?symbol.pointerCode:'&'+value.code});}
    const address=n.kind==='unary'&&n.op==='&'&&n.value.kind==='index'?n.value:null;
    const parts=pointerParts(n),base=parts?.base;
    if(base?.kind==='id'&&!['true','false','NULL','nullptr'].includes(base.name)){
      const symbol=this.lookup(base.name,base);
      if(symbol.kind==='thread-block'&&n.kind==='id'){n.symbol=symbol;return {type:'thread-block',code:'',pre:[],rootSymbol:symbol};}
      if(['local','reference'].includes(symbol.kind)&&isArray(symbol.type)&&!isArray(symbol.type.element)){
        if(n.kind!=='id'||symbol.constant)this.fail('Local array helper pointers require a whole mutable array.',n);
        const value=this.expr(n);return this.result(n,symbol.type,value.code,value.pre,{rootSymbol:symbol,localArrayPointer:true,pointerCode:symbol.kind==='reference'?symbol.pointerCode:'&'+symbol.code});
      }
      if(symbol.kind==='shared'&&isArray(symbol.type)&&!isArray(symbol.type.element)){
        if(symbol.atomic)this.fail('Shared helper pointers do not support atomic arrays.',n);
        const node=parts.offset,offset=node?this.expr(node):{type:'i32',code:'0i',pre:[]};
        if(!['i32','u32'].includes(offset.type))this.fail('Shared helper offsets must be 32-bit integers.',n);
        n.pointerBaseSymbol=symbol;n.pointerOffset=node;
        return this.result(n,symbol.type,symbol.code,offset.pre,{rootSymbol:{...symbol,sharedPointer:symbol.code},pointerCode:this.convert(offset.code,offset.type,'i32',n)});
      }
      if(['buffer','buffer-alias'].includes(symbol.kind)){
        const node=parts.offset,offset=node?this.expr(node):{type:'i32',code:'0i',pre:[]};
        if(!['i32','u32'].includes(offset.type))this.fail('Helper buffer offsets must be 32-bit integers.',n);
        n.pointerBaseSymbol=symbol;n.pointerOffset=node;
        const pointerCode=symbol.offsetCode?`(${symbol.offsetCode} + ${this.convert(offset.code,offset.type,'i32',n)})`:this.convert(offset.code,offset.type,'i32',n);
        return this.result(n,symbol.type,symbol.code,offset.pre,{rootSymbol:symbol,pointerCode});
      }
    }
    return this.expr(n,false,n.kind==='index');
  }
  bindPointerHelper(helper,args,n){
    const uses=analyse([helper],helper.params),roots=[];
    for(const [i,p]of helper.params.entries())if(p.pointer){
      const a=args[i],symbol=a?.rootSymbol,root=symbol?.rootBufferName;
      if(p.volatileParameter&&!root)this.fail('Volatile helper pointers require storage buffers.',n.args[i]);
      if(a?.localArrayPointer){if(a.type.element!==p.type)this.fail('Local array element type must match the helper pointer.',n.args[i]);roots.push([i,'@array:'+a.type.length,false]);continue;}
      if(a?.localPointer){if(a.type.element!==p.type)this.fail('Local pointer type must exactly match the helper parameter.',n.args[i]);roots.push([i,'@local',false]);continue;}
      if(symbol?.sharedPointer){if(!isArray(a.type)||a.type.element!==p.type)this.fail('Shared pointer type must exactly match the helper parameter.',n.args[i]);if(symbol.constant&&!p.constant)this.fail('Cannot discard const through a shared helper pointer.',n.args[i]);roots.push([i,'@shared:'+symbol.sharedPointer,!!symbol.constant]);continue;}
      if(!root||!isArray(a.type)||a.type.element!==p.type)this.fail('Helper pointers require a same-type storage buffer or buffer offset.',n.args[i]);
      if(symbol.constant&&!p.constant)this.fail('Cannot discard const through a helper pointer argument.',n.args[i]);
      roots.push([i,root,!!symbol.constant]);
      if(uses.reads.has(p.name))this.usage.reads.add(root);
      if(uses.writes.has(p.name))this.usage.writes.add(root);
      if(uses.atomicBindings.has(p.name)||p.volatileParameter||['cw_uchar','bool'].includes(p.type)&&uses.writes.has(p.name)){this.usage.atomic.add(root);this.bufferSymbols.get(root).atomic=true;}
    }
    this.usage.storageBarrier=[...this.usage.writes].some(x=>this.usage.reads.has(x));
    const key=JSON.stringify([helper.name,roots]);
    if(!this.pointerHelpers.has(key)){
      if(this.pointerHelpers.size>=128)this.fail('At most 128 helper buffer specializations are supported.',n);
      const clone=cloneAst(helper);let name='cw_buffer_helper_'+this.pointerHelpers.size;while(this.functions.has(name))name+='_';
      clone.name=name;clone.pointerOrigin=helper.name;
      const localNames=new Set(roots.filter(([,root])=>root==='@local').map(([i])=>clone.params[i].name));
      // Preserve forwarding of a scalar pointer through another typed device helper.
      // The specialization represents that pointer as a function-space reference.
      walk(clone.body,node=>{if(node.kind!=='call')return;const member=node.callee.kind==='member',callees=member?[...this.functions.values()].filter(f=>f.classMethod===node.callee.member):[this.functions.get(node.callee.name)].filter(Boolean);if(!callees.length)return;node.args=node.args.map((arg,i)=>{if(arg.kind!=='id'||!localNames.has(arg.name))return arg;const source=clone.params.find(p=>p.name===arg.name);if(!callees.every(callee=>{const target=callee.params[i+(member?1:0)];return target?.pointer&&target.type===source.type;}))return arg;return {kind:'unary',op:'&',value:arg,token:arg.token,forwardedLocalAddress:true};});});
      // Arrow access to a local record pointer aliases that record directly.
      walk(clone.body,node=>{if(node.kind==='object-deref'&&node.value?.kind==='id'&&localNames.has(node.value.name)){const name=node.value.name;delete node.value;node.kind='id';node.name=name;}});
      const inspect=(node,parent)=>{if(!node||typeof node!=='object')return;if(node.kind==='decl'&&localNames.has(node.name))this.fail('Shadowed local pointer parameters are unsupported.',node);if(node.kind==='id'&&localNames.has(node.name)&&!(parent?.kind==='unary'&&parent.op==='&'&&parent.forwardedLocalAddress)&&!(parent?.kind==='member'&&parent.base===node&&this.structs.has(clone.params.find(p=>p.name===node.name)?.type))&&!(parent?.kind==='index'&&parent.base===node&&parent.index.kind==='literal'&&Number(parent.index.value.replace(/[uU]$/,''))===0))this.fail('Local pointers support only dereference or index zero; arithmetic and escapes are unsupported.',node);for(const [key,value]of Object.entries(node))if(!['token','type'].includes(key)){if(Array.isArray(value))value.forEach(v=>inspect(v,node));else if(value&&typeof value==='object')inspect(value,node);}};inspect(clone.body,null);
      walk(clone.body,node=>{if(node.kind==='index'&&node.base.kind==='id'&&localNames.has(node.base.name)){const name=node.base.name;delete node.base;delete node.index;delete node.dereference;node.kind='id';node.name=name;}});
      for(const [i,root,constant]of roots){if(root.startsWith('@array:')){clone.params[i].type=arrayOf(clone.params[i].type,Number(root.slice(7)));clone.params[i].pointer=false;clone.params[i].reference=true;clone.params[i].localArray=true;}else if(root==='@local'){clone.params[i].pointer=false;clone.params[i].reference=true;clone.params[i].localPointer=true;}else{if(root.startsWith('@shared:'))clone.params[i].boundShared=root.slice(8);else clone.params[i].boundBuffer=root;clone.params[i].boundConstant=constant;}}
      this.pointerHelpers.set(key,clone);this.functions.set(name,clone);this.helpers.push(clone);this.ast.functions.push(clone);
    }
    return this.pointerHelpers.get(key);
  }
  bindReferenceHelper(helper,args,n){
    const storageRoots=helper.params.map((p,i)=>p.reference&&args[i].storageReferenceRoot?args[i].storageReferenceRoot:p.reference&&args[i].rootSymbol?.referenceSpace==='storage'&&args[i].type===args[i].rootSymbol.type&&(args[i].storageReferenceIndexCode!==undefined||args[i].code===args[i].rootSymbol.code&&args[i].rootSymbol.storageReferenceIndexCode!==undefined)?args[i].rootSymbol.storageReferenceRoot||args[i].rootSymbol.code:null);
    for(let i=0;i<storageRoots.length;i++){const buffer=[...this.bufferSymbols.values()].find(b=>b.code===storageRoots[i]);if(buffer){this.usage.reads.add(buffer.rootBufferName);if(!helper.params[i].constant)this.usage.writes.add(buffer.rootBufferName);}}
    const storagePaths=storageRoots.map((root,i)=>root?(args[i].storageReferencePath??args[i].rootSymbol?.storageReferencePath??''):null);
    const spaces=helper.params.map((p,i)=>p.reference?(storageRoots[i]?'storage':args[i].rootSymbol?.referenceSpace==='private'?'private':args[i].rootSymbol?.kind==='shared'||args[i].rootSymbol?.referenceSpace==='workgroup'?'workgroup':'function'):null);
    if(spaces.every((space,i)=>space===null||space===(helper.params[i].referenceSpace||'function'))&&storageRoots.every((root,i)=>root===(helper.params[i].boundReferenceStorage||null)&&storagePaths[i]===(root?(helper.params[i].boundReferencePath||''):null))&&helper.params.every((p,i)=>!p.reference||args[i].sharedReferenceIndexCode===undefined&&!args[i].rootSymbol?.sharedReferenceRoot))return helper;
    const candidates=helper.params.map((p,i)=>p.reference?(args[i].sharedReferenceIndexCode!==undefined?args[i].rootSymbol.code:args[i].rootSymbol?.sharedReferenceRoot||null):null),roots=candidates.map((root,i)=>root&&(candidates.filter(r=>r===root).length>1||args[i].rootSymbol?.sharedReferenceRoot)?root:null);
    const origin=helper.referenceOrigin||helper.name,key=JSON.stringify([origin,spaces,roots,storageRoots,storagePaths]);
    if(!this.referenceHelpers.has(key)){
      if(this.referenceHelpers.size>=128)this.fail('At most 128 helper reference specializations are supported.',n);
      const base=this.functions.get(origin)||helper;walk(base.body,node=>{if(node.kind==='decl'&&node.shared)this.fail('Reference address-space specialization of helpers with shared declarations is unsupported.',node);});
      const clone=cloneAst(base);let name='cw_reference_helper_'+this.referenceHelpers.size;while(this.functions.has(name))name+='_';clone.name=name;clone.referenceOrigin=origin;
      spaces.forEach((space,i)=>{if(space)clone.params[i].referenceSpace=space;if(roots[i])clone.params[i].boundReferenceShared=roots[i];if(storageRoots[i]){clone.params[i].boundReferenceStorage=storageRoots[i];clone.params[i].boundReferencePath=storagePaths[i];}});this.referenceHelpers.set(key,clone);this.functions.set(name,clone);this.helpers.push(clone);this.ast.functions.push(clone);
    }
    return this.referenceHelpers.get(key);
  }
  call(n) {
    const allocation=deviceHeapCall(this,n);if(allocation)return allocation;
    if(n.callee.kind==='member'&&n.callee.base.kind==='object-deref'&&!n.callee.base.concreteHeap){
      const pointer=n.callee.base.value,value=this.expr(pointer),baseName=String(value.type).replace('cw_objectptr_',''),iface=this.ast.interfaces?.find(i=>i.name===baseName);
      if(iface){
        const method=iface.abstractMethods.find(m=>m.name===n.callee.member);if(!method)this.fail('Unknown interface method.',n);
        const slot='cw_dispatch_'+this.temp++,result=slot+'_result',pre=[...value.pre,`let ${slot} = ${value.code};`];if(method.result!=='void')pre.push(`var ${result}: ${method.result};`);
        const branches=[];
        for(const heap of this.objectHeaps.values())if(this.structs.get(heap.type)?.base===baseName&&(!value.pointerTargets||value.pointerTargets.includes(heap.name))){
          const receiver={kind:'object-deref',token:n.token,value:structuredClone(pointer),concreteHeap:heap.name,virtualHandleCode:slot},call={kind:'call',token:n.token,callee:{kind:'member',token:n.token,base:receiver,member:n.callee.member},args:structuredClone(n.args)},emitted=this.call(call);
          branches.push({tag:heap.tag,call});pre.push(`if ((${slot} >> 20u) == ${heap.tag}u) {`,...indent([...emitted.pre,method.result==='void'?`${emitted.code};`:`${result} = ${emitted.code};`]),'}');
        }
        if(!branches.length)this.fail('No allocated implementation of this interface is available.',n);
        n.virtualDispatch={pointer,slot,branches};return this.result(n,method.result,method.result==='void'?'':result,pre);
      }
    }

    if(n.callee.kind==='id'){
      const record=[...this.structs.values()].find(s=>s.valueClass&&s.name===n.callee.name);
      if(record&&n.classMemberInit&&n.args.length===1&&!this.ast.functions.some(f=>f.classConstructor&&f.classOwner===record.name&&f.params.length===1&&f.params[0].type===record.type)){const value=this.expr(n.args[0]);if(value.type===record.type){n.classIdentity=n.args[0];return this.result(n,record.type,value.code,value.pre);}}
      if(record){if(!record.constructors.length)this.fail('No value-class constructor is declared.',n);n.callee={kind:'id',token:n.token,name:'cw_ctor_'+record.name};}
    }
    if(n.callee.kind==='member'&&n.callee.member!=='sync'){
      const receiver=n.callee.base,value=this.expr(receiver),record=this.structs.get(value.type),method=record?.methods?.find(m=>m.name===n.callee.member&&!m.indexedReference);
      if(!method)this.fail('Unknown or unsupported value-class method.',n);
      if(method.fieldReference){
        if(n.args.length||!value.rootSymbol||!['local','reference','buffer','buffer-alias','shared'].includes(value.rootSymbol.kind))this.fail('Field reference getter requires a stable receiver and no arguments.',n);
        if(method.access==='private'&&this.currentFunction.classOwner!==record.name)this.fail('Private class getter is inaccessible here.',n);
        const field={kind:'member',base:receiver,member:method.field,token:n.token,accessorOwner:record.name};n.classIdentity=field;n.fieldReference=true;
        const result=this.expr(field);return this.result(n,result.type,result.code,result.pre,{rootSymbol:{...result.rootSymbol,constant:true},fieldReference:true});
      }
      n.args=[receiver,...n.args];n.callee={kind:'id',token:n.token,name:method.helper};
    }
    const groupSync=n.callee.kind==='id'&&n.callee.name==='cooperative_groups::sync';
    const memberSync=n.callee.kind==='member'&&n.callee.member==='sync';
    if(groupSync||memberSync){
      const group=groupSync?n.args[0]:n.callee.base;
      if(n.args.length!==(groupSync?1:0)||group?.kind!=='id'||this.lookup(group.name,group).kind!=='thread-block')this.fail('Block sync requires a local thread_block handle from this_thread_block().',n);
      n.callName='__syncthreads';return this.result(n,'void','workgroupBarrier()');
    }
    if (n.callee.kind !== 'id') this.fail('Only named functions are supported.', n);
    const name = n.callee.name; n.callName = name;
    if(name.startsWith('cw_native_tile_'))return emitNativeTile(this,n,name);
    if(name==='__clz'){if(n.args.length!==1)this.fail('__clz requires one 32-bit integer.',n);const a=this.expr(n.args[0]);if(!['i32','u32'].includes(a.type))this.fail('__clz requires one 32-bit integer.',n);return this.result(n,'i32',`i32(countLeadingZeros(u32(${a.code})))`,a.pre);}
    if(name==='__popc'){if(n.args.length!==1)this.fail('__popc requires one integer.',n);const a=this.expr(n.args[0]);if(!['i32','u32'].includes(a.type))this.fail('__popc requires an integer.',n);return this.result(n,'i32',`i32(countOneBits(u32(${a.code})))`,a.pre);}
    if(name==='__ffs'){
      if(n.args.length!==1)this.fail('__ffs requires one 32-bit integer.',n);const a=this.expr(n.args[0]);if(!['i32','u32'].includes(a.type))this.fail('__ffs requires one 32-bit integer.',n);return this.result(n,'i32',`(i32(firstTrailingBit(u32(${a.code}))) + 1i)`,a.pre);
    }
    if(['sincosf','__sincosf'].includes(name)){
      if(n.args.length!==3||n.args.slice(1).some(a=>a.kind!=='unary'||a.op!=='&'||!['id','member'].includes(a.value.kind)))this.fail('sincosf requires a phase and addresses of two mutable local float values or components.',n);
      const phase=this.expr(n.args[0]),outputs=n.args.slice(1).map(a=>this.expr(a.value,true));
      if(!numeric(phase.type))this.fail('sincosf phase must convert to float.',n);
      for(const [i,out]of outputs.entries()){this.writable(out,n.args[i+1].value);if(out.type!=='f32'||!['local','reference'].includes(out.rootSymbol?.kind)||out.atomic||out.pre.length)this.fail('sincosf outputs require mutable local float values or components.',n.args[i+1]);}
      if(outputs[0].code===outputs[1].code)this.fail('sincosf outputs must be distinct.',n);
      const input='cw_sincos_phase_'+this.temp++,turns='cw_sincos_turns_'+this.temp++,reduced='cw_sincos_reduced_'+this.temp++,sin='cw_sincos_sin_'+this.temp++,cos='cw_sincos_cos_'+this.temp++;
      n.localPointerArgs=[false,true,true];
      return this.result(n,'void',`${outputs[1].code} = ${cos}`,[...phase.pre,`let ${input}: f32 = ${this.convert(phase.code,phase.type,'f32',n)};`,`let ${turns} = round(${input} * 0.15915494309189535f);`,`let ${reduced} = (${input} - ${turns} * 6.28125f) - ${turns} * 0.001935307179586477f;`,`let ${sin} = sin(select(${input}, ${reduced}, abs(${input}) <= 8192.0f));`,`let ${cos} = cos(select(${input}, ${reduced}, abs(${input}) <= 8192.0f));`,`${outputs[0].code} = ${sin};`]);
    }
    if(name==='tex1D'){if(n.callee.templateArgument!=='float4'||n.args.length!==2||n.args[0].kind!=='id')this.fail('tex1D supports a bound float4 texture and one float coordinate.',n);const texture=this.lookup(n.args[0].name,n.args[0]),coordinate=this.expr(n.args[1]);if(texture.kind!=='texture'||texture.dimension!=='2d'||!numeric(coordinate.type))this.fail('tex1D requires a matching kernel texture parameter and float coordinate.',n);return this.result(n,'vec4<f32>',`textureSampleLevel(${texture.code}, ${texture.sampler}, vec2<f32>(${this.convert(coordinate.code,coordinate.type,'f32',n)}, 0.5f), 0.0f)`,coordinate.pre);}
    if(name==='texCubemap'){
      if(n.callee.templateArgument!=='float'||n.args.length!==4||n.args[0]?.kind!=='id')this.fail('texCubemap requires a float cubemap and three scalar direction coordinates.',n);
      const texture=this.lookup(n.args[0].name,n.args[0]),coords=n.args.slice(1).map(a=>this.expr(a));if(texture.kind!=='texture'||texture.dimension!=='2d-array'||texture.format!=='r32float'||!coords.every(c=>numeric(c.type)))this.fail('Cubemap sampling requires six float faces and scalar coordinates.',n);
      const id=this.temp++,d='cw_cube_d_'+id,a='cw_cube_a_'+id,uv='cw_cube_uv_'+id,face='cw_cube_face_'+id;
      const pre=[...coords.flatMap(c=>c.pre),`let ${d} = vec3<f32>(${coords.map(c=>this.convert(c.code,c.type,'f32',n)).join(', ')});`,`let ${a} = abs(${d});`,`var ${uv}: vec2<f32>;`,`var ${face}: i32;`,
        `if (${a}.x >= ${a}.y && ${a}.x >= ${a}.z) {`,`  ${face} = select(1i, 0i, ${d}.x >= 0.0f);`,`  ${uv} = vec2<f32>(select(${d}.z, -${d}.z, ${d}.x >= 0.0f), -${d}.y) / ${a}.x;`,
        `} else if (${a}.y >= ${a}.z) {`,`  ${face} = select(3i, 2i, ${d}.y >= 0.0f);`,`  ${uv} = vec2<f32>(${d}.x, select(-${d}.z, ${d}.z, ${d}.y >= 0.0f)) / ${a}.y;`,
        '} else {',`  ${face} = select(5i, 4i, ${d}.z >= 0.0f);`,`  ${uv} = vec2<f32>(select(-${d}.x, ${d}.x, ${d}.z >= 0.0f), -${d}.y) / ${a}.z;`,'}'];
      return this.result(n,'f32',`textureSampleLevel(${texture.code}, ${texture.sampler}, ${uv} * 0.5f + vec2<f32>(0.5f), ${face}, 0.0f).x`,pre);
    }
    if(name==='tex2DLayered'){
      const scalar=n.callee.templateArgument==='float';
      if(!['float','float4'].includes(n.callee.templateArgument)||n.args.length!==4||n.args[0]?.kind!=='id')this.fail('tex2DLayered requires a float or float4 texture, two coordinates and an integer layer.',n);
      const texture=this.lookup(n.args[0].name,n.args[0]),x=this.expr(n.args[1]),y=this.expr(n.args[2]),layer=this.expr(n.args[3]);
      if(texture.kind!=='texture'||texture.dimension!=='2d-array'||texture.format!==(scalar?'r32float':'rgba32float')||![x.type,y.type].every(numeric)||!['i32','u32'].includes(layer.type))this.fail('Layered sampling requires matching float/float4 layers and scalar coordinates.',n);
      return this.result(n,scalar?'f32':'vec4<f32>',`textureSampleLevel(${texture.code}, ${texture.sampler}, vec2<f32>(${this.convert(x.code,x.type,'f32',n)}, ${this.convert(y.code,y.type,'f32',n)}), i32(${layer.code}), 0.0f)${scalar?'.x':''}`,[...x.pre,...y.pre,...layer.pre]);
    }
    if(name==='surf2DLayeredwrite'){
      if(![5,6].includes(n.args.length)||n.args[1]?.kind!=='id'||n.args.length===6&&(n.args[5]?.kind!=='id'||n.args[5].name!=='cudaBoundaryModeTrap'))this.fail('Layered stores require float4, global XY byte coordinates and trap mode.',n);
      const surface=this.lookup(n.args[1].name,n.args[1]),value=this.expr(n.args[0]),x=this.expr(n.args[2]),y=this.expr(n.args[3]),layer=this.expr(n.args[4]);
      if(surface.kind!=='surface'||surface.dimension!=='2d-array'||value.type!=='vec4<f32>'||![x.type,y.type].every(t=>['i32','u32','cw_size64'].includes(t))||!['i32','u32'].includes(layer.type)||!this.surfaceGridCoordinate(n.args[2],'x',16)||!this.surfaceGridCoordinate(n.args[3],'y',1))this.fail('Layered stores require float4 and checked global XY byte coordinates.',n);
      const selector=n.args[4].kind==='id'&&layer.rootSymbol?.kind==='uniform'?{scalar:layer.rootSymbol.name}:n.args[4].kind==='literal'?{value:n.args[4].numericValue}:null;
      if(!selector||surface.metadata.layer&&JSON.stringify(surface.metadata.layer)!==JSON.stringify(selector))this.fail('Layered stores require one uniform or literal layer per surface.',n);
      surface.metadata.layer=selector;surface.metadata.format='rgba32float';surface.metadata.coordinates='global-xy-layer';
      return this.result(n,'void',`textureStore(${surface.code}, vec2<i32>(${this.convert(x.code,x.type,'i32',n)} / 16i, ${this.convert(y.code,y.type,'i32',n)}), i32(${layer.code}), ${value.code})`,[...value.pre,...x.pre,...y.pre,...layer.pre]);
    }
    if(name==='surf1Dwrite'){

      if(![3,4].includes(n.args.length)||n.args[1]?.kind!=='id'||n.args.length===4&&(n.args[3]?.kind!=='id'||n.args[3].name!=='cudaBoundaryModeTrap'))this.fail('surf1Dwrite requires a float4 value, checked global X byte offset and default or explicit trap mode.',n);
      const surface=this.lookup(n.args[1].name,n.args[1]),value=this.expr(n.args[0]),x=this.expr(n.args[2]);
      if(surface.kind!=='surface'||surface.dimension!=='2d'||value.type!=='vec4<f32>'||!['i32','u32','cw_size64'].includes(x.type)||!this.surfaceGridCoordinate(n.args[2],'x',16))this.fail('surf1Dwrite requires float4 and globalX * sizeof(float4); other offsets cannot be checked before dispatch.',n);
      if(surface.storeFormat&&surface.storeFormat!=='rgba32float')this.fail('A surface cannot mix storage formats.',n);
      surface.storeFormat='rgba32float';surface.metadata.format='rgba32float';surface.metadata.coordinates='global-x';
      return this.result(n,'void',`textureStore(${surface.code}, vec2<i32>(${this.convert(x.code,x.type,'i32',n)} / 16i, 0i), ${value.code})`,[...value.pre,...x.pre]);
    }
    if(name==='surf3Dwrite'){

      if(![5,6].includes(n.args.length)||n.args[1].kind!=='id'||n.args.length===6&&(n.args[5].kind!=='id'||n.args[5].name!=='cudaBoundaryModeTrap'))this.fail('surf3Dwrite supports float or byte surfaces with checked global XYZ coordinates and default or explicit cudaBoundaryModeTrap.',n);
      const surface=this.lookup(n.args[1].name,n.args[1]),value=this.expr(n.args[0]),coords=n.args.slice(2,5).map(a=>this.expr(a)),bytes=value.type==='cw_uchar'?1:4;
      if(surface.kind!=='surface'||surface.dimension!=='3d'||!['f32','cw_uchar'].includes(value.type)||coords.some(c=>!['i32','u32','cw_size64'].includes(c.type))||!['x','y','z'].every((axis,i)=>this.surfaceGridCoordinate(n.args[i+2],axis,i===0?bytes:1)))this.fail('Surface writes require float or byte values and checked global XYZ byte offsets; other coordinates cannot be bounds-checked before dispatch.',n);
      const format=bytes===1?'rgba8unorm':'r32float';if(surface.storeFormat&&surface.storeFormat!==format)this.fail('A surface cannot mix float and byte writes.',n);surface.storeFormat=format;surface.metadata.format=format;if(bytes===1)surface.metadata.sourceElementType='cw_uchar';
      const code=coords.map(c=>this.convert(c.code,c.type,'i32',n)),colour=bytes===1?`vec4<f32>(f32(${value.code}) / 255.0f, 0.0f, 0.0f, 1.0f)`:`vec4<f32>(${value.code}, 0.0f, 0.0f, 0.0f)`;
      return this.result(n,'void',`textureStore(${surface.code}, vec3<i32>(${code[0]} / ${bytes}i, ${code[1]}, ${code[2]}), ${colour})`,[...value.pre,...coords.flatMap(c=>c.pre)]);
    }
    if(name==='surf2Dwrite'){
      if(n.args.length!==5||n.args[1].kind!=='id'||n.args[4].kind!=='id'||n.args[4].name!=='cudaBoundaryModeTrap')this.fail('surf2Dwrite supports float surfaces with checked global XY coordinates and cudaBoundaryModeTrap.',n);
      const surface=this.lookup(n.args[1].name,n.args[1]),value=this.expr(n.args[0]),x=this.expr(n.args[2]),y=this.expr(n.args[3]);
      if(surface.kind!=='surface'||surface.dimension!=='2d'||value.type!=='f32'||!['i32','u32'].includes(x.type)||!['i32','u32'].includes(y.type)||!this.surfaceGridCoordinate(n.args[2],'x',4)||!this.surfaceGridCoordinate(n.args[3],'y',1))this.fail('Surface writes require float values, byte offset globalX * 4 and row globalY; other coordinates cannot be bounds-checked before dispatch.',n);
      return this.result(n,'void',`textureStore(${surface.code}, vec2<i32>(i32(${x.code}) / 4i, i32(${y.code})), vec4<f32>(${value.code}, 0.0f, 0.0f, 0.0f))`,[...value.pre,...x.pre,...y.pre]);
    }
    if(name==='tex2D'&&['uchar','unsigned char','uint','unsigned','unsigned int'].includes(n.callee.templateArgument)) {
      if(n.args.length!==3||n.args[0].kind!=='id')this.fail('Integer tex2D requires a bound texture and two coordinates.',n);
      const wide=['uint','unsigned','unsigned int'].includes(n.callee.templateArgument),texture=this.lookup(n.args[0].name,n.args[0]),coords=n.args.slice(1).map(a=>this.expr(a));
      if(texture.kind!=='texture'||texture.dimension!=='2d'||texture.format!==(wide?'r32uint':'r8uint')||coords.some(c=>!numeric(c.type)))this.fail('Integer tex2D requires a matching unsigned texture and numeric coordinates.',n);
      this.byte2DSamplingUsed=true;
      return this.result(n,wide?'u32':'cw_uchar',`cw_sample_byte2d(${texture.code}, vec2<f32>(${coords.map(c=>this.convert(c.code,c.type,'f32',n)).join(', ')}))`,coords.flatMap(c=>c.pre));
    }
    if(name==='tex2D'&&n.callee.templateArgument==='uchar2'){
      if(n.args.length!==3||n.args[0].kind!=='id')this.fail('uchar2 tex2D requires a byte-pair texture and two coordinates.',n);
      const texture=this.lookup(n.args[0].name,n.args[0]),coords=n.args.slice(1).map(a=>this.expr(a));
      if(texture.kind!=='texture'||texture.format!=='rg8uint'||texture.dimension!=='2d'||coords.some(c=>!numeric(c.type)))this.fail('uchar2 tex2D requires an rg8uint texture and numeric coordinates.',n);
      const temp='cw_sample_pair_'+this.temp++,xy=`vec2<i32>(floor(vec2<f32>(${coords.map(c=>this.convert(c.code,c.type,'f32',n)).join(', ')})))`;
      return this.result(n,'cw_uchar2',`(${temp}.x | (${temp}.y << 8u))`,[...coords.flatMap(c=>c.pre),`let ${temp} = textureLoad(${texture.code}, clamp(${xy}, vec2<i32>(0), vec2<i32>(textureDimensions(${texture.code})) - vec2<i32>(1)), 0).xy;`]);
    }
    if(name==='tex2D'){const type=n.callee.templateArgument,vector=['float2','float4'].includes(type),format=type==='float2'?'rg32float':type==='float4'?'rgba32float':'r32float';if(!['float','float2','float4'].includes(type)||n.args.length!==3||n.args[0].kind!=='id')this.fail('tex2D supports a bound float, float2 or float4 texture and two numeric coordinates.',n);const texture=this.lookup(n.args[0].name,n.args[0]),coords=n.args.slice(1).map(a=>this.expr(a));if(texture.kind!=='texture'||texture.dimension!=='2d'||texture.format!==format||coords.some(c=>!['f32','i32','u32','cw_uchar'].includes(c.type)))this.fail('tex2D requires a matching kernel texture parameter and numeric coordinates.',n);this[vector?'rgba2DSamplingUsed':'float2DSamplingUsed']=true;return this.result(n,type==='float2'?'vec2<f32>':type==='float4'?'vec4<f32>':'f32',`cw_sample_${vector?'rgba':'float'}2d(${texture.code}, ${texture.sampler}, vec2<f32>(${coords.map(c=>this.convert(c.code,c.type,'f32',n)).join(', ')}), ${texture.coordinateScale}, ${texture.pixelPoint})${type==='float2'?'.xy':''}`,coords.flatMap(c=>c.pre));}
    if(name==='tex1Dfetch'){
      const kind=n.callee.templateArgument;if(!['float','float2','float4','uint'].includes(kind)||n.args.length!==2||n.args[0].kind!=='id')this.fail('tex1Dfetch supports float, float2, float4 or uint records and one integer index.',n);
      const texture=this.lookup(n.args[0].name,n.args[0]),index=this.expr(n.args[1]);if(texture.kind!=='texture'||texture.linearFetch!==kind||!['i32','u32'].includes(index.type))this.fail('tex1Dfetch requires a matching linear texture and an integer index.',n);
      this.linearFetchUsed ||= new Set();this.linearFetchUsed.add(kind);return this.result(n,kind==='uint'?'u32':kind==='float2'?'vec2<f32>':kind==='float4'?'vec4<f32>':'f32',`cw_fetch_${kind}(${texture.code}, u32(${index.code}), ${texture.lengthCode})`,index.pre);
    }
    if(name==='tex3D'){if(!['float','float4'].includes(n.callee.templateArgument)||n.args.length!==4||n.args[0].kind!=='id')this.fail('tex3D supports a bound texture object and three float coordinates, returning float or float4.',n);const texture=this.lookup(n.args[0].name,n.args[0]);if(texture.kind!=='texture'||texture.dimension!=='3d')this.fail('tex3D requires a kernel texture parameter.',n);const coords=n.args.slice(1).map(a=>this.expr(a));if(coords.some(c=>c.type!=='f32'))this.fail('tex3D coordinates must be floats.',n);const vector=n.callee.templateArgument==='float4';if(vector)this.float4_3DSamplingUsed=true;else this.float3DSamplingUsed=true;return this.result(n,vector?'vec4<f32>':'f32',`${vector?'cw_sample_float4_3d':'cw_sample_float3d'}(${texture.code}, ${texture.sampler}, vec3<f32>(${coords.map(c=>c.code).join(', ')}), ${texture.coordinateScale}, ${texture.pixelPoint})`,coords.flatMap(c=>c.pre));}
    if (name === '__syncthreads') { if (n.args.length) this.fail('__syncthreads takes no arguments.', n); return this.result(n, 'void', 'workgroupBarrier()'); }
    if(name==='atomicCAS'){
      if(n.args.length!==3||n.args[0].kind!=='unary'||n.args[0].op!=='&'||!['index','id'].includes(n.args[0].value.kind))this.fail('atomicCAS requires &buffer[index] or &sharedScalar, compare and replacement.',n);
      const target=this.expr(n.args[0].value,true),compare=this.expr(n.args[1]),replacement=this.expr(n.args[2]);
      if(!target.atomic||!['i32','u32'].includes(target.type))this.fail('atomicCAS requires 32-bit integer atomic storage.',n);this.writable(target,n.args[0].value);
      const id=`cw_cas_${this.temp++}`,type=target.type;
      return this.result(n,type,id+'_old',[...target.pre,`let ${id}_ptr = &${target.code};`,...compare.pre,`let ${id}_compare = ${this.convert(compare.code,compare.type,type,n)};`,...replacement.pre,`let ${id}_value = ${this.convert(replacement.code,replacement.type,type,n)};`,`var ${id}_old: ${type};`,'loop {',`  let ${id}_result = atomicCompareExchangeWeak(${id}_ptr, ${id}_compare, ${id}_value);`,`  ${id}_old = ${id}_result.old_value;`,`  if (${id}_result.exchanged || ${id}_old != ${id}_compare) { break; }`,'}']);
    }
    const atomics = {atomicAdd: 'atomicAdd', atomicMin: 'atomicMin', atomicMax: 'atomicMax', atomicExch: 'atomicExchange'};
    if (atomics[name]) {
      if (n.args.length !== 2 || n.args[0].kind !== 'unary' || n.args[0].op !== '&' || !['index','id'].includes(n.args[0].value.kind)) this.fail(`${name} requires &buffer[index] or &sharedScalar and a scalar value.`, n);
      const target = this.expr(n.args[0].value, true), value = this.expr(n.args[1]);
      if (!target.atomic || !['u32', 'i32'].includes(target.type)) this.fail('Only integer buffer/shared-array atomics are supported; CUDA float atomicAdd is not silently emulated.', n);
      this.writable(target, n.args[0].value);
      return this.result(n, target.type, `${atomics[name]}(&${target.code}, ${this.convert(value.code, value.type, target.type, n)})`, [...target.pre, ...value.pre]);
    }
    const aliasType=n.aliasType;if(aliasType&&['f32','i32','u32','bool','cw_uchar'].includes(aliasType)){if(n.args.length!==1)this.fail('Scalar type aliases require one constructor argument.',n);const value=n.args[0];delete n.args;delete n.callee;Object.assign(n,{kind:'cast',target:aliasType,value});return this.expr(n);}
    const casts = {uchar:'cw_uchar',float: 'f32', int: 'i32', uint: 'u32', bool: 'bool'};
    if(name==='float'&&n.args.length===1){const value=this.expr({kind:'cast',target:'f32',value:n.args[0],token:n.token});return this.result(n,'f32',value.code,value.pre);}
    const args = n.args.map(a => this.argument(a)), pre = args.flatMap(a => a.pre);
    if(n.printfIntegerArguments&&args.some(a=>!['i32','u32','cw_short','cw_ushort','cw_uchar','bool'].includes(a.type)))this.fail('Diagnostic integer formats require integer arguments.',n);
    if (['__float_as_uint','__uint_as_float','__float_as_int','__int_as_float'].includes(name)) {
      const types = {__float_as_uint:['f32','u32'],__uint_as_float:['u32','f32'],__float_as_int:['f32','i32'],__int_as_float:['i32','f32']};
      const [from,to] = types[name];
      if(args.length!==1 || args[0].type!==from) this.fail(name+' requires one '+from+' argument.',n);
      return this.result(n,to,`bitcast<${to}>(${args[0].code})`,pre);
    }
    const halfConversions={__float2half_rn:['f32','f16'],__half2float:['f16','f32'],__half22float2:['vec2<f16>','vec2<f32>'],__float22half2_rn:['vec2<f32>','vec2<f16>']};
    if(halfConversions[name]){const [from,to]=halfConversions[name];if(args.length!==1||args[0].type!==from)this.fail(name+' requires one '+from+' argument.',n);return this.result(n,to,`${to}(${args[0].code})`,pre);}
    if(name==='__floats2half2_rn'){if(args.length!==2||args.some(a=>a.type!=='f32'))this.fail(name+' requires two float arguments.',n);return this.result(n,'vec2<f16>',`vec2<f16>(${args.map(a=>'f16('+a.code+')').join(', ')})`,pre);}
    if(['__hmul','__hadd','__hsub','__hmul2','__hadd2','__hsub2'].includes(name)){const type=name.endsWith('2')?'vec2<f16>':'f16';if(args.length!==2||args.some(a=>a.type!==type))this.fail(name+' requires two '+type+' arguments.',n);const op=name.startsWith('__hmul')?'*':name.startsWith('__hadd')?'+':'-';return this.result(n,type,`(${args[0].code} ${op} ${args[1].code})`,pre);}
    if(name==='isfinite'){if(args.length!==1||args[0].type!=='f32')this.fail('isfinite currently supports one float argument.',n);return this.result(n,'bool',`((bitcast<u32>(${args[0].code}) & 2139095040u) != 2139095040u)`,pre);}
    if(name==='__mul24'||name==='__umul24'){
      if(args.length!==2||args.some(a=>!['i32','u32'].includes(a.type)))this.fail(`${name} requires two 32-bit integer arguments.`,n);
      const signed=name==='__mul24',type=signed?'i32':'u32';
      this.integerIntrinsics.add(name);
      return this.result(n,type,`cw_${signed?'mul24':'umul24'}(${args.map(a=>this.convert(a.code,a.type,type,n)).join(', ')})`,pre);
    }
    if(['make_uchar2','make_uchar4'].includes(name)){const count=name==='make_uchar2'?2:4;if(args.length!==count||args.some(a=>!numeric(a.type)&&a.type!=='bool'))this.fail('Packed byte constructor requires '+count+' scalar components.',n);const components=args.map((a,i)=>{const value=a.type==='f32'?`u32(i32(${a.code}))`:this.convert(a.code,a.type,'u32',n);return `((${value} & 255u) << ${i*8}u)`;});return this.result(n,count===2?'cw_uchar2':'cw_uchar4',`(${components.join(' | ')})`,pre);}
    if (casts[name]) { if (args.length !== 1) this.fail('Scalar casts require one argument.', n); return this.result(n, casts[name], this.convert(args[0].code, args[0].type, casts[name], n), pre); }
    if (/^make_(float|uint|int)[234]$/.test(name)) {
      if(/^make_float[234]$/.test(name)&&args.length===Number(name.at(-1))){const vector=vectorConstructor(this,n.args);if(vector!==null)return this.result(n,`vec${args.length}<f32>`,vector,pre);}
      if(name==='make_float3'&&args.length===1&&args[0].type==='vec4<f32>')return this.result(n,'vec3<f32>',`${args[0].code}.xyz`,pre);
      if(name==='make_float4'&&args.length===2&&args[0].type==='vec3<f32>'&&args[1].type==='f32')return this.result(n,'vec4<f32>',`vec4<f32>(${args[0].code}, ${args[1].code})`,pre);
      const count = Number(name.at(-1)); if (args.length !== count && args.length !== 1) this.fail(`${name} needs ${count} arguments.`, n);
      const element=name.startsWith('make_uint')?'u32':name.startsWith('make_int')?'i32':'f32';
      return this.result(n, `vec${count}<${element}>`, `vec${count}<${element}>(${args.map(a => this.convert(a.code, a.type, element, n)).join(', ')})`, pre);
    }
    if(name==='length'&&!args.some(a=>this.structs.has(a.type))){if(args.length!==1||!vectorLength(args[0].type)||vectorElement(args[0].type)!=='f32')this.fail('length requires a float vector.',n);return this.result(n,'f32',`length(${args[0].code})`,pre);}
    if((name==='dot'||name==='normalize')&&!args.some(a=>this.structs.has(a.type))){const type=args[0]?.type;if(args.length!==(name==='dot'?2:1)||!vectorLength(type)||vectorElement(type)!=='f32'||(name==='dot'&&args[1].type!==type))this.fail(name+' requires matching float vectors.',n);return this.result(n,name==='dot'?'f32':type,`${name}(${args.map(a=>a.code).join(', ')})`,pre);}
    if(['fminf','fmaxf'].includes(name)&&args.some(a=>vectorLength(a.type))){const type=args[0]?.type;if(args.length!==2||!vectorLength(type)||vectorElement(type)!=='f32'||args[1].type!==type)this.fail(name+' requires matching float vectors.',n);return this.result(n,type,`${name==='fminf'?'min':'max'}(${args.map(a=>a.code).join(', ')})`,pre);}
    if(name==='__fdividef'){if(args.length!==2)this.fail('__fdividef requires two arguments.',n);return this.result(n,'f32',`(${args.map(a=>this.convert(a.code,a.type,'f32',n)).join(' / ')})`,pre);}
    if(name==='exp'){if(args.length!==1||args[0].type!=='f32')this.fail('exp supports the float overload only.',n);return this.result(n,'f32',`exp(${args[0].code})`,pre);}
    if(name==='sqrt'&&args.length===1&&args[0].type==='cw_f64'){this.float64Used=true;return this.result(n,'cw_f64',`cw_d_sqrt(${args[0].code})`,pre);}
    if(['fmin','fmax'].includes(name)&&args.length===2&&args.some(a=>a.type==='cw_f64')){if(args.some(a=>!['cw_f64','f32','i32','u32'].includes(a.type)))this.fail('Double min/max require scalar numeric arguments.',n);this.float64Used=true;return this.result(n,'cw_f64',`cw_d_${name}(${args.map(a=>this.convert(a.code,a.type,'cw_f64',n)).join(', ')})`,pre);}
    if(name==='sqrt'){if(args.length!==1||args[0].type!=='f32')this.fail('sqrt supports the single float overload only; double/integer overloads are unavailable.',n);return this.result(n,'f32',`sqrt(${args[0].code})`,pre);}
    if(['rint','rintf'].includes(name)){if(args.length!==1||args[0].type!=='f32')this.fail(name+' requires one float argument.',n);this.roundEvenUsed=true;return this.result(n,'f32',`cw_round_even(${args[0].code})`,pre);}
    if(name==='roundf'){if(args.length!==1||args[0].type!=='f32')this.fail('roundf requires one float argument.',n);this.roundAwayUsed=true;return this.result(n,'f32',`cw_round_away(${args[0].code})`,pre);}
    if(name==='abs'&&args.length===1&&args[0].type==='f32')return this.result(n,'f32',`abs(${args[0].code})`,pre);
    if(name==='abs'){if(args.length!==1||!(args[0].type==='i32'||narrow(args[0].type)))this.fail('abs requires a signed integer or promoted narrow integer.',n);return this.result(n,'i32',`abs(${this.convert(args[0].code,args[0].type,'i32',n)})`,pre);}
    if(name==='__saturatef'){if(n.args.length!==1)this.fail('__saturatef requires one float argument.',n);const a=args[0];if(a.type!=='f32')this.fail('__saturatef requires a float argument.',n);return this.result(n,'f32',`clamp(${a.code}, 0.0f, 1.0f)`,a.pre);}
    if(['fabs','floor'].includes(name)&&(args.length!==1||args[0].type!=='f32'))this.fail(name+' supports the CUDA float overload; double precision is unavailable.',n);
    const unary = {sinf: 'sin', cosf: 'cos', tanf: 'tan', tan: 'tan', sqrtf: 'sqrt', rsqrtf: 'inverseSqrt', expf: 'exp', __expf:'exp', exp2f: 'exp2', logf: 'log', __logf:'log', log2f: 'log2', fabs:'abs', fabsf: 'abs', floor: 'floor', floorf: 'floor', ceilf: 'ceil', truncf: 'trunc'};
    if(['fmin','fmax'].includes(name)&&(args.length!==2||args.some(a=>a.type!=='f32')))this.fail(name+' requires two float arguments.',n);
    if(['pow','powf'].includes(name)){
      if(args.length!==2)this.fail(name+' requires two arguments.',n);
      this.float64Used=true;this.integerPowerUsed=true;
      return this.result(n,'f32',`cw_pow_f32(${args.map(a=>this.convert(a.code,a.type,'f32',n)).join(', ')})`,pre);
    }
    const binary = {fmin: 'min', fmax: 'max', fminf: 'min', fmaxf: 'max', powf: 'pow', pow: 'pow', atan2f: 'atan2'};
    if (unary[name] || binary[name] || name === 'fmaf') {
      const count = unary[name] ? 1 : binary[name] ? 2 : 3;
      if (args.length !== count) this.fail(`${name} requires ${count} arguments.`, n);
      const target = unary[name] || binary[name] || 'fma';
      return this.result(n, 'f32', `${target}(${args.map(a => this.convert(a.code, a.type, 'f32', n)).join(', ')})`, pre);
    }
    if (['min', 'max'].includes(name)) {
      if (args.length !== 2) this.fail(`${name} needs two arguments.`, n);
      const type = this.common(args[0].type==='cw_uchar'?'i32':args[0].type, args[1].type==='cw_uchar'?'i32':args[1].type, n);
      if (!numeric(type)) this.fail('min/max accept scalars.', n);
      return this.result(n, type, `${name}(${args.map(a => this.convert(a.code, a.type, type, n)).join(', ')})`, pre);
    }
    if(name==='curand_init'&&this.options.libraries?.includes('curand-xorwow')){
      let zeroPosition=false;try{zeroPosition=args.length===4&&constantValue(n.args[1])===0&&constantValue(n.args[2])===0;}catch{}if(!zeroPosition)this.fail('XORWOW currently requires compile-time zero subsequence and offset.',n);
      if(!['i32','u32','cw_size64'].includes(args[0].type))this.fail('XORWOW requires an integer seed.',n);
    }
    let helper = this.functions.get(name);
    if(this.overloads.has(name)){const matches=this.overloads.get(name).filter(f=>args.length<=f.params.length&&f.params.every((p,i)=>i>=args.length?p.defaultValue!==undefined:(typeName(p.pointer?(isArray(args[i].type)?args[i].type.element:null):args[i].type)===typeName(p.type)||String(p.type).startsWith('cw_objectptr_')&&['0i','0u'].includes(args[i].code)||String(p.type).startsWith('cw_objectlist_')&&args[i].rootSymbol?.objectImport?.type===p.type.replace('cw_objectlist_','cw_objectptr_')||String(p.type).startsWith('cw_bufferref_')&&args[i].rootSymbol?.objectImport?.type===p.type.replace('cw_bufferref_',''))));if(!matches.length){const convertible=this.overloads.get(name).filter(f=>(f.classConstructor||f.params.every(p=>!p.pointer&&!p.reference&&(numeric(p.type)||p.type==='cw_f64')))&&args.length===f.params.length&&f.params.every((p,i)=>typeName(args[i].type)===typeName(p.type)||String(args[i].type).startsWith('cw_objectptr_')&&this.structs.get('cw_struct_'+args[i].type.slice(13))?.base===String(p.type).slice(13)||!p.pointer&&!p.reference&&(numeric(args[i].type)||args[i].type==='cw_f64')&&(numeric(p.type)||p.type==='cw_f64')));if(convertible.length===1)matches.push(convertible[0]);}if(matches.length!==1)this.fail('Overload '+name+' requires one exact parameter-type match or an unambiguous scalar conversion.',n);helper=matches[0];}
    if(!helper&&this.templates){
      helper=this.templates.deduce(name,args.map(a=>a.type),n.callee);
      if(helper)for(const fn of this.ast.functions)if(fn.qualifier==='__device__'&&!this.functions.has(fn.name)){this.functions.set(fn.name,fn);if(!fn.params.some(p=>p.pointer))this.helpers.push(fn);}
    }
    if (!helper || helper.qualifier !== '__device__') this.fail(`Unsupported function '${name}'. CUDA host APIs, warp intrinsics, dynamic launches and libraries are not available.`, n);
    if(args.length>helper.params.length||helper.params.slice(args.length).some(p=>p.defaultValue===undefined))this.fail(`Wrong number of arguments for '${name}'.`,n);
    for(const param of helper.params)if(param.defaultValue!==undefined&&!['f32','i32','u32','bool','cw_uchar'].includes(param.type))this.fail('Default arguments require scalar value parameters.',param);
    while(args.length<helper.params.length){const value=structuredClone(helper.params[args.length].defaultValue);n.args.push(value);const argument=this.argument(value);args.push(argument);pre.push(...argument.pre);}
    if(helper.params.some(p=>p.pointer))helper=this.bindPointerHelper(helper,args,n);
    if(helper.params.some(p=>p.reference))helper=this.bindReferenceHelper(helper,args,n);
    if(helper.classAccess==='private'&&this.currentFunction.classOwner!==helper.classOwner)this.fail('Private class method or constructor is inaccessible here.',n);
    const caller=this.currentFunction.name,edges=this.helperCalls.get(caller)||new Set();edges.add(helper.name);this.helperCalls.set(caller,edges);
    const reaches=(from,target,seen=new Set())=>{if(from===target)return true;if(seen.has(from))return false;seen.add(from);return [...(this.helperCalls.get(from)||[])].some(next=>reaches(next,target,seen));};
    if(reaches(helper.name,caller))this.fail('Recursive helper calls are unsupported.',n);
    n.callee.name=helper.name;n.callName=helper.name;n.userHelper=true;
    const references=new Map();n.constRefTemporaries=[];n.localPointerArgs=helper.params.map(p=>!!p.localPointer);n.referenceArgs=helper.params.map(p=>!!p.reference);n.groupArgs=helper.params.map(p=>p.type==='thread-block');n.pointerArgs=helper.params.map(p=>!!p.pointer);
    const codes=args.map((a,i)=>{const p=helper.params[i];if(p.type==='texture3d'){const shape=textureShape(p.textureSampling),symbol=a.rootSymbol;if(symbol?.kind!=='texture'||symbol.dimension!==shape.dimension||symbol.format!==shape.format)this.fail('Texture helper argument requires the same inferred sampling format.',n.args[i]);return symbol.code+', '+symbol.sampler+(symbol.linearFetch?', '+symbol.lengthCode:'')+(['tex2D','tex2Dfloat2','tex2Dfloat4','tex3D','tex3Dfloat4'].includes(p.textureSampling)?', '+symbol.coordinateScale+', '+symbol.pixelPoint:'');}if(p.localArray){if(references.has(a.rootSymbol))this.fail('Aliased local array arguments are unsupported.',n.args[i]);references.set(a.rootSymbol,false);return a.pointerCode;}if(p.localPointer){if(references.has(a.rootSymbol))this.fail('Aliased local pointer/reference arguments are unsupported.',n.args[i]);references.set(a.rootSymbol,false);return a.pointerCode;}if(p.pointer)return a.pointerCode;if(p.type==='thread-block'){if(a.type!=='thread-block'||a.rootSymbol?.kind!=='thread-block')this.fail('thread_block arguments require a block handle.',n.args[i]);return null;}if(String(p.type).startsWith('cw_objectlist_')&&isArray(a.type)){const imported=a.rootSymbol?.objectImport;if(!imported||imported.type!==p.type.replace('cw_objectlist_','cw_objectptr_'))this.fail('Captured list arguments require a registered object pointer buffer.',n.args[i]);return `${imported.id*1048576}u + u32(${a.pointerCode||'0i'})`;}if(String(p.type).startsWith('cw_bufferref_')&&isArray(a.type)){const captured=bufferReferenceArgument(this,a,p.type,n);pre.push(...captured.pre.slice(a.pre.length));return captured.code;}if(!p.reference)return this.convert(a.code,a.type,p.type,n);
      const node=n.args[i],s=a.rootSymbol;
      if(p.boundReferenceStorage){if(a.type!==p.type||!p.constant&&s?.constant)this.fail('Persistent reference requires a mutable object of the same type.',node);const index=a.storageReferenceIndexCode??s?.storageReferenceIndexCode;if(index===undefined)this.fail('Persistent object index is unavailable.',node);return `i32(${index})`;}
      if(p.boundReferenceShared){if(s?.atomic||a.type!==p.type||!p.constant&&s?.constant)this.fail('Shared reference requires a matching non-atomic array element.',node);const index=a.sharedReferenceIndexCode??s?.sharedReferenceIndexCode;if(index===undefined)this.fail('Shared reference index is unavailable.',node);return `i32(${index})`;}
      if(s?.kind==='shared'&&s.atomic)this.fail('Atomic shared values cannot bind ordinary references.',node);
      if(s?.referenceSpace==='storage'){if(!p.constant)this.fail('Mutable helper references into persistent objects are not yet supported.',node);const temp='cw_storage_ref_'+this.temp++;pre.push(`var ${temp}: ${p.type} = ${a.code};`);n.constRefTemporaries[i]=true;return '&'+temp;}
      if(p.constant){if(node.kind==='member'&&['cw_uchar2','cw_uchar4'].includes(node.base.type)&&p.type==='cw_uchar'){const temp='cw_const_ref_'+this.temp++;pre.push(`var ${temp}: cw_uchar = ${a.code};`);n.constRefTemporaries[i]=true;return '&'+temp;}if(s?.rootBufferName)this.fail('Const references to storage elements are unsupported; copy the value to a local first.',node);if(a.type!==p.type||!(numeric(p.type)||['cw_uchar2','cw_uchar4'].includes(p.type)||vectorLength(p.type)||this.structs.has(p.type)))this.fail('Const references require the exact scalar, vector or struct type.',node);if(s&&references.has(s)&&!references.get(s))this.fail('Aliased reference arguments are unsupported.',node);if(s)references.set(s,true);if(s?.kind==='reference')return node.kind==='id'?s.pointerCode:'&'+a.code;if(s?.kind==='shared'){if(!['id','index'].includes(node.kind))this.fail('Shared references require whole scalars/vectors or array elements.',node);return '&'+a.code;}if(s?.kind==='local'&&!s.constant&&['id','member','object-deref'].includes(node.kind))return '&'+a.code;const temp='cw_const_ref_'+this.temp++;pre.push(`var ${temp}: ${p.type} = ${a.code};`);n.constRefTemporaries[i]=true;return '&'+temp;}
      if(!(['id','index','object-deref'].includes(node.kind)||node.kind==='member'&&this.structs.has(a.type))||!s||!['local','reference','shared'].includes(s.kind)||s.constant||isArray(a.type)||!(numeric(a.type)||vectorLength(a.type)||this.structs.has(a.type))||a.type!==p.type)this.fail('Reference arguments require a mutable scalar/vector or array element in local or shared memory, of the exact type.',node);
      if(references.has(s))this.fail('Aliased reference arguments are unsupported.',node);references.set(s,false);return s.kind==='reference'&&node.kind==='id'?s.pointerCode:`&${a.code}`;
    });
    return this.result(n, helper.result, `f_${helper.name}(${[...codes.filter(c=>c!==null),'cw_thread','cw_block','cw_grid'].join(', ')})`, pre);
  }
  writable(target, n) {
    const s = target.rootSymbol;
    if(target.packedBase&&!target.packedAtomic&&(n.base?.kind!=='id'||s?.kind!=='local'||!['cw_uchar2','cw_uchar4'].includes(s.type)))this.fail('Byte component writes require a named local uchar4; write complete uchar4 records to storage or shared memory.',n);
    if (!s || !['id', 'index', 'member'].includes(n.kind) || isArray(target.type)) this.fail('Assignment requires a scalar/vector variable or array element.', n);
    const containsConst=type=>this.structs.has(type)&&this.structs.get(type).fields.some(f=>f.constant||containsConst(f.resolvedType));if(containsConst(target.type)&&!n.initializingField)this.fail('Cannot assign a record containing const fields.',n);
    if (s.constant) this.fail(`Cannot write through const '${s.name}'.`, n);
    if (s.kind === 'uniform') this.fail('Scalar kernel parameters are read-only in this subset. Copy the parameter to a local variable first.', n);
  }
  aggregateCopy(type,path){
    if(isArray(type))return `${typeName(type)}(${Array.from({length:type.length},(_,i)=>this.aggregateCopy(type.element,`${path}[${i}i]`)).join(', ')})`;
    if(this.structs.has(type))return `${type}(${this.structs.get(type).fields.map(f=>this.aggregateCopy(f.resolvedType,`${path}.cw_field_${f.name}`)).join(', ')})`;
    return path;
  }
  effect(n) {
    if(n.kind==='object-delete'||n.kind==='device-launch')return this.expr(n).pre;
    if(n.kind==='sequence')return n.expressions.flatMap(e=>this.effect(e));
    if (n.kind === 'assign') {
      if(['+=','-=','*=','/='].includes(n.op)){const left=this.expr(n.left,true),method=this.structs.get(left.type)?.methods?.find(m=>m.name==='operator'+n.op);if(method){const receiver=n.left,right=n.right;delete n.left;delete n.right;delete n.op;Object.assign(n,{kind:'call',callee:{kind:'member',base:receiver,member:method.name,token:n.token},args:[right]});return this.effect(n);}}
      const destinationUpdates=[];walk(n.left,node=>{if(node.kind==='unary'&&['++','--'].includes(node.op))destinationUpdates.push(node);});if(destinationUpdates.length){if(n.op!=='='||n.left.kind!=='index'||destinationUpdates.length!==1)this.fail('Increment/decrement in assignment destinations requires one indexed = store.',n);n.destinationEffects=true;}
      if(n.op==='='&&n.right.kind==='assign'){let current=n;while(current.kind==='assign'){if(current.op!=='='||current.left.kind!=='id'||!['local','reference'].includes(this.lookup(current.left.name,current.left).kind))this.fail('Chained assignment requires named local variables or references and =.',current);current=current.right;}const inner=this.effect(n.right),target=this.expr(n.left,true);this.writable(target,n.left);const value=this.expr(n.right.left);n.type=target.type;return [...inner,`${target.code} = ${this.convert(value.code,value.type,target.type,n)};`];}
      if(n.left.kind==='id'&&n.op==='='){
        const symbol=this.lookup(n.left.name,n.left);
        if(['buffer','buffer-alias'].includes(symbol.kind)){
          const value=this.argument(n.right);if(!symbol.offsetCode||!value.pointerCode||value.rootSymbol?.code!==symbol.code||typeName(value.type)!==typeName(symbol.type))this.fail('Pointer reassignment must stay within the same typed allocation.',n);
          if(value.rootSymbol.constant&&!symbol.constant)this.fail('Cannot discard const through pointer reassignment.',n);
          n.left.symbol=symbol;n.pointerRebind=true;n.type=symbol.type;
          return [...value.pre,`${symbol.offsetCode} = ${value.pointerCode};`];
        }
      }
      if(n.left.kind==='id'&&['+=','-='].includes(n.op)){
        const symbol=this.lookup(n.left.name,n.left);
        if(['buffer','buffer-alias'].includes(symbol.kind)){
          const value=this.expr(n.right);if(!['i32','u32'].includes(value.type)||!symbol.offsetCode)this.fail('Pointer updates require a 32-bit integer offset.',n);
          n.left.symbol=symbol;n.pointerShift=true;n.type=symbol.type;
          return [...value.pre,`${symbol.offsetCode} ${n.op} ${this.convert(value.code,value.type,'i32',n)};`];
        }
      }
      if(n.left.kind==='index'&&n.left.base.kind==='id'&&this.lookup(n.left.base.name,n.left).kind==='pointer-array'){
        if(n.destinationEffects)this.fail('Increment/decrement in shared pointer-array destinations is unsupported.',n);
        const slots=this.lookup(n.left.base.name,n.left);let rhs=n.right;if(rhs.kind==='pointer-cast'){if(rhs.target!==slots.elementType||rhs.constant&&!slots.constant||rhs.volatilePointer&&!slots.volatilePointer)this.fail('Pointer array casts must preserve pointee type and qualifiers.',n);rhs=rhs.value;}const address=rhs.kind==='unary'&&rhs.op==='&'?rhs.value:null;
        if(n.op!=='='||address?.kind!=='index'||address.base.kind!=='id')this.fail('Pointer array assignments require a shared-array element address.',n);
        const root=this.lookup(address.base.name,address);if(root.kind!=='shared'||(root.atomic&&!slots.volatilePointer)||!isArray(root.type)||root.type.element!==slots.elementType)this.fail('Pointer arrays require a matching non-atomic shared array.',n);
        if(root.constant&&!slots.constant)this.fail('Cannot discard const in a pointer array.',n);
        if(slots.pointerRoot&&slots.pointerRoot!==root)this.fail('A pointer array must refer to one shared allocation.',n);
        slots.pointerRoot=root;const left=this.expr(n.left.base),index=this.expr(n.left.index),offset=this.expr(address.index);if(!['i32','u32'].includes(index.type)||!['i32','u32'].includes(offset.type))this.fail('Pointer array offsets must be 32-bit integers.',n);
        n.pointerArrayAssignment=true;n.pointerArrayOffset=address.index;n.left.type='i32';n.type='i32';
        return [...left.pre,...index.pre,...offset.pre,`${slots.code}[${index.code}] = ${this.convert(offset.code,offset.type,'i32',n)};`];
      }
      const target = this.expr(n.left, true); this.writable(target, n.left); let value = String(target.type).startsWith('cw_bufferref_')?this.argument(n.right):this.expr(n.right);if(String(target.type).startsWith('cw_bufferref_')&&isArray(value.type)){const captured=bufferReferenceArgument(this,value,target.type,n);value={...value,...captured,type:target.type};}if(n.destinationEffects){if(target.packedAtomic||target.packedBase||target.packedPairComponents||target.scalarVectorComponents)this.fail('Increment/decrement in packed assignment destinations is unsupported.',n);const snapshot='cw_assignment_rhs_'+this.temp++;target.pre=[...value.pre,`let ${snapshot}: ${typeName(value.type)} = ${value.code};`,...target.pre];value={...value,code:snapshot,pre:[]};}let packedPre;if(target.packedAtomic){n.packedAtomicAssignment=true;const tmp='cw_byte_value_'+this.temp++;packedPre=[...value.pre,`let ${tmp}: ${typeName(value.type)} = ${value.code};`,...target.pre];value={...value,code:tmp,pre:[]};}
      if(['+=','-=','*=','/='].includes(n.op)&&this.structs.has(target.type)){
        const name='cw_compound_'+{'+':'add','-':'subtract','*':'multiply','/':'divide'}[n.op[0]];
        if(!this.functions.has(name)&&!this.overloads.has(name))this.fail('No supported free compound operator is declared for '+n.op,n);
        const left=n.left,right=n.right;delete n.left;delete n.right;delete n.op;
        Object.assign(n,{kind:'call',callee:{kind:'id',name,token:n.token},args:[left,right]});return this.effect(n);
      }
      if(n.op==='='&&this.structs.has(target.type)&&value.type===target.type){const snapshot='cw_value_copy_'+this.temp++;n.type=target.type;return [...target.pre,...value.pre,`let ${snapshot} = ${value.code};`,`${target.code} = ${this.aggregateCopy(target.type,snapshot)};`];}
      if(target.packedPairComponents){
        if(n.op!=='='||value.type!=='cw_uchar2')this.fail('Packed pair stores require a whole uchar2 assignment.',n);
        this.packedAtomicUsed=true;const temp='cw_pair_store_'+this.temp++;n.type='cw_uchar2';
        return [...target.pre,...value.pre,`let ${temp} = ${value.code};`,...target.packedPairComponents.map(({word,shift},i)=>`cw_store_byte(&${word}, ${shift}, (${temp} >> ${i*8}u) & 255u);`)];
      }
      if(target.scalarVectorComponents){
        if(n.op!=='='||value.type!==target.type)this.fail('Vector pointer stores require a matching vector assignment.',n);
        const temp='cw_vector_store_'+this.temp++;n.type=target.type;
        return [...target.pre,...value.pre,`let ${temp} = ${value.code};`,...target.scalarVectorComponents.map((code,i)=>`${code} = ${temp}.${'xyzw'[i]};`)];
      }
      if(n.op!=='='&&vectorLength(target.type)){const op=n.op.slice(0,-1);if(n.left.kind!=='id'||vectorElement(target.type)!=='f32'||!['+','-','*','/'].includes(op)||![target.type,'f32'].includes(value.type))this.fail('Vector compound assignments require a named float vector and matching vector or float scalar.',n);const rhs=value.type===target.type?value.code:`${target.type}(${value.code})`;n.operandType=target.type;n.type=target.type;return [...target.pre,...value.pre,`${target.code} = ${target.code} ${op} ${rhs};`];}
      if(['cw_uchar2','cw_uchar4'].includes(target.type)&&n.op!=='=')this.fail('uchar4 compound arithmetic requires explicit byte components.',n);
      let code = this.convert(value.code, value.type, target.type, n);
      if (n.op !== '=') {
        const op = n.op.slice(0, -1); if (target.atomic&&!target.rootSymbol?.volatileShared) this.fail('Use explicit atomicAdd/Min/Max/Exch rather than compound assignments to atomic arrays.', n);
        // CUDA volatile compound updates are a load followed by a store, not
        // an atomic read-modify-write. Capture the destination address once.
        let current=target.code;
        if(target.atomic){const address='cw_volatile_address_'+this.temp++;target.pre.push(`let ${address} = &${target.code};`);target.code=`(*${address})`;current=target.type==='f32'?`bitcast<f32>(atomicLoad(${address}))`:`atomicLoad(${address})`;}

        const type = ['<<', '>>'].includes(op) ? (narrow(target.type)?'i32':target.type) : this.common(narrow(target.type)?'i32':target.type, value.type==='cw_uchar'?'i32':value.type, n), rhsType = ['<<', '>>'].includes(op) ? 'u32' : type;
        if (['<<', '>>', '%', '&', '|', '^'].includes(op) && !['i32', 'u32'].includes(type)) this.fail('Integer operator requires integer operands.', n);
        if(type==='cw_f64'){const fn={'+':'cw_d_add','-':'cw_d_sub','*':'cw_d_mul','/':'cw_d_div'}[op];if(!fn)this.fail('Unsupported double compound operator.',n);this.float64Used=true;code=this.convert(`${fn}(${this.convert(current,target.type,type,n)}, ${this.convert(value.code,value.type,type,n)})`,type,target.type,n);}else if(op==='/'&&type==='f32'){this.compensatedDivisionUsed=true;code=this.convert(`cw_divide_f32(${this.convert(current,target.type,type,n)}, ${this.convert(value.code,value.type,rhsType,n)})`,type,target.type,n);}else code = this.convert(`(${this.convert(current, target.type, type, n)} ${op} ${this.convert(value.code, value.type, rhsType, n)})`, type, target.type, n);
        n.operandType = type;
      }
      n.type = target.type;
      if(target.packedAtomic){this.packedAtomicUsed=true;return [...packedPre,`cw_store_byte(&${target.packedBase}, ${target.packedShiftCode??target.packedShift+'u'}, ${target.type==='bool'?`select(0u, 1u, ${code})`:`u32(${code})`});`];}
      if(target.packedBase)return [...target.pre,...value.pre,`${target.packedBase} = (${target.packedBase} & ${(~(255<<target.packedShift))>>>0}u) | ((u32(${code}) & 255u) << ${target.packedShift}u);`];
      if(target.bufferReferenceTargets)return [...target.pre,...value.pre,...target.bufferReferenceTargets.map(t=>`if(${t.guard}) { ${t.code} = ${code}; }`)];
      if(target.devicePointerGuard)return [...target.pre,...value.pre,`if(${target.devicePointerGuard}) { ${target.code} = ${code}; }`];
      return [...target.pre, ...value.pre, target.atomic ? `atomicStore(&${target.code}, ${target.rootSymbol?.volatileShared&&target.type==='f32'?`bitcast<u32>(${code})`:code});` : `${target.code} = ${code};`];
    }
    if (n.kind === 'unary' && ['++', '--'].includes(n.op)) {
      const target = this.expr(n.value, true); this.writable(target, n.value); if(target.rootSymbol?.deviceAllocation)this.fail('Device heap increments require an explicit indexed assignment.',n); if (target.atomic || !numeric(target.type)) this.fail('Increment/decrement require a non-atomic scalar.', n);
      n.type = target.type;if(target.packedAtomic){this.packedAtomicUsed=true;return [...target.pre,`cw_store_byte(&${target.packedBase}, ${target.packedShiftCode??target.packedShift+'u'}, u32(i32(${target.code}) ${n.op==='++'?'+':'-'} 1i));`];}if(['cw_short','cw_ushort'].includes(target.type))return [...target.pre,`${target.code} = ${this.convert(`i32(${target.code}) ${n.op==='++'?'+':'-'} 1i`,'i32',target.type,n)};`];if(target.type==='cw_uchar'&&!target.packedBase)return [...target.pre,`${target.code} = (${target.code} ${n.op==='++'?'+':'-'} 1u) & 255u;`];if(target.packedBase)return [...target.pre,`${target.packedBase} = (${target.packedBase} & ${(~(255<<target.packedShift))>>>0}u) | ((u32(i32(${target.code}) ${n.op==='++'?'+':'-'} 1i) & 255u) << ${target.packedShift}u);`]; return [...target.pre, `${target.code} ${n.op === '++' ? '+=' : '-='} ${target.type}(1);`];
    }
    const value = this.expr(n);
    if(value.type==='void'&&!value.code)return value.pre;
    if (n.kind !== 'call') this.fail('Only assignments, increments and function calls may stand alone as statements.', n);
    if (n.callName === '__syncthreads') return [...value.pre, 'workgroupBarrier();', ...(this.usage.storageBarrier ? ['storageBarrier();'] : [])];
    return [...value.pre, value.type === 'void' ? `${value.code};` : `_ = ${value.code};`];
  }
  declare(n) {
    // Class pointers normally denote heap handles. A pointer into an explicit
    // value buffer instead retains that buffer's element type and offset.
    if(!n.pointer&&!n.reference&&!n.dimensions.length&&String(n.type).startsWith('cw_objectptr_')){
      const parts=pointerParts(n.init),record=this.ast.structs.find(r=>'cw_objectptr_'+r.name===n.type);
      if(parts&&record?.valueClass){const base=this.lookup(parts.base.name,parts.base);if(['buffer','buffer-alias'].includes(base.kind)&&base.type.element===record.type){n.type=record.type;n.pointer=true;}}
    }
    if(n.type==='cw_f64'){this.float64Used=true;if(n.pointer||n.reference||n.shared||n.dimensions.length)this.fail('Double locals support scalar values only.',n);}
    if(n.type==='cw_size64'){this.extentUsed=true;if(n.pointer||n.shared||n.dimensions.length)this.fail('size_t locals support scalar values only.',n);}
    if(n.type==='cw_extent')this.fail('cudaExtent supports read-only by-value kernel parameters only.',n);
    if(['texture3d','surface2d'].includes(n.type))this.fail('Texture and surface handles require kernel or supported helper parameters, not local aliases.',n);
    if(n.init?.kind==='shared-conversion'){
      if(!n.pointer||n.reference||n.shared||n.external||n.dimensions.length||n.type!==n.init.target)this.fail('Shared conversion requires a matching local pointer declaration.',n);
      if(!n.init.conversions.includes(false)&&!n.constant)this.fail('Cannot discard const from shared wrapper conversion.',n);
      // A stateless conversion exposes the dispatch's dynamic shared allocation.
      n.pointer=false;n.shared=true;n.external=true;n.dimensions=[null];n.init=null;
      if(n.constant)this.fail('Const shared wrapper views are not yet supported.',n);
    }
    if(n.external&&(!n.shared||n.pointer||n.reference||n.constant||n.init||n.dimensions.length!==1||n.dimensions[0]!==null))this.fail('extern is supported only as extern __shared__ T name[].',n);
    if(n.reference){
      if(n.pointer||n.shared||n.external||n.dimensions.length||!n.init)this.fail('Local references currently require const and a stable initializer.',n);
      const value=this.expr(n.init),root=value.rootSymbol;
      if(value.type!==n.type||!root||!['local','reference','buffer','buffer-alias','shared'].includes(root.kind)||root.atomic||!(['id','member','index'].includes(n.init.kind)||value.fieldReference))this.fail('Const local reference needs an addressable value of the same type.',n);
      if(!n.constant&&root.constant)this.fail('Cannot discard const through a local reference.',n);
      if(!n.constant&&(!this.structs.has(n.type)||value.storageReferenceIndexCode===undefined))this.fail('Mutable local references require a writable storage record.',n);
      const pointer='cw_ref_'+this.temp++,index=value.storageReferenceIndexCode,referencePre=[...value.pre],storageRoot=value.storageReferenceRoot;
      let address=value.code;if(storageRoot&&index!==undefined){referencePre.push(`let ${pointer}_index = i32(${index});`);address=storageRoot+'['+pointer+'_index]'+(value.storageReferencePath||'');}
      const symbol={...root,name:n.name,type:n.type,code:`(*${pointer})`,pointerCode:pointer,kind:'reference',constant:n.constant||root.constant,referenceSpace:root.referenceSpace||(['buffer','buffer-alias'].includes(root.kind)?'storage':root.kind==='shared'?'workgroup':'function'),...(storageRoot?{storageReferenceRoot:storageRoot,storageReferencePath:value.storageReferencePath||'',storageReferenceIndexCode:pointer+'_index'}:{})};
      if(!storageRoot&&value.type!==root.type){delete symbol.storageReferenceRoot;delete symbol.storageReferenceIndexCode;}
      this.add(n.name,symbol,n);n.symbol=symbol;n.localReference=true;n.resolvedType=n.type;return [...referencePre,`let ${pointer} = &${address};`];
    }
    if(n.pointer&&n.dimensions.length){
      const length=n.dimensions.length===1?constantValue(n.dimensions[0]):null;
      if(n.shared||n.external||n.init||!Number.isInteger(length)||length<1||length>256||!(numeric(n.type)||vectorLength(n.type)))this.fail('Pointer arrays require 1..256 local slots and scalar/vector shared pointees.',n);
      const code='v_'+n.name,symbol={name:n.name,kind:'pointer-array',type:arrayOf(arrayOf(n.type),length),elementType:n.type,code,constant:n.constant,volatilePointer:!!n.volatilePointer};this.add(n.name,symbol,n);n.symbol=symbol;n.resolvedDimensions=[length];n.resolvedType=arrayOf('i32',length);
      return [`var ${code}: array<i32, ${length}>;`];
    }
    if(n.pointer){
      const parts=pointerParts(n.init),baseNode=parts?.base;let offsetNode=parts?.offset;
      if(n.shared||n.dimensions.length||baseNode?.kind!=='id')this.fail('Local pointers require a buffer alias with an optional integer offset.',n);
      const base=this.lookup(baseNode.name,baseNode);if(!['buffer','buffer-alias','shared'].includes(base.kind)||base.type.element!==n.type||(base.kind==='shared'&&base.atomic))this.fail('Local pointers can alias only same-type storage buffers or non-atomic shared arrays.',n);
      if(base.constant&&!n.constant)this.fail('Cannot discard const through a buffer alias.',n);
      const offset=offsetNode?this.expr(offsetNode):{type:'i32',code:'0i',pre:[]};if(!['i32','u32'].includes(offset.type))this.fail('Buffer alias offsets must be 32-bit integers.',n);
      const offsetCode=`cw_offset_${this.temp++}`,symbol={...base,...(base.kind==='shared'?{sharedPointer:base.code}:{}),name:n.name,constant:n.constant||base.constant,kind:'buffer-alias',offsetCode};this.add(n.name,symbol,n);n.symbol=symbol;n.aliasBase=base;n.aliasOffset=offsetNode;
      return [...offset.pre,`var ${offsetCode} = ${base.offsetCode?base.offsetCode+' + ':''}${this.convert(offset.code,offset.type,'i32',n)};`];
    }
    if(this.structs.has(n.type)&&(n.shared||n.dimensions.length))this.fail('Structs currently support local values only, not shared memory or arrays of structs.',n);
    if (n.type === 'void') this.fail('Variables cannot have void type.', n);
    let type = n.type;const sharedOwner=(this.currentFunction.pointerOrigin||this.currentFunction.name)+':'+n.token.offset+':'+n.name;
    const dims = n.dimensions.map(d => {if(d===null){if(!n.external||!n.shared)this.fail('Unsized arrays require extern __shared__.',n);if(this.dynamicSharedUsed&&this.dynamicSharedOwner!==sharedOwner)this.fail('Only one dynamic shared array is supported; CUDA declarations alias the same allocation.',n);const stride=n.type==='cw_uchar'?1:typeStride(n.type);if(n.type==='bool'||['cw_short','cw_ushort'].includes(n.type)||vectorLength(n.type)===3)this.fail('Dynamic shared arrays require 32-bit scalars or two/four-component vectors.',n);if(!this.dynamicSharedBytes||this.dynamicSharedBytes%stride)this.fail('Set sharedMemoryBytes to a positive multiple of the dynamic shared element size.',n);this.dynamicSharedUsed=true;this.dynamicSharedOwner=sharedOwner;return this.dynamicSharedBytes/stride;}const value = constantValue(d); if (!Number.isSafeInteger(value) || value < 1 || value > 65536) this.fail('Invalid fixed array dimension (1..65536).', n); return value; });
    for (let i = dims.length - 1; i >= 0; i--) type = arrayOf(type, dims[i]);
    if (n.shared && n.init) this.fail('__shared__ variables cannot have an initializer.', n);
    if (isArray(type) && n.init&&n.init.kind!=='initializer') this.fail('Local arrays require a brace initializer.', n);
    let volatileAliased=false;if(n.shared){const slots=new Set();walk(this.currentFunction.body,x=>{if(x.kind==='decl'&&x.volatilePointer)slots.add(x.name);});walk(this.currentFunction.body,x=>{if(x.kind!=='assign'||x.left.kind!=='index'||!slots.has(x.left.base.name))return;const rhs=x.right.kind==='pointer-cast'?x.right.value:x.right;if(rhs.kind==='unary'&&rhs.op==='&'&&rhs.value.kind==='index'&&rhs.value.base.name===n.name)volatileAliased=true;});}
    const atomic = n.shared && (volatileAliased||n.volatileShared||analyse([this.currentFunction],[]).atomic.has(n.name));
    if (atomic && !['i32', 'u32'].includes(n.type)&&!(n.volatileShared&&n.type==='f32')) this.fail('Shared atomics require int or unsigned int.', n);
    if(n.init?.kind==='initializer')n.init.target=type;
    const init = n.init ? this.expr(n.init) : null;
    if(!init&&this.structs.get(type)?.fields.some(f=>f.constant)&&!(this.currentFunction.classConstructor&&n.name==='cw_object_'+this.currentFunction.classOwner))this.fail('Const fields require constructor initialization.',n);
    if (n.constant && !init && !n.shared) this.fail('A const local variable needs an initializer.', n);
    const code = `${n.shared ? (this.currentFunction===this.kernel?'s':'s_'+(this.currentFunction.pointerOrigin||this.currentFunction.name)) : 'v'}_${n.name}`;
    const symbol = {name: n.name, type, code, constant: n.constant, atomic, kind: n.shared ? 'shared' : 'local',...(n.shared?{sharedOwner,volatileShared:!!n.volatileShared||volatileAliased}:{})};
    const existing=n.shared&&this.shared.find(x=>x.code===code);
    if(existing){if(existing.sharedOwner!==sharedOwner||typeName(existing.type)!==typeName(type)||existing.atomic!==atomic)this.fail('Shared declaration conflicts across helper specializations.',n);this.add(n.name,existing,n);n.symbol=existing;n.resolvedDimensions=dims;n.resolvedType=type;return [];}
    this.add(n.name, symbol, n); n.symbol = symbol; n.resolvedDimensions = dims; n.resolvedType = type;
    if (n.shared) {
      if (this.shared.some(x => x.code === code)) this.fail('Shared array names must be unique.', n);
      this.shared.push(symbol); return [];
    }
    if(init&&this.structs.has(type)&&init.type===type){
      // Explicit aggregate construction preserves value-copy semantics when
      // backend lowering materializes dynamically indexed array fields.
      const snapshot='cw_value_copy_'+this.temp++;
      return [...init.pre,`let ${snapshot} = ${init.code};`,`${n.constant?'let':'var'} ${code}: ${type} = ${this.aggregateCopy(type,snapshot)};`];
    }
    if(n.workgroupUniformSnapshot){
      const shared='cw_uniform_record_'+this.temp++,snapshot=shared+'_value';
      this.shared.push({code:shared,type,atomic:false});
      return [...init.pre,`let ${snapshot} = ${this.convert(init.code,init.type,type,n)};`,`if (all(cw_thread == vec3<u32>(0u))) { ${shared} = ${snapshot}; }`,`${n.constant?'let':'var'} ${code}: ${typeName(type)} = workgroupUniformLoad(&${shared});`];
    }
    return [...(init?.pre || []), `${n.constant ? 'let' : 'var'} ${code}: ${typeName(type)}${init ? ` = ${isArray(type)?init.code:this.convert(init.code, init.type, type, n)}` : ''};`];
  }
  body(n) {
    this.scopes.push(new Map());
    const lines = n.kind === 'block' ? n.body.flatMap(s => this.statement(s)) : this.statement(n);
    this.scopes.pop(); return lines;
  }
  statement(n) {
    switch (n.kind) {
      case 'empty': return [];
      case 'thread-block':
        n.symbol=this.add(n.name,{name:n.name,kind:'thread-block',constant:true},n);return [];
      case 'block': return ['{', ...indent(this.body(n)), '}'];
      case 'decl': return this.declare(n);
      case 'decls': return n.declarations.flatMap(d=>this.declare(d));
      case 'expr': return this.effect(n.value);
      case 'if': { const condition = this.expr(n.condition); return [...condition.pre, `if (${this.convert(condition.code, condition.type, 'bool', n)}) {`, ...indent(this.body(n.yes)), ...(n.no ? ['} else {', ...indent(this.body(n.no))] : []), '}']; }
      case 'switch': {
        const selector=this.expr(n.selector),type=selector.type==='bool'?'i32':selector.type;
        if(!['i32','u32'].includes(type))this.fail('Switch selector must be a 32-bit integer or enum.',n);
        const seen=new Set(),labels=n.cases.map(c=>{
          if(c.value===null)return 'default';
          const value=this.expr(c.value),integer=constantValue(c.value);
          if(!['i32','u32'].includes(value.type)||value.pre.length||!Number.isInteger(integer)||integer<(type==='i32'?-2147483648:0)||integer>(type==='i32'?2147483647:4294967295))this.fail('Case labels must be integer constants in the selector range.',c);
          if(seen.has(integer))this.fail('Duplicate switch case.',c);seen.add(integer);c.constant=integer;
          return 'case '+integer+(type==='u32'?'u':'i');
        });
        // WGSL cases do not fall through. Repeat each reachable suffix inside its
        // native switch case, preserving break, return and outer-loop continue.
        const lines=[...selector.pre,`switch (${this.convert(selector.code,selector.type,type,n)}) {`];
        this.switchDepth=(this.switchDepth||0)+1;
        for(let i=0;i<n.cases.length;i++){
          const statements=[];let done=false;
          for(let j=i;j<n.cases.length&&!done;j++)for(const statement of n.cases[j].body){statements.push(statement);if(['break','return','continue'].includes(statement.kind)){done=true;break;}}
          lines.push('  '+labels[i]+': {',...indent(indent(this.body({kind:'block',body:statements}))), '  }');
        }
        this.switchDepth--;
        if(!n.cases.some(c=>c.value===null))lines.push('  default: {}');
        return [...lines,'}'];
      }
      case 'do': {
        const condition=this.expr(n.condition);this.loopDepth++;const inner=this.body(n.body);this.loopDepth--;
        if(!condition.pre.length&&((n.condition.kind==='literal'&&constantValue(n.condition)!==0)||(n.condition.kind==='id'&&n.condition.name==='true')))return ['loop {',...indent(inner),'}'];
        return ['loop {',...indent(inner),'  continuing {',...indent(indent([...condition.pre,`break if !${this.convert(condition.code,condition.type,'bool',n)};`])),'  }','}'];
      }
      case 'for': case 'while': {
        this.scopes.push(new Map()); this.loopDepth++;
        const init = n.kind === 'for' && n.init ? (['decl','decls'].includes(n.init.kind) ? this.statement(n.init) : this.effect(n.init)) : [];
        const condition = n.condition ? this.expr(n.condition) : {code: 'true', type: 'bool', pre: []};
        const inner = this.body(n.body), step = n.kind === 'for' && n.step ? this.effect(n.step) : [];
        this.loopDepth--; this.scopes.pop();
        const copies=unrollCount(n);
        if(copies!==null)return ['{',...indent(init),...Array.from({length:copies},()=>['  {',...indent(indent(inner)),'  }',...indent(step)]).flat(),'}'];
        return ['{', ...indent(init),...(n.provenTileVoteLoop?['  _ = subgroupBallot(true);','  @diagnostic(off, subgroup_uniformity)']:[]), '  loop {', ...indent(indent([...condition.pre, `if (!${this.convert(condition.code, condition.type, 'bool', n)}) { break; }`, ...inner, ...(step.length ? ['continuing {', ...indent(step), '}'] : [])])), '  }', '}'];
      }
      case 'return': {
        if (n.value) { const value = this.expr(n.value); if (this.currentFunction.result === 'void') this.fail('Void functions cannot return a value.', n); return [...value.pre, `return ${this.convert(value.code, value.type, this.currentFunction.result, n)};`]; }
        if (this.currentFunction.result !== 'void') this.fail('Non-void functions must return a value.', n); return ['return;'];
      }
      case 'break': case 'continue': if (!this.loopDepth&&!(n.kind==='break'&&this.switchDepth)) this.fail(`${n.kind} is only valid in a loop or an applicable switch.`, n); return [`${n.kind};`];
      default: this.fail(`Unsupported statement ${n.kind}.`, n);
    }
  }
  checkRecursion() {
    const edges = new Map(this.ast.functions.map(f => { const e = []; walk(f.body, n => { if (n.kind === 'call' && n.callee.kind === 'id' && this.functions.has(n.callee.name)) e.push(n.callee.name); }); return [f.name, e]; }));
    const visiting = new Set(), done = new Set();
    const dfs = name => { if (visiting.has(name)) this.fail(`Recursive function call involving '${name}' is unsupported.`, this.functions.get(name)); if (done.has(name)) return; visiting.add(name); edges.get(name).forEach(dfs); visiting.delete(name); done.add(name); };
    for (const name of edges.keys()) dfs(name);
  }
  emit() {
    this.checkRecursion();
    if (this.kernel.result !== 'void') this.fail('__global__ kernels must return void.', this.kernel);
    const textures=[],surfaces=[],bufferCount=this.kernel.params.filter(p=>p.pointer&&!Object.hasOwn(this.bufferAliases,p.name)&&!this.objectImports.some(i=>i.name===p.name)).length+this.deviceParams.length;
    if(this.launchConsumer&&this.dynamicSharedBytes!==this.launchConsumer.sharedMemoryBytes)this.fail('Queue consumer shared bytes must match the original child launch.',this.kernel);
    if(this.launchConsumer&&this.workgroupSize.some((v,i)=>v!==this.launchConsumer.block[i]))this.fail('Queue consumer block size must match the original child launch.',this.kernel);
    const recordParameterLines=[];
    const bindings = [], scalars = this.launchConsumer?[{name:'cw_launch_slot',type:'u32',offset:0}]:[], header = [`// CUDA WebShader ${COMPILER_VERSION}. Generated from kernel ${this.kernel.name}.`];
    for(const {type} of this.ast.bufferReferenceTypes||[])header.push(`alias ${type} = u32;`);
    for(const type of this.ast.objectListTypes||[])header.push(`alias ${type} = u32;`);
    for(const imported of this.objectImports)header.push(`@group(1) @binding(${imported.binding}) var<storage, read_write> cw_import_${imported.id}: array<${imported.type}>;`);
    for(const type of this.ast.devicePointerTypes||[])header.push(`alias ${type} = u32;`);
    for(const type of this.ast.objectPointerTypes||[])header.push(`alias ${type} = u32;`);
    for(const s of this.structs.values()){if(s.fields.some(f=>f.type==='cw_f64'))this.float64Used=true;if(s.fields.some(f=>f.type==='cw_size64'))this.extentUsed=true;}
    for(const s of this.structs.values())header.push(`struct ${s.type} {`,...s.fields.map(f=>`  cw_field_${f.name}: ${typeName(f.resolvedType)},`),'}');
    const sharedAtomicType = t => isArray(t) ? `array<${sharedAtomicType(t.element)}, ${t.length}>` : `atomic<${t==='f32'?'u32':t}>`;
    for (const p of [...this.kernel.params,...this.deviceParams]) {
      if(vectorElement(p.type)==='f16'&&!p.pointer)this.fail('Half kernel parameters passed by value are unsupported; use half storage pointers or local/shared half values.',p);
      const imported=this.objectImports.find(i=>i.name===p.name);if(imported&&p.pointer){const symbol={name:p.name,rootBufferName:p.name,type:arrayOf(p.type),code:'cw_import_'+imported.id,constant:p.constant,atomic:false,kind:'buffer',objectImport:imported,...(shiftedPointers(this.kernel).has(p.name)||this.launchConsumer?.buffers.some(b=>b.name===p.name)?{offsetCode:'cw_pointer_'+p.name}:{})};this.add(p.name,symbol,p,true);p.symbol=symbol;this.bufferSymbols.set(p.name,symbol);continue;}
      if (p.shared || p.reference || p.external || p.type === 'void') this.fail('Invalid kernel parameter type.', p);
      if(!p.pointer&&this.structs.has(p.type)){
        const leaves=recordLeaves(this,p.type,p),codes=new Map();
        for(const leaf of leaves){const name=p.name+'.'+leaf.path.join('.'),queued=this.launchConsumer?.scalars.find(s=>s.name===name);let code;
          if(queued){const word=this.launchConsumer.frontier?`cw_launch_words[${queued.word}u]`:`${this.launchConsumer.variable}.words[cw_params.p_cw_launch_slot*${this.launchConsumer.stride}u+${queued.word}u]`;code=leaf.type==='u32'?word:leaf.type==='bool'?`(${word} != 0u)`:`bitcast<${leaf.type}>(${word})`;}
          else {const field='cw_record_'+scalars.length;scalars.push({name,type:leaf.type==='bool'?'u32':leaf.type,...(leaf.type==='bool'?{sourceType:'bool'}:{}),field,offset:scalars.length*4});code='cw_params.'+field;if(leaf.type==='bool')code=`(${code} != 0u)`;}
          codes.set(leaf.path.join('.'),code);
        }
        const code='v_'+p.name,symbol={name:p.name,type:p.type,code,constant:p.constant,atomic:false,kind:'local',recordFields:leaves};this.add(p.name,symbol,p,true);p.symbol=symbol;
        recordParameterLines.push(`var ${code}: ${p.type} = ${recordConstructor(this,p.type,path=>codes.get(path.join('.')))};`);continue;
      }
      if(p.type==='cw_f64'){
        if(p.pointer||p.reference||p.defaultValue!==undefined||this.launchConsumer)this.fail('Double parameters support explicit host-bound values only.',p);
        this.float64Used=true;
        for(const word of ['lo','hi'])scalars.push({name:p.name+'.'+word,type:'u32',sourceType:'double-word',field:'cw_double_'+p.name+'_'+word,offset:scalars.length*4});
        const symbol={name:p.name,type:p.type,code:`vec2<u32>(cw_params.cw_double_${p.name}_lo, cw_params.cw_double_${p.name}_hi)`,constant:true,atomic:false,kind:'uniform'};
        this.add(p.name,symbol,p,true);p.symbol=symbol;continue;
      }
      if(p.type==='cw_size64'){
        if(p.pointer)this.fail('size_t supports value parameters, not storage buffers.',p);
        this.extentUsed=true;scalars.push({name:p.name,type:'u32',sourceType:'size_t',offset:scalars.length*4});
        const symbol={name:p.name,type:p.type,code:`vec2<u32>(cw_params.p_${p.name},0u)`,constant:true,atomic:false,kind:'uniform'};this.add(p.name,symbol,p,true);p.symbol=symbol;continue;
      }
      if(p.type==='cw_extent'){
        if(p.pointer)this.fail('cudaExtent supports read-only by-value kernel parameters only.',p);
        this.extentUsed=true;const fields=['width','height','depth'];for(const field of fields)scalars.push({name:p.name+'.'+field,type:'u32',sourceType:'cudaExtent-component',field:'cw_extent_'+p.name+'_'+field,offset:scalars.length*4});
        const symbol={name:p.name,type:p.type,code:`cw_extent(${fields.map(field=>`vec2<u32>(cw_params.cw_extent_${p.name}_${field}, 0u)`).join(', ')})`,constant:true,atomic:false,kind:'uniform'};
        this.add(p.name,symbol,p,true);p.symbol=symbol;continue;
      }
      if(!p.pointer&&vectorLength(p.type)){
        const fields='xyzw'.slice(0,vectorLength(p.type)).split(''),element=vectorElement(p.type);
        for(const field of fields)scalars.push({name:p.name+'.'+field,type:element,sourceType:'vector-component',field:'cw_vector_'+p.name+'_'+field,offset:scalars.length*4});
        const symbol={name:p.name,type:p.type,code:`${p.type}(${fields.map(field=>`cw_params.cw_vector_${p.name}_${field}`).join(', ')})`,components:fields.map(field=>p.name+'.'+field),constant:true,atomic:false,kind:'uniform'};this.add(p.name,symbol,p,true);p.symbol=symbol;continue;
      }
      if(p.type==='surface2d'){if(p.pointer)this.fail('Surface objects must be passed by value.',p);const dimensions=new Set();walk(this.kernel.body,n=>{if(n.kind==='call'&&['surf1Dwrite','surf2Dwrite','surf3Dwrite','surf2DLayeredwrite'].includes(n.callee?.name)&&n.args[1]?.kind==='id'&&n.args[1].name===p.name)dimensions.add(n.callee.name==='surf2DLayeredwrite'?'2d-array':n.callee.name==='surf1Dwrite'?'1d':n.callee.name==='surf3Dwrite'?'3d':'2d');});if(dimensions.size>1)this.fail('A surface cannot mix 2D and 3D writes.',p);const dimension=dimensions.has('2d-array')?'2d-array':dimensions.has('3d')?'3d':'2d',binding=bufferCount+this.kernel.params.filter(p=>p.type==='texture3d').length*2+surfaces.length,symbol={name:p.name,type:p.type,code:'cw_surface_'+p.name,kind:'surface',dimension,constant:true};this.add(p.name,symbol,p,true);p.symbol=symbol;const metadata={name:p.name,binding,dimension,format:'r32float',access:'write-only',coordinates:dimension==='3d'?'global-xyz':'global-xy'};symbol.metadata=metadata;surfaces.push(metadata);continue;}
      if(p.type==='texture3d'){if(p.pointer)this.fail('Texture objects must be passed by value.',p);const binding=bufferCount+textures.length*2,samplerBinding=binding+1,sampling=p.textureSampling||'tex3D',{dimension,format}=textureShape(sampling);const symbol={name:p.name,type:p.type,code:'t_'+p.name,sampler:'s_'+p.name,coordinateScale:dimension==='3d'?`vec3<f32>(cw_params.cw_tex_${p.name}_sx, cw_params.cw_tex_${p.name}_sy, cw_params.cw_tex_${p.name}_sz)`:`vec2<f32>(cw_params.cw_tex_${p.name}_sx, cw_params.cw_tex_${p.name}_sy)`,pixelPoint:`cw_params.cw_tex_${p.name}_point`,...(sampling.startsWith('fetch_')?{linearFetch:sampling.slice(6),lengthCode:`cw_params.cw_tex_${p.name}_length`}:{}),dimension,format,kind:'texture',constant:true};this.add(p.name,symbol,p,true);p.symbol=symbol;textures.push({name:p.name,binding,samplerBinding,dimension,format,...(sampling==='texCubemap'?{coordinates:'cube-direction'}:{}),...(['tex2Duchar','tex2Duchar2','tex2Duint'].includes(sampling)?{coordinates:sampling==='tex2Duint'?'pixel-uint':'pixel-byte'}:['tex2Dfloat2','tex2Dfloat4'].includes(sampling)?{coordinates:'2d'}:sampling.startsWith('fetch_')?{coordinates:'linear'}:{})});header.push(`@group(0) @binding(${binding}) var ${symbol.code}: texture_${dimension.replace('-','_')}<${['r32uint','r8uint','rg8uint'].includes(format)?'u32':'f32'}>;`,`@group(0) @binding(${samplerBinding}) var ${symbol.sampler}: sampler;`);continue;}
      if (p.pointer) {
        if(['cw_short','cw_ushort'].includes(p.type))this.fail('Short storage pointers need a packed 16-bit ABI; only short values are supported.',p);
        if(String(p.type).startsWith('cw_objectptr_')&&!this.persistentObjects)this.fail('Object pointer buffers require objectHeap: persistent.',p);
        if(this.structs.has(p.type)&&p.origin!=='constant-struct-storage'){
          if(containsNativeBool(this,p.type)){
            if(!p.constant)this.fail('Native bool fields are not host-shareable; packed record buffers require const pointers.',p);
            p.nativeLayout=nativeRecordLayout(this,p.type,p);
            if(p.nativeLayout.size%4)this.fail('Native bool record strides must be a multiple of four bytes.',p);
            p.storageStride=p.nativeLayout.size;
          }else{const layout=this.storageLayout(p.type);p.storageStride=Math.ceil(layout.size/layout.align)*layout.align;}
        }
        if(p.type==='cw_uchar2')this.fail('Use byte storage with uchar2 pointer views; direct uchar2 buffer parameters are unsupported.',p);
        if (vectorLength(p.type)===3) this.fail('Three-component vector pointers have incompatible CUDA/WGSL layouts. Use 32-bit scalars or two/four-component vectors.', p);
        const canonical=Object.hasOwn(this.bufferAliases,p.name)?this.bufferAliases[p.name]:p.name,atomic=this.usage.atomic.has(canonical),readOnly=!this.usage.writes.has(canonical)&&!atomic;
        if (p.constant && this.usage.writes.has(p.name)) this.fail(`Cannot write through const buffer '${p.name}'.`, p);
        if (atomic && !['i32', 'u32','cw_uchar4','cw_uchar','bool'].includes(p.type)) this.fail('Only 32-bit integer atomics are supported.', p);
        const binding = bindings.length;
        if(canonical===p.name)bindings.push({name: p.name, elementType: p.type, stride: p.storageStride||(['cw_uchar','bool'].includes(p.type)?1:typeStride(p.type)), binding, readOnly, atomic,...(p.type==='bool'?{storageType:'u32',packedBoolean:true,volatile:!!p.volatileParameter}:{}),...(p.nativeLayout?{storageType:'u32',nativeLayout:p.nativeLayout}:{}),...(p.origin?{origin:p.origin,count:p.count,minBindingSize:p.count*(p.storageStride||typeStride(p.type)),...(p.fields?{fields:p.fields,storageType:'u32'}:{})}:{})});
        const symbol = {name: p.name, rootBufferName:canonical, type: arrayOf(p.type), code: `b_${canonical}`, constant: p.constant, atomic, kind: 'buffer',...(p.nativeLayout?{nativeLayout:p.nativeLayout}:{}),...(p.origin?{deviceGlobal:true}:{}),...(!p.origin&&(shiftedPointers(this.kernel).has(p.name)||this.launchConsumer?.buffers.some(b=>b.name===p.name))?{offsetCode:'cw_pointer_'+p.name}:{})};
        if(p.origin)this.globalSymbols.set(p.name,symbol);else this.add(p.name, symbol, p, true); p.symbol = symbol;if(p.origin)(p.origin==='constant-struct-storage'?this.ast.constantGlobals:this.ast.deviceGlobals).find(g=>g.name===p.name).symbol=symbol;this.bufferSymbols.set(p.name,symbol);
        if(canonical===p.name)header.push(`@group(0) @binding(${binding}) var<storage, ${readOnly ? 'read' : 'read_write'}> b_${p.name}: array<${atomic ? `atomic<${p.type==='bool'?'u32':p.type}>` : ['cw_uchar','bool'].includes(p.type)||p.nativeLayout?'u32':p.type}>;`);
      } else {
        if ((!numeric(p.type)&&!['bool','cw_uchar4'].includes(p.type))||p.type==='cw_uchar') this.fail('Scalar kernel parameters must be float, int, unsigned int, bool or packed uchar4. Put other vectors in buffers.', p);
        let defaultMetadata={};if(p.defaultValue!==undefined){let value=p.defaultValue.kind==='id'?Number(p.defaultValue.name==='true'):constantValue(p.defaultValue);if(!Number.isFinite(value)||p.type==='i32'&&(Math.trunc(value)<-2147483648||Math.trunc(value)>2147483647)||p.type==='u32'&&(Math.trunc(value)<0||Math.trunc(value)>4294967295))this.fail('Kernel default is outside its supported scalar range.',p);value=p.type==='f32'?Math.fround(value):p.type==='bool'?Number(!!value):p.type==='u32'?value>>>0:value|0;if(!Number.isFinite(value))this.fail('Kernel default overflows its scalar type.',p);defaultMetadata={defaultValue:value};}
        const queuedScalar=this.launchConsumer?.scalars.find(s=>s.name===p.name);
        if(queuedScalar){
          const word=this.launchConsumer.frontier?`cw_launch_words[${queuedScalar.word}u]`:`${this.launchConsumer.variable}.words[cw_params.p_cw_launch_slot*${this.launchConsumer.stride}u+${queuedScalar.word}u]`;
          const symbol={name:p.name,type:p.type,code:p.type==='u32'?word:`bitcast<${p.type}>(${word})`,constant:false,atomic:false,kind:'uniform'};
          this.add(p.name,symbol,p,true);p.symbol=symbol;continue;
        }
        scalars.push({...defaultMetadata,name:p.name,type:p.type==='cw_short'?'i32':['bool','cw_uchar4','cw_ushort'].includes(p.type)?'u32':p.type,...(['bool','cw_uchar4','cw_short','cw_ushort'].includes(p.type)?{sourceType:p.type}:{}),offset:scalars.length*4});
        const symbol = {name: p.name, type: p.type, code: p.type==='bool'?`(cw_params.p_${p.name} != 0u)`:`cw_params.p_${p.name}`, constant: false, atomic: false, kind: 'uniform'};
        this.add(p.name, symbol, p, true); p.symbol = symbol;
      }
    }
    for(const heap of this.objectHeaps.values()){
      if(this.persistentObjects)header.push(`struct CWHeap_${heap.name} { alive: array<atomic<u32>, ${heap.capacity}>, objects: array<${heap.type}, ${heap.capacity}>, }`,`@group(1) @binding(${heap.binding}) var<storage, read_write> ${heap.variable}: CWHeap_${heap.name};`);
      else header.push(`var<private> ${heap.code}: array<${heap.type}, ${heap.capacity}>;`,`var<private> ${heap.aliveCode}: array<bool, ${heap.capacity}>;`);
    }
    header.push(...deviceHeapDeclarations(this.deviceHeaps),...launchQueueDeclarations(this.launchQueues));
    const moduleShared=new Map(),usedGlobalNames=new Set();for(const fn of [this.kernel,...this.helpers])walk(fn.body,n=>{if(n.kind==='id')usedGlobalNames.add(n.name);});
    for(const declaration of this.ast.sharedGlobals||[])if(usedGlobalNames.has(declaration.name)){this.declare(declaration);moduleShared.set(declaration.name,declaration.symbol);}
    const helperLines = [];
    // Helpers cannot capture kernel arguments; explicit scalar arguments only.
    const kernelScope = this.scopes;
    let emittedHelpers=0;
    const emitHelpers=()=>{while(emittedHelpers<this.helpers.length){const helper=this.helpers[emittedHelpers++];
      this.scopes = [new Map(moduleShared)]; this.currentFunction = helper;
      for (const p of helper.params) {
        if (p.shared || p.external || p.type === 'void'||p.type==='surface2d'||p.type==='cw_extent') this.fail('Invalid helper parameter.', p);
        if(p.type==='texture3d'){if(p.pointer||p.reference)this.fail('Texture helper parameters must be passed by value.',p);const {dimension,format}=textureShape(p.textureSampling);p.symbol=this.add(p.name,{name:p.name,type:p.type,code:'cw_texture_'+p.name,sampler:'cw_sampler_'+p.name,coordinateScale:'cw_scale_'+p.name,pixelPoint:'cw_point_'+p.name,...(p.textureSampling.startsWith('fetch_')?{linearFetch:p.textureSampling.slice(6),lengthCode:'cw_length_'+p.name}:{}),dimension,format,kind:'texture',constant:true},p);continue;}
        if(p.pointer){const base=p.boundShared?this.shared.find(s=>s.code===p.boundShared):this.bufferSymbols.get(p.boundBuffer);if(!base)this.fail('Helper buffer pointer was not specialized.',p);p.symbol=this.add(p.name,{...base,...(p.boundShared?{sharedPointer:p.boundShared}:{}),name:p.name,kind:'buffer-alias',constant:p.constant||p.boundConstant,offsetCode:'cw_buffer_offset_'+helper.params.indexOf(p)},p);continue;}
        if(p.type==='thread-block'){if(p.reference)this.fail('thread_block helper parameters must be passed by value.',p);p.symbol=this.add(p.name,{name:p.name,type:p.type,kind:'thread-block',constant:true},p);continue;}
        if(p.reference&&!p.localArray&&!numeric(p.type)&&!vectorLength(p.type)&&!this.structs.has(p.type)&&!(p.constant&&(['cw_uchar2','cw_uchar4'].includes(p.type)||vectorLength(p.type)||this.structs.has(p.type))))this.fail('Helper references require numeric scalars or vectors.',p);
        if(p.boundReferenceStorage){const index='v_'+p.name,code=p.boundReferenceStorage+'['+index+']'+(p.boundReferencePath||'');p.symbol=this.add(p.name,{name:p.name,type:p.type,code,pointerCode:'&'+code,storageReferenceRoot:p.boundReferenceStorage,storageReferencePath:p.boundReferencePath||'',storageReferenceIndexCode:index,referenceSpace:'storage',constant:p.constant,atomic:false,kind:'reference'},p);continue;}
        if(p.boundReferenceShared){const index='v_'+p.name,code=p.boundReferenceShared+'['+index+']';p.symbol=this.add(p.name,{name:p.name,type:p.type,code,pointerCode:'&'+code,sharedReferenceRoot:p.boundReferenceShared,sharedReferenceIndexCode:index,referenceSpace:'workgroup',constant:p.constant,atomic:false,kind:'reference'},p);continue;}
        p.symbol = this.add(p.name, {name: p.name, type: p.type, code: p.reference?`(*v_${p.name})`:`v_${p.name}`,pointerCode:p.reference?`v_${p.name}`:undefined,...(p.reference?{referenceSpace:p.referenceSpace||'function'}:{}), constant:p.constant, atomic: false, kind: p.reference?'reference':'local'}, p);
      }
      const body = this.body(helper.body);
      helperLines.push(`fn f_${helper.name}(${[...helper.params.filter(p=>p.type!=='thread-block').map(p => p.type==='texture3d'?`cw_texture_${p.name}: texture_${textureShape(p.textureSampling).dimension.replace('-','_')}<${['fetch_uint','tex2Duchar','tex2Duchar2','tex2Duint'].includes(p.textureSampling)?'u32':'f32'}>, cw_sampler_${p.name}: sampler${p.textureSampling.startsWith('fetch_')?`, cw_length_${p.name}: u32`:''}${['tex2D','tex2Dfloat2','tex2Dfloat4','tex3D','tex3Dfloat4'].includes(p.textureSampling)?`, cw_scale_${p.name}: vec${p.textureSampling.startsWith('tex3D')?3:2}<f32>, cw_point_${p.name}: f32`:''}`:p.pointer?`cw_buffer_arg_${helper.params.indexOf(p)}: i32`:`${p.reference||p.constant?'v_':'cw_arg_'}${p.name}: ${p.boundReferenceShared||p.boundReferenceStorage?'i32':p.reference?`ptr<${p.referenceSpace||'function'}, ${typeName(p.type)}>`:p.type}`),'cw_thread: vec3<u32>','cw_block: vec3<u32>','cw_grid: vec3<u32>'].join(', ')})${helper.result === 'void' ? '' : ` -> ${helper.result}`} {`,...indent(helper.params.filter(p=>p.pointer).map(p=>`var cw_buffer_offset_${helper.params.indexOf(p)}: i32 = cw_buffer_arg_${helper.params.indexOf(p)};`)),...indent(helper.params.filter(p=>p.type!=='thread-block'&&p.type!=='texture3d'&&!p.pointer&&!p.reference&&!p.constant).map(p=>`var v_${p.name}: ${p.type} = cw_arg_${p.name};`)), ...indent(body), '}');
    }};
    emitHelpers();
    this.scopes = kernelScope; this.currentFunction = this.kernel;
    const launchPrefix=[];
    if(this.launchConsumer?.frontier){const q=this.launchConsumer;this.shared.push({code:'cw_launch_shared',type:arrayOf('u32',q.stride),atomic:false});launchPrefix.push(`if(all(cw_thread == vec3<u32>(0u))) {`,...Array.from({length:q.stride},(_,i)=>`cw_launch_shared[${i}u] = ${q.frontier.variable}.words[cw_params.p_cw_launch_slot*${q.stride}u+${i}u];`),'}','let cw_launch_words = workgroupUniformLoad(&cw_launch_shared);');}
    const main = [...launchPrefix,...recordParameterLines,...this.kernel.params.filter(p=>p.pointer&&p.symbol.offsetCode).map(p=>`var ${p.symbol.offsetCode}: i32 = ${this.launchConsumer?.buffers.some(b=>b.name===p.name)?'bitcast<i32>('+(this.launchConsumer.frontier?'cw_launch_words['+this.launchConsumer.buffers.find(b=>b.name===p.name).offsetWord+'u]':this.launchConsumer.variable+'.words[cw_params.p_cw_launch_slot*'+this.launchConsumer.stride+'u+'+this.launchConsumer.buffers.find(b=>b.name===p.name).offsetWord+'u]')+')':'0i'};`),...this.body(this.kernel.body)];
    emitHelpers();this.scopes=kernelScope;this.currentFunction=this.kernel;
    for(const surface of surfaces)header.push(`@group(0) @binding(${surface.binding}) var cw_surface_${surface.name}: texture_storage_${surface.dimension.replace('-','_')}<${surface.format}, write>;`);
    for(const scalar of this.constantScalars)scalars.push({...scalar,offset:scalars.length*4});
    let uniformBytes=scalars.length*4;const textureScales=textures.filter(t=>t.format==='r32float'||t.coordinates==='2d'||t.dimension==='3d').map(t=>{const offset=uniformBytes;uniformBytes+=t.dimension==='3d'?16:12;return {name:t.name,offset,pointOffset:offset+(t.dimension==='3d'?12:8),...(t.dimension==='3d'?{dimension:'3d'}:{})};}),textureLengths=textures.filter(t=>t.coordinates==='linear').map(t=>{const offset=uniformBytes;uniformBytes+=4;return {name:t.name,offset};}),uniformSize=uniformBytes?Math.ceil(uniformBytes/16)*16:0;
    if(uniformSize){
      header.push('struct CWParams {',...scalars.map(s=>`  ${s.field||'p_'+s.name}: ${s.type},`));
      for(const scale of textureScales)header.push(`  cw_tex_${scale.name}_sx: f32,`,`  cw_tex_${scale.name}_sy: f32,`,...(scale.dimension==='3d'?[`  cw_tex_${scale.name}_sz: f32,`]:[]),`  cw_tex_${scale.name}_point: f32,`);
      for(const length of textureLengths)header.push(`  cw_tex_${length.name}_length: u32,`);
      for(let i=uniformBytes;i<uniformSize;i+=4)header.push(`  cw_pad_${i}: u32,`);
      header.push('}',`@group(0) @binding(${bindings.length+textures.length*2+surfaces.length}) var<uniform> cw_params: CWParams;`);
    }
    header.push(`const cw_block_size: vec3<u32> = vec3<u32>(${this.workgroupSize.map(x => `${x}u`).join(', ')});`);
    if(this.dynamicSharedBytes&&!this.dynamicSharedUsed)this.fail('sharedMemoryBytes was supplied but the kernel has no dynamic shared array.',this.kernel);
    // Runtime parameters preserve CUDA wraparound even when call arguments are literals;
    // WGSL rejects overflowing constant expressions in an inline multiply.
    if(this.integerIntrinsics.has('__mul24'))helperLines.unshift('fn cw_mul24(a: i32, b: i32) -> i32 { return ((a << 8u) >> 8u) * ((b << 8u) >> 8u); }');
    if(this.integerIntrinsics.has('__umul24'))helperLines.unshift('fn cw_umul24(a: u32, b: u32) -> u32 { return (a & 16777215u) * (b & 16777215u); }');
    // A float32 significand times a 16-bit integer fits exactly in 40 bits.
    // Integer limbs preserve the original double product before truncating to a byte.
    if(this.integerPowerUsed)helperLines.push(`
fn cw_pow_f32(base: f32, exponent: f32) -> f32 {
  if(exponent >= -64.0f && exponent <= 64.0f && exponent == trunc(exponent)) {
    var count=u32(abs(exponent));
    var value=cw_d_from_f32(base);var product=cw_d_from_u32(1u);
    loop {
      if(count == 0u) { break; }
      if((count & 1u) != 0u) { product=cw_d_mul(product,value); }
      count >>= 1u;
      if(count != 0u) { value=cw_d_mul(value,value); }
    }
    if(exponent < 0.0f) { product=cw_d_div(cw_d_from_u32(1u),product); }
    return cw_d_to_f32(product);
  }
  return pow(base,exponent);
}`);
    if(this.float64Used)helperLines.unshift(FLOAT64_WGSL);
    if(this.sizeMultiplyUsed)helperLines.unshift(`fn cw_size_multiply(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let a0 = a.x & 65535u; let a1 = a.x >> 16u;
  let b0 = b.x & 65535u; let b1 = b.x >> 16u;
  let p0 = a0 * b0;
  let p1 = a1 * b0 + (p0 >> 16u);
  let p2 = a0 * b1 + (p1 & 65535u);
  let low = (p2 << 16u) | (p0 & 65535u);
  let high = a1 * b1 + (p1 >> 16u) + (p2 >> 16u) + a.x * b.y + a.y * b.x;
  return vec2<u32>(low, high);
}`);
    if(this.extentUsed){header.unshift('alias cw_size64 = vec2<u32>;','struct cw_extent { width: vec2<u32>, height: vec2<u32>, depth: vec2<u32>, }');helperLines.unshift('fn cw_size_less(a: vec2<u32>, b: vec2<u32>) -> bool { return a.y < b.y || (a.y == b.y && a.x < b.x); }');}
    if(this.exactByteScaleUsed)helperLines.unshift(`fn cw_exact_byte_scale(value: f32, scale: u32) -> u32 {
  let bits = bitcast<u32>(value);
  let exponent = (bits >> 23u) & 255u;
  let mantissa = (bits & 8388607u) | select(0u, 8388608u, exponent != 0u);
  let lowProduct = (mantissa & 65535u) * scale;
  let upper = (mantissa >> 16u) * scale + (lowProduct >> 16u);
  let low = (lowProduct & 65535u) | (upper << 16u);
  let high = upper >> 16u;
  let shift = select(149i, 150i - i32(exponent), exponent != 0u);
  var result = 0u;
  if (exponent == 255u) { return 0u; }
  if (shift >= 64i) { result = 0u; }
  else if (shift >= 32i) { result = high >> u32(shift - 32i); }
  else if (shift > 0i) { result = (low >> u32(shift)) | (high << u32(32i - shift)); }
  else if (shift == 0i) { result = low; }
  else if (shift > -8i) { result = low << u32(-shift); }
  if ((bits & 2147483648u) != 0u) { result = 0u - result; }
  return result & 255u;
}`);
    if(this.packedAtomicUsed)helperLines.unshift('fn cw_store_byte(word: ptr<storage, atomic<u32>, read_write>, shift: u32, value: u32) { var old = atomicLoad(word); loop { let next = (old & ~(255u << shift)) | ((value & 255u) << shift); let result = atomicCompareExchangeWeak(word, old, next); if (result.exchanged) { return; } old = result.old_value; } }');
    for(const kind of this.linearFetchUsed||[]){const scalar=kind==='uint'?'u32':'f32',type=kind==='float2'?'vec2<f32>':kind==='float4'?'vec4<f32>':scalar,swizzle=kind==='float2'?'.rg':kind==='float4'?'':'.r';helperLines.unshift(`fn cw_fetch_${kind}(tex: texture_2d<${scalar}>, index: u32, length: u32) -> ${type} { if (index >= length) { return ${type}(0); } let width = textureDimensions(tex).x; return textureLoad(tex, vec2<i32>(i32(index % width), i32(index / width)), 0)${swizzle}; }`);}
    if(this.float3DSamplingUsed)helperLines.unshift('fn cw_sample_float3d(tex: texture_3d<f32>, texSampler: sampler, coords: vec3<f32>, scale: vec3<f32>, pixelPoint: f32) -> f32 { if (pixelPoint > 0.0f) { let maximum = vec3<f32>(textureDimensions(tex)) - vec3<f32>(1.0f); let pixel = vec3<i32>(clamp(floor(coords), vec3<f32>(0.0f), maximum)); return textureLoad(tex, pixel, 0).r; } return textureSampleLevel(tex, texSampler, coords * scale, 0.0f).r; }');
    if(this.float4_3DSamplingUsed)helperLines.unshift('fn cw_sample_float4_3d(tex: texture_3d<f32>, texSampler: sampler, coords: vec3<f32>, scale: vec3<f32>, pixelPoint: f32) -> vec4<f32> { if (pixelPoint > 0.0f) { let maximum = vec3<f32>(textureDimensions(tex)) - vec3<f32>(1.0f); let pixel = vec3<i32>(clamp(floor(coords), vec3<f32>(0.0f), maximum)); return textureLoad(tex, pixel, 0); } return textureSampleLevel(tex, texSampler, coords * scale, 0.0f); }');
    if(this.float2DSamplingUsed)helperLines.unshift('fn cw_sample_float2d(tex: texture_2d<f32>, texSampler: sampler, coords: vec2<f32>, scale: vec2<f32>, pixelPoint: f32) -> f32 { if (pixelPoint > 0.0f) { let maximum = vec2<f32>(textureDimensions(tex)) - vec2<f32>(1.0f); let pixel = vec2<i32>(clamp(floor(coords), vec2<f32>(0.0f), maximum)); return textureLoad(tex, pixel, 0).r; } return textureSampleLevel(tex, texSampler, coords * scale, 0.0f).r; }');
    // Correct the approximate native WGSL quotient using its fused residual.
    // Nonfinite quotients retain ordinary WGSL behavior; this is not a software IEEE divider.
    if(this.compensatedDivisionUsed)helperLines.unshift('fn cw_divide_f32(a: f32, b: f32) -> f32 { let q = a / b; if ((bitcast<u32>(q) & 0x7f800000u) == 0x7f800000u || (bitcast<u32>(q) & 0x7fffffffu) == 0u || (bitcast<u32>(b) & 0x7f800000u) == 0x7f800000u) { return q; } let residual = fma(-q, b, a); return q + residual / b; }');
    if(this.roundEvenUsed)helperLines.unshift('fn cw_round_even(x: f32) -> f32 { let lo = floor(x); let fraction = x - lo; let odd = lo - 2.0f * floor(lo * 0.5f) != 0.0f; let result = select(lo, lo + 1.0f, fraction > 0.5f || (fraction == 0.5f && odd)); return select(result, bitcast<f32>(bitcast<u32>(x) & 2147483648u), result == 0.0f); }');
    if(this.roundAwayUsed)helperLines.unshift('fn cw_round_away(x: f32) -> f32 { let whole = trunc(x); let fraction = abs(x - whole); return select(whole, whole + select(-1.0f, 1.0f, x >= 0.0f), fraction >= 0.5f); }');
    if(this.byte2DSamplingUsed)helperLines.unshift('fn cw_sample_byte2d(tex: texture_2d<u32>, coords: vec2<f32>) -> u32 { let maximum = vec2<f32>(textureDimensions(tex)) - vec2<f32>(1.0f); let pixel = vec2<i32>(clamp(floor(coords), vec2<f32>(0.0f), maximum)); return textureLoad(tex, pixel, 0).x; }');
    if(this.rgba2DSamplingUsed)helperLines.unshift('fn cw_sample_rgba2d(tex: texture_2d<f32>, texSampler: sampler, coords: vec2<f32>, scale: vec2<f32>, pixelPoint: f32) -> vec4<f32> { if (pixelPoint > 0.0f) { let maximum = vec2<f32>(textureDimensions(tex)) - vec2<f32>(1.0f); let pixel = vec2<i32>(clamp(floor(coords), vec2<f32>(0.0f), maximum)); return textureLoad(tex, pixel, 0); } return textureSampleLevel(tex, texSampler, coords * scale, 0.0f); }');
    for (const s of this.shared) header.push(`var<workgroup> ${s.code}: ${s.atomic ? sharedAtomicType(s.type) : typeName(s.type)};`);
    const storageSize = this.shared.reduce((n, s) => n + Math.ceil(typeStride(s.type) / 16) * 16, 0);
    let usesPackedBytes=(this.ast.structs||[]).some(s=>s.fields.some(f=>['cw_uchar','cw_uchar2','cw_uchar4'].includes(f.type)));walk(this.ast,n=>{if(['cw_uchar','cw_uchar2','cw_uchar4'].includes(n.type)||['cw_uchar','cw_uchar2','cw_uchar4'].includes(n.result))usesPackedBytes=true;});let usesShort=false;walk(this.ast,n=>{if(['cw_short','cw_ushort'].includes(n.type)||['cw_short','cw_ushort'].includes(n.result))usesShort=true;});if(usesShort)header.unshift('alias cw_short = i32;','alias cw_ushort = u32;');if(usesPackedBytes)header.unshift('alias cw_uchar = u32;','alias cw_uchar2 = u32;','alias cw_uchar4 = u32;');
    if(this.kernel.nativeTiles&&(this.workgroupSize[0]%32||this.workgroupSize[1]!==1||this.workgroupSize[2]!==1))this.fail('Static tiles require a one-dimensional workgroup containing complete 32-thread tiles.',this.kernel);
    let usesHalf=false;walk(this.ast,n=>{if(vectorElement(n.type)==='f16'||vectorElement(n.result)==='f16'||vectorElement(n.target)==='f16')usesHalf=true;});
    const wgsl = [...(usesHalf?['enable f16;']:[]),...(this.kernel.nativeTiles?['enable subgroups;','enable subgroup_size_control;','requires subgroup_id, subgroup_uniformity;']:[]),...header, '', ...helperLines, '', `${this.kernel.nativeTiles?'@subgroup_size(32) ':''}@compute @workgroup_size(${this.workgroupSize.join(', ')})`, 'fn main(', ...(this.kernel.nativeTiles?['  @builtin(subgroup_id) cw_subgroup: u32,','  @builtin(subgroup_invocation_id) cw_lane: u32,']:['  @builtin(local_invocation_id) cw_thread: vec3<u32>,']), '  @builtin(workgroup_id) cw_block: vec3<u32>,', '  @builtin(num_workgroups) cw_grid: vec3<u32>', ') {', ...(this.kernel.nativeTiles?['  let cw_thread=vec3<u32>(cw_subgroup*32u+cw_lane,0u,0u);']:[]),...indent(main), '}', ''].join('\n');
    return {version: COMPILER_VERSION, name: this.kernel.name, entryPoint: 'main', wgsl, metadata: {...(usesHalf||this.kernel.nativeTiles?{requiredFeatures:[...(usesHalf?['shader-f16']:[]),...(this.kernel.nativeTiles?['subgroups','subgroup-size-control']:[])]}:{}),...(this.kernel.nativeTiles?{nativeTiles:{size:32,logicalThreadRemapping:true},requiredWgslFeatures:['subgroup_id','subgroup_uniformity']}:{}),...(this.launchQueues.length?{deviceLaunchQueue:{producerOnly:true,queues:this.launchQueues.map(({variable,...q})=>q)}}:{}),...(this.objectHeaps.size||this.deviceHeaps.size||this.launchQueues.length||this.objectImports.length?{objectHeap:{scope:this.persistentObjects?'arena':'invocation',persistent:this.persistentObjects,imports:this.objectImports,pointerBuffers:this.deviceHeaps.size||this.ast.bufferReferenceTypes?.length?bindings.filter(b=>this.containsDevicePointer(b.elementType)).map(b=>b.name):[],types:[...launchQueueStorageTypes(this.launchQueues),...[...this.deviceHeaps.values()].map(h=>({name:h.name,capacity:h.capacity,binding:h.binding,byteLength:h.byteLength,recordLayout:h.recordLayout})),...[...this.objectHeaps.values()].map(h=>({name:h.name,capacity:h.capacity,tag:h.tag,...(this.persistentObjects?{binding:h.binding,byteLength:h.byteLength,recordLayout:h.recordLayout}:{})}))]}}:{}),workgroupSize: this.workgroupSize, bindings,...(Object.keys(this.bufferAliases).length?{bufferAliases:{...this.bufferAliases}}:{}), scalars, uniformSize, uniformBinding: uniformSize ? bindings.length+textures.length*2+surfaces.length : null,...(textures.length?{textures}:{}),...(textureScales.length?{textureScales}:{}),...(textureLengths.length?{textureLengths}:{}),...(surfaces.length?{surfaces}:{}), workgroupStorageBytes: storageSize,...(this.dynamicSharedUsed?{dynamicSharedMemoryBytes:this.dynamicSharedBytes}:{}), barrier: this.usage.storageBarrier ? 'workgroup-and-storage' : 'workgroup'}, ast: this.ast, kernel: this.kernel};
  }
}
function resolveTraitTypes(fn, ast, parameter, argument) {
  const resolve=(type,node)=>{
    if(type?.kind!=='trait-type')return type;
    const key=type.argument===parameter?argument:type.argument;
    if(!key||!builtinType(key)||key==='void')throw new CompileError('Type-trait arguments must resolve to supported built-in value types.',node.token,ast.source);
    const trait=ast.typeTraits.find(t=>t.name===type.name),specialization=trait?.specializations.find(s=>s.argument===key);
    const member=(specialization?.members||trait?.members)?.find(m=>m.name===type.member);
    if(!member)throw new CompileError(`Unknown type-trait member '${type.name}::${type.member}'.`,node.token,ast.source);
    const name=!specialization&&member.type===trait.parameter?key:member.type,resolved=builtinType(name);
    if(!resolved||resolved==='void')throw new CompileError(`Type-trait member resolves to unsupported value type '${name}'.`,node.token,ast.source);
    return resolved;
  };
  fn.result=resolve(fn.result,fn);
  for(const p of fn.params)p.type=resolve(p.type,p);
  walk(fn.body,n=>{if(n.type)n.type=resolve(n.type,n);if(n.target)n.target=resolve(n.target,n);});
}
function instantiateHelperTemplates(ast, kernel) {
  const parameters=fn=>fn.templateParameters?.length?fn.templateParameters:[fn.templateParameter];
  const canonical=(fn,argument,node)=>{const values=argument.split(',').map(x=>x.trim());if(values.length!==parameters(fn).length||values.some(v=>!v))throw new CompileError('Template argument count must match its parameters.',node.token,ast.source);if(fn.templateKind==='int'){try{return values.map(v=>String(integerExpression(v))).join(',');}catch(e){throw new CompileError(e.message,node.token,ast.source);}}return values.join(',');};
  const definitions=new Map(),specializations=new Map(),instances=new Map(),visiting=new Set(),done=new Set(),clones=[];
  for(const fn of ast.functions){
    if(fn.specializationArgument!==undefined){
      const primary=definitions.get(fn.name);if(primary?.templateParameter)fn.specializationArgument=canonical(primary,fn.specializationArgument,fn);const key=fn.name+'<'+fn.specializationArgument+'>';
      if(!primary?.templateParameter||primary.qualifier!=='__device__')throw new CompileError('Declare a primary device helper template before its specialization.',fn.token,ast.source);
      if(specializations.has(key))throw new CompileError('Duplicate device helper specialization.',fn.token,ast.source);
      if(fn.params.length!==primary.params.length)throw new CompileError('Device helper specialization signature does not match its primary template.',fn.token,ast.source);
      specializations.set(key,fn);continue;
    }
    if(definitions.has(fn.name))throw new CompileError(`Duplicate function '${fn.name}'.`,fn.token,ast.source);
    definitions.set(fn.name,fn);
  }
  const fail=(message,node)=>{throw new CompileError(message,node.token,ast.source);};
  function process(fn){
    if(done.has(fn))return;
    if(visiting.has(fn))fail('Recursive helper calls are unsupported.',fn);
    visiting.add(fn);
    pruneConstexpr(fn.body,(message,n)=>fail(message,n));
    resolveTraitTypes(fn,ast);
    walk(fn.body,node=>{
      if(node.kind!=='call'||node.callee.kind!=='id')return;
      if(['tex3D','tex1D','tex1Dfetch','tex2D','tex2DLayered','texCubemap'].includes(node.callee.name))return;
      const callee=node.callee,definition=definitions.get(callee.name);let argument=callee.templateArgument;if(definition?.templateParameter&&argument!==undefined)argument=canonical(definition,argument,callee);
      if(!definition?.templateParameter){
        if(argument!==undefined)fail('Explicit template arguments require a templated device helper.',callee);
        if(definition?.qualifier==='__device__')process(definition);
        return;
      }
      if(definition.qualifier!=='__device__')fail('Device-side kernel launches are unsupported.',callee);
      if(argument===undefined)return; // Deduced after argument expression types are known to the emitter.
      const key=definition.name+'<'+argument+'>';
      let instance=instances.get(key);
      if(!instance){
        if(instances.size>=128)fail('At most 128 device helper template specializations are supported.',callee);
        const selected=specializations.get(key);
        instance=structuredClone(selected||definition);
        if(selected)for(let i=0;i<instance.params.length;i++)if(definition.params[i].defaultValue!==undefined)instance.params[i].defaultValue=structuredClone(definition.params[i].defaultValue);
        const names=parameters(definition),argumentsList=argument.split(','),replacements=new Map();
        for(let i=0;i<names.length;i++){const value=argumentsList[i],type=definition.templateKind==='type'?builtinType(value):null;if(definition.templateKind==='type'&&(!type||['void','texture3d','surface2d'].includes(type)))fail('Template type argument must be a supported built-in value type.',callee);if(!type&&(!Number.isSafeInteger(Number(value))||Number(value)<-2147483648||Number(value)>2147483647))fail('Template argument must be a signed 32-bit integer.',callee);replacements.set(names[i],{value,type});}
        const replaceType=type=>{if(typeof type==='string'&&type.startsWith('template:'))return replacements.get(type.slice(9))?.type??type;if(type?.kind==='trait-type')return {...type,argument:replacements.get(type.argument)?.value??type.argument};return type;};
        const substitute=fn=>{fn.result=replaceType(fn.result);for(const param of fn.params){if(replacements.has(param.name))fail('Template parameter shadowing is unsupported.',param);param.type=replaceType(param.type);}
          walk(fn.body,n=>{if(['decl','thread-block'].includes(n.kind)&&replacements.has(n.name))fail('Template parameter shadowing is unsupported.',n);if(n.type)n.type=replaceType(n.type);if(n.target)n.target=replaceType(n.target);if(n.kind==='call'&&n.callee.kind==='id'){const replacement=replacements.get(n.callee.name);if(replacement?.type)n.callee.name=replacement.value;}
            if(n.templateArgument!==undefined)n.templateArgument=n.templateArgument.replace(/[A-Za-z_]\w*/g,name=>{const r=replacements.get(name);return r?(r.type?r.value:'('+r.value+')'):name;});
            if(n.kind==='id'){const r=replacements.get(n.name);if(r&&!r.type){n.kind='literal';n.value=r.value;delete n.name;}}
          });resolveTraitTypes(fn,ast);
        };
        substitute(instance);
        if(selected){const expected={...definition,params:structuredClone(definition.params),body:{kind:'block',body:[]}};substitute(expected);
          if(instance.result!==expected.result||instance.params.some((p,i)=>p.type!==expected.params[i].type||p.pointer!==expected.params[i].pointer||p.reference!==expected.params[i].reference||(p.reference&&p.constant!==expected.params[i].constant)))fail('Device helper specialization signature does not match its primary template.',selected);
          walk(instance.body,n=>{if([n.type,n.target].some(t=>typeof t==='string'&&t.startsWith('unsupported:')))fail('Double-precision value types are unsupported in selected helper specializations.',n);});
        }
        let name='cw_specialized_'+instances.size;
        while(definitions.has(name))name+='_';
        instance.name=name;instance.templateParameter=null;instance.templateKind=null;delete instance.specializationArgument;
        instances.set(key,instance);definitions.set(name,instance);clones.push(instance);
      }
      process(instance);
      callee.name=instance.name;delete callee.templateArgument;
    });
    visiting.delete(fn);done.add(fn);
  }
  process(kernel);
  for(const fn of ast.functions)if(fn.qualifier==='__device__'&&!fn.templateParameter&&fn.specializationArgument===undefined)process(fn);
  ast.functions=ast.functions.filter(fn=>fn.qualifier!=='__device__'||(!fn.templateParameter&&fn.specializationArgument===undefined)).concat(clones);
  return {deduce(name,types,callee){
    const definition=definitions.get(name);
    if(!definition?.templateParameter)return null;
    if(definition.qualifier!=='__device__')fail('Device-side kernel launches are unsupported.',callee);
    if(definition.templateKind!=='type')fail('Integer helper templates require an explicit template argument.',callee);
    if(parameters(definition).length>1)fail('Multiple helper template types require explicit template arguments.',callee);
    if(types.length>definition.params.length||definition.params.slice(types.length).some(p=>p.defaultValue===undefined))fail(`Wrong number of arguments for '${name}'.`,callee);
    const matches=definition.params.flatMap((p,i)=>i<types.length&&p.type==='template:'+definition.templateParameter?[p.pointer?(isArray(types[i])?types[i].element:undefined):types[i]]:[]);
    if(!matches.length)fail('Cannot deduce helper template type from these parameters; supply an explicit argument.',callee);
    const type=matches[0];
    if(matches.some(t=>typeName(t)!==typeName(type)))fail('Conflicting deduced helper template argument types.',callee);
    const names=['float','int','uint','bool','uchar','uchar4','short','ushort',...['float','int','uint'].flatMap(p=>[2,3,4].map(n=>p+n))],argument=names.find(n=>builtinType(n)===type);
    if(!argument)fail('Deduced helper template argument must be a supported built-in value type.',callee);
    const call={kind:'call',token:callee.token,callee:{...callee,templateArgument:argument},args:[]};
    process({kind:'function',name:'deduction',token:callee.token,result:'void',params:[],body:{kind:'block',body:[call]}});
    for(const fn of clones)if(!ast.functions.includes(fn))ast.functions.push(fn);
    return definitions.get(call.callee.name);
  }};
}
/** Every entry is still a separate artifact. optimize:false is the diagnostic baseline. */
export function compile(source, options = {}, bufferUsage = null) {
  const mode = options.optimize === undefined || options.optimize === true ? 'dependencies'
    : options.optimize === false ? 'none' : options.optimize;
  if (!['none', 'dependencies', 'specialize'].includes(mode))
    throw new CompileError('optimize must be false, true, none, dependencies, or specialize.');
  // Validate the ORIGINAL program before pruning any source branches, so errors
  // in dead code cannot silently disappear. No driver/pipeline compilation here.
  const baseline = compileCore(source, {...options, optimize: 'none'}, bufferUsage);
  if (mode === 'none') return baseline;
  let result = baseline, specializationReport = null;
  if (mode === 'specialize') {
    try {
      const candidate = compileCore(source, {...options, optimize: 'specialize'}, bufferUsage);
      specializationReport = candidate.metadata.optimization?.specialization;
      const {optimization: ignored, ...candidateInterface} = candidate.metadata;
      // Keep ALL runtime metadata, not just binding numbers. Uniform records,
      // barriers, storage access, resource footprints and layouts must agree.
      if (JSON.stringify(candidateInterface) === JSON.stringify(baseline.metadata)) result = candidate;
      else specializationReport = {...specializationReport, fallback: 'interface-change'};
    } catch (error) {
      // A rejected optimization is not a rejected source program. Baseline has
      // already validated. Report the fallback rather than swallowing it.
      specializationReport = {fallback: 'optimization-rejected', message: String(error.message)};
    }
  }
  const baselineBytes = new TextEncoder().encode(baseline.wgsl).length;
  const trimmed = trimWgslDependencies(result.wgsl, result.entryPoint);
  result.wgsl = trimmed.wgsl;
  result.metadata.optimization = {mode, baselineBytes,
    ...trimmed.report, ...(specializationReport ? {specialization: specializationReport} : {})};
  // Scheduled children remain independent compilation units.
  const trimChildren = artifact => {
    for (const {artifact: child} of artifact.children || []) {
      if (!child.metadata.optimization) {
        const t = trimWgslDependencies(child.wgsl, child.entryPoint);
        child.wgsl = t.wgsl;
        child.metadata.optimization = {mode: 'dependencies', baselineBytes: t.report.beforeBytes, ...t.report};
      }
      trimChildren(child);
    }
  };
  trimChildren(result);
  return result;
}
function compileCore(source, options = {},bufferUsage=null) {
  const ast = parse(source, options);lowerPrintf(ast,options);const kernels = ast.functions.filter(f => f.qualifier === '__global__');
  if(options.valueBuffers!==undefined){
    if(!Array.isArray(options.valueBuffers)||new Set(options.valueBuffers).size!==options.valueBuffers.length)throw new CompileError('valueBuffers requires unique kernel parameter names.');
    for(const name of options.valueBuffers){const params=kernels.flatMap(f=>f.params.filter(p=>p.name===name));if(!params.length)throw new CompileError('Unknown value buffer '+name);for(const p of params){const record=ast.structs.find(r=>'cw_objectptr_'+r.name===p.type);if(p.pointer||!record?.valueClass)throw new CompileError('Value buffers require a concrete class pointer parameter.',p.token,source);p.type=record.type;p.pointer=true;}}
  }

  // Lower same-allocation byte-address round trips while retaining typed offsets.
  const bytePointer=node=>{
    if(node?.kind==='binary'&&['+','-'].includes(node.op)){const p=bytePointer(node.left);if(p)return {...p,index:{...node,left:p.index}};return null;}
    if(node?.kind!=='pointer-cast')return null;
    const offset=node.value,inner=offset?.left;
    if(offset?.kind!=='binary'||offset.op!=='+'||inner?.kind!=='pointer-cast'||inner.target!=='byte-address'||inner.value?.kind!=='id')return null;
    return {base:inner.value,target:node.target,constant:node.constant,index:{kind:'byte-index',token:node.token,baseName:inner.value.name,target:node.target,constant:node.constant,offset:offset.right}};
  };
  walk(ast,n=>{
    if(n.kind==='decl'&&n.pointer){const p=bytePointer(n.init);if(!p)return;if(p.target!==n.type)throw new CompileError('Byte pointer alias must retain its pointee type.',n.token,source);if(p.constant&&!n.constant)throw new CompileError('Cannot discard const through a pointer cast.',n.token,source);n.init={kind:'binary',op:'+',left:p.base,right:p.index,token:n.token};}
    if(n.kind==='unary'&&n.op==='*'){const p=bytePointer(n.value),base=n.value;delete n.value;delete n.op;Object.assign(n,p?{kind:'index',base:p.base,index:p.index,pointerTarget:p.target,pointerConstant:p.constant,dereference:true}:{kind:'index',base,index:{kind:'literal',value:'0',token:n.token},dereference:true});}
  });
  const specialization=options.entry?.match(/^([A-Za-z_]\w*)<\s*([^<>]+)\s*>$/),entry=specialization?specialization[1]:options.entry;if(specialization)specialization[2]=specialization[2].trim();if(specialization&&ast.typeAliases?.[specialization[2]])specialization[2]=['float','int','uint','bool','uchar','uchar4','short','ushort',...['float','int','uint'].flatMap(p=>[2,3,4].map(n=>p+n))].find(n=>builtinType(n)===ast.typeAliases[specialization[2]]);
  const kernel = entry ? kernels.find(k => k.name === entry) : kernels.length === 1 ? kernels[0] : null;
  if (!kernel) throw new CompileError(options.entry ? `Kernel '${options.entry}' was not found.` : 'Multiple kernels found; specify options.entry.');
  if(!!kernel.templateParameter!==!!specialization)throw new CompileError(kernel.templateParameter?'Specify a template entry, for example '+kernel.name+(kernel.templateKind==='type'?'<float>.':'<16>.'):'This kernel does not have a template parameter.',kernel.token,source);
  if(specialization&&specialization[2].split(',').length!==(kernel.templateParameters?.length||1))throw new CompileError('Template argument count must match its parameters.',kernel.token,source);
  if(specialization&&kernel.templateKind==='type'){
    const type=builtinType(specialization[2]),name=kernel.templateParameter,placeholder='template:'+name;
    if(!type||type==='void')throw new CompileError('Template type argument must be a supported built-in value type.',kernel.token,source);
    for(const p of kernel.params){if(p.name===name)throw new CompileError('Template parameter shadowing is unsupported.',p.token,source);if(p.type===placeholder)p.type=type;}
    if(kernel.result===placeholder)kernel.result=type;
    walk(kernel.body,n=>{if(['decl','thread-block'].includes(n.kind)&&n.name===name)throw new CompileError('Template parameter shadowing is unsupported.',n.token,source);if(n.type===placeholder)n.type=type;if(n.target===placeholder)n.target=type;if(n.kind==='call'&&n.callee.kind==='id'&&n.callee.name===name)n.callee.name=specialization[2];});
  }else if(specialization){
    specialization[2]=specialization[2].split(',').map(v=>ast.enumValues?.[v.trim()]?.value??v.trim()).join(',');
    const names=kernel.templateParameters?.length?kernel.templateParameters:[kernel.templateParameter],values=specialization[2].split(',').map(v=>Number(v.trim())),replacements=new Map(names.map((name,i)=>[name,values[i]]));
    if(specialization[2].split(',').some(v=>!/^\d+$/.test(v.trim()))||values.some(value=>!Number.isSafeInteger(value)||value<0||value>2147483647))throw new CompileError('Template arguments must be nonnegative 32-bit signed integers.',kernel.token,source);
    for(const p of kernel.params)if(replacements.has(p.name))throw new CompileError('Template parameter shadowing is unsupported.',p.token,source);
    walk(kernel.body,n=>{if(['decl','thread-block'].includes(n.kind)&&replacements.has(n.name))throw new CompileError('Template parameter shadowing is unsupported.',n.token,source);if(n.kind==='id'&&replacements.has(n.name)){n.kind='literal';n.value=String(replacements.get(n.name));delete n.name;}if(n.templateArgument!==undefined)n.templateArgument=n.templateArgument.replace(/[A-Za-z_]\w*/g,name=>replacements.has(name)?'('+replacements.get(name)+')':name);});
  }
  if(specialization&&kernel.templateKind==='type')walk(kernel.body,n=>{if(n.templateArgument!==undefined)n.templateArgument=n.templateArgument.replace(/[A-Za-z_]\w*/g,name=>name===kernel.templateParameter?specialization[2]:name);});
  pruneConstexpr(kernel.body,(message,n)=>{throw new CompileError(message,n.token,source);});
  resolveTraitTypes(kernel,ast,kernel.templateParameter,specialization?.[2]);
  lowerDeferredPointers(ast,walk,(message,n)=>{throw new CompileError(message,n?.token,source);});
  lowerConstantRows(ast,walk,(message,n)=>{throw new CompileError(message,n?.token,source);});
  markUniformRecordSnapshots(ast,kernel,walk);
  lowerNativeTiles(ast,walk,(message,n)=>{throw new CompileError(message,n?.token,source);});
  const tiledGroups=lowerTiledGroups(ast,options,walk,(message,n)=>{throw new CompileError(message,n?.token,source);});
  const scalarConstraints=uniformBlockGuards(kernel,options,walk,message=>{throw new CompileError(message,kernel.token,source);});
  const returnPhases=lowerReturnPhases(kernel,options,walk,(message,n)=>{throw new CompileError(message,n?.token,source);});
  const overloadGroups=new Map();for(const f of ast.functions)if(f.specializationArgument===undefined){const group=overloadGroups.get(f.name)||[];group.push(f);overloadGroups.set(f.name,group);}let overloadIndex=0;const occupied=new Set(ast.functions.map(f=>f.name));for(const [name,group]of overloadGroups)if(group.length>1){if(group.some(f=>f.qualifier!=='__device__'||f.templateParameter))throw new CompileError('Overloads support non-template device helpers only.',group[0].token,source);const signatures=new Set();for(const f of group){const signature=JSON.stringify(f.params.map(p=>[p.type,p.pointer,p.reference,(p.pointer||p.reference)&&p.constant]));if(signatures.has(signature))throw new CompileError('Duplicate function signature '+name,f.token,source);signatures.add(signature);let unique='cw_overload_'+overloadIndex+++'_'+name;while(occupied.has(unique))unique+='_';occupied.add(unique);f.overloadName=name;f.name=unique;}}
  const templates=instantiateHelperTemplates(ast,kernel);
  const dependencySpecialization=options.optimize==='specialize'?specializeDependencies(ast,kernel):null;
  const emitter=new Emitter(ast,kernel,options,templates,bufferUsage),result=emitter.emit();if(options.libraries?.length)result.metadata.libraries=options.libraries.map(name=>({name,seedBits:64,subsequence:0,offset:0,stateLayout:'compiler-owned',operations:['curand_init','curand','curand_uniform']}));
  if(ast.diagnostics)result.metadata.diagnostics=ast.diagnostics;
  emitter.checkRecursion(); // Class calls have now resolved to concrete helpers.
  if(tiledGroups)result.metadata.tiledGroups='predicated-first-tile';
  if(returnPhases)result.metadata.predicatedReturns=true;
  if(scalarConstraints.length||emitter.pointerConstraints.length)result.metadata.scalarConstraints=[...scalarConstraints,...emitter.pointerConstraints];
  const changed=['reads','writes','atomic'].some(k=>[...emitter.usage[k]].some(name=>!emitter.initialBufferUsage[k].has(name)));
  if(changed){if(bufferUsage)throw new CompileError('Helper buffer access analysis did not converge.');return compileCore(source,options,Object.fromEntries(['reads','writes','atomic'].map(k=>[k,[...emitter.usage[k]]])));}if(specialization)result.metadata.templateArguments={[kernel.templateParameter]:kernel.templateKind==='type'?specialization[2]:Number(specialization[2])};if(options.scheduleDeviceLaunches){
    const queues=result.metadata.deviceLaunchQueue?.queues.filter(q=>q.caller===result.name)||[];
    if(!queues.length)throw new CompileError('Scheduled execution requires a parent with child launches.');
    result.children=queues.map(q=>({queueId:q.id,artifact:compile(source,{...options,scheduleDeviceLaunches:false,deviceLaunchConsumer:q.id,entry:q.childEntry,workgroupSize:q.block,sharedMemoryBytes:q.sharedMemoryBytes})}));
  }
  if(dependencySpecialization)result.metadata.optimization={specialization:dependencySpecialization};
  return result;
}
export function serializableArtifact(compiled) {
  return {version: compiled.version, name: compiled.name, entryPoint: compiled.entryPoint, wgsl: compiled.wgsl, metadata: compiled.metadata,...(compiled.children?{children:compiled.children.map(c=>({queueId:c.queueId,artifact:serializableArtifact(c.artifact)}))}:{})};
}
