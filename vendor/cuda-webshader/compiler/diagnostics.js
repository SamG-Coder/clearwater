import {parse,CompileError} from './parser.js';
// Capture bounded integer printf events; decoding happens explicitly after GPU completion.
export function lowerPrintf(ast,options){
 const calls=[],seen=new WeakSet();const visit=(node,parent)=>{if(!node||typeof node!=='object'||seen.has(node))return;seen.add(node);if(node.kind==='call'&&node.callee?.name==='printf')calls.push({node,parent});for(const [key,v]of Object.entries(node))if(key!=='token'){if(Array.isArray(v))v.forEach(n=>visit(n,node));else if(v&&typeof v==='object')visit(v,node);}};for(const fn of ast.functions)visit(fn.body,null);
 if(!calls.length)return;
 const fail=(message,n)=>{throw new CompileError(message,n.token,ast.source);};
 const capacity=options.diagnosticCapacity??64;if(!Number.isInteger(capacity)||capacity<1||capacity>65536)fail('Diagnostic capacity must be in 1..65536.',calls[0].node);
 const name='cw_printf_storage';let conflict=false;const inspect=n=>{if(!n||typeof n!=='object')return;if(n.name?.startsWith('cw_printf_'))conflict=true;for(const [k,v]of Object.entries(n))if(k!=='token'){if(Array.isArray(v))v.forEach(inspect);else if(v&&typeof v==='object')inspect(v);}};inspect(ast);if(conflict)fail('Names beginning cw_printf_ are reserved for diagnostic capture.',calls[0].node);
 const formats=[],strideWords=9;let helpers=`__device__ unsigned int ${name}[${2+capacity*strideWords}];\n`;
 for(const {node,parent}of calls){
  if(parent?.kind!=='expr'||parent.value!==node)fail('printf supports standalone diagnostic statements only; its return value is unsupported.',node);
  const literal=node.args[0];if(literal?.kind!=='string')fail('printf requires a literal format string.',node);
  const format=literal.value.slice(1,-1).replace(/\\([nrt"\\])/g,(_,c)=>({n:'\n',r:'\r',t:'\t','"':'"','\\':'\\'})[c]),types=[];if(format.length>1024)fail('Diagnostic formats are limited to 1024 characters.',node);
  for(let i=0;i<format.length;i++)if(format[i]==='%'){const c=format[++i];if(c==='%')continue;if(!['u','d'].includes(c))fail('Diagnostic printf currently supports %u, %d and %% only.',node);types.push(c==='u'?'u32':'i32');}
  if(types.length>8||node.args.length!==types.length+1)fail('Diagnostic printf requires a matching list of at most eight integer arguments.',node);
  let id=formats.findIndex(f=>f.format===format);if(id<0){id=formats.length;if(id>=64)fail('At most 64 diagnostic formats are supported.',node);formats.push({format,types});
   const args=types.map((_,i)=>`unsigned int a${i}`).join(',');helpers+=`__device__ void cw_printf_${id}(${args}){unsigned int slot=atomicAdd(&${name}[0],1u);if(slot<${capacity}u){atomicExch(&${name}[2u+slot*${strideWords}u],${id+1}u);${types.map((_,i)=>`atomicExch(&${name}[${3+i}u+slot*${strideWords}u],a${i});`).join('')}}else{atomicAdd(&${name}[1],1u);}}\n`;
  }
  node.callee={kind:'id',name:'cw_printf_'+id,token:node.callee.token};node.args=node.args.slice(1);node.printfIntegerArguments=true;
 }
 const generated=parse(helpers+"__global__ void cw_printf_dummy(){}\n");ast.functions.push(...generated.functions.filter(f=>f.qualifier==='__device__'));ast.deviceGlobals.push(...generated.deviceGlobals);ast.diagnostics={buffer:name,capacity,strideWords,formats};
}
export function decodeDiagnostics(metadata,words){
 const info=metadata.diagnostics;if(!info)throw Error('Kernel has no diagnostic capture metadata.');
 if(!(words instanceof Uint32Array)||words.length<2+info.capacity*info.strideWords)throw Error('Incomplete diagnostic capture buffer.');
 const messages=[];for(let i=0;i<Math.min(words[0],info.capacity);i++){const offset=2+i*info.strideWords,spec=info.formats[words[offset]-1];if(!spec)throw Error('Unknown or incomplete diagnostic record.');let arg=0;messages.push(spec.format.replace(/%[%ud]/g,s=>s==='%%'?'%':String(s==='%d'?words[offset+1+arg++]|0:words[offset+1+arg++])));}
 return {attempted:words[0],dropped:words[1],messages};
}
