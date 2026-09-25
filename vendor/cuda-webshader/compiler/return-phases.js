// CUDA permits exited threads at a block barrier. Keep those lanes alive in
// WGSL, with all subsequent effects guarded, for straight-line barrier phases.
export function lowerReturnPhases(kernel,options,walk,fail){
 if(options.predicatedReturns===undefined||options.predicatedReturns===false)return false;
 if(options.predicatedReturns!==true)fail('predicatedReturns must be boolean.',kernel);
 const sync=n=>n?.kind==='expr'&&n.value?.kind==='call'&&n.value.callee?.name==='__syncthreads'&&n.value.args.length===0;
 const has=(n,p)=>{let found=false;walk(n,x=>{if(p(x))found=true;});return found;};
 const ret=n=>n?.kind==='return'&&!n.value;
 const guardedReturn=n=>n.kind==='if'&&!n.no&&(ret(n.yes)||n.yes?.kind==='block'&&n.yes.body.length===1&&ret(n.yes.body[0]));
 const statements=kernel.body.body;
 if(!statements.some(sync))fail('Predicated returns require a top-level block barrier.',kernel);
 for(const n of statements){
  if(sync(n)||guardedReturn(n)||ret(n))continue;
  if(has(n,x=>x.kind==='return'||x.kind==='call'&&['__syncthreads','__syncthreads_count','__syncthreads_and','__syncthreads_or'].includes(x.callee?.name)))fail('Predicated returns require top-level barriers and direct return guards.',n);
  if(n.kind==='decl'&&(n.pointer||n.reference||!n.shared&&(n.dimensions.length||!['i32','u32','f32','bool'].includes(n.type))))fail('Predicated phase locals must be scalars or shared arrays.',n);
  if(!['decl','expr','if','empty'].includes(n.kind))fail('Unsupported statement in predicated return phases: '+n.kind,n);
 }
 const names=new Set(kernel.params.map(p=>p.name));walk(kernel.body,n=>{if(n.name)names.add(n.name);});let name='cw_return_active';while(names.has(name))name+='_';
 const token=kernel.token,id=(name,token)=>({kind:'id',name,token}),active=()=>id(name,token);
 const block=(body,token)=>({kind:'block',body,token});
 const guard=n=>({kind:'if',condition:active(),yes:block([n],n.token),no:null,token:n.token});
 const assign=(left,right,token)=>({kind:'expr',value:{kind:'assign',op:'=',left,right,token},token});
 const inactive=()=>assign(active(),id('false',token),token);
 const output=[{kind:'decl',name,type:'bool',init:id('true',token),token,constant:false,shared:false,pointer:false,reference:false,external:false,dimensions:[]}];
 for(const n of statements){
  if(sync(n)||n.kind==='decl'&&n.shared){output.push(n);continue;}
  if(ret(n)){output.push(inactive());continue;}
  if(guardedReturn(n)){output.push(guard({...n,yes:block([inactive()],n.token)}));continue;}
  if(n.kind==='decl'){
   const init=n.init;output.push({...n,constant:false,init:null});
   if(init)output.push(guard(assign(id(n.name,n.token),init,n.token)));
  }else output.push(guard(n));
 }
 kernel.body.body=output;return true;
}
