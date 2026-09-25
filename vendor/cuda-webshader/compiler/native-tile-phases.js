// Promote a complete tile's memory phases to workgroup phases. Only uniform,
// literal-bound loops are lifted; inactive threads never evaluate memory reads.
export function lowerNativeTilePhases(fn,handles,walk,fail){
 const sync=n=>n?.kind==='call'&&n.callee?.name==='cooperative_groups::sync'&&n.args.length===1&&handles.has(n.args[0].name);
 const contains=n=>{let yes=false;walk(n,x=>{if(sync(x))yes=true;});return yes;};
 if(!contains(fn.body))return;
 let warpSizeShadowed=fn.params.some(p=>p.name==='warpSize');walk(fn.body,n=>{if(n.kind==='decl'&&n.name==='warpSize')warpSizeShadowed=true;});
 const names=new Set(),warps=new Set();walk(fn.body,n=>{if(n.name)names.add(n.name);});let serial=0;
 const fresh=()=>{let name;do{name='cw_native_phase_'+serial++;}while(names.has(name));names.add(name);return name;};
 const id=(name,token)=>({kind:'id',name,token}),block=(body,token)=>({kind:'block',body,token});
 const declaration=(name,type,init,token)=>({kind:'decl',name,type,init,token,constant:true,shared:false,pointer:false,reference:false,external:false,dimensions:[]});
 const guard=(active,value,type)=>({kind:'conditional',condition:structuredClone(active),yes:value,no:type==='bool'?id('false',value.token):{kind:'literal',value:type==='f32'?'0.0f':type==='u32'?'0u':'0',token:value.token},token:value.token});
 const barrier=token=>({kind:'expr',value:{kind:'call',callee:id('__syncthreads',token),args:[],token},token});
 const root=n=>n?.kind==='id'?n.name:['member','index'].includes(n?.kind)?root(n.base):null;
 const warpIndex=n=>n?.kind==='binary'&&n.op==='/'&&n.left?.kind==='member'&&n.left.base?.name==='threadIdx'&&n.left.member==='x'&&(n.right?.name==='warpSize'&&!warpSizeShadowed||n.right?.kind==='literal'&&Number(n.right.value.replace(/[uU]$/,''))===32);
 function masked(n,active){
  const token=n.token;
  if(n.kind==='block')return block(n.body.map(x=>masked(x,active)),token);
  if(n.kind==='empty')return n;
  if(n.kind==='decls')return {...n,declarations:n.declarations.map(x=>masked(x,active))};
  if(n.kind==='decl'){
   if(n.shared||n.pointer||n.reference||n.dimensions.length||!['i32','u32','f32','bool'].includes(n.type))fail('Predicated native tile locals require scalar values.',n);
   if(n.init)n.init=guard(active,n.init,n.type);return n;
  }
  if(n.kind==='if'){
   // Without a barrier, preserve the original nested scope and lazy condition.
   if(contains(n))fail('Nested tile synchronization requires a complete tile phase.',n);
   return {kind:'if',condition:structuredClone(active),yes:block([n],token),no:null,token};
  }
  if(n.kind==='for'){
   const a=n.init,c=n.condition,s=n.step,name=a?.name;
   if(a?.kind!=='decl'||!['i32','u32'].includes(a.type)||a.init?.kind!=='literal'||c?.kind!=='binary'||c.left?.name!==name||c.right?.kind!=='literal'||!['<','<=','>','>='].includes(c.op)||s?.kind!=='unary'||!['++','--'].includes(s.op)||s.value?.name!==name)fail('Native tile phase loops require uniform literal bounds.',n);
   let changes=false;walk(n.body,x=>{if(x.kind==='assign'&&root(x.left)===name||x.kind==='unary'&&['++','--','&'].includes(x.op)&&root(x.value)===name)changes=true;});
   if(changes)fail('Native tile phase induction variables cannot change or escape.',n);
   n.body=masked(n.body,active);return n;
  }
  if(n.kind==='expr'){
   if(sync(n.value))return barrier(token);
   return {kind:'if',condition:structuredClone(active),yes:block([n],token),no:null,token};
  }
  fail('Unsupported native tile phase control flow: '+n.kind,n);
 }
 fn.body.body=fn.body.body.map(n=>{
  if(n.kind==='decl'&&n.constant&&warpIndex(n.init))warps.add(n.name);
  if(!contains(n))return n;
  if(n.kind==='expr'&&sync(n.value))return barrier(n.token);
  const c=n.condition;
  if(n.kind!=='if'||n.no||c?.kind!=='binary'||!['==','<','<=','>','>='].includes(c.op)||!(warpIndex(c.left)||c.left?.kind==='id'&&warps.has(c.left.name))||c.right?.kind!=='literal')fail('Static tile memory synchronization requires a complete tile branch at workgroup scope.',n);
  walk(n.yes,x=>{if(['return','break','continue'].includes(x.kind))fail('Native tile phases cannot exit or continue early.',x);if(x.kind==='call'&&x.callee?.name?.startsWith('cw_native_tile_')&&x.callee.name!=='cw_native_tile_thread_rank')fail('Native tile memory phases cannot contain collectives.',x);});
  const mask=fresh();return block([declaration(mask,'bool',c,n.token),masked(n.yes,id(mask,n.token))],n.token);
 });
 if(contains(fn.body))fail('Static tile memory synchronization requires predicated workgroup phases.',fn);
}
