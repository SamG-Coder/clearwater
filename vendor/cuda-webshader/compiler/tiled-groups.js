// Lower a bounded first CUDA tile to predicated workgroup-wide phases.
// Inactive lanes participate in barriers, but never evaluate tile memory accesses.
export function lowerTiledGroups(ast,options,walk,fail){
 const tileFunctions=new Map(ast.functions.filter(f=>f.params.some(p=>p.type==='thread-tile')).map(f=>[f.name,f]));
 let hasTiles=false;walk(ast,n=>{if(n.kind==='thread-tile')hasTiles=true;});if(!hasTiles&&!tileFunctions.size)return false;
 // Group handles are synchronization tokens, never ordinary boolean values.
 for(const f of ast.functions){
  const handles=new Set(f.params.filter(p=>p.type==='thread-tile').map(p=>p.name));walk(f.body,n=>{if(n.kind==='thread-tile')handles.add(n.name);});
  const inspect=(node,parent=null)=>{if(!node||typeof node!=='object')return;
   if(node.kind==='id'&&handles.has(node.name)){
    const index=parent?.kind==='call'?parent.args.indexOf(node):-1,target=parent?.kind==='call'?tileFunctions.get(parent.callee.name):null;
    if(!(index===0&&parent.callee.name==='cooperative_groups::sync'&&parent.args.length===1)&&!(index>=0&&target?.params[index]?.type==='thread-tile'))fail('Tile handles may only be passed to tile helpers or sync.',node);
   }
   if(node.kind==='call'&&tileFunctions.has(node.callee.name)){const target=tileFunctions.get(node.callee.name);if(node.args.length!==target.params.length)fail('Tile helper argument count must match.',node);for(let i=0;i<node.args.length;i++){const p=target.params[i],a=node.args[i];if(p.type==='thread-tile'&&!(a.kind==='id'&&handles.has(a.name)))fail('Tile helper requires its group handle.',a);if(p.pointer&&a.kind!=='id')fail('Predicated tile helpers require named pointer arguments.',a);}}
   for(const [key,value]of Object.entries(node))if(key!=='token'){if(Array.isArray(value))value.forEach(x=>inspect(x,node));else if(value&&typeof value==='object')inspect(value,node);}
  };inspect(f.body);
 }
 const names=new Set();walk(ast,n=>{if(n.name)names.add(n.name);});let sequence=0;const fresh=()=>{let n;do{n='cw_tile_phase_'+sequence++;}while(names.has(n));names.add(n);return n;};
 const clone=n=>structuredClone(n),id=(name,token)=>({kind:'id',name,token}),literal=(value,token)=>({kind:'literal',value:String(value),token}),block=(body,token)=>({kind:'block',body,token}),expr=(value,token=value.token)=>({kind:'expr',value,token});
 const zero=(type,token)=>type==='bool'?id('false',token):type.startsWith('vec')?{kind:'initializer',target:type,items:[],token}:literal(type==='f32'?'0.0f':type==='u32'?'0u':'0',token);
 const decl=(name,type,init,token)=>({kind:'decl',name,type,init,token,constant:true,shared:false,pointer:false,reference:false,external:false,dimensions:[]});
 const guarded=(active,value,type)=>({kind:'conditional',condition:clone(active),yes:value,no:zero(type,value.token),token:value.token});
 const barrier=token=>expr({kind:'call',callee:id('__syncthreads',token),args:[],token});
 const root=n=>n?.kind==='id'?n.name:['index','member'].includes(n?.kind)?root(n.base):null;
 const hasTileCall=n=>{let found=false;walk(n,x=>{if(x.kind==='call'&&(tileFunctions.has(x.callee.name)||x.callee.name==='cooperative_groups::sync'))found=true;});return found;};
 function calls(node,active){
  if(!node||typeof node!=='object')return node;
  if(node.kind==='call'&&tileFunctions.has(node.callee.name)){
   const f=tileFunctions.get(node.callee.name);node.args=node.args.map((a,i)=>f.params[i]?.tileParameter?clone(active):f.params[i]?.pointer?a:guarded(active,calls(a,active),f.params[i].type));return node;
  }
  for(const [key,v]of Object.entries(node))if(key!=='token'){if(Array.isArray(v))node[key]=v.map(x=>calls(x,active));else if(v&&typeof v==='object')node[key]=calls(v,active);}return node;
 }
 for(const f of tileFunctions.values()){
  const params=f.params.filter(p=>p.type==='thread-tile');if(params.length!==1||params[0].pointer||params[0].reference)fail('Tile helpers require one group passed by value.',f);
  params[0].tileParameter=true;params[0].type='bool';params[0].constant=true;f.tilePredicate=params[0].name;
 }
 function maskedStatements(statements,active,types){return statements.flatMap(n=>masked(n,active,types));}
 function masked(n,active,types){
  const token=n.token;
  if(n.kind==='empty')return [];
  if(n.kind==='block')return [block(maskedStatements(n.body,active,new Map(types)),token)];
  if(n.kind==='decls')return n.declarations.flatMap(d=>masked(d,active,types));
  if(n.kind==='decl'){
   types.set(n.name,n.type);if(n.shared)return [n];if(n.pointer||n.reference||n.dimensions.length)fail('Predicated tile locals require scalar or vector values.',n);
   if(n.init){const cooperative=hasTileCall(n.init);n.init=calls(n.init,active);if(!cooperative)n.init=guarded(active,n.init,n.type);}return [n];
  }
  if(n.kind==='if'){
   const mask=fresh(),condition={kind:'binary',op:'&&',left:clone(active),right:n.condition,token},out=[decl(mask,'bool',condition,token)];
   out.push(...masked(n.yes,id(mask,token),new Map(types)));
   if(n.no){const other=fresh();out.push(decl(other,'bool',{kind:'binary',op:'&&',left:clone(active),right:{kind:'unary',op:'!',value:id(mask,token),token},token},token),...masked(n.no,id(other,token),new Map(types)));}return out;
  }
  if(n.kind==='for'){
   // Loop control must remain independent of the per-lane predicate.
   const init=n.init,c=n.condition,step=n.step,loopName=init?.name;
   if(init?.kind!=='decl'||init.init?.kind!=='literal'||!['i32','u32'].includes(init.type)||c?.kind!=='binary'||c.left.kind!=='id'||c.left.name!==loopName||c.right.kind!=='literal'||!['<','<=','>','>='].includes(c.op)||!((step?.kind==='unary'&&['++','--'].includes(step.op)&&step.value.name===loopName)||(step?.kind==='assign'&&['>>=','<<=','+=','-='].includes(step.op)&&step.left.name===loopName&&step.right.kind==='literal')))fail('Tile loops require uniform literal bounds and a simple induction variable.',n);
   let changesInduction=false;walk(n.body,x=>{if(x.kind==='assign'&&root(x.left)===loopName||x.kind==='unary'&&['++','--','&'].includes(x.op)&&root(x.value)===loopName)changesInduction=true;});if(changesInduction)fail('Tile loop induction variables cannot change or escape in the loop body.',n);n.body=block(masked(n.body,active,new Map(types)),token);return [n];
  }
  if(n.kind==='return'){
   if(!n.value)return [n];if(hasTileCall(n.value))n.value=calls(n.value,active);else n.value=guarded(active,n.value,types.get('@result'));return [n];
  }
  if(n.kind==='expr'){
   const v=n.value;
   if(v.kind==='call'&&v.callee.name==='cooperative_groups::sync'){
    if(v.args.length!==1||v.args[0].kind!=='id'||v.args[0].name!==types.get('@tile'))fail('Tile sync requires its group handle.',v);return [barrier(token)];
   }
   if(hasTileCall(v))return [expr(calls(v,active),token)];
   if(v.kind==='assign'&&v.op!=='='&&v.left.kind==='index'&&types.has(root(v.left))){
    let impure=false;walk([v.left,v.right],x=>{if(x.kind==='call'||x.kind==='assign'||x.kind==='unary'&&['++','--'].includes(x.op))impure=true;});if(impure)fail('Tile memory updates require side-effect-free operands.',v);
    const temp=fresh(),type=types.get(root(v.left)),value={kind:'binary',op:v.op.slice(0,-1),left:clone(v.left),right:v.right,token};
    return [decl(temp,type,guarded(active,value,type),token),barrier(token),{kind:'if',condition:clone(active),yes:block([expr({kind:'assign',op:'=',left:v.left,right:id(temp,token),token})],token),no:null,token}];
   }
   return [{kind:'if',condition:clone(active),yes:block([n],token),no:null,token}];
  }
  fail('Unsupported control flow inside a predicated tile: '+n.kind,n);
 }
 for(const f of tileFunctions.values()){
  let early=false;walk(f.body,n=>{if(n.kind==='return'&&n!==f.body.body.at(-1))early=true;});if(early)fail('Tile helpers cannot return early.',f);
  const types=new Map(f.params.filter(p=>p.type!=='bool').map(p=>[p.name,p.type]));types.set('@result',f.result);types.set('@tile',f.tilePredicate);
  f.body.body=maskedStatements(f.body.body,id(f.tilePredicate,f.token),types);
 }
 for(const f of ast.functions){if(tileFunctions.has(f.name))continue;const tiles=new Map(),blocks=new Set(f.params.filter(p=>p.type==='thread-block').map(p=>p.name)),lanes=new Set(),types=new Map(f.params.map(p=>[p.name,p.type]));
  for(const n of f.body.body){if(n.kind==='thread-block')blocks.add(n.name);if(n.kind==='decl'){types.set(n.name,n.type);if(n.constant&&n.init?.kind==='member'&&n.init.base.name==='threadIdx'&&n.init.member==='x')lanes.add(n.name);}if(n.kind==='thread-tile'){
   if(!blocks.has(n.parent))fail('A tile must be partitioned from an existing thread block.',n);const size=n.size.kind==='literal'?Number(n.size.value.replace(/[uU]$/,'')):0,w=options.workgroupSize||[128,1,1];if(!Number.isInteger(size)||size<1||(size&(size-1))||w[0]%size||w[1]!==1||w[2]!==1)fail('Tiled groups require a literal power-of-two size dividing a one-dimensional block.',n);tiles.set(n.name,size);
  }}
  if(!tiles.size)continue;const output=[];
  for(const n of f.body.body){if(n.kind==='thread-tile'){output.push(decl(n.name,'bool',id('true',n.token),n.token));continue;}
   if(hasTileCall(n)){
    const c=n.condition,used=[...tiles].filter(([name])=>{let yes=false;walk(n,x=>{if(x.kind==='id'&&x.name===name)yes=true;});return yes;});
    if(n.kind!=='if'||n.no||c?.kind!=='binary'||c.op!=='<'||c.left.kind!=='id'||!lanes.has(c.left.name)||c.right.kind!=='literal'||used.length!==1||Number(c.right.value.replace(/[uU]$/,''))!==used[0][1])fail('A local tile must execute in a complete first-tile lane < size branch.',n);
    const mask=fresh(),tileTypes=new Map(types);tileTypes.set('@tile',used[0][0]);output.push(decl(mask,'bool',c,n.token),...masked(n.yes,id(mask,n.token),tileTypes));
   }else output.push(n);
  }f.body.body=output;
 }
 return true;
}
