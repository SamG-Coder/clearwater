// Storage reads do not carry WGSL uniformity, even when every lane addresses
// the same record. Recognize pure scalar getters at kernel block scope and
// explicitly publish their value through workgroupUniformLoad.
export function markUniformRecordSnapshots(ast,kernel,walk){
 let barriers=false;walk(kernel.body,n=>{if(n.kind==='call'&&['__syncthreads','cooperative_groups::sync'].includes(n.callee?.name))barriers=true;});if(!barriers)return;
 const records=new Map(ast.structs.map(r=>[r.type,r])),refs=new Map();
 const uniformIndex=n=>n?.kind==='literal'||n?.kind==='member'&&n.base?.kind==='id'&&n.base.name==='blockIdx'&&['x','y','z'].includes(n.member)||n?.kind==='binary'&&['+','-','*','/','%','<<','>>','&','|','^'].includes(n.op)&&uniformIndex(n.left)&&uniformIndex(n.right);
 const changed=new Set();walk(kernel.body,n=>{if(n.kind==='assign'&&n.left.kind==='id')changed.add(n.left.name);if(n.kind==='unary'&&['++','--','&'].includes(n.op)&&n.value?.kind==='id')changed.add(n.value.name);});
 const pureGetter=(record,name)=>{
  const method=record.methods.find(m=>m.name===name),fn=ast.functions.find(f=>f.name===method?.helper);
  if(!fn?.classConstant||fn.params.length!==1||fn.body.body.length!==1||fn.body.body[0].kind!=='return')return false;
  const self=fn.params[0].name;
  const fieldType=n=>{if(n?.kind==='id'&&n.name===self)return record.type;if(n?.kind==='member'){const parent=records.get(fieldType(n.base));return parent?.fields.find(f=>f.name===n.member&&!f.dimensions.length)?.type;}};
  const pure=n=>n?.kind==='literal'||n?.kind==='member'&&['i32','u32','f32','bool'].includes(fieldType(n))||n?.kind==='binary'&&pure(n.left)&&pure(n.right)||n?.kind==='unary'&&['+','-','!','~'].includes(n.op)&&pure(n.value)||n?.kind==='cast'&&['i32','u32','f32','bool'].includes(n.target)&&pure(n.value);
  return pure(fn.body.body[0].value);
 };
 for(const n of kernel.body.body){
  if(n.kind!=='decl')continue;
  refs.delete(n.name);
  const at=n.init;
  if(n.reference&&at?.kind==='index'&&at.base.kind==='id'&&uniformIndex(at.index)&&!changed.has(at.base.name)){
   const param=kernel.params.find(p=>p.name===at.base.name&&p.pointer&&p.type===n.type),record=records.get(n.type);
   if(param&&record?.valueClass)refs.set(n.name,record);
  }
  if(n.pointer||n.reference||n.shared||n.dimensions.length||!['i32','u32','f32','bool'].includes(n.type))continue;
  if(at?.kind==='call'&&at.callee.kind==='member'&&at.callee.base.kind==='id'&&!at.args.length){const record=refs.get(at.callee.base.name);if(record&&pureGetter(record,at.callee.member))n.workgroupUniformSnapshot=true;}
 }
}
