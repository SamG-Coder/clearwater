// Snapshot a named pointer to the first element of a constant-table row.
// All uses remain indexed reads of the original immutable table.
export function lowerConstantRows(ast,walk,fail){
 const globals=new Map(ast.constantGlobals.filter(g=>g.dimensions.length===2).map(g=>[g.name,g]));
 for(const f of ast.functions){
  const aliases=new Map(),counts=new Map();walk(f.body,n=>{if(n.kind==='decl')counts.set(n.name,(counts.get(n.name)||0)+1);});
  for(const n of f.body.body){
   const address=n.init,row=address?.value?.base,table=row?.base;
   if(n.kind!=='decl'||!n.pointer||address?.kind!=='unary'||address.op!=='&'||address.value.kind!=='index'||row?.kind!=='index'||table?.kind!=='id'||!globals.has(table.name))continue;
   const g=globals.get(table.name),column=address.value.index;
   if(n.type!==g.type||n.shared||n.reference||n.dimensions.length||column.kind!=='literal'||!/^0[uU]?$/.test(column.value)||counts.get(n.name)!==1||counts.has(table.name)||f.params.some(p=>[table.name,n.name].includes(p.name)))fail('Constant row aliases require one unshadowed same-type pointer to column zero.',n);
   aliases.set(n.name,{table:table.name,declaration:n});n.pointer=false;n.constant=true;n.type='i32';n.init=row.index;
  }
  if(!aliases.size)continue;
  function visit(n,parent=null){
   if(!n||typeof n!=='object')return;
   if(n.kind==='index'&&n.base.kind==='id'&&aliases.has(n.base.name)){
    const a=aliases.get(n.base.name),token=n.base.token;n.base={kind:'index',base:{kind:'id',name:a.table,token},index:n.base,token};visit(n.index,n);return;
   }
   if(n.kind==='id'&&aliases.has(n.name))fail('Constant row pointers support indexed access only; reassignment and escapes are unsupported.',n);
   for(const [key,v]of Object.entries(n))if(key!=='token'){if(Array.isArray(v))v.forEach(x=>visit(x,n));else if(v&&typeof v==='object')visit(v,n);}
  }
  visit(f.body);
 }
}
