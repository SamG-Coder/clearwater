// Plain scalar records cross launch boundaries as explicit 32-bit fields.
export function recordLeaves(e,type,node,path=[],seen=new Set()){
 if(['i32','u32','f32','bool'].includes(type))return [{path,type}];
 const record=e.structs.get(type);if(!record||seen.has(type)||seen.size>=16)e.fail('Record launch arguments require finite scalar-only records.',node);
 const next=new Set(seen);next.add(type);const leaves=record.fields.flatMap(f=>recordLeaves(e,f.resolvedType||f.type,node,[...path,f.name],next));
 if(leaves.length>64)e.fail('Record launch arguments support at most 64 scalar fields.',node);return leaves;
}
export function recordConstructor(e,type,leaf,path=[]){
 const record=e.structs.get(type);return record?`${type}(${record.fields.map(f=>recordConstructor(e,f.resolvedType||f.type,leaf,[...path,f.name])).join(', ')})`:leaf(path,type);
}
