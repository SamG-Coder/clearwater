// Clone compiler data without recursing through the JavaScript call stack.
// Preserve shared references and cycles just as structuredClone does.
export function cloneAst(value){
 if(value===null||typeof value!=='object')return value;
 const seen=new Map(),pending=[];
 const make=source=>{if(source===null||typeof source!=='object')return source;if(seen.has(source))return seen.get(source);const proto=Object.getPrototypeOf(source);if(!Array.isArray(source)&&proto!==Object.prototype&&proto!==null){const copy=structuredClone(source);seen.set(source,copy);return copy;}const copy=Array.isArray(source)?new Array(source.length):Object.create(proto);seen.set(source,copy);pending.push([source,copy]);return copy;};
 const result=make(value);
 while(pending.length){const [source,target]=pending.pop();for(const key of Object.keys(source))Object.defineProperty(target,key,{value:make(source[key]),enumerable:true,writable:true,configurable:true});}
 return result;
}
