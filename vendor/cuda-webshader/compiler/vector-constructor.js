// Recombine explicitly component-wise CUDA helpers into vector WGSL. Only
// side-effect-free reads of local float vectors are eligible; evaluation order
// and the arithmetic tree remain unchanged.
export function vectorConstructor(emitter, nodes) {
  const width=nodes.length;
  if(width<2||width>4)return null;
  function combine(parts){
    if(parts.every((n,i)=>n.kind==='member'&&n.member==='xyzw'[i]&&n.base.kind==='id'&&n.base.name===parts[0].base.name)){
      const v=emitter.expr(parts[0].base);
      return v.type===`vec${width}<f32>`&&!v.pre.length&&v.rootSymbol?.kind==='local'?v.code:null;
    }
    const first=parts[0];
    if(first.kind==='binary'&&['+','-','*','/'].includes(first.op)&&parts.every(n=>n.kind==='binary'&&n.op===first.op)){
      const a=combine(parts.map(n=>n.left)),b=combine(parts.map(n=>n.right));
      return a!==null&&b!==null?`(${a} ${first.op} ${b})`:null;
    }
    const unary={truncf:'trunc',floorf:'floor',ceilf:'ceil',fabsf:'abs'};
    if(first.kind==='call'&&first.callee.kind==='id'&&unary[first.callee.name]&&parts.every(n=>n.kind==='call'&&n.callee.kind==='id'&&n.callee.name===first.callee.name&&n.args.length===1)){
      const a=combine(parts.map(n=>n.args[0]));return a===null?null:`${unary[first.callee.name]}(${a})`;
    }
    const binary={fmaxf:'max',fminf:'min'};
    if(first.kind==='call'&&first.callee.kind==='id'&&binary[first.callee.name]&&parts.every(n=>n.kind==='call'&&n.callee.kind==='id'&&n.callee.name===first.callee.name&&n.args.length===2)){
      const a=combine(parts.map(n=>n.args[0])),b=combine(parts.map(n=>n.args[1]));
      return a===null||b===null?null:`${binary[first.callee.name]}(${a}, ${b})`;
    }
    return null;
  }
  return combine(nodes);
}
