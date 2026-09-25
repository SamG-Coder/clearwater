// Evaluate scalar constant initializers with C++ floating-point promotion.
export function evaluateModuleConstant(node, globals, fail, depth=0) {
  if(depth>64)fail('Module constexpr initializer is too deeply nested.',node);
  const checked=(value,type)=>{
    if(type==='f32')value=Math.fround(value);
    if(type==='u32')value=value>>>0;
    if(!Number.isFinite(value)||type==='i32'&&(!Number.isInteger(value)||value<-2147483648||value>2147483647))
      fail('Module constexpr arithmetic is outside its scalar range.',node);
    return {value,type};
  };
  if(node.kind==='literal'){
    const text=node.value,hex=/^0[xX]/.test(text),type=/[uU]$/.test(text)?'u32':!hex&&/[fF]$/.test(text)?'f32':!hex&&/[.eE]/.test(text)?'cw_f64':'i32';
    const value=Number(text.replace(hex?/[uU]$/:/[fFuU]$/,''));
    if(type==='u32'&&(!Number.isInteger(value)||value<0||value>4294967295))fail('Module constexpr literal exceeds unsigned 32-bit range.',node);
    return checked(value,type);
  }
  if(node.kind==='id'){
    const g=globals.find(g=>g.name===node.name&&g.constexprValue!==undefined);
    if(g)return {value:g.constexprValue,type:g.type};
  }
  if(node.kind==='unary'&&['+','-'].includes(node.op)){
    const a=evaluateModuleConstant(node.value,globals,fail,depth+1);
    return checked(node.op==='-'?-a.value:a.value,a.type);
  }
  if(node.kind==='binary'&&['+','-','*','/','%'].includes(node.op)){
    const left=evaluateModuleConstant(node.left,globals,fail,depth+1),right=evaluateModuleConstant(node.right,globals,fail,depth+1);
    const type=['cw_f64','f32','u32','i32'].find(t=>left.type===t||right.type===t);
    const convert=v=>type==='f32'?Math.fround(v):type==='u32'?v>>>0:v;
    const a=convert(left.value),b=convert(right.value);
    if((node.op==='/'||node.op==='%')&&b===0)fail('Division by zero in module constexpr initializer.',node);
    if(node.op==='%'&&['f32','cw_f64'].includes(type))fail('Floating remainder is not a constant integer expression.',node);
    const value=node.op==='+'?a+b:node.op==='-'?a-b:node.op==='*'?(type==='u32'?Math.imul(a,b):a*b):node.op==='/'?(['i32','u32'].includes(type)?Math.trunc(a/b):a/b):a%b;
    return checked(value,type);
  }
  fail('Module constexpr initializers require scalar constant arithmetic.',node);
}
