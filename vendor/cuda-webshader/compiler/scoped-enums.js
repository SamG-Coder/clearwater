// Scoped enums use their declared 32-bit integer representation in WGSL.
export function parseScopedEnum(p){
 const token=p.take('enum');if(!p.match('class')&&!p.match('struct'))p.fail('Only scoped enum declarations are supported.',token);
 const collision=p.startsType(),name=p.name();if(collision||p.typeAliases.has(name)||p.structs.has(name)||p.enumTypes.has(name))p.fail('Duplicate enum type.',token);
 let type='i32';if(p.match(':')){const spec=p.type();if(!['i32','u32'].includes(spec.type)||spec.pointer||spec.reference||spec.constant)p.fail('Enums require a 32-bit integer underlying type.',token);type=spec.type;}
 if(p.enumTypes.size>=64)p.fail('At most 64 scoped enums are supported.',token);
 p.enumTypes.set(name,type);p.take('{');const members=new Map();let next=0;
 const checked=(value,n)=>{if(!Number.isSafeInteger(value)||value<(type==='i32'?-2147483648:0)||value>(type==='i32'?2147483647:4294967295))p.fail('Enum value exceeds its underlying range.',n.token);return value;};
 const evaluate=(n,depth=0)=>{
  if(depth>32)p.fail('Enum initializer is too deeply nested.',n.token);
  const result=(value,unsigned)=>{if(unsigned)value>>>=0;else if(!Number.isInteger(value)||value<-2147483648||value>2147483647)p.fail('Signed enum initializer arithmetic overflow.',n.token);return {value,unsigned};};
  if(n.kind==='literal'&&/^-?(?:0[xX][\da-fA-F]+|\d+)[uU]?$/.test(n.value)){
   const value=Number(n.value.replace(/[uU]$/,'')),unsigned=/[uU]$/.test(n.value)||/^0[xX]/.test(n.value)&&value>2147483647;
   if(!Number.isSafeInteger(value)||value< -2147483648||value>(unsigned?4294967295:2147483647))p.fail('Enum initializer literal requires a 32-bit integer.',n.token);
   return {value,unsigned};
  }
  if(n.kind==='id'&&members.has(n.name))return {value:members.get(n.name),unsigned:type==='u32'};
  if(n.kind==='unary'&&['+','-','~'].includes(n.op)){const a=evaluate(n.value,depth+1);return result(n.op==='-'?-a.value:n.op==='~'?~a.value:a.value,a.unsigned);}
  if(n.kind==='binary'&&['+','-','*','/','%','<<','>>','|','&','^'].includes(n.op)){
   const left=evaluate(n.left,depth+1),right=evaluate(n.right,depth+1),shift=['<<','>>'].includes(n.op),unsigned=shift?left.unsigned:left.unsigned||right.unsigned;
   const a=unsigned?left.value>>>0:left.value,b=shift?right.value:unsigned?right.value>>>0:right.value;
   if((n.op==='/'||n.op==='%')&&b===0)p.fail('Division by zero in enum initializer.',n.token);
   if(shift&&(!Number.isInteger(b)||b<0||b>31))p.fail('Enum shift is outside 0..31.',n.token);
   if(n.op==='<<'&&!unsigned&&(a<0||a*2**b>4294967295))p.fail('Invalid signed enum left shift.',n.token);
   return result(n.op==='+'?a+b:n.op==='-'?a-b:n.op==='*'?(unsigned?Math.imul(a,b):a*b):n.op==='/'?Math.trunc(a/b):n.op==='%'?a%b:n.op==='<<'?a<<b:n.op==='>>'?(unsigned?a>>>b:a>>b):n.op==='|'?(a|b):n.op==='&'?(a&b):(a^b),unsigned);
  }
  p.fail('Enum initializers require bounded integer constant expressions.',n.token);
 };
 while(!p.is('}')){
  const member=p.name();if(members.has(member)||members.size>=256)p.fail('Duplicate or excessive enum member.',token);
  const value=p.match('=')?evaluate(p.expression(2)).value:next;checked(value,{token});members.set(member,value);p.enumValues.set(name+'::'+member,{value:String(value)+(type==='u32'?'u':''),type});next=value+1;
  if(!p.match(','))break;
 }
 p.take('}');p.take(';');if(!members.size)p.fail('Scoped enum requires at least one member.',token);
 p.enumTypes.set(name,type);p.typeAliases.set(name,type);
 for(const [member,value] of members)p.enumValues.set(name+'::'+member,{value:String(value)+(type==='u32'?'u':''),type});
}
