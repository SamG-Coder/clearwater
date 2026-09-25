// Bounded compile-time integer arithmetic. No JavaScript evaluation or C++ side effects.
export function integerExpression(source){
 const text=String(source),tokens=text.match(/\d+|[()+*/%\-]/g)||[];
 if(tokens.join('')!==text.replace(/\s/g,'')||!tokens.length||tokens.length>64)throw Error('Template argument requires a bounded integer arithmetic expression.');
 let index=0,depth=0;
 const checked=n=>{if(!Number.isSafeInteger(n)||n< -2147483648||n>2147483647)throw Error('Template argument arithmetic exceeds the signed 32-bit range.');return n;};
 const unary=()=>{if(++depth>32)throw Error('Template argument expression is too deeply nested.');let value;const t=tokens[index++];if(t==='+'||t==='-'){const operand=unary();value=checked(t==='-'?-operand:operand);}else if(t==='('){value=sum();if(tokens[index++]!==')')throw Error('Unbalanced template argument parentheses.');}else if(/^\d+$/.test(t||'')){value=Number(t);if(!Number.isSafeInteger(value)||value>2147483648)throw Error('Template argument literal exceeds the signed 32-bit range.');}else throw Error('Expected an integer template argument operand.');depth--;return value;};
 const product=()=>{let value=unary();while(['*','/','%'].includes(tokens[index])){const op=tokens[index++],right=unary();checked(value);checked(right);if((op==='/'||op==='%')&&right===0)throw Error('Division by zero in template argument.');value=checked(op==='*'?value*right:op==='/'?Math.trunc(value/right):value%right);}return value;};
 const sum=()=>{let value=product();while(['+','-'].includes(tokens[index])){const op=tokens[index++],right=product();checked(value);checked(right);value=checked(op==='+'?value+right:value-right);}return value;};
 const result=sum();if(index!==tokens.length)throw Error('Unsupported integer template argument expression.');return checked(result);
}
export const substituteTemplateArgument=(argument,parameter,value)=>String(argument).replace(new RegExp('\\b'+parameter+'\\b','g'),'('+value+')');
