// Supported function-like macros forward every argument once, in order, to a named call.
// Forwarders do not perform general textual substitution.
export function forwardingMacro(directive) {
 const wrapped=directive.match(/^(#\s*define\s+[A-Za-z_]\w*\([^()]*\))\s+\(\s*([A-Za-z_]\w*\([^()]*\))\s*\)\s*(?:\/\/[^\n]*)?$/);if(wrapped)return forwardingMacro(wrapped[1]+' '+wrapped[2]);
 const m=directive.match(/^#\s*define\s+([A-Za-z_]\w*)\(([^()]*)\)\s+([A-Za-z_]\w*)\(([^()]*)\)\s*(?:\/\/[^\n]*)?$/);
 if(!m)return null;
 const split=s=>s.trim()?s.split(',').map(p=>p.trim()):[],params=split(m[2]),args=split(m[4]);
 if(params.some(p=>!/^[A-Za-z_]\w*$/.test(p))||new Set(params).size!==params.length||params.length!==args.length||params.some((p,i)=>p!==args[i]))return null;
 return {name:m[1],target:m[3],arity:params.length};
}

// Parenthesized expressions, subscripts and complete calls preserve precedence when expanded
// into the AST. Other textual C preprocessing remains outside this subset.
export function expressionMacro(directive){
 if(forwardingMacro(directive))return null;
 const m=directive.match(/^#\s*define\s+([A-Za-z_]\w*)\(([^()]*)\)\s+(.*?)\s*(?:\/\/[^\n]*)?$/);if(!m)return null;
 const params=m[2].trim()?m[2].split(',').map(p=>p.trim()):[],body=m[3];
 if(params.length>16||params.some(p=>!/^[A-Za-z_]\w*$/.test(p))||new Set(params).size!==params.length||body.length>1024||/[#;{}"'\\]/.test(body))return null;
 // A complete parenthesized expression or named subscript is a primary
 // expression in CUDA, so substituting its AST preserves surrounding precedence.
 const indexed=body.match(/^[A-Za-z_]\w*\s*\[/),call=body.match(/^[A-Za-z_]\w*(?:<[^<>]+>)?\s*\(/),begin=indexed?indexed[0].length-1:call?call[0].length-1:0;
 if(!indexed&&!call&&(!body.startsWith('(')||!body.endsWith(')')))return null;
 const stack=[];for(let i=begin;i<body.length;i++){const c=body[i];if(c==='('||c==='[')stack.push(c);else if(c===')'||c===']'){if(stack.pop()!==(c===')'?'(':'['))return null;}if(!stack.length&&i<body.length-1)return null;}if(stack.length)return null;
 for(const match of body.matchAll(/[A-Za-z_]\w*/g))if(params.includes(match[0])){const before=body.slice(0,match.index).trimEnd(),after=body.slice(match.index+match[0].length).trimStart(),wrapped=before.endsWith('(')&&after.startsWith(')'),argument=call&&/[,(]$/.test(before)&&/^[,)]/.test(after);if(!wrapped&&!argument)return null;}
 return {name:m[1],params,body};
}
