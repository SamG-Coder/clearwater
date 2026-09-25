import {evaluateModuleConstant} from './module-constexpr.js';
import {parseScopedEnum} from './scoped-enums.js';
import {CURAND_XORWOW_SOURCE} from './curand-xorwow.js';
/** A deliberately bounded CUDA C frontend. No eval, regex transpilation, or source-specific rewrites. */
import {parseValueClass,finishValueClasses,parseExternalValueMethod} from './value-classes.js';
import {forwardingMacro,expressionMacro} from './macros.js';
import {integerExpression} from './integer-expression.js';
export class CompileError extends Error {
  constructor(message, token = {}, source = '') {
    const line = token.line || 1, column = token.column || 1;
    super(`${message} (${line}:${column})${source ? `\n${source.split('\n')[line - 1] || ''}\n${' '.repeat(column - 1)}^` : ''}`);
    this.name = 'CompileError'; this.line = line; this.column = column;
  }
}
const NUM = /^(?:0[xX][\da-fA-F]+(?:[uU][lL]?|[lL][uU])?|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?(?:[uU][lL]?|[lL][uU]|[fF])?)/;
const WORD = /^[A-Za-z_]\w*/;
const OPERATORS = ['<<<', '>>>', '<<=', '>>=', '::', '++', '--', '+=', '-=', '*=', '/=', '%=', '==', '!=', '<=', '>=', '&&', '||', '<<', '>>', '&=', '|=', '^=', '->'];
const TYPES = new Set(['__half','__half2','float', 'int', 'uint', 'unsigned', 'bool', 'void', 'float2', 'float3', 'float4']);
const QUALIFIERS = new Set(['const', '__shared__', '__restrict__', '__restrict', 'restrict','extern']);
const MAP = { __half:'f16',__half2:'vec2<f16>', float: 'f32', int: 'i32', uint: 'u32', bool: 'bool', void: 'void', float2: 'vec2<f32>', float3: 'vec3<f32>', float4: 'vec4<f32>' };
TYPES.add('uchar');MAP.uchar='cw_uchar';
TYPES.add('short');MAP.short='cw_short';TYPES.add('ushort');MAP.ushort='cw_ushort';
TYPES.add('uchar4');MAP.uchar4='cw_uchar4';
TYPES.add('uchar2');MAP.uchar2='cw_uchar2';
TYPES.add('size_t');MAP.size_t='cw_size64';
TYPES.add('cudaExtent');MAP.cudaExtent='cw_extent';
TYPES.add('cudaTextureObject_t');MAP.cudaTextureObject_t='texture3d';
TYPES.add('cudaSurfaceObject_t');MAP.cudaSurfaceObject_t='surface2d';
for(const [prefix,type] of [['uint','u32'],['int','i32']])for(const size of [2,3,4]){TYPES.add(prefix+size);MAP[prefix+size]=`vec${size}<${type}>`;}
export const builtinType = name => Object.hasOwn(MAP,name)?MAP[name]:null;
function unwrapCondition(text){for(let wraps=0;wraps<32&&text.startsWith('(')&&text.endsWith(')');wraps++){let depth=0,whole=true;for(let i=0;i<text.length;i++){if(text[i]==='(')depth++;if(text[i]===')')depth--;if(depth<0||(depth===0&&i<text.length-1))whole=false;}if(!whole||depth!==0)break;text=text.slice(1,-1).trim();}return text;}
export function tokenize(source, defines = {}) {
  if (typeof source !== 'string' || source.length > 1_000_000) throw new CompileError('Source must be a string of at most 1 MB.');
  const macros = new Map(Object.entries(defines).map(([k, v]) => {
    if (!/^[A-Za-z_]\w*$/.test(k) || !Number.isFinite(v)) throw new CompileError('Defines must be named finite numbers.');
    return [k, String(v)];
  }));
  const conditionals=[];let enabled=true;
  const tokens = [],forwarders=new Map(),expressions=new Map(),objectExpressions=new Map(),aliases=new Map();let numericMacroSnapshot=null; let i = 0, line = 1, column = 1;
  const resolveAlias=(name,token)=>{const seen=new Set();while(aliases.has(name)){if(seen.has(name)||seen.size>=32)throw new CompileError('Identifier macro alias cycle or chain limit exceeded.',token,source);seen.add(name);name=aliases.get(name);}return name;};
  const advance = str => { for (const c of str) { if (c === '\n') { line++; column = 1; } else column++; } i += str.length; };
  while (i < source.length) {
    const rest = source.slice(i), token = {line, column, offset: i};
    if (/^\s/.test(rest)) { advance(rest.match(/^\s+/)[0]); continue; }
    if (rest.startsWith('//')) { advance(rest.split('\n')[0]); continue; }
    if (rest.startsWith('/*')) { const end = rest.indexOf('*/'); if (end < 0) throw new CompileError('Unclosed comment.', token, source); advance(rest.slice(0, end + 2)); continue; }
    if (rest[0] === '#') {
      const directive = rest.split('\n')[0];
      const conditional=directive.trimEnd().match(/^#\s*(ifdef|ifndef|if|else|endif)\b(.*)$/);
      if(conditional){const [,kind,tail]=conditional,expression=kind==='if'?unwrapCondition(tail.replace(/\/\/.*$/,'').trim()):tail.replace(/\/\/.*$/,'').trim();
        if(kind==='ifdef'||kind==='ifndef'){if(!/^[A-Za-z_]\w*$/.test(expression))throw new CompileError('Conditional definition test requires one macro name.',token,source);const defined=macros.has(expression)||forwarders.has(expression)||expressions.has(expression)||objectExpressions.has(expression)||aliases.has(expression),selected=kind==='ifdef'?defined:!defined;conditionals.push({parent:enabled,selected,otherwise:false});enabled=enabled&&selected;}
        else if(kind==='if'){const match=expression.match(/^(!)?\s*([A-Za-z_]\w*|[0-9]+)$/);if(!match){let number;try{const expanded=expression.replace(/[A-Za-z_]\w*/g,name=>{if(objectExpressions.has(name)||forwarders.has(name)||expressions.has(name))throw Error('Conditional macro must be an integer.');const value=macros.get(resolveAlias(name,token))??'0';if(!/^[+-]?\d+[uU]?$/.test(value))throw Error('Conditional macro must be an integer.');return '('+value.replace(/[uU]$/,'')+')';});number=integerExpression(expanded);}catch(error){throw new CompileError('Conditional preprocessing requires bounded integer arithmetic: '+error.message,token,source);}const selected=!!number;conditionals.push({parent:enabled,selected,otherwise:false});enabled=enabled&&selected;advance(directive);continue;}if(objectExpressions.has(match[2]))throw new CompileError('Conditional macro must be an integer.',token,source);const raw=/^[0-9]+$/.test(match[2])?match[2]:macros.get(resolveAlias(match[2],token))??'0',number=Number(raw.replace(/[uU]$/,''));if(!Number.isSafeInteger(number))throw new CompileError('Conditional macro must be an integer.',token,source);const selected=match[1]?!number:!!number;conditionals.push({parent:enabled,selected,otherwise:false});enabled=enabled&&selected;}
        else{const frame=conditionals.at(-1);if(!frame||expression)throw new CompileError('Unmatched or malformed conditional directive.',token,source);if(kind==='else'){if(frame.otherwise)throw new CompileError('Duplicate #else.',token,source);frame.otherwise=true;enabled=frame.parent&&!frame.selected;}else{conditionals.pop();enabled=frame.parent;}}
        advance(directive);continue;
      }
      if(/^#\s*(elif)\b/.test(directive))throw new CompileError('Unsupported conditional directive; preprocess it first.',token,source);
      if(!enabled){advance(directive);continue;}
      const unroll=directive.trimEnd().match(/^#\s*pragma\s+unroll(?:\s+([1-9]\d*))?\s*(?:\/\/.*)?$/);if(unroll){tokens.push({...token,kind:'pragma-unroll',value:'#pragma unroll',factor:unroll[1]?Number(unroll[1]):true});advance(directive);continue;}
      const alias=directive.trimEnd().match(/^#\s*define\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*(?:\/\/.*)?$/);
      const definedName=directive.match(/^#\s*define\s+([A-Za-z_]\w*)/)?.[1];if(definedName&&aliases.has(definedName))throw new CompileError('Macro redefinition is unsupported.',token,source);
      if(alias){if(macros.has(alias[1])){advance(directive);continue;}if(forwarders.has(alias[1])||expressions.has(alias[1])||objectExpressions.has(alias[1])||aliases.size>=128)throw new CompileError('Duplicate or excessive identifier macro alias.',token,source);aliases.set(alias[1],alias[2]);advance(directive);continue;}
      const expression=expressionMacro(directive.trimEnd());if(expression){if(macros.has(expression.name)||forwarders.has(expression.name)||expressions.has(expression.name)||objectExpressions.has(expression.name))throw new CompileError('Macro redefinition is unsupported.',token,source);expressions.set(expression.name,expression);advance(directive);continue;}
      const forward=forwardingMacro(directive.trimEnd());
      if(forward){if(macros.has(forward.name)||forwarders.has(forward.name)||expressions.has(forward.name)||objectExpressions.has(forward.name))throw new CompileError('Macro redefinition is unsupported.',token,source);forwarders.set(forward.name,forward);advance(directive);continue;}
      const m = directive.trimEnd().match(/^#\s*define\s+([A-Za-z_]\w*)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFuU]?)\s*(?:\/\/.*)?$/);
      if(!m){const object=directive.trimEnd().match(/^#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*(?:\/\/.*)?$/);if(object){try{const value=integerExpression(object[2].replace(/[A-Za-z_]\w*/g,name=>macros.has(name)?'('+macros.get(name)+')':name));if(forwarders.has(object[1])||expressions.has(object[1])||objectExpressions.has(object[1]))throw Error('Macro redefinition is unsupported.');if(!macros.has(object[1])){macros.set(object[1],String(value));numericMacroSnapshot=null;}advance(directive);continue;}catch(error){
          const body=object[2].trim(),primary=expressionMacro('#define cw_object_macro() '+body);if(body.length>1024||!primary)throw new CompileError(error.message,token,source);
          if(objectExpressions.has(object[1])||forwarders.has(object[1])||expressions.has(object[1]))throw new CompileError('Macro redefinition is unsupported.',token,source);
          if(macros.has(object[1])){advance(directive);continue;}
          const expanded=body.replace(/0[xX][\da-fA-F]+[uU]?|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFuU]?|[A-Za-z_]\w*/g,name=>/^[A-Za-z_]\w*$/.test(name)?(objectExpressions.has(name)?objectExpressions.get(name).map(t=>t.value).join(' '):macros.has(name)?macros.get(name):name):name);
          if(expanded.length>8192||objectExpressions.size>=128)throw new CompileError('Object macro expansion limit exceeded.',token,source);
          let parts;try{parts=tokenize(expanded).filter(t=>t.kind!=='eof');}catch{throw new CompileError(error.message,token,source);}
          if(parts.length>256||parts.some(t=>t.kind==='symbol'&&!'()[],.+-*/%<>&|^~!?:'.includes(t.value)))throw new CompileError('Object expressions require bounded primary expressions.',token,source);
          objectExpressions.set(object[1],parts);advance(directive);continue;
        }}}
      if (!m) throw new CompileError('Only numeric object-like #define directives and direct function-forwarding macros are supported; preprocess other directives first.', token, source);
      if(forwarders.has(m[1])||expressions.has(m[1])||objectExpressions.has(m[1]))throw new CompileError('Macro redefinition is unsupported.',token,source);
      if (!macros.has(m[1])){macros.set(m[1], m[2]);numericMacroSnapshot=null;} advance(directive); continue;
    }
    if(!enabled){advance(rest.split('\n')[0]);continue;}
    if(rest[0]==='"') {const literal=rest.match(/^"(?:[^"\n\r\\]|\\[nrt"\\])*"/);if(!literal)throw new CompileError('Unsupported or unterminated string literal.',token,source);tokens.push({...token,kind:'string',value:literal[0]});advance(literal[0]);continue;}
    const number = rest.match(NUM);
    if (number) { tokens.push({...token, kind: 'number', value: number[0].replace(/(?:[uU][lL]|[lL][uU])$/,'u')}); advance(number[0]); continue; }
    const word = rest.match(WORD);
    if (word) {
      const value = resolveAlias(word[0],token), expanded = macros.get(value);
      if(objectExpressions.has(value)){if(tokens.length+objectExpressions.get(value).length>200000)throw new CompileError('Object macro token budget exceeded.',token,source);for(const part of objectExpressions.get(value)){if(part.kind==='word'&&aliases.has(part.value))throw new CompileError('Identifier aliases inside object-expression macros require preprocessing.',token,source);tokens.push({...part,...token});}}
      else if (expanded !== undefined) {
        let body = expanded;
        if (body.startsWith('-') || body.startsWith('+')) { tokens.push({...token, kind: 'symbol', value: body[0]}); body = body.slice(1); }
        tokens.push({...token, kind: 'number', value: body});
      } else {
        const chain=[];let target=value;const seen=new Set();
        while(forwarders.has(target)&&!seen.has(target)&&chain.length<32){seen.add(target);const f=forwarders.get(target);chain.push(f);target=resolveAlias(f.target,token);}
        tokens.push({...token, kind: 'word', value,...(expressions.has(value)?{expressionMacro:{...expressions.get(value),defines:(numericMacroSnapshot??=Object.fromEntries(macros)),definitions:[...expressions.values()].map(m=>'#define '+m.name+'('+m.params.join(',')+') '+m.body).concat([...forwarders.values()].map(m=>{const args=Array.from({length:m.arity},(_,i)=>'cw_arg'+i).join(',');return '#define '+m.name+'('+args+') '+m.target+'('+args+')';})).concat([...aliases].map(([name,target])=>'#define '+name+' '+target))}}:{}),...(chain.length?{forward:{chain,target,tooDeep:forwarders.has(target)&&!seen.has(target),recursive:seen.has(target),numericTarget:macros.has(target)}}:{})});
      }
      advance(word[0]); continue;
    }
    const op = OPERATORS.find(x => rest.startsWith(x));
    if (op) { tokens.push({...token, kind: 'symbol', value: op}); advance(op); continue; }
    if ('{}[]();,.?:+-*/%<>=!~&|^'.includes(rest[0])) { tokens.push({...token, kind: 'symbol', value: rest[0]}); advance(rest[0]); continue; }
    throw new CompileError(`Unsupported character ${JSON.stringify(rest[0])}.`, token, source);
  }
  if(conditionals.length)throw new CompileError('Unclosed #if directive.',{line,column},source);
  tokens.push({kind: 'eof', value: '<eof>', line, column, offset: i}); return tokens;
}
const PRECEDENCE = {'=': 1, '+=': 1, '-=': 1, '*=': 1, '/=': 1, '%=': 1, '&=': 1, '|=': 1, '^=': 1, '<<=': 1, '>>=': 1, '||': 3, '&&': 4, '|': 5, '^': 6, '&': 7, '==': 8, '!=': 8, '<': 9, '>': 9, '<=': 9, '>=': 9, '<<': 10, '>>': 10, '+': 11, '-': 11, '*': 12, '/': 12, '%': 12};
export class Parser {
  constructor(source, defines) { this.source = source; this.tokens = tokenize(source, defines); this.i = 0; this.groupNamespaces = new Set(['cooperative_groups']); this.functionNames=new Set(['tex3D','tex1D','tex1Dfetch','tex2D','tex2Dgather','tex2DLayered','texCubemap']); this.staticTemplates=new Map();this.staticFunctions=[];this.staticCache=new Map();this.staticResolving=new Set();this.typeAliases=new Map();this.enumTypes=new Map();this.enumValues=new Map();this.typeTraits=new Map();this.structs=new Map(); this.sharedWrappers=new Map();this.expandedMacroNodes=0; }
  peek(offset = 0) { return this.tokens[this.i + offset] || this.tokens.at(-1); }
  is(value) { return this.peek().value === value; }
  take(value) { if (value && !this.is(value)) this.fail(`Expected '${value}', found '${this.peek().value}'.`); return this.tokens[this.i++]; }
  match(value) { if (this.is(value)) { this.i++; return true; } return false; }
  fail(message, token = this.peek()) { throw new CompileError(message, token, this.source); }
  name() { const t = this.take(); if (t.kind !== 'word') this.fail('Expected an identifier.', t); return t.value; }
  qualifiedName() {
    const token=this.peek(),name=this.name();
    if(!this.match('::'))return name;
    if(this.enumTypes.has(name)){const qualified=name+'::'+this.name();if(!this.enumValues.has(qualified))this.fail('Unknown scoped enum member.',token);return qualified;}
    if(!this.groupNamespaces.has(name))this.fail(`Unsupported namespace '${name}'. Only cooperative_groups namespace aliases are supported.`,token);
    return 'cooperative_groups::'+this.name();
  }
  deferredType(name){return this.deferUnsupportedTypes&&/^double[234]?$/.test(name);}
  startsType() { return this.peek().value==='double'||this.peek().value==='volatile'||this.typeAliases.has(this.peek().value)||this.structs.has(this.peek().value)||TYPES.has(this.peek().value) || this.deferredType(this.peek().value) || this.peek().value==='typename' || this.typeTraits.has(this.peek().value) || this.templateTypeNames?.has(this.peek().value) || QUALIFIERS.has(this.peek().value); }
  type({recordField=false,parameter=false}={}) {
    const volatileParameter=parameter&&this.match('volatile');
    let constant = false, shared = false,external=false;
    while (QUALIFIERS.has(this.peek().value)) { const q = this.take().value; constant ||= q === 'const'; shared ||= q === '__shared__';external ||= q==='extern'; }
    const tok = this.take(); let type;
    if(this.groupNamespaces.has(tok.value)){this.take('::');const group=this.name();if(!['thread_block','thread_group'].includes(group))this.fail('Supported cooperative group types are thread_block and thread_group.',tok);type=group==='thread_block'?'thread-block':'thread-tile';}
    else if(tok.value==='typename'||this.typeTraits.has(tok.value)){
      const name=tok.value==='typename'?this.name():tok.value;
      if(!this.typeTraits.has(name))this.fail(`Unknown type trait '${name}'.`,tok);
      this.take('<');const argument=this.name();this.take('>');this.take('::');const member=this.name();
      type={kind:'trait-type',name,argument,member};
    }else if (tok.value === 'unsigned') { if(this.match('char'))type='cw_uchar';else if(this.match('short')){this.match('int');type='cw_ushort';}else{this.match('int'); type = 'u32';} } else type = this.structs.has(tok.value)?this.structs.get(tok.value).type:this.templateTypeNames?.has(tok.value)?'template:'+tok.value:(tok.value==='double'?'cw_f64':this.typeAliases.get(tok.value)||builtinType(tok.value));
    if(tok.value==='short')this.match('int');
    if(!type&&this.deferredType(tok.value))type='unsupported:'+tok.value;
    if (!type) this.fail(`Unsupported type '${tok.value}'. Use float, int, unsigned int, bool or float2/3/4.`, tok);
    if (this.match('const')) constant = true;
    let pointer = this.match('*');
    const reference=this.match('&');
    if(pointer&&reference)this.fail('Pointer references are unsupported.');
    while (['__restrict__', '__restrict', 'restrict'].includes(this.peek().value)) this.take();
    if(pointer&&(this.structs.get(tok.value)?.valueClass||this.structs.get(tok.value)?.forward)){type='cw_objectptr_'+tok.value;pointer=!!this.match('*');(this.objectPointerTypes??=new Set()).add(type);}
    if(this.is('*'))this.fail('Only class pointer arrays support a second pointer level.');
    if(volatileParameter&&(type!=='bool'||!pointer||reference||shared||external))this.fail('Volatile kernel parameters currently require bool pointers.',tok);
    return {type, constant, shared, pointer,reference,external,...(volatileParameter?{volatileParameter:true}:{})};
  }
  zipFunctorAhead(){
    if(!this.is('struct')||this.peek(2).value!=='{')return false;
    let depth=0;for(let i=2;i<4096;i++){const t=this.peek(i);if(t.kind==='eof')return false;if(t.value==='{')depth++;if(t.value==='}'&&--depth===0)return false;if(t.value==='operator'&&this.peek(i+1).value==='('&&this.peek(i+2).value===')')return true;}return false;
  }
  zipFunctor(){
    const token=this.take('struct'),name=this.name(),fields=[];this.take('{');
    const reserved=n=>['tuple0','tuple1','tuple2','tuple3','count','cw_zip_index'].includes(n);
    while(!this.is('__host__')){
      const spec=this.type(),field=this.name();this.take(';');
      if(fields.length>=4||!['f32','i32','u32','bool','vec2<f32>','vec3<f32>','vec4<f32>','texture3d'].includes(spec.type)||spec.pointer||spec.reference||spec.constant||spec.shared||spec.external)this.fail('Zip functor state requires up to four scalar, float-vector or texture fields.',token);
      if(reserved(field)||fields.some(f=>f.name===field))this.fail('Zip functor state conflicts with launch parameters.',token);
      fields.push({...spec,name:field});
    }
    if(!fields.length)this.fail('Zip functor requires explicit value state.',token);
    this.take('__host__');this.take('__device__');this.take(name);this.take('(');const argumentsByName=new Map();
    for(let i=0;i<fields.length;i++){if(i)this.take(',');const spec=this.type(),argument=this.name();if(spec.pointer||spec.reference||argumentsByName.has(argument))this.fail('Zip constructor requires distinct value parameters.',token);argumentsByName.set(argument,spec.type);}
    this.take(')');this.take(':');const initialized=new Set(),usedArguments=new Set();
    for(let i=0;i<fields.length;i++){if(i)this.take(',');const field=this.name(),spec=fields.find(f=>f.name===field);this.take('(');const argument=this.name();this.take(')');if(!spec||initialized.has(field)||usedArguments.has(argument)||argumentsByName.get(argument)!==spec.type)this.fail('Zip constructor must initialize each field from a matching value parameter.',token);initialized.add(field);usedArguments.add(argument);}
    this.take('{');this.take('}');this.take('template');this.take('<');if(!this.match('typename'))this.take('class');const tupleType=this.name();this.take('>');this.match('__host__');this.take('__device__');this.take('void');this.take('operator');this.take('(');this.take(')');this.take('(');this.take(tupleType);const tuple=this.name();this.take(')');
    this.zipTuple=tuple;const body=this.block();this.zipTuple=null;this.take('}');this.take(';');
    const visit=(n,fn)=>{if(!n||typeof n!=='object')return;fn(n);for(const [key,value] of Object.entries(n)){if(key==='token')continue;if(Array.isArray(value))value.forEach(v=>visit(v,fn));else if(value&&typeof value==='object')visit(value,fn);}};
    const declarations=new Map(),types=new Map(),used=new Set();
    for(const f of fields)declarations.set(f.name,new Set([f.type]));
    visit(body,n=>{if(n.kind==='decl'){if(!declarations.has(n.name))declarations.set(n.name,new Set());declarations.get(n.name).add(n.type);}if(n.zipElement!==undefined)used.add(n.zipElement);});
    const requireType=(n,type)=>{if(n?.zipElement===undefined)return;if(!['f32','vec4<f32>'].includes(type))this.fail('Zip elements require an unambiguous float or float4 value type.',n.token);const previous=types.get(n.zipElement);if(previous&&previous!==type)this.fail('Zip element has conflicting value types.',n.token);types.set(n.zipElement,type);};
    const infer=n=>{if(!n)return null;if(n.zipElement!==undefined)return types.get(n.zipElement)||null;if(n.kind==='id'){const set=declarations.get(n.name);return set?.size===1?[...set][0]:null;}if(n.kind==='cast')return n.target;if(n.kind==='call')return n.callee?.name==='make_float4'?'vec4<f32>':n.callee?.name==='float'?'f32':null;return null;};
    for(let pass=0;pass<4;pass++)visit(body,n=>{if(n.kind==='decl'&&n.init?.zipElement!==undefined)requireType(n.init,n.type);if(n.kind==='assign'&&n.left?.zipElement!==undefined){const type=infer(n.right);if(type)requireType(n.left,type);}if(n.kind==='assign'&&n.right?.zipElement!==undefined){const type=infer(n.left);if(type)requireType(n.right,type);}});
    const count=used.size;if(!count||count>4||[...used].some(i=>i>=count)||[...used].some(i=>!types.has(i)))this.fail('Zip launch needs contiguous, explicitly typed float or float4 elements starting at zero.',token);
    const spelling=type=>Object.keys(MAP).find(key=>MAP[key]===type);
    const parameters=[...Array.from({length:count},(_,i)=>(types.get(i)==='f32'?'float':'float4')+'*tuple'+i),...fields.map(f=>spelling(f.type)+' '+f.name),'unsigned count'];
    const launch=new Parser('__global__ void '+name+'('+parameters.join(',')+'){unsigned cw_zip_index=blockIdx.x*blockDim.x+threadIdx.x;if(cw_zip_index>=count)return;}').parse().functions[0];
    launch.token=token;launch.body.body.push(...body.body);launch.zipFunctor=true;this.functionNames.add(name);return launch;
  }
  parse() {
    const functions = [],constantGlobals=[],sharedGlobals=[],deviceGlobals=[];
    while (this.peek().kind !== 'eof') {
      const token = this.peek();
      if(this.is('enum')){parseScopedEnum(this);continue;}
      if(this.match('static')&&!['constexpr','__constant__','__global__','__device__'].includes(this.peek().value))this.fail('Static module declarations require CUDA constant storage or a device function.',token);
      if(this.match('constexpr')){
        const spec=this.type({parameter:true}),name=this.name();
        if(spec.pointer||spec.reference||spec.constant||spec.shared||spec.external||!['cw_f64','f32','i32','u32'].includes(spec.type))this.fail('Module constexpr requires a plain numeric scalar.',token);
        if(constantGlobals.some(g=>g.name===name)||functions.some(f=>f.name===name)||deviceGlobals.some(g=>g.name===name)||sharedGlobals.some(g=>g.name===name)||this.typeAliases.has(name)||TYPES.has(name))this.fail('Duplicate module constexpr name.',token);
        this.take('=');const init=this.expression(2);this.take(';');
        const evaluated=evaluateModuleConstant(init,constantGlobals,(message,n)=>this.fail(message,n.token));
        let value=evaluated.value;
        if(spec.type==='f32')value=Math.fround(value);
        if(['i32','u32'].includes(spec.type)){value=Math.trunc(value);if(value<(spec.type==='i32'?-2147483648:0)||value>(spec.type==='i32'?2147483647:4294967295))this.fail('Module constexpr conversion is outside its integer range.',token);}
        if(!Number.isFinite(value))this.fail('Module constexpr value must be finite.',token);
        constantGlobals.push({kind:'constant-global',token,name,type:spec.type,dimensions:[],init,constexprValue:value});continue;
      }
      let templateParameter=null,templateKind=null,templateParameters=[];this.templateTypeNames=new Set();this.templateParameterName=null;this.deferUnsupportedTypes=false;
      if(this.is('class')&&this.peek(2).value===';'){this.take();const name=this.name();this.take(';');if(builtinType(name)||this.typeAliases.has(name)||this.typeTraits.has(name)||this.structs.size>=64)this.fail('Invalid or excessive forward class declaration.',token);if(!this.structs.has(name))this.structs.set(name,{name,type:'cw_struct_'+name,fields:[],methods:[],forward:true});continue;}
      if(this.is('class')){if(builtinType(this.peek(1).value))this.fail('Class name conflicts with a built-in type.');functions.push(...parseValueClass(this));continue;}
      if(this.zipFunctorAhead()){functions.push(this.zipFunctor());continue;}
      if(this.is('struct')&&this.peek(2).value==='{'){
        let depth=0,methods=false;for(let i=2;i<100000;i++){const t=this.peek(i);if(t.kind==='eof')break;if(t.value==='{')depth++;if(t.value==='}')if(--depth===0)break;if(depth===1&&t.value==='__device__')methods=true;}
        if(methods){functions.push(...parseValueClass(this));continue;}
      }
      if(this.is('typedef')&&this.peek(1).value==='struct'||this.is('struct')&&this.peek(2).value==='{'){
        const alias=this.match('typedef');this.take('struct');let name=this.is('{')?null:this.name();this.take('{');const fields=[];
        while(!this.is('}')){const fieldToken=this.peek(),spec=this.type({recordField:true}),fieldName=this.name(),dimensions=[];if(spec.pointer){if(!['f32','i32','u32','vec2<f32>','vec3<f32>','vec4<f32>'].includes(spec.type))this.fail('Pointer fields require 32-bit scalar or float-vector elements.',fieldToken);spec.pointerElement=spec.type;spec.type='cw_deviceptr_'+spec.type.replace(/[<>]/g,'_');(this.devicePointerTypes??=new Set()).add(spec.type);}if(spec.reference||spec.shared||spec.external||spec.constant||(['void','texture3d','surface2d','thread-block','cw_extent'].includes(spec.type)))this.fail('Struct fields require plain scalar/vector value types.',fieldToken);while(this.match('[')){dimensions.push(this.expression(2));this.take(']');}this.take(';');if(dimensions.length>1||fields.length>=256)this.fail('Structs support at most 256 fields and one-dimensional field arrays.',fieldToken);if(fields.some(f=>f.name===fieldName))this.fail('Duplicate struct field.',fieldToken);if(spec.pointer&&dimensions.length)this.fail('Arrays of pointer fields are unsupported.',fieldToken);fields.push({name:fieldName,type:spec.type,...(spec.pointer?{pointerElement:spec.pointerElement}:{}),dimensions,token:fieldToken});}
        this.take('}');if(alias){const aliasName=this.name();if(name&&name!==aliasName)this.fail('Distinct struct tag/typedef aliases are unsupported.',token);name=aliasName;}this.take(';');if(!name||!fields.length||this.structs.has(name)||this.typeAliases.has(name)||TYPES.has(name)||this.typeTraits.has(name))this.fail('Structs require a distinct name and at least one field.',token);if(this.structs.size>=64)this.fail('At most 64 plain structs are supported.',token);this.structs.set(name,{name,type:'cw_struct_'+name,fields,token,complete:true});continue;
      }
      if(this.is('typedef')&&this.peek(1).value!=='struct'){
        this.take('typedef');const spec=this.type(),name=this.name();this.take(';');
        if(spec.pointer||spec.reference||spec.constant||spec.shared||spec.external||!Object.values(MAP).includes(spec.type)||['void','texture3d','surface2d'].includes(spec.type))this.fail('Typedef aliases require unqualified built-in scalar or vector value types.',token);
        if(this.structs.has(name)||this.typeTraits.has(name)||this.typeAliases.has(name)||TYPES.has(name)&&(!['uint','uchar'].includes(name)||builtinType(name)!==spec.type))this.fail('Duplicate or conflicting type alias.',token);
        if(this.typeAliases.size>=128)this.fail('At most 128 type aliases are supported.',token);this.typeAliases.set(name,spec.type);continue;
      }
      if(this.is('__shared__')||this.is('extern')&&this.peek(1).value==='__shared__'){
        const declaration=this.declaration();if(declaration.kind!=='decl'||!declaration.shared||declaration.pointer||declaration.reference||declaration.constant||declaration.init)this.fail('Module shared storage requires a plain shared declaration.',token);
        if(sharedGlobals.some(g=>g.name===declaration.name))this.fail('Duplicate module shared declaration.',token);sharedGlobals.push(declaration);continue;
      }
      if(this.match('__constant__')){
        this.match('static');this.deferUnsupportedTypes=true;const valueType=this.type(),name=this.name();
        if(valueType.pointer||valueType.reference||valueType.shared||valueType.external)this.fail('Constant globals support scalar values and fixed scalar arrays only.',token);
        const dimensions=[];while(this.match('[')){dimensions.push(this.is(']')?null:this.expression(2));this.take(']');}if(dimensions.length>2||dimensions.length===2&&dimensions.includes(null))this.fail('Constant arrays support at most two fixed dimensions.',token);
        const init=this.match('=')?this.initializer():null;this.take(';');
        if(dimensions[0]===null){if(init?.kind!=='initializer'||!init.items.length)this.fail('Inferred constant arrays require a nonempty initializer.',token);dimensions[0]={kind:'literal',value:String(init.items.length),token};}
        if(constantGlobals.some(g=>g.name===name))this.fail('Duplicate constant global.',token);
        constantGlobals.push({kind:'constant-global',token,name,type:valueType.type,dimensions,init});continue;
      }
      if(this.match('extern')){const linkage=this.take();if(linkage.kind!=='string'||linkage.value!=='"C"')this.fail('Only extern "C" linkage on a single device function definition is supported.',linkage);if(!['__global__','__device__'].includes(this.peek().value))this.fail('extern "C" must precede a single __global__ or __device__ function definition; linkage blocks and templates are unsupported.');}
      if(this.match('template')){
        this.take('<');
        if(this.match('>'))templateKind='specialization';
        else{do{const kind=this.take();if(!['int','class','typename'].includes(kind.value)&&!this.enumTypes.has(kind.value))this.fail('Template parameters require int, a scoped enum, class or typename.',kind);if(this.enumTypes.has(kind.value)&&this.enumTypes.get(kind.value)!=='i32')this.fail('Enum template parameters currently require a signed 32-bit underlying type.',kind);const parameterKind=kind.value==='int'||this.enumTypes.has(kind.value)?'int':'type',name=this.name();if(TYPES.has(name)||templateParameters.includes(name))this.fail('Template parameters must have distinct names.',token);if(templateParameters.length>=4)this.fail('At most four template parameters are supported.',token);if(templateParameters.length&&templateKind!==parameterKind)this.fail('Template parameters must be all integer or all type parameters.',kind);templateKind=parameterKind;templateParameters.push(name);if(parameterKind==='type')this.templateTypeNames.add(name);}while(this.match(','));this.take('>');templateParameter=templateParameters[0];}
      }
      this.templateParameterName=templateParameter;
      if(this.match('struct')){
        if(templateParameters.length>1)this.fail('Type-trait and shared-wrapper structs require one template parameter.',token);
        if(!['type','specialization'].includes(templateKind))this.fail('Only type-trait template structs containing typedef members are supported.',token);
        const name=this.name();let argument=null;
        if(templateKind==='specialization')argument=this.templateArgument().replace(/^unsigned char$/,'uchar').replace(/^unsigned int$/,'uint');
        this.take('{');const members=[];
        if(this.is('static')||this.is('}')||this.staticTemplates.has(name)){this.staticStruct(name,argument,templateParameter,token);continue;}
        if(this.is('__device__')){
          if(templateKind!=='type')this.fail('Shared-memory conversion wrappers require one type parameter.',token);
          const conversions=[];
          while(!this.is('}')){
            this.take('__device__');while(['inline','__inline__','__forceinline__'].includes(this.peek().value))this.take();
            this.take('operator');const constant=this.match('const');this.take(templateParameter);this.take('*');this.take('(');this.take(')');const methodConst=this.match('const');
            this.take('{');this.take('extern');this.take('__shared__');const storage=this.type();const storageName=this.name();this.take('[');this.take(']');this.take(';');
            if(!['i32','u32','f32'].includes(storage.type)||storage.pointer||storage.reference||storage.constant)this.fail('Shared wrapper backing storage must be an unsized 32-bit scalar array.',token);
            this.take('return');this.take('(');this.take(templateParameter);this.take('*');this.take(')');this.take(storageName);this.take(';');this.take('}');
            if(constant!==methodConst||conversions.includes(constant))this.fail('Shared wrapper conversions require distinct mutable and const overloads.',token);conversions.push(constant);
          }
          this.take('}');this.take(';');if(this.sharedWrappers.has(name)||this.typeTraits.has(name)||TYPES.has(name))this.fail('Duplicate or reserved shared wrapper name.',token);
          this.sharedWrappers.set(name,{name,conversions});continue;
        }
        while(!this.is('}')){
          this.take('typedef');let type=this.name();if(type==='unsigned'){this.match('int');type='uint';}
          const member=this.name();this.take(';');if(members.some(m=>m.name===member))this.fail('Duplicate type-trait member.',token);members.push({name:member,type});
        }
        this.take('}');this.take(';');
        if(!members.length)this.fail('Type traits require at least one typedef member.',token);
        if(argument===null){if(this.typeTraits.has(name)||TYPES.has(name))this.fail('Duplicate or reserved type-trait name.',token);this.typeTraits.set(name,{name,parameter:templateParameter,members,specializations:[]});}
        else{const trait=this.typeTraits.get(name);if(!trait)this.fail('Declare the primary type trait before its specializations.',token);if(trait.specializations.some(s=>s.argument===argument))this.fail('Duplicate type-trait specialization.',token);trait.specializations.push({argument,members});}
        continue;
      }
      this.deferUnsupportedTypes=templateKind==='specialization';
      if(this.match('namespace')){const alias=this.name();this.take('=');const target=this.name();this.take(';');if(target!=='cooperative_groups'||this.groupNamespaces.has(alias))this.fail('Only distinct aliases of cooperative_groups are supported.',token);this.groupNamespaces.add(alias);continue;}
      let hostQualified=false;while (['static','inline','__inline__', '__forceinline__','__host__'].includes(this.peek().value)) {if(this.take().value==='__host__')hostQualified=true;}
      let launchThreads=null;
      const launchBounds=()=>{this.take('__launch_bounds__');this.take('(');const t=this.take();if(t.kind!=='number'||!/^[0-9]+[uU]?$/.test(t.value))this.fail('Launch bounds require a positive integer thread count.',t);launchThreads=Number(t.value.replace(/[uU]$/,''));if(launchThreads<1||launchThreads>1024)this.fail('Launch bounds thread count must be in [1,1024].',t);this.take(')');};
      if(this.is('__launch_bounds__'))launchBounds();
      const qualifier = this.take().value;
      if(templateKind==='specialization'&&qualifier!=='__device__')this.fail('Explicit function specializations support only device helpers.',token);
      if(templateParameters.length>1&&qualifier!=='__device__'&&templateKind!=='int')this.fail('Multiple template type parameters are supported on device helpers only.',token);
      if(templateParameter&&!['__global__','__device__'].includes(qualifier))this.fail('Templates are supported only on kernels and device helpers.',token);
      if (!['__global__', '__device__'].includes(qualifier)) this.fail('Only __global__ kernels and __device__ helper functions are accepted. Host CUDA APIs, structs, templates and PTX are not supported.', token);
      while (['static','inline','__inline__', '__forceinline__','__host__'].includes(this.peek().value)) {if(this.take().value==='__host__')hostQualified=true;}
      if(hostQualified&&qualifier!=='__device__')this.fail('__host__ is supported only alongside __device__ helpers.',token);
      if(this.is('__launch_bounds__')){if(launchThreads!==null)this.fail('Duplicate launch bounds.');launchBounds();}
      if(launchThreads!==null&&qualifier!=='__global__')this.fail('Launch bounds apply only to kernels.',token);
      while(this.match('__declspec')){this.take('(');this.take('noinline');this.take(')');}
      const result = this.type();
      if(this.structs.has(this.peek().value)&&this.peek(1).value==='::'){parseExternalValueMethod(this,result,qualifier,token);continue;}
      if (result.pointer || result.shared || result.reference || result.external || result.type==='cw_extent') this.fail('Function return pointers/references/shared/extern qualifiers are unsupported.');
      if(this.peek().forward||this.peek().expressionMacro)this.fail('Function-like macros are supported at call sites, not in function declarations.');
      let name = this.name(),freeOperator=null;
      if(name==='operator'){freeOperator=this.take().value;if(qualifier!=='__device__'||templateKind||!['+','-','*','/','+=','-=','*=','/='].includes(freeOperator))this.fail('Free operators support non-template device binary arithmetic.',token);name=(freeOperator.endsWith('=')?'cw_compound_':'cw_binary_')+{'+':'add','-':'subtract','*':'multiply','/':'divide'}[freeOperator[0]];}
      if(qualifier==='__device__'&&this.is('[')){
        if(templateKind||hostQualified||result.constant||!['f32','i32','u32'].includes(result.type))this.fail('Device globals require non-template mutable 32-bit scalar arrays.',token);
        this.take('[');const length=this.expression(2);this.take(']');this.take(';');
        if(deviceGlobals.some(g=>g.name===name)||constantGlobals.some(g=>g.name===name)||sharedGlobals.some(g=>g.name===name))this.fail('Duplicate global storage name.',token);
        deviceGlobals.push({kind:'device-global',name,type:result.type,length,token});continue;
      }
      if(this.typeAliases.has(name))this.fail('Functions cannot shadow a type alias.',token);this.functionNames.add(name);let specializationArgument;
      if(templateKind==='specialization')specializationArgument=this.templateArgument();
      this.take('('); const params = [];
      if (!this.is(')')) do { const token = this.peek(), type = this.type({parameter:true}), name = this.name();if(this.match('[')){const size=this.take();if(size.kind!=='number'||!/^\d+[uU]?$/.test(size.value)||Number(size.value.replace(/[uU]$/,''))<1||Number(size.value.replace(/[uU]$/,''))>65536||type.pointer||type.reference)this.fail('Array parameters require one positive fixed dimension.',size);this.take(']');type.pointer=true;}if(type.type==='cw_f64'&&(type.pointer||type.reference))this.fail('Unsupported type: double pointers and references require a storage ABI.',token);const defaultValue=this.match('=')?this.expression(2):undefined;
        if(this.typeAliases.has(name))this.fail('Parameters cannot shadow a type alias.',token);if(defaultValue!==undefined){const literal=defaultValue.kind==='unary'&&['+','-'].includes(defaultValue.op)?defaultValue.value:defaultValue;if(!['__device__','__global__'].includes(qualifier)||templateKind==='specialization')this.fail('Default arguments belong on primary device or kernel definitions only.',token);if(type.pointer||type.reference||(!['f32','i32','u32','bool','cw_uchar'].includes(type.type)&&!String(type.type).startsWith('template:')))this.fail('Default arguments require scalar value parameters.',token);if(literal.kind!=='literal'&&!(literal===defaultValue&&literal.kind==='id'&&['true','false'].includes(literal.name)))this.fail('Default arguments support numeric or boolean literals with an optional numeric sign.',defaultValue.token);}
        else if(params.some(p=>p.defaultValue!==undefined))this.fail('Parameters after a default argument must also have defaults.',token);
        params.push({kind: 'param', token, name, ...type,...(defaultValue!==undefined?{defaultValue}:{})});
      } while (this.match(','));
      this.take(')');
      if(['+','-'].includes(freeOperator)&&params.length===1){if(params[0].pointer||!String(params[0].type).startsWith('cw_struct_'))this.fail('Free unary operators require a class value or reference.',token);name='cw_unary_'+(freeOperator==='+'?'plus':'minus');this.functionNames.add(name);}
      else if(freeOperator?.endsWith('=')){if(result.type!=='void'||params.length!==2||!params[0].reference||params[0].constant||!String(params[0].type).startsWith('cw_struct_')||params.some(p=>p.pointer)||params[1].reference&&!params[1].constant)this.fail('Compound operators require void return, a mutable class reference and a value or const-reference operand.',token);}
      else if(freeOperator&&(params.length!==2||!params.some(p=>String(p.type).startsWith('cw_struct_')||/^vec[234]</.test(p.type))||params.some(p=>p.pointer||p.reference&&!p.constant)))this.fail('Free binary operators require two value or const-reference parameters, including a class or vector value.',token);
      const body = this.block();
      functions.push({kind: 'function', token, name, qualifier, result: result.type, params, body,freeOperator,launchThreads,templateParameter,templateParameters,templateKind,...(specializationArgument!==undefined?{specializationArgument}:{})});
    }
    finishValueClasses(this);
    if (!functions.some(f => f.qualifier === '__global__')) this.fail('No __global__ kernel was found.');
    for(const g of deviceGlobals)if(constantGlobals.some(c=>c.name===g.name)||sharedGlobals.some(c=>c.name===g.name)||functions.some(f=>f.name===g.name))this.fail('Duplicate global storage name.',g.token);
    return {kind: 'module', functions:functions.concat(this.staticFunctions), constantGlobals,sharedGlobals,deviceGlobals,typeAliases:Object.fromEntries(this.typeAliases),enumValues:Object.fromEntries(this.enumValues),devicePointerTypes:[...(this.devicePointerTypes||[])],objectListTypes:[...(this.objectListTypes||[])],bufferReferenceTypes:[...(this.bufferReferenceTypes||[])].map(([type,element])=>({type,element})),interfaces:[...this.structs.values()].filter(s=>s.interfaceOnly),objectPointerTypes:[...(this.objectPointerTypes||[])],structs:[...this.structs.values()].filter(s=>!s.forward&&!s.interfaceOnly),typeTraits:[...this.typeTraits.values()], source: this.source};
  }
  staticStruct(name,argument,parameter,token){
    let owner=this.staticTemplates.get(name);
    if(argument===null){if(owner||this.typeTraits.has(name)||this.structs.has(name)||this.typeAliases.has(name)||TYPES.has(name))this.fail('Duplicate or reserved static template name.',token);owner={parameter,primary:null,specializations:new Map()};this.staticTemplates.set(name,owner);}
    else if(!owner)this.fail('Declare the primary static template before its specializations.',token);
    if(argument!==null&&[...this.staticCache.keys()].some(key=>key.startsWith(name+'<'+argument+'>::')))this.fail('Static specializations must precede their first use.',token);
    if(argument!==null&&owner.specializations.has(argument))this.fail('Duplicate static template specialization.',token);
    const methods=new Map(),constants=new Set();
    while(!this.is('}')){
      const start=this.i;this.take('static');
      while(!['{',';','<eof>'].includes(this.peek().value)&&this.peek().kind!=='eof')this.take();
      if(this.peek().kind==='eof')this.fail('Unclosed static template member.',token);
      if(this.match(';')){
        const header=this.tokens.slice(start,this.i-1),equal=header.findIndex(t=>t.value==='=');
        if(header[1]?.value!=='const'||equal<3||header.some(t=>['(',')','[',']'].includes(t.value)))this.fail('Static template data members require a simple const declaration.',header[0]);
        const constant=header[equal-1].value;if(constants.has(constant)||methods.has(constant))this.fail('Duplicate static template member.',header[0]);constants.add(constant);continue;
      }
      const header=this.tokens.slice(start,this.i),open=header.findIndex(t=>t.value==='('),method=header[open-1]?.value;
      if(open<1||!header.some(t=>t.value==='__device__')||!method||methods.has(method)||constants.has(method))this.fail('Static templates require distinct static device method definitions.',header[0]);
      this.take('{');let depth=1;while(depth){const t=this.take();if(t.kind==='eof')this.fail('Unclosed static device method.',token);if(t.value==='{')depth++;if(t.value==='}')depth--;}
      methods.set(method,{tokens:this.tokens.slice(start,this.i),token:header[0]});
      if(methods.size>64)this.fail('At most 64 static methods per template are supported.',token);
    }
    this.take('}');this.take(';');const record={methods,constants,aliases:new Map(this.typeAliases)};
    if(argument===null)owner.primary=record;else owner.specializations.set(argument,record);
  }
  staticMethod(name,argument,method,token){
    argument=argument.replace(/^unsigned char$/,'uchar').replace(/^unsigned int$/,'uint');
    const type=builtinType(argument);if(!type||['void','texture3d','surface2d'].includes(type))this.fail('Static method calls require an explicit supported built-in type argument.',token);
    const owner=this.staticTemplates.get(name),selected=owner.specializations.get(argument)||owner.primary,record=selected?.methods.get(method),key=name+'<'+argument+'>::'+method;
    if(!record)this.fail('No static device method '+key+'. Static data member access is unsupported.',token);
    if(this.staticResolving.has(key))this.fail('Recursive static device methods are unsupported.',token);
    if(this.staticCache.has(key))return this.staticCache.get(key);
    if(this.staticCache.size+this.staticResolving.size>=128)this.fail('At most 128 static method instances are supported.',token);
    this.staticResolving.add(key);
    const parser=Object.assign(Object.create(Parser.prototype),this,{tokens:[...record.tokens,{kind:'eof',value:'<eof>',line:token.line,column:token.column}],i:0,staticMemberNames:new Set([...selected.methods.keys(),...selected.constants]),typeAliases:new Map(selected.aliases),templateTypeNames:new Set(),deferUnsupportedTypes:false});
    if(!owner.specializations.has(argument))parser.typeAliases.set(owner.parameter,type);
    while(['static','inline','__inline__','__forceinline__','__device__'].includes(parser.peek().value))parser.take();
    const result=parser.type();if(result.pointer||result.reference||result.shared||result.external)parser.fail('Static method return values cannot be pointers or references.',token);parser.take(method);parser.take('(');const params=[];
    if(!parser.is(')'))do{const t=parser.peek(),spec=parser.type(),paramName=parser.name();if(parser.typeAliases.has(paramName))parser.fail('Parameters cannot shadow a type alias.',t);params.push({kind:'param',token:t,name:paramName,...spec});}while(parser.match(','));
    parser.take(')');const body=parser.block();if(parser.peek().kind!=='eof')parser.fail('Unexpected static method suffix.');
    const generated='cw_static_method_'+this.staticFunctions.length;
    this.staticFunctions.push({kind:'function',token:record.token,name:generated,qualifier:'__device__',result:result.type,params,body,templateParameter:null,templateParameters:[],templateKind:null});
    this.staticResolving.delete(key);this.staticCache.set(key,generated);return generated;
  }
  block() { const token = this.take('{'), body = []; while (!this.is('}')) { if (this.peek().kind === 'eof') this.fail('Unclosed block.'); body.push(this.statement()); } this.take('}'); return {kind: 'block', token, body}; }
  initializer(){
    if(this.sharedWrappers.has(this.peek().value)){
      const token=this.take(),wrapper=this.sharedWrappers.get(token.value);this.take('<');const type=this.type();this.take('>');this.take('(');this.take(')');
      if(type.pointer||type.reference||type.shared||type.external||type.constant)this.fail('Shared wrapper arguments must be value types.',token);
      return {kind:'shared-conversion',token,target:type.type,conversions:wrapper.conversions};
    }
    if(!this.is('{'))return this.expression(2);
    const token=this.take('{'),items=[];
    while(!this.is('}')){items.push(this.initializer());if(!this.match(','))break;}
    this.take('}');return {kind:'initializer',token,items};
  }
  declaration(semicolon = true) {
    const token = this.peek(),volatileSnapshot=this.match('volatile'),d = this.type(),declarations=[];
    do {const name=this.name(),dimensions=[];if(this.zipTuple&&['tuple0','tuple1','tuple2','tuple3','count','cw_zip_index'].includes(name))this.fail('Zip functor local name conflicts with generated launch storage.',token);if(this.typeAliases.has(name))this.fail('Value declarations cannot shadow a type alias.',token);while(this.match('[')){dimensions.push(this.is(']')?null:this.expression(2));this.take(']');}let init=this.match('=')?this.initializer():null;const record=[...this.structs.values()].find(s=>s.type===d.type&&s.valueClass);if(record&&!d.pointer&&!d.reference&&!dimensions.length&&!init){const args=[];if(this.match('(')){if(!this.is(')'))do{args.push(this.expression(2));}while(this.match(','));this.take(')');}if(record.constructors.length)init={kind:'call',token,callee:{kind:'id',name:'cw_ctor_'+record.name,token},args};else if(args.length)this.fail('No value-class constructor is declared.',token);}const volatileShared=volatileSnapshot&&d.shared&&!d.pointer&&!d.reference&&!d.external&&!d.constant&&!init&&['f32','i32','u32'].includes(d.type);const volatilePointer=volatileSnapshot&&d.pointer&&!d.shared&&!d.external&&!d.reference&&dimensions.length===1&&!init&&['i32','u32'].includes(d.type);if(volatileSnapshot&&!volatileShared&&!volatilePointer&&(d.pointer||d.reference||d.shared||d.external||dimensions.length||!init))this.fail('Volatile is supported only on initialized local value snapshots.',token);declarations.push({kind:'decl',token,name,...d,...(volatilePointer?{volatilePointer:true}:volatileShared?{volatileShared:true}:volatileSnapshot?{constant:true,volatileSnapshot:true}:{}),dimensions,init});if(this.is(',')&&(d.pointer||d.reference))this.fail('Pointer/reference declaration lists are unsupported.');}while(this.match(','));
    if (semicolon) this.take(';');return declarations.length===1?declarations[0]:{kind:'decls',token,declarations};
  }
  statement() {
    const token = this.peek();
    if(token.kind==='pragma-unroll'){this.take();const loop=this.statement();if(!['for','while','do'].includes(loop.kind))this.fail('#pragma unroll must precede a loop.',token);loop.unroll=token.factor;return loop;}
    if(this.match('asm')||this.match('__asm__')){
      this.match('volatile');this.take('(');let instruction='';while(this.peek().kind==='string')instruction+=this.take().value.slice(1,-1);
      if(!/^\s*vabsdiff4\.u32\.u32\.u32\.add\s+%0\s*,\s*%1\s*,\s*%2\s*,\s*%3\s*;\s*$/.test(instruction))this.fail('Inline PTX supports only vabsdiff4.u32.u32.u32.add with four positional registers.',token);
      this.take(':');this.take('"=r"');this.take('(');const left=this.expression();this.take(')');this.take(':');const args=[];
      for(let i=0;i<3;i++){if(i)this.take(',');this.take('"r"');this.take('(');args.push(this.expression());this.take(')');}
      this.take(')');this.take(';');if(left.kind!=='id')this.fail('PTX output requires a named 32-bit integer register.',token);return {kind:'expr',token,value:{kind:'assign',op:'=',token,left,right:{kind:'ptx-sad4',token,args,outputName:left.name}}};
    }
    if(this.match('do')){const body=this.statement();this.take('while');this.take('(');const condition=this.expression();this.take(')');this.take(';');return {kind:'do',token,body,condition};}
    if(this.groupNamespaces.has(token.value)&&this.peek(1).value==='::'&&this.peek(2).value==='thread_block_tile'){
      this.qualifiedName();this.take('<');this.take('32');this.take('>');const name=this.name();this.take('=');const factory=this.qualifiedName();this.take('<');this.take('32');this.take('>');this.take('(');const parent=this.name();this.take(')');this.take(';');if(factory!=='cooperative_groups::tiled_partition')this.fail('Static tiles require tiled_partition<32>(block).',token);return {kind:'thread-warp',token,name,parent};
    }
    if(this.groupNamespaces.has(token.value)&&this.peek(1).value==='::'&&this.peek(2).value==='thread_group'){
      this.qualifiedName();const name=this.name();this.take('=');const factory=this.qualifiedName();this.take('(');const parent=this.name();this.take(',');const size=this.expression(2);this.take(')');this.take(';');if(factory!=='cooperative_groups::tiled_partition')this.fail('thread_group requires tiled_partition(block, size).',token);return {kind:'thread-tile',token,name,parent,size};
    }
    if(this.groupNamespaces.has(token.value)&&this.peek(1).value==='::'&&this.peek(2).value==='thread_block'){
      this.qualifiedName();const name=this.name();this.take('=');const factory=this.qualifiedName();this.take('(');this.take(')');this.take(';');
      if(factory!=='cooperative_groups::this_thread_block')this.fail('thread_block must be initialized with cooperative_groups::this_thread_block().',token);
      return {kind:'thread-block',token,name};
    }
    if (this.is('{')) return this.block();
    if (this.match(';')) return {kind: 'empty', token};
    if (this.match('if')) { const constexpr = this.match('constexpr'); this.take('('); const condition = this.expression(); this.take(')'); const yes = this.statement(), no = this.match('else') ? this.statement() : null; return {kind: 'if', token, condition, yes, no, constexpr}; }
    if(this.match('switch')){
      this.take('(');const selector=this.expression();this.take(')');this.take('{');const cases=[];let current;
      while(!this.is('}')){
        if(this.peek().kind==='eof')this.fail('Unclosed switch.',token);
        if(this.is('case')||this.is('default')){
          const label=this.take(),value=label.value==='case'?this.expression(2):null;this.take(':');
          if(value===null&&cases.some(c=>c.value===null))this.fail('Duplicate switch default.',label);
          if(cases.length>=64)this.fail('Switch supports at most 64 labels.',label);
          current={kind:'case',token:label,value,body:[]};cases.push(current);
        }else{
          if(!current)this.fail('Switch statements must follow a case or default label.',this.peek());
          const statement=this.statement();if(['decl','decls'].includes(statement.kind))this.fail('Enclose switch case declarations in braces.',statement.token);
          current.body.push(statement);
        }
      }
      this.take('}');return {kind:'switch',token,selector,cases};
    }
    if (this.match('for')) { this.take('('); const init = this.is(';') ? null : this.startsType() ? this.declaration(false) : this.expression(); this.take(';'); const condition = this.is(';') ? null : this.expression(); this.take(';'); const steps=[];if(!this.is(')'))do{steps.push(this.expression());}while(this.match(','));const step=steps.length>1?{kind:'sequence',token,expressions:steps}:steps[0]||null; this.take(')'); return {kind: 'for', token, init, condition, step, body: this.statement()}; }
    if (this.match('while')) { this.take('('); const condition = this.expression(); this.take(')'); return {kind: 'while', token, condition, body: this.statement()}; }
    if (this.match('return')) { const value = this.is(';') ? null : this.expression(); this.take(';'); return {kind: 'return', token, value}; }
    if (this.match('break') || this.match('continue')) { this.take(';'); return {kind: token.value, token}; }
    if (this.startsType()) return this.declaration();
    const value = this.expression(); this.take(';'); return {kind: 'expr', token, value};
  }
  expression(min = 1) {
    let left = this.unary();
    while (true) {
      const token = this.peek(), op = token.value;
      if (op === '?' && min <= 2) { this.take(); const yes = this.expression(); this.take(':'); const no = this.expression(2); left = {kind: 'conditional', token, condition: left, yes, no}; continue; }
      const p = PRECEDENCE[op]; if (p === undefined || p < min) break;
      this.take(); const right = this.expression(p === 1 ? p : p + 1); left = {kind: p === 1 ? 'assign' : 'binary', token, op, left, right};
    }
    return left;
  }
  expandExpressionMacro(macro,args,token){
    if(args.length!==macro.params.length)this.fail('Wrong argument count for expression macro '+macro.name,token);
    const stack=this.expressionMacroStack||[];if(stack.includes(macro.name)||stack.length>=8)this.fail('Recursive or excessively deep expression macro expansion is unsupported.',token);
    const parser=new Parser([...Object.entries(macro.defines).map(([name,value])=>'#define '+name+' '+value),...(macro.definitions||[]),macro.body].join('\n'));parser.functionNames=new Set(this.functionNames);parser.typeAliases=new Map(this.typeAliases);parser.expressionMacroStack=[...stack,macro.name];
    const expression=parser.expression();if(parser.peek().kind!=='eof')this.fail('Expression macro must contain one expression.',token);
    const copy=(node,substitute)=>{if(!node||typeof node!=='object')return node;if(node.kind&&++this.expandedMacroNodes>65536)this.fail('Expression macro expansion exceeds 65,536 AST nodes.',token);if(substitute&&node.kind==='id'&&macro.params.includes(node.name))return copy(args[macro.params.indexOf(node.name)],false);if(Array.isArray(node))return node.map(n=>copy(n,substitute));return Object.fromEntries(Object.entries(node).map(([key,value])=>[key,key==='token'?(substitute?token:value):copy(value,substitute)]));};return copy(expression,true);
  }
  templateCallAhead(){for(let offset=1;offset<=65;offset++){const token=this.peek(offset);if(token.value==='>')return ['(','<<<'].includes(this.peek(offset+1).value);if(!['word','number'].includes(token.kind)&&!['+','-','*','/','%','(',')',',','::'].includes(token.value))return false;}return false;}
  templateArgument(){this.take('<');const parts=[];while(!this.is('>')){const token=this.peek();if(parts.length>=64||(!['word','number'].includes(token.kind)&&!['+','-','*','/','%','(',')',','].includes(token.value)))this.fail('Template arguments support type lists or bounded integer arithmetic.',token);if(this.enumTypes.has(token.value)&&this.peek(1).value==='::'){const name=this.qualifiedName();parts.push(this.enumValues.get(name).value);continue;}const word=this.take().value;parts.push(this.typeAliases.has(word)?Object.keys(MAP).find(k=>MAP[k]===this.typeAliases.get(word)):word);}this.take('>');if(!parts.length)this.fail('Missing template argument.');return parts.join(' ');}
  unary() {
    if(this.is('new')){const token=this.take(),name=this.name(),record=this.structs.get(name);if(!record?.complete||record.interfaceOnly)this.fail('new requires a complete concrete value class.',token);this.take('(');const args=[];if(!this.is(')'))do{args.push(this.expression(2));}while(this.match(','));this.take(')');const pointerType='cw_objectptr_'+name;(this.objectPointerTypes??=new Set()).add(pointerType);return {kind:'object-new',token,name,pointerType,args};}
    if(this.is('delete')){const token=this.take();return {kind:'object-delete',token,value:this.unary()};}

    const token = this.peek();
    if(this.match('sizeof')){this.take('(');const spec=this.type();if(spec.pointer||spec.reference||spec.shared||spec.external)this.fail('sizeof supports built-in value types only.',token);this.take(')');return {kind:'sizeof',token,target:spec.type};}
    if(this.match('static_cast')){this.take('<');const type=this.type();if(type.pointer||type.reference||type.shared||type.external)this.fail('static_cast supports value types only.',token);this.take('>');this.take('(');const value=this.expression();this.take(')');return {kind:'cast',token,target:type.type,value};}
    if (['+', '-', '!', '~', '&', '++', '--', '*'].includes(token.value)) { this.take(); return {kind: 'unary', token, op: token.value, value: this.unary(), prefix: true}; }
    if(this.is('(')&&this.peek(1).value==='void'&&this.peek(2).value==='*'&&this.peek(3).value==='*'&&this.peek(4).value===')'){for(let i=0;i<5;i++)this.take();return {kind:'allocation-output',token,value:this.unary()};}
    if(this.is('(')&&this.peek(1).value==='char'&&this.peek(2).value==='*'&&this.peek(3).value===')'){this.take('(');this.take('char');this.take('*');this.take(')');return {kind:'pointer-cast',token,target:'byte-address',constant:false,value:this.unary()};}
    if (this.is('(') && this.peek(2).value!=='(' && (this.peek(1).value==='double'||this.structs.has(this.peek(1).value)||this.typeAliases.has(this.peek(1).value)||TYPES.has(this.peek(1).value) || this.deferredType(this.peek(1).value) || this.peek(1).value==='typename' || this.typeTraits.has(this.peek(1).value) || this.templateTypeNames?.has(this.peek(1).value) || this.peek(1).value === 'const' || this.peek(1).value === 'volatile')) { this.take('('); const volatilePointer=this.match('volatile'); const type = this.type(); if(volatilePointer&&(!type.pointer||!['i32','u32'].includes(type.type)))this.fail('Volatile casts require integer pointers.',token); if(type.reference)this.fail('Reference casts are unsupported.'); this.take(')'); return {kind:type.pointer?'pointer-cast':'cast',token,target:type.type,constant:type.constant,volatilePointer,value:this.unary()}; }
    let value;
    if (token.kind === 'number') { this.take(); value = {kind: 'literal', token, value: token.value}; }
    else if(token.kind==='string'){this.take();value={kind:'string',token,value:token.value};}
    else if (this.match('(')) { value = this.expression(); this.take(')'); }
    else if(this.zipTuple&&token.value==='cuda'){
      this.take('cuda');this.take('::');this.take('std');this.take('::');this.take('get');this.take('<');const element=this.take();if(!['0','1','2','3'].includes(element.value))this.fail('Zip functors support up to four typed tuple elements.',element);this.take('>');this.take('(');this.take(this.zipTuple);this.take(')');value={kind:'index',token,zipElement:Number(element.value),base:{kind:'id',token,name:'tuple'+element.value},index:{kind:'id',token,name:'cw_zip_index'}};
    }
    else if (token.kind === 'word') { const name=this.qualifiedName();if(this.zipTuple&&['tuple0','tuple1','tuple2','tuple3','count','cw_zip_index'].includes(name))this.fail('Zip functor name conflicts with generated launch storage.',token);if(this.staticMemberNames?.has(name))this.fail('Unqualified static member references require explicit template qualification.',token);value = this.enumValues.has(name)?{kind:'literal',token,value:this.enumValues.get(name).value}:{kind: 'id', token, name}; }
    else this.fail('Expected an expression.', token);
    while (true) {
      if(value.kind==='id'&&this.staticTemplates.has(value.name)&&this.is('<')){const argument=this.templateArgument();this.take('::');const method=this.name();if(!this.is('('))this.fail('Static data member access is unsupported.',token);value={...value,name:this.staticMethod(value.name,argument,method,token)};}
      else if(value.kind==='id'&&this.functionNames.has(value.name)&&this.is('<')&&this.templateCallAhead())value.templateArgument=this.templateArgument();
      else if(value.kind==='id'&&this.match('<<<')){const configuration=[];do{configuration.push(this.expression(2));}while(this.match(','));this.take('>>>');if(configuration.length<2||configuration.length>4)this.fail('Device launches require grid, block, optional shared bytes and stream.',token);this.take('(');const args=[];if(!this.is(')'))do{args.push(this.expression(2));}while(this.match(','));this.take(')');value={kind:'device-launch',token,callee:value,configuration,args};}
      else if (this.match('[')) { const index = this.expression(); this.take(']'); value = {kind: 'index', token, base: value, index}; }
      else if(this.match('->')){value={kind:'member',token,base:{kind:'object-deref',token,value},member:this.name()};}
      else if (this.match('.')) { value = {kind: 'member', token, base: value, member: this.name()}; }
      else if (this.match('(')) { const args = []; if (!this.is(')')) do { args.push(this.expression(2)); } while (this.match(',')); this.take(')');
        if(value.kind==='id'&&value.token.forward){const f=value.token.forward;if(f.tooDeep)this.fail('Forwarding macro chains are limited to 32 calls.',value.token);if(f.recursive||f.numericTarget)this.fail('Recursive or non-function forwarding macro target is unsupported.',value.token);if(f.chain.some(m=>m.arity!==args.length))this.fail(`Wrong argument count for forwarding macro '${value.name}'.`,value.token);value={...value,name:f.target};}
        value = value.kind==='id'&&value.token.expressionMacro?this.expandExpressionMacro(value.token.expressionMacro,args,value.token):{kind: 'call', token, callee: value, args,...(value.kind==='id'&&this.typeAliases.has(value.name)?{aliasType:this.typeAliases.get(value.name)}:{})}; }
      else if (this.is('++') || this.is('--')) { value = {kind: 'unary', token, op: this.take().value, value, prefix: false}; }
      else break;
    }
    return value;
  }
}
export function parse(source, options = {}) {
 const libraries=options.libraries||[];
 if(!Array.isArray(libraries)||libraries.some(n=>n!=='curand-xorwow')||new Set(libraries).size!==libraries.length)throw new CompileError('Supported libraries: curand-xorwow (once).');
 const parser=new Parser(source,options.defines);
 if(libraries.includes('curand-xorwow'))parser.tokens=[...tokenize(CURAND_XORWOW_SOURCE).filter(t=>t.kind!=='eof').map(t=>({...t,library:'curand-xorwow'})),...parser.tokens];
 return parser.parse();
}
