/**
 * Bounded, per-entry-point specialization for the plain CUDA value subset.
 *
 * A helper parameter is constant only when ALL reachable call sites agree and
 * it is never written or addressed. A record field is constant only in a closed
 * by-value world: every construction initializes it, every write agrees, and no
 * pointer/reference/global/kernel input can supply or mutate such a record.
 *
 * We prune control flow, not arithmetic expressions. In particular this does
 * not turn runtime integer operations into overflowing WGSL const expressions,
 * approximate floats, freeze uniforms, or clone functions. Mixed modes remain
 * generic. The caller validates the original program BEFORE invoking this pass.
 */
const integerTypes = new Set(['i32', 'u32', 'bool']);
const excludedKeys = new Set(['token', 'source', 'symbol', 'type', 'resolved', 'pointerBaseSymbol']);
function walk(n, visit, parent = null, key = null) {
  if (!n || typeof n !== 'object') return;
  if (n.kind) visit(n, parent, key);
  for (const [k, v] of Object.entries(n)) if (!excludedKeys.has(k)) {
    if (Array.isArray(v)) v.forEach(x => walk(x, visit, n, k));
    else if (v && typeof v === 'object') walk(v, visit, n, k);
  }
}
const equal = (a, b) => !!a && !!b && a.type === b.type && a.value === b.value;
function converted(c, type) {
  if (!c || !integerTypes.has(type)) return null;
  if (type === 'bool') return {type, value: Number(c.value !== 0)};
  // No signed-overflow or implementation-defined narrowing assumptions.
  if (type === 'i32' && (c.value < -2147483648 || c.value > 2147483647)) return null;
  return {type, value: type === 'u32' ? c.value >>> 0 : c.value};
}
function literal(n) {
  if (n?.kind === 'id' && ['true', 'false'].includes(n.name)) return {type: 'bool', value: Number(n.name === 'true')};
  if (n?.kind !== 'literal') return null;
  if (['true', 'false'].includes(n.value)) return {type: 'bool', value: Number(n.value === 'true')};
  if (!/^-?(?:0[xX][0-9a-fA-F]+|\d+)[uU]?$/.test(n.value)) return null;
  const u = /[uU]$/.test(n.value), value = Number(n.value.replace(/[uU]$/, ''));
  if (!Number.isInteger(value) || value < -2147483648 || value > (u ? 4294967295 : 2147483647)) return null;
  return {type: u ? 'u32' : 'i32', value};
}
function binary(op, a, b) {
  if (!a || !b) return null;
  const type = a.type === 'u32' || b.type === 'u32' ? 'u32' : 'i32';
  a = converted(a, type); b = converted(b, type); if (!a || !b) return null;
  const x = a.value, y = b.value;
  const tests = {'==': x === y, '!=': x !== y, '<': x < y, '<=': x <= y, '>': x > y, '>=': x >= y};
  if (Object.hasOwn(tests, op)) return {type: 'bool', value: Number(tests[op])};
  let v;
  switch (op) {
    case '+': v = BigInt(x) + BigInt(y); break;
    case '-': v = BigInt(x) - BigInt(y); break;
    case '*': v = BigInt(x) * BigInt(y); break;
    case '/': if (!y || type === 'i32' && x === -2147483648 && y === -1) return null; v = BigInt(x) / BigInt(y); break;
    case '%': if (!y || type === 'i32' && x === -2147483648 && y === -1) return null; v = BigInt(x) % BigInt(y); break;
    case '&': v = BigInt(x) & BigInt(y); break;
    case '|': v = BigInt(x) | BigInt(y); break;
    case '^': v = BigInt(x) ^ BigInt(y); break;
    default: return null; // Shifts, float promotion, and other operators are not folded.
  }
  if (type === 'u32') return {type, value: Number(BigInt.asUintN(32, v))};
  return v >= -2147483648n && v <= 2147483647n ? {type, value: Number(v)} : null;
}

function indexFunctions(functions) {
  const info = new Map();
  for (const fn of functions) {
    const ids = new WeakMap(), parent = new WeakMap(), bindings = [], calls = [], nodes = [];
    let scopes = [new Map()];
    const resolve = name => { for (let i = scopes.length - 1; i >= 0; i--) if (scopes[i].has(name)) return scopes[i].get(name); return null; };
    const bind = (n, parameter = false) => {
      const b = {node: n, fn, type: n.type, parameter, refs: [], written: false, escaped: false};
      scopes.at(-1).set(n.name, b); bindings.push(b); return b;
    };
    fn.params.forEach(p => bind(p, true));
    const expr = (n, p = null, key = null) => {
      if (!n || typeof n !== 'object') return;
      if (n.kind) { nodes.push(n); parent.set(n, {node: p, key}); }
      if (n.kind === 'id') { const b = resolve(n.name); if (b) { ids.set(n, b); b.refs.push(n); } return; }
      if (n.kind === 'call') calls.push(n);
      for (const [k, v] of Object.entries(n)) if (!excludedKeys.has(k)) {
        if (Array.isArray(v)) v.forEach(x => expr(x, n, k));
        else if (v && typeof v === 'object') expr(v, n, k);
      }
    };
    const body = (n, container = null) => {
      if (!n) return;
      nodes.push(n); parent.set(n, {node: container});
      if (n.kind === 'block') { scopes.push(new Map()); n.body.forEach(s => body(s, n)); scopes.pop(); }
      else if (n.kind === 'decl') { const b = bind(n); b.container = container; expr(n.init, n, 'init'); }
      else if (n.kind === 'decls') n.declarations.forEach(d => body(d, n));
      else if (n.kind === 'for') { scopes.push(new Map()); body(n.init, n); expr(n.condition, n, 'condition'); expr(n.step, n, 'step'); body(n.body, n); scopes.pop(); }
      else if (n.kind === 'if') { expr(n.condition, n, 'condition'); scopes.push(new Map()); body(n.yes, n); scopes.pop(); scopes.push(new Map()); body(n.no, n); scopes.pop(); }
      else if (['while', 'do'].includes(n.kind)) { expr(n.condition, n, 'condition'); scopes.push(new Map()); body(n.body, n); scopes.pop(); }
      else if (n.kind === 'switch') { expr(n.selector, n, 'selector'); for (const c of n.cases) { scopes.push(new Map()); c.body.forEach(s => body(s, c)); scopes.pop(); } }
      else { nodes.pop(); expr(n, container); }
    };
    body(fn.body);
    const root = n => n?.kind === 'id' ? ids.get(n) : ['member', 'index'].includes(n?.kind) ? root(n.base) : null;
    for (const n of nodes) {
      if (n.kind === 'assign' || n.kind === 'unary' && ['++', '--'].includes(n.op)) {
        const dest = n.kind === 'assign' ? n.left : n.value;
        if (dest?.kind === 'id') { const b = root(dest); if (b) b.written = true; }
      }
      if (n.kind === 'unary' && n.op === '&') { const b = root(n.value); if (b) b.escaped = true; }
    }
    info.set(fn.name, {fn, ids, parent, bindings, calls, nodes, root});
  }
  return info;
}

export function specializeDependencies(ast, kernel, {maxPasses = 6} = {}) {
  const report = {passes: 0, branchesPruned: 0, conditionalExpressionsPruned: 0,
    scalarParameters: [], recordFields: [], skipped: null};
  const functions = ast.functions, byName = new Map(functions.map(f => [f.name, f]));
  const structs = new Map((ast.structs || []).map(s => [s.type, s]));
  const seenScalars = new Set(), seenFields = new Set();
  let unsupported = byName.size !== functions.length || functions.some(f => f.templateParameter || f.overloadName || f.classOwner || f.params.some(p => p.reference || !p.pointer && (p.type === 'thread-block' || String(p.type).startsWith('texture')))) || [...structs.values()].some(s => s.valueClass || s.methods?.length) || (ast.objectPointerTypes?.length || ast.devicePointerTypes?.length || ast.bufferReferenceTypes?.length || ast.sharedGlobals?.length);
  walk(ast, n => { if (n.kind === 'decl' && (n.reference || n.shared || n.volatileShared) || ['object-new', 'object-deref', 'device-launch', 'thread-block'].includes(n.kind)) unsupported = true; });
  // New language features must opt IN to this analysis. Inline PTX, implicit
  // receivers and lowered pointer/collective nodes may have writes not captured
  // by ordinary assignment syntax; never guess their effects.
  const plainNodes = new Set(['block','decl','decls','for','while','do','if','switch','case',
    'expr','return','break','continue','empty','sequence','assign','binary','unary',
    'index','member','id','literal','cast','conditional','call']);
  for (const fn of functions) walk(fn.body, n => { if (!plainNodes.has(n.kind)) unsupported = true; });
  if (unsupported) { report.skipped = 'requires-plain-by-value-call-graph'; return report; }
  const reachable = infos => {
    const reached = new Set(), pending = [kernel.name];
    while (pending.length) {
      const name = pending.pop(); if (reached.has(name)) continue; reached.add(name);
      for (const call of infos.get(name)?.calls || []) if (call.callee?.kind === 'id' && byName.has(call.callee.name)) pending.push(call.callee.name);
    }
    return reached;
  };
  for (let pass = 0; pass < maxPasses; pass++) {
    report.passes++;
    const all = indexFunctions(functions), reached = reachable(all), infos = [...reached].map(n => all.get(n));
    if (pass === 0) report.reachableFunctionsBefore = reached.size;
    const constants = new Map(), fields = new Map();
    const typeOf = (n, info) => {
      if (n?.kind === 'id') return info.ids.get(n)?.type;
      if (n?.kind === 'member') return structs.get(typeOf(n.base, info))?.fields.find(f => f.name === n.member)?.type;
      if (n?.kind === 'call' && n.callee.kind === 'id') return byName.get(n.callee.name)?.result;
      if (n?.kind === 'conditional') { const a = typeOf(n.yes, info), b = typeOf(n.no, info); return a === b ? a : null; }
      return null;
    };
    const evaluate = (n, info, depth = 0) => {
      if (!n || depth > 32) return null;
      const l = literal(n); if (l) return l;
      if (n.kind === 'id') {
        const b = info.ids.get(n);
        if (b?.escaped || b?.written || b?.node.pointer) return null;
        if (constants.has(b)) return constants.get(b);
        if (b?.node.constant && b.node.init) return converted(evaluate(b.node.init, info, depth + 1), b.type);
        return null;
      }
      if (n.kind === 'member' && info.root(n) && !info.root(n).node.pointer)
        return fields.get(typeOf(n.base, info) + '.' + n.member) || null;
      if (n.kind === 'cast' && integerTypes.has(n.target)) return converted(evaluate(n.value, info, depth + 1), n.target);
      if (n.kind === 'unary' && ['!', '+', '-', '~'].includes(n.op)) {
        const a = evaluate(n.value, info, depth + 1); if (!a) return null;
        if (n.op === '!') return {type: 'bool', value: Number(!a.value)};
        if (n.op === '+') return converted(a, a.type === 'bool' ? 'i32' : a.type);
        if (n.op === '~') return {type: a.type === 'u32' ? 'u32' : 'i32', value: a.type === 'u32' ? (~a.value) >>> 0 : ~a.value};
        return a.type === 'u32' ? {type: 'u32', value: (-a.value) >>> 0} : converted({type: 'i32', value: -a.value}, 'i32');
      }
      if (n.kind === 'binary') {
        const a = evaluate(n.left, info, depth + 1); if (!a) return null;
        if (n.op === '&&' && !a.value || n.op === '||' && a.value) return {type: 'bool', value: Number(!!a.value)};
        const b = evaluate(n.right, info, depth + 1);
        if (['&&', '||'].includes(n.op)) return b ? {type: 'bool', value: Number(!!b.value)} : null;
        return binary(n.op, a, b);
      }
      // Calls, assignments, increments, loads, and floating point are NOT constants.
      return null;
    };
    for (let iteration = 0; iteration < 32; iteration++) {
      let changed = false;
      for (const info of infos) if (info.fn !== kernel) {
        const callers = infos.flatMap(c => c.calls.filter(n => n.callee?.kind === 'id' && n.callee.name === info.fn.name).map(n => [c, n]));
        if (!callers.length) continue;
        for (let i = 0; i < info.fn.params.length; i++) {
          const p = info.fn.params[i], b = info.bindings[i];
          if (!integerTypes.has(p.type) || p.pointer || b.written || b.escaped || constants.has(b)) continue;
          const values = callers.map(([c, n]) => converted(evaluate(n.args[i] ?? p.defaultValue, c), p.type));
          if (values[0] && values.every(v => equal(v, values[0]))) { constants.set(b, values[0]); changed = true; }
        }
      }
      if (!changed) break;
    }
    for (const [b, c] of constants) {
      const key = b.fn.name + '.' + b.node.name;
      if (!seenScalars.has(key)) { seenScalars.add(key); report.scalarParameters.push({function: b.fn.name, parameter: b.node.name, ...c}); }
    }
    for (const [type, record] of structs) for (const field of record.fields) {
      if (!integerTypes.has(field.type) || field.dimensions?.length) continue;
      let candidate = null, valid = true, writes = 0;
      // Nested records and externally supplied records require alias/dataflow
      // analysis beyond this closed-world pass; leave them generic.
      if ([...structs.values()].some(s => s.fields.some(f => f.type === type)) ||
          [...(ast.constantGlobals || []), ...(ast.deviceGlobals || [])].some(g => g.type === type)) continue;
      for (const info of infos) {
        const local = info.bindings.filter(b => b.type === type);
        const closedValue = n => {
          if (typeOf(n, info) !== type) return false;
          if (n.kind === 'id') { const b = info.ids.get(n); return !!b && !b.node.pointer && !b.node.reference && !b.node.dimensions?.length; }
          if (n.kind === 'call') return n.callee.kind === 'id' && byName.has(n.callee.name);
          return n.kind === 'conditional' && closedValue(n.yes) && closedValue(n.no);
        };
        for (const b of local) {
          if (b.node.pointer || b.node.reference || b.node.dimensions?.length || b.escaped || b.parameter && info.fn === kernel) { valid = false; break; }
          if (b.parameter) continue;
          if (b.node.init) {
            if (!closedValue(b.node.init)) valid = false;
            walk(b.node.init, n => { if (n.kind === 'id' && info.ids.get(n) === b) valid = false; });
            continue;
          }
          // A fresh record must initialize this field as its first use in the
          // same lexical block. Conditional/partial initialization is refused.
          const id = b.refs[0], member = info.parent.get(id)?.node, assignment = info.parent.get(member)?.node,
            statement = info.parent.get(assignment)?.node;
          let declaration = b.node;
          if (info.parent.get(declaration)?.node?.kind === 'decls') declaration = info.parent.get(declaration).node;
          const block = info.parent.get(declaration)?.node;
          if (!id || member?.kind !== 'member' || member.base !== id || member.member !== field.name ||
              assignment?.kind !== 'assign' || assignment.op !== '=' || assignment.left !== member ||
              statement?.kind !== 'expr' || block?.kind !== 'block' || info.parent.get(statement)?.node !== block) { valid = false; break; }
          const start = block.body.indexOf(declaration), end = block.body.indexOf(statement);
          if (end <= start || block.body.slice(start + 1, end).some(s => !['expr', 'decl', 'decls', 'empty'].includes(s.kind))) { valid = false; break; }
        }
        if (!valid) break;
        for (const n of info.nodes) {
          const dest = n.kind === 'assign' ? n.left : n.kind === 'unary' && ['++', '--'].includes(n.op) ? n.value : null;
          if (dest?.kind === 'member' && typeOf(dest.base, info) === type && dest.member === field.name) {
            const value = n.kind === 'assign' && n.op === '=' ? converted(evaluate(n.right, info), field.type) : null;
            if (!value || candidate && !equal(candidate, value)) { valid = false; break; }
            candidate = value; writes++;
          }
          if (dest && typeOf(dest, info) === type && (n.kind !== 'assign' || n.op !== '=' || !closedValue(n.right))) { valid = false; break; }
          if (n.kind === 'return' && info.fn.result === type && !closedValue(n.value)) { valid = false; break; }
        }
        if (!valid) break;
      }
      if (valid && writes && candidate) {
        const key = type + '.' + field.name; fields.set(key, candidate);
        if (!seenFields.has(key)) { seenFields.add(key); report.recordFields.push({record: record.name, field: field.name, ...candidate}); }
      }
    }
    let changed = 0;
    for (const info of infos) {
      const prune = n => {
        if (!n || typeof n !== 'object') return;
        if (n.kind === 'if' || n.kind === 'conditional') {
          const value = evaluate(n.condition, info);
          if (value) {
            const selected = value.value ? n.yes : n.no;
            const replacement = n.kind === 'if' ? {kind: 'block', token: n.token, body: selected ? [selected] : []} : selected;
            if (replacement) {
              if (n.kind === 'if') report.branchesPruned++; else report.conditionalExpressionsPruned++;
              for (const k of Object.keys(n)) delete n[k]; Object.assign(n, replacement); changed++;
              prune(n); return;
            }
          }
        }
        for (const [key, v] of Object.entries(n)) if (!excludedKeys.has(key)) {
          if (Array.isArray(v)) v.forEach(prune); else if (v && typeof v === 'object') prune(v);
        }
      };
      prune(info.fn.body);
    }
    report.reachableFunctionsAfter = reachable(indexFunctions(functions)).size;
    if (!changed) break;
  }
  return report;
}
