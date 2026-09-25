/**
 * Conservative dependency pruning for compiler-owned WGSL.
 * Only whole function declarations are removed. Resource bindings, types,
 * constants, overrides, entry signatures and retained function bodies are not
 * rewritten. Identifiers outside functions are roots too (e.g. const calls).
 */
export function trimWgslDependencies(wgsl, entryPoint = 'main') {
  if (typeof wgsl !== 'string' || typeof entryPoint !== 'string')
    throw new TypeError('WGSL and entryPoint must be strings.');
  const tokens = [];
  // Tokenize identifiers and delimiters, retaining original character ranges. Nested
  // WGSL comments must not introduce fake calls, braces, or fn declarations.
  for (let i = 0; i < wgsl.length;) {
    const start = i;
    if (/\s/.test(wgsl[i])) { i++; continue; }
    if (wgsl.startsWith('//', i)) {
      const end = wgsl.indexOf('\n', i + 2); i = end < 0 ? wgsl.length : end; continue;
    }
    if (wgsl.startsWith('/*', i)) {
      let depth = 1; i += 2;
      while (i < wgsl.length && depth) {
        if (wgsl.startsWith('/*', i)) { depth++; i += 2; }
        else if (wgsl.startsWith('*/', i)) { depth--; i += 2; }
        else i++;
      }
      if (depth) return unchanged('unterminated-comment');
      continue;
    }
    if (/[A-Za-z_]/.test(wgsl[i])) {
      while (++i < wgsl.length && /[A-Za-z_0-9]/.test(wgsl[i])) {}
    } else i++;
    tokens.push({text: wgsl.slice(start, i), start, end: i});
  }
  const functions = [], outside = [], names = new Set();
  let level = 0, declarationStart = 0;
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (level === 0 && t.text === 'fn') {
      const name = tokens[i + 1]?.text;
      if (!name || !/^[A-Za-z_]\w*$/.test(name) || names.has(name))
        return unchanged('ambiguous-function-declaration');
      names.add(name);
      let open = i + 2;
      while (open < tokens.length && tokens[open].text !== '{') open++;
      if (open === tokens.length) return unchanged('missing-function-body');
      let depth = 1, end = open + 1;
      for (; end < tokens.length && depth; end++) {
        if (tokens[end].text === '{') depth++;
        else if (tokens[end].text === '}') depth--;
      }
      if (depth) return unchanged('unbalanced-function-body');
      // Attribute tokens belong to this declaration, not the preceding one.
      const prefix = tokens.slice(declarationStart, i);
      const start = prefix.length && prefix[0].text === '@' ? declarationStart : i;
      const attributes = tokens.slice(start, i).map(t => t.text);
      functions.push({name, start: tokens[start].start, end: tokens[end - 1].end,
        refs: new Set(tokens.slice(i + 2, end).map(t => t.text)),
        entry: attributes.some(t => ['compute', 'vertex', 'fragment'].includes(t))});
      // Remove the attribute tokens previously considered outside the function.
      if (start < i) outside.splice(outside.length - (i - start), i - start);
      i = end - 1; declarationStart = end;
      continue;
    }
    outside.push(t.text);
    if (t.text === '{') level++;
    else if (t.text === '}') {
      level--; if (level < 0) return unchanged('unbalanced-module');
      if (level === 0) declarationStart = i + 1;
    } else if (level === 0 && t.text === ';') declarationStart = i + 1;
  }
  if (level !== 0 || !names.has(entryPoint)) return unchanged('missing-entry-or-unbalanced-module');
  const byName = new Map(functions.map(f => [f.name, f]));
  const roots = new Set([entryPoint, ...functions.filter(f => f.entry).map(f => f.name),
    ...outside.filter(t => names.has(t))]);
  const reached = new Set(), pending = [...roots];
  while (pending.length) {
    const name = pending.pop();
    if (reached.has(name)) continue;
    reached.add(name);
    for (const ref of byName.get(name)?.refs || []) if (names.has(ref) && !reached.has(ref)) pending.push(ref);
  }
  const removed = functions.filter(f => !reached.has(f.name));
  let result = '', offset = 0;
  for (const f of removed) { result += wgsl.slice(offset, f.start); offset = f.end; }
  result += wgsl.slice(offset);
  const encoder = new TextEncoder();
  const functionBytes = new Map(functions.map(f => [f.name, encoder.encode(wgsl.slice(f.start, f.end)).length]));
  const weights = functions.filter(f => reached.has(f.name)).map(f => {
    const seen = new Set(), todo = [f.name]; let transitiveBytes = 0;
    while (todo.length) {
      const name = todo.pop(); if (seen.has(name)) continue; seen.add(name);
      const fn = byName.get(name); if (!fn) continue;
      transitiveBytes += functionBytes.get(name);
      for (const ref of fn.refs) if (names.has(ref) && !seen.has(ref)) todo.push(ref);
    }
    return {name: f.name, ownBytes: functionBytes.get(f.name),
      transitiveBytes, reachableFunctions: seen.size};
  }).sort((a, b) => b.transitiveBytes - a.transitiveBytes || a.name.localeCompare(b.name));
  return {wgsl: result, report: {beforeBytes: encoder.encode(wgsl).length,
    afterBytes: encoder.encode(result).length, functionsBefore: functions.length,
    functionsAfter: reached.size, removedFunctions: removed.map(f => f.name),
    roots: [...roots], largestDependencies: weights.slice(0, 12)}};

  function unchanged(reason) {
    return {wgsl, report: {beforeBytes: new TextEncoder().encode(wgsl).length,
      afterBytes: new TextEncoder().encode(wgsl).length, removedFunctions: [], skipped: reason}};
  }
}
