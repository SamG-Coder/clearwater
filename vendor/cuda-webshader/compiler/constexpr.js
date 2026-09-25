// Select branches after template substitution, before helper instantiation.
// Deliberately bounded to boolean and signed integer constant expressions.
export function pruneConstexpr(root, fail) {
  const evaluate = (n, depth = 0) => {
    if (depth > 64) fail('if constexpr condition is too deeply nested.', n);
    const checked = value => {
      if (!Number.isInteger(value) || value < -2147483648 || value > 2147483647)
        fail('if constexpr arithmetic requires signed 32-bit constants.', n);
      return value;
    };
    if (n.kind === 'id' && ['true', 'false'].includes(n.name)) return Number(n.name === 'true');
    if (n.kind === 'literal') {
      if (n.value === 'true') return 1;
      if (n.value === 'false') return 0;
      if (/^-?(?:\d+|0[xX][\da-fA-F]+)$/.test(n.value)) return checked(Number(n.value));
    }
    if (n.kind === 'unary' && ['!', '+', '-', '~'].includes(n.op)) {
      const a = evaluate(n.value, depth + 1);
      return checked(n.op === '!' ? Number(!a) : n.op === '-' ? -a : n.op === '~' ? ~a : a);
    }
    if (n.kind === 'binary') {
      const a = evaluate(n.left, depth + 1);
      if (n.op === '&&' && !a) return 0;
      if (n.op === '||' && a) return 1;
      const b = evaluate(n.right, depth + 1);
      if (['/', '%'].includes(n.op) && b === 0) fail('Division by zero in if constexpr condition.', n);
      const ops = {
        '+': () => a + b, '-': () => a - b, '*': () => a * b,
        '/': () => Math.trunc(a / b), '%': () => a % b,
        '==': () => Number(a === b), '!=': () => Number(a !== b),
        '<': () => Number(a < b), '<=': () => Number(a <= b),
        '>': () => Number(a > b), '>=': () => Number(a >= b),
        '&&': () => Number(!!b), '||': () => Number(!!b),
        '&': () => a & b, '|': () => a | b, '^': () => a ^ b,
      };
      if (ops[n.op]) return checked(ops[n.op]());
    }
    fail('if constexpr requires a supported integer or boolean constant expression.', n);
  };
  const visit = node => {
    if (!node || typeof node !== 'object') return;
    if (node.kind === 'if' && node.constexpr) {
      const selected = evaluate(node.condition) ? node.yes : node.no;
      const replacement = {kind: 'block', token: node.token, body: selected ? [selected] : []};
      for (const key of Object.keys(node)) delete node[key];
      Object.assign(node, replacement);
      visit(node);
      return;
    }
    for (const [key, value] of Object.entries(node)) {
      if (key === 'token') continue;
      if (Array.isArray(value)) value.forEach(visit);
      else if (value && typeof value === 'object') visit(value);
    }
  };
  visit(root);
}
