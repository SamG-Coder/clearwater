// Honor small, provably counted CUDA unroll hints. Unsupported loops retain
// their original control flow; a pragma is an optimization hint, not a promise.
export function unrollCount(loop) {
  if (!loop.unroll || loop.unroll === 1 || loop.kind !== 'for') return null;
  const d=loop.init,c=loop.condition,s=loop.step;
  const literal=n=>n?.kind==='literal'&&/^(?:0x[\da-f]+|\d+)[uU]?$/.test(n.value)?Number(n.value.replace(/[uU]$/,'')):NaN;
  if(d?.kind!=='decl'||!['i32','u32'].includes(d.type)||d.pointer||d.reference||d.dimensions?.length||c?.kind!=='binary'||c.left?.kind!=='id'||c.left.name!==d.name||!['<','<='].includes(c.op))return null;
  const start=literal(d.init),end=literal(c.right);
  const increment=s?.kind==='unary'&&s.op==='++'&&s.value?.name===d.name?1:s?.kind==='assign'&&s.op==='+='&&s.left?.name===d.name?literal(s.right):NaN;
  if(![start,end,increment].every(Number.isSafeInteger)||increment<=0)return null;
  const count=Math.max(0,Math.ceil((end-start+(c.op==='<='?1:0))/increment));
  if(count>32||(loop.unroll!==true&&loop.unroll!==count)||start+count*increment>(d.type==='u32'?4294967295:2147483647))return null;
  let safe=true,nodes=0;
  function visit(n){
    if(!n||typeof n!=='object')return;
    if(Array.isArray(n)){n.forEach(visit);return;}
    nodes++;
    // Conservatively retain loops with exits, aliases, or a modified/shadowed counter.
    if(['break','continue'].includes(n.kind)||(n.kind==='decl'&&n.name===d.name)||(n.kind==='assign'&&n.left?.kind==='id'&&n.left.name===d.name)||(n.kind==='unary'&&['&','++','--'].includes(n.op)&&n.value?.kind==='id'&&n.value.name===d.name))safe=false;
    if((n.kind==='call'&&n.args?.some(a=>a.kind==='id'&&a.name===d.name))||(n.kind==='decl'&&n.reference&&n.init?.kind==='id'&&n.init.name===d.name))safe=false;
    for(const [k,v] of Object.entries(n))if(!['token','symbol'].includes(k))visit(v);
  }
  visit(loop.body);
  return safe&&nodes*Math.max(1,count)<=24000?count:null;
}
