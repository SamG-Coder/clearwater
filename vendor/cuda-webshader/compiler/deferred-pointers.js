// A null-initialized pointer can use a fixed storage binding only when every
// use is dominated by an assignment and all assignments select that binding.
export function lowerDeferredPointers(ast,walk,fail) {
  for(const fn of ast.functions) {
    const declarations=[];
    walk(fn.body,n=>{if(n.kind==='decl'&&n.pointer&&!n.dimensions.length&&
      (n.init?.kind==='id'&&n.init.name==='NULL'||n.init?.kind==='literal'&&/^0[uUlL]*$/.test(n.init.value)))declarations.push(n);});
    for(const declaration of declarations) {
      const name=declaration.name;
      if(fn.params.some(p=>p.name===name))fail('Deferred pointer parameter shadowing is unsupported.',declaration);
      if(declaration.init.kind==='id') {
        if(fn.params.some(p=>p.name==='NULL')||[...(ast.constantGlobals||[]),...(ast.deviceGlobals||[]),...(ast.sharedGlobals||[])].some(p=>p.name==='NULL'))fail('NULL must be an unshadowed null pointer constant.',declaration);
        walk(fn.body,n=>{if(n.kind==='decl'&&n.name==='NULL')fail('NULL must be an unshadowed null pointer constant.',n);});
      }
      let base=null;
      const pointerRoot=n=>n?.kind==='id'?n:['+','-'].includes(n?.op)&&n.kind==='binary'?pointerRoot(n.left):null;
      walk(fn.body,n=>{
        if(n.kind==='decl'&&n.name===name&&n!==declaration)fail('Deferred pointer shadowing is unsupported.',n);
        if(['break','continue','do'].includes(n.kind))fail('Deferred pointers do not yet support break, continue or do loops.',n);
        if(n.kind==='assign'&&n.left.kind==='id'&&n.left.name===name&&n.op==='=') {
          const root=pointerRoot(n.right),parameter=fn.params.find(p=>p.name===root?.name&&p.pointer);
          if(!parameter||parameter.type!==declaration.type)fail('Deferred pointers require assignments from a same-type buffer parameter.',n);
          if(base&&base!==parameter.name)fail('Deferred pointers must retain one buffer binding.',n);
          base=parameter.name;
        }
      });
      if(!base)fail('Null pointer requires a buffer assignment before use.',declaration);
      walk(fn.body,n=>{if(n.kind==='decl'&&n.name===base)fail('Deferred pointer buffer shadowing is unsupported.',n);});
      const read=(node,assigned)=>walk(node,n=>{if(n.kind==='id'&&n.name===name&&!assigned)fail('Pointer may be null: assign its buffer before use.',n);});
      const expression=(node,assigned)=>{
        if(node?.kind==='assign'&&node.op==='='&&node.left.kind==='id'&&node.left.name===name){read(node.right,assigned);return true;}
        read(node,assigned);return assigned;
      };
      const statement=(node,assigned)=>{
        if(!node)return assigned;
        if(node===declaration)return false;
        if(node.kind==='block')return node.body.reduce((state,n)=>statement(n,state),assigned);
        if(node.kind==='decls')return node.declarations.reduce((state,n)=>statement(n,state),assigned);
        if(node.kind==='expr')return expression(node.value,assigned);
        if(node.kind==='if') {
          read(node.condition,assigned);
          const yes=statement(node.yes,assigned),no=node.no?statement(node.no,assigned):assigned;
          return yes&&no;
        }
        if(node.kind==='for'||node.kind==='while') {
          let initial=assigned;
          if(node.init)initial=['decl','decls'].includes(node.init.kind)?statement(node.init,assigned):expression(node.init,assigned);
          read(node.condition,initial);
          const body=statement(node.body,initial);
          expression(node.step,body);
          return initial; // The loop may execute zero times.
        }
        read(node,assigned);return assigned;
      };
      statement(fn.body,false);
      declaration.init={kind:'id',name:base,token:declaration.init.token};
    }
  }
}
