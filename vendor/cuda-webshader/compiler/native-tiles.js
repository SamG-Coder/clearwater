import {lowerNativeTilePhases} from './native-tile-phases.js';
// A CUDA tile maps to an explicitly sized WGSL subgroup. Logical CUDA thread
// indices are remapped from subgroup IDs, never assumed to equal physical IDs.
export function lowerNativeTiles(ast,walk,fail){
 for(const fn of ast.functions){const handles=new Set(),blocks=new Set();walk(fn.body,n=>{if(n.kind==='thread-block')blocks.add(n.name);});
  walk(fn.body,n=>{if(n.kind==='thread-warp'){if(fn.qualifier!=='__global__'||!blocks.has(n.parent))fail('Static tiles require a kernel-local thread block.',n);if(handles.has(n.name))fail('Static tile names must be unique.',n);handles.add(n.name);fn.nativeTiles=true;n.kind='empty';}});
  if(!handles.size)continue;
  let warpSizeShadowed=fn.params.some(p=>p.name==='warpSize');walk(fn.body,n=>{if(n.kind==='decl'&&n.name==='warpSize')warpSizeShadowed=true;});
  walk(fn.body,n=>{if(n.kind==='decl'&&handles.has(n.name))fail('Static tile handles cannot be shadowed.',n);
   if(n.kind==='call'&&n.callee.kind==='member'&&n.callee.base.kind==='id'&&handles.has(n.callee.base.name)){const method=n.callee.member;if(!['thread_rank','any','ballot','shfl','shfl_up'].includes(method))fail('Unsupported static tile operation '+method,n);n.callee={kind:'id',name:'cw_native_tile_'+method,token:n.token};}
  });
  lowerNativeTilePhases(fn,handles,walk,fail);
  // The vote itself makes the loop decision identical within a tile. Some
  // validators retain its predicate's lane dependence around the back-edge.
  // Restrict the diagnostic override to loops with no divergent collectives
  // or early exits, and leave an unsuppressed entry collective outside.
  const collective=n=>n?.kind==='call'&&n.callee?.name?.startsWith('cw_native_tile_')&&!['cw_native_tile_thread_rank','cw_native_tile_warp_index'].includes(n.callee.name);
  walk(fn.body,loop=>{if(!['for','while'].includes(loop.kind)||loop.condition?.callee?.name!=='cw_native_tile_any')return;
    const prove=(node,conditional=false)=>{if(!node||typeof node!=='object')return;if(Array.isArray(node)){node.forEach(n=>prove(n,conditional));return;}
      if(['break','continue','return'].includes(node.kind))fail('Vote-controlled tile loops cannot exit or continue early.',node);
      if(collective(node)&&conditional)fail('Vote-controlled tile loops require unconditional collectives.',node);
      const next=conditional||['if','for','while','do','conditional'].includes(node.kind)||node.kind==='binary'&&['&&','||'].includes(node.op);
      for(const [key,value]of Object.entries(node))if(key!=='token')prove(value,next);
    };prove(loop.body);prove(loop.step);prove(loop.condition.args);loop.provenTileVoteLoop=true;
  });
  // Logical CUDA thread x is subgroup_id * 32 + subgroup lane. Preserve
  // the subgroup ID explicitly so validators need not prove that identity.
  walk(fn.body,n=>{if(n.kind==='binary'&&n.op==='/'&&n.left?.kind==='member'&&n.left.base?.name==='threadIdx'&&n.left.member==='x'&&(n.right?.name==='warpSize'&&!warpSizeShadowed||n.right?.kind==='literal'&&Number(n.right.value.replace(/[uU]$/,''))===32)){delete n.left;delete n.right;delete n.op;Object.assign(n,{kind:'call',callee:{kind:'id',name:'cw_native_tile_warp_index',token:n.token},args:[]});}});
 }
}
export function emitNativeTile(e,n,name){
 const op=name.slice('cw_native_tile_'.length),args=n.args.map(a=>e.expr(a));
 const count=['thread_rank','warp_index'].includes(op)?0:['any','ballot'].includes(op)?1:2;if(args.length!==count)e.fail('Wrong static tile argument count.',n);
 if(op==='thread_rank')return e.result(n,'u32','(cw_thread.x % 32u)');
 if(op==='warp_index')return e.result(n,'u32','cw_subgroup');
 if(['any','ballot'].includes(op)){const a=args[0],predicate=e.convert(a.code,a.type,'bool',n);return e.result(n,op==='any'?'bool':'u32',op==='any'?`subgroupAny(${predicate})`:`subgroupBallot(${predicate}).x`,a.pre);}
 const [value,index]=args;if(!['i32','u32','f32'].includes(value.type)||!['i32','u32'].includes(index.type))e.fail('Tile shuffle requires a scalar value and integer lane.',n);
 const temp='cw_shuffle_'+e.temp++,pre=[...value.pre,`let ${temp}=${value.code};`,...index.pre];
 if(op==='shfl')return e.result(n,value.type,`subgroupShuffle(${temp}, u32(${index.code}) % 32u)`,pre);
 const delta=temp+'_delta',shuffled=temp+'_value';pre.push(`let ${delta}=u32(${index.code});`,`let ${shuffled}=subgroupShuffle(${temp}, select(0u,(cw_thread.x % 32u)-${delta},(cw_thread.x % 32u)>=${delta}));`);
 return e.result(n,value.type,`select(${temp},${shuffled},(cw_thread.x % 32u)>=${delta})`,pre);
}
