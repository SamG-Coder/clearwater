// Read-only CUDA records with byte-sized bool fields use their native byte layout.
// Keep computational WGSL records separate from host storage, which cannot contain bool.
const array=t=>t&&typeof t==='object'&&t.kind==='array';
const alignUp=(n,a)=>Math.ceil(n/a)*a;
export function containsNativeBool(emitter,type){
 return type==='bool'||array(type)&&containsNativeBool(emitter,type.element)||emitter.structs.has(type)&&emitter.structs.get(type).fields.some(f=>containsNativeBool(emitter,f.resolvedType));
}
export function nativeRecordLayout(emitter,type,node){
 if(array(type)){const element=nativeRecordLayout(emitter,type.element,node);return {kind:'array',type,element,align:element.align,size:alignUp(element.size,element.align)*type.length};}
 if(emitter.structs.has(type)){
  let size=0,align=1;const fields=[];
  for(const f of emitter.structs.get(type).fields){const layout=nativeRecordLayout(emitter,f.resolvedType,node);size=alignUp(size,layout.align);fields.push({name:f.name,offset:size,layout});size+=layout.size;align=Math.max(align,layout.align);}
  return {kind:'record',type,fields,align,size:alignUp(size,align)};
 }
 const vector=/^vec([234])<(f32|i32|u32)>$/.exec(type);
 if(vector){const width=Number(vector[1]);return {kind:'vector',type,width,element:vector[2],align:width===3?4:width*4,size:width*4};}
 if(['bool','f32','i32','u32'].includes(type))return {kind:'scalar',type,align:type==='bool'?1:4,size:type==='bool'?1:4};
 emitter.fail('Native bool records support bool, 32-bit scalars, vectors and nested plain records only.',node);
}
export function decodeNativeRecord(layout,buffer,byteBase,offset=0){
 const read=(type,at)=>{const word=`${buffer}[(${byteBase} + ${at&~3}u) >> 2u]`;return type==='bool'?`(((${word} >> ${(at%4)*8}u) & 255u) != 0u)`:type==='u32'?word:`bitcast<${type}>(${word})`;};
 if(layout.kind==='scalar')return read(layout.type,offset);
 if(layout.kind==='vector')return `${layout.type}(${Array.from({length:layout.width},(_,i)=>read(layout.element,offset+i*4)).join(', ')})`;
 if(layout.kind==='record')return `${layout.type}(${layout.fields.map(f=>decodeNativeRecord(f.layout,buffer,byteBase,offset+f.offset)).join(', ')})`;
 const typeName=t=>array(t)?`array<${typeName(t.element)}, ${t.length}>`:t;
 return `${typeName(layout.type)}(${Array.from({length:layout.type.length},(_,i)=>decodeNativeRecord(layout.element,buffer,byteBase,offset+i*alignUp(layout.element.size,layout.element.align))).join(', ')})`;
}
