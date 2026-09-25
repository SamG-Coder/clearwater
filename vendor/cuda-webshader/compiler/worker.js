import {compile,serializableArtifact} from './compiler.js';
self.onmessage=event=>{const {id,source,options}=event.data;try{const start=performance.now(),artifact=serializableArtifact(compile(source,options));self.postMessage({id,artifact,compileMs:performance.now()-start});}catch(error){self.postMessage({id,error:{message:error.message,line:error.line,column:error.column}});}};
