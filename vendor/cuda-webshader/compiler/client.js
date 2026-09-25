/** Compiler work happens away from the render thread. Generated artifacts are serializable. */
export class CompilerClient{
  constructor(){this.id=0;this.pending=new Map();this.spawn();}
  spawn(){this.worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});this.worker.onmessage=e=>{const task=this.pending.get(e.data.id);if(!task)return;clearTimeout(task.timer);this.pending.delete(e.data.id);if(e.data.error)task.reject(new Error(e.data.error.message));else task.resolve(e.data);};this.worker.onerror=e=>this.failAll(new Error(e.message||'Compiler worker failed.'));}
  failAll(error){for(const t of this.pending.values()){clearTimeout(t.timer);t.reject(error);}this.pending.clear();}
  compile(source,options={}){const id=++this.id;return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{this.worker.terminate();this.failAll(new Error('Compilation exceeded the worker time limit. Simplify the source and retry.'));this.spawn();},10000);this.pending.set(id,{resolve,reject,timer});this.worker.postMessage({id,source,options});});}
  dispose(){this.worker.terminate();this.failAll(new Error('Compiler client disposed.'));}
}
