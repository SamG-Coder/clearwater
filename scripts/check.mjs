import { readFile } from "node:fs/promises";
import { compile } from "../vendor/cuda-webshader/compiler/compiler.js";
let source = await readFile(
  new URL("../src/clearwater.cu", import.meta.url),
  "utf8",
);
source += "\n" + await readFile(new URL("../pool/pool.cu", import.meta.url), "utf8");
source += "\n" + await readFile(new URL("../cube/cube.cu", import.meta.url), "utf8");
source += "\n" + await readFile(new URL("../study/study.cu", import.meta.url), "utf8");
for (const entry of [...source.matchAll(/__global__ void (\w+)/g)].map(
  (m) => m[1],
)) {
  const artifact = compile(source, {
    entry,
    workgroupSize: ["spectrum_rows", "spectrum_norm", "lens_rows"].includes(
      entry,
    )
      ? [64, 1, 1]
      : [8, 8, 1],
  });
  console.log(`${entry}: ${artifact.wgsl.length} WGSL bytes`);
}
