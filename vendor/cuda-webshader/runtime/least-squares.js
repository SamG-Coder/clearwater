// SPDX-License-Identifier: MIT
// Streaming normal equations: GPU accumulation, small dense solve on the host.
export const NORMAL_EQUATIONS_SOURCE = `
__global__ void normal_equations(const float* rows, float* partial,
    int count, int features, int outputs, int stride, int offset) {
  int entry = blockIdx.x * blockDim.x + threadIdx.x;
  int entries = features * (features + outputs);
  if (entry >= entries) return;
  int a = entry / (features + outputs);
  int b = entry % (features + outputs);
  int lane = blockIdx.y;
  float sum = 0.0f;
  float correction = 0.0f;
  for (int row = lane; row < count; row += 16) {
    int start = offset + row * stride;
    float value = rows[start + a] * rows[start + b] - correction;
    float next = sum + value;
    correction = (next - sum) - value;
    sum = next;
  }
  partial[lane * entries + entry] += sum;
}`;

/** Partial-pivoted Float64 host solve. Rows of the result correspond to outputs. */
export function solveNormalEquations(equations, features, outputs, ridge = 1e-6) {
  if (!Number.isInteger(features) || features < 1 || features > 64 ||
      !Number.isInteger(outputs) || outputs < 1 || outputs > 64 ||
      equations.length !== features * (features + outputs) ||
      !Number.isFinite(ridge) || ridge < 0 || Array.from(equations).some(v => !Number.isFinite(v))) {
    throw new RangeError('Invalid normal equations or ridge.');
  }
  const stride = features + outputs, a = Float64Array.from(equations);
  let trace = 0;
  for (let i = 0; i < features; i++) trace += a[i * stride + i];
  if (!(trace > 0)) throw new Error('Least-squares system has no nonzero observations.');
  const lambda = ridge * trace / features;
  for (let i = 0; i < features; i++) a[i * stride + i] += lambda;
  for (let k = 0; k < features; k++) {
    let pivot = k;
    for (let i = k + 1; i < features; i++) if (Math.abs(a[i * stride + k]) > Math.abs(a[pivot * stride + k])) pivot = i;
    if (!(Math.abs(a[pivot * stride + k]) > Number.EPSILON * trace)) throw new Error('Singular least-squares system; use a positive ridge.');
    for (let j = k; j < stride; j++) {
      const t = a[k * stride + j]; a[k * stride + j] = a[pivot * stride + j]; a[pivot * stride + j] = t;
    }
    const diagonal = a[k * stride + k];
    for (let j = k; j < stride; j++) a[k * stride + j] /= diagonal;
    for (let i = 0; i < features; i++) if (i !== k) {
      const factor = a[i * stride + k];
      for (let j = k; j < stride; j++) a[i * stride + j] -= factor * a[k * stride + j];
    }
  }
  const weights = new Float32Array(features * outputs);
  for (let o = 0; o < outputs; o++) for (let i = 0; i < features; i++) weights[o * features + i] = a[i * stride + features + o];
  if (weights.some(v => !Number.isFinite(v))) throw new Error('Least-squares solution overflows float32.');
  return weights;
}

/** Each GPU row is [features..., targets...]; include a constant feature for a bias.
 * Rows may be streamed through a reusable buffer. All appends use one queue in order.
 * f32 products/accumulation are not a substitute for native f64 normal equations.
 */
export class NormalEquations {
  static async create(runtime, features, outputs) {
    if (![features, outputs].every(n => Number.isInteger(n) && n >= 1 && n <= 64)) throw new RangeError('Feature and output counts must be 1..64.');
    const kernel = await runtime.kernel(NORMAL_EQUATIONS_SOURCE, {entry:'normal_equations', workgroupSize:[64,1,1]});
    return new NormalEquations(runtime, features, outputs, kernel);
  }
  constructor(runtime, features, outputs, kernel) {
    this.runtime = runtime; this.features = features; this.outputs = outputs; this.kernel = kernel;
    this.entries = features * (features + outputs);
    this.partial = runtime.createBuffer(this.entries * 16 * 4);
    this.rows = 0;
  }
  append(rows, count, {stride = this.features + this.outputs, offset = 0} = {}) {
    this.runtime.checkResource(this.partial); this.runtime.checkResource(rows);
    if (!Number.isInteger(count) || count < 1 || !Number.isInteger(stride) || stride < this.features + this.outputs ||
        !Number.isInteger(offset) || offset < 0 || offset + (count - 1) * stride + this.features + this.outputs > rows.byteLength / 4 ||
        offset + count * stride > 0x7fffffff) throw new RangeError('Observation rows do not fit the buffer or signed indexing range.');
    this.runtime.batch().dispatch(this.kernel.bind({rows, partial:this.partial}, {
      count, features:this.features, outputs:this.outputs, stride, offset
    }), [Math.ceil(this.entries / 64),16,1]).submit();
    this.rows += count;
    return this;
  }
  async solve({ridge = 1e-6} = {}) {
    if (!this.rows) throw new Error('No observations were appended.');
    const partial = await this.runtime.read(this.partial), equations = new Float64Array(this.entries);
    for (let lane = 0; lane < 16; lane++) for (let i = 0; i < this.entries; i++) equations[i] += partial[lane * this.entries + i];
    return solveNormalEquations(equations, this.features, this.outputs, ridge);
  }
  dispose() { if (!this.partial.destroyed) this.runtime.destroyBuffer(this.partial); }
}
