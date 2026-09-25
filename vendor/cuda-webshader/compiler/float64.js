// SPDX-License-Identifier: MIT
// Binary64 expression arithmetic implemented with integer limbs. No shader f64
// extension, float32 demotion, or GPU-dependent denormal arithmetic is used.
export const FLOAT64_WGSL = `
alias cw_f64 = vec2<u32>;
fn cw_d_shl(a: vec2<u32>, n: u32) -> vec2<u32> {
  if(n == 0u) { return a; }
  if(n >= 64u) { return vec2<u32>(0u); }
  if(n >= 32u) { return vec2<u32>(0u, a.x << (n - 32u)); }
  return vec2<u32>(a.x << n, (a.y << n) | (a.x >> (32u - n)));
}
fn cw_d_shr(a: vec2<u32>, n: u32) -> vec2<u32> {
  if(n == 0u) { return a; }
  if(n >= 64u) { return vec2<u32>(0u); }
  if(n >= 32u) { return vec2<u32>(a.y >> (n - 32u), 0u); }
  return vec2<u32>((a.x >> n) | (a.y << (32u - n)), a.y >> n);
}
fn cw_d_jam(a: vec2<u32>, n: u32) -> vec2<u32> {
  let shifted = cw_d_shr(a, n);
  return shifted | vec2<u32>(select(0u, 1u, any(cw_d_shl(shifted, n) != a)), 0u);
}
fn cw_d_uadd(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let low = a.x + b.x;
  return vec2<u32>(low, a.y + b.y + select(0u, 1u, low < a.x));
}
fn cw_d_usub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}
fn cw_d_uless(a: vec2<u32>, b: vec2<u32>) -> bool {
  return a.y < b.y || (a.y == b.y && a.x < b.x);
}
fn cw_d_nan(a: cw_f64) -> bool { return (a.y & 2147483647u) > 2146435072u || ((a.y & 2147483647u) == 2146435072u && a.x != 0u); }
fn cw_d_inf(a: cw_f64) -> bool { return (a.y & 2147483647u) == 2146435072u && a.x == 0u; }
fn cw_d_zero(a: cw_f64) -> bool { return (a.y & 2147483647u) == 0u && a.x == 0u; }
fn cw_d_neg(a: cw_f64) -> cw_f64 { return a ^ vec2<u32>(0u, 2147483648u); }
struct CWDoubleParts { significand: vec2<u32>, exponent: i32, }
fn cw_d_parts(a: cw_f64) -> CWDoubleParts {
  let raw = (a.y >> 20u) & 2047u;
  var s = vec2<u32>(a.x, a.y & 1048575u);
  var e = i32(raw) - 1023i;
  if(raw != 0u) { s.y |= 1048576u; }
  else {
    e = -1022i;
    if(any(s != vec2<u32>(0u))) {
      loop { if((s.y & 1048576u) != 0u) { break; } s = cw_d_shl(s, 1u); e--; }
    }
  }
  return CWDoubleParts(s,e);
}
// Input has its leading bit at bit 55 and three guard/round/sticky bits.
fn cw_d_pack(sign: u32, exponent: i32, value: vec2<u32>) -> cw_f64 {
  var e = exponent; var v = value;
  if(e < -1022i) { v = cw_d_jam(v, u32(-1022i-e)); e = -1022i; }
  let tail = v.x & 7u;
  var s = cw_d_shr(v, 3u);
  if(tail > 4u || (tail == 4u && (s.x & 1u) != 0u)) { s = cw_d_uadd(s,vec2<u32>(1u,0u)); }
  if((s.y & 2097152u) != 0u) { s = cw_d_shr(s,1u); e++; }
  if(e > 1023i) { return vec2<u32>(0u,sign | 2146435072u); }
  let field = select(0u, u32(e+1023i), (s.y & 1048576u) != 0u);
  return vec2<u32>(s.x, sign | (field << 20u) | (s.y & 1048575u));
}
fn cw_d_from_f32(a: f32) -> cw_f64 {
  let bits = bitcast<u32>(a); let sign = bits & 2147483648u;
  let field = (bits >> 23u) & 255u; var mantissa = bits & 8388607u;
  if(field == 255u) { return vec2<u32>(0u,sign | 2146435072u | select(0u,524288u,mantissa != 0u)); }
  if(field == 0u && mantissa == 0u) { return vec2<u32>(0u,sign); }
  var e = i32(field)-127i;
  if(field == 0u) { e = -126i; loop { if((mantissa & 8388608u) != 0u) { break; } mantissa <<= 1u; e--; } }
  let s = cw_d_shl(vec2<u32>(mantissa & 8388607u,0u),29u);
  return vec2<u32>(s.x,sign | (u32(e+1023i) << 20u) | s.y);
}
fn cw_d_from_u32(a: u32) -> cw_f64 {
  if(a == 0u) { return vec2<u32>(0u); }
  let e = 31u-countLeadingZeros(a); let s = cw_d_shl(vec2<u32>(a,0u),52u-e);
  return vec2<u32>(s.x,((e+1023u) << 20u) | (s.y & 1048575u));
}
fn cw_d_from_i32(a: i32) -> cw_f64 {
  if(a < 0i) { return cw_d_neg(cw_d_from_u32(0u-u32(a))); }
  return cw_d_from_u32(u32(a));
}
fn cw_d_to_f32(a: cw_f64) -> f32 {
  let sign = a.y & 2147483648u;
  if(cw_d_nan(a)) { return bitcast<f32>(sign | 2143289344u); }
  if(cw_d_inf(a)) { return bitcast<f32>(sign | 2139095040u); }
  if(cw_d_zero(a)) { return bitcast<f32>(sign); }
  let p = cw_d_parts(a); var e = p.exponent;
  let shift = 29u + u32(max(-126i-e,0i));
  // Keep three rounding bits while reducing the 53-bit significand to 24 bits.
  let v = cw_d_jam(p.significand,shift-3u); let tail = v.x & 7u;
  var s = cw_d_shr(v,3u).x;
  if(tail > 4u || (tail == 4u && (s & 1u) != 0u)) { s++; }
  e = max(e,-126i);
  if(s >= 16777216u) { s >>= 1u; e++; }
  if(e > 127i) { return bitcast<f32>(sign | 2139095040u); }
  let field = select(0u,u32(e+127i),s >= 8388608u);
  return bitcast<f32>(sign | (field << 23u) | (s & 8388607u));
}
fn cw_d_u64_to_f32(a: vec2<u32>) -> f32 {
  if(all(a == vec2<u32>(0u))) { return 0.0f; }
  var e = select(31u-countLeadingZeros(a.x),63u-countLeadingZeros(a.y),a.y != 0u);
  var v: vec2<u32>;
  if(e > 26u) { v = cw_d_jam(a,e-26u); } else { v = cw_d_shl(a,26u-e); }
  let tail = v.x & 7u; var s = v.x >> 3u;
  if(tail > 4u || (tail == 4u && (s & 1u) != 0u)) { s++; }
  if(s >= 16777216u) { s >>= 1u; e++; }
  return bitcast<f32>(((e+127u) << 23u) | (s & 8388607u));
}
fn cw_d_eq(a: cw_f64,b: cw_f64) -> bool { return !cw_d_nan(a) && !cw_d_nan(b) && (all(a == b) || (cw_d_zero(a) && cw_d_zero(b))); }
fn cw_d_lt(a: cw_f64,b: cw_f64) -> bool {
  if(cw_d_nan(a) || cw_d_nan(b) || cw_d_eq(a,b)) { return false; }
  let sa = a.y >> 31u; let sb = b.y >> 31u;
  if(sa != sb) { return sa != 0u; }
  return select(cw_d_uless(a,b),cw_d_uless(b,a),sa != 0u);
}
fn cw_d_le(a: cw_f64,b: cw_f64) -> bool { return cw_d_lt(a,b) || cw_d_eq(a,b); }
fn cw_d_add(a: cw_f64,b: cw_f64) -> cw_f64 {
  if(cw_d_nan(a) || cw_d_nan(b) || (cw_d_inf(a) && cw_d_inf(b) && ((a.y ^ b.y) >> 31u) != 0u)) { return vec2<u32>(0u,2146959360u); }
  if(cw_d_inf(a)) { return a; } if(cw_d_inf(b)) { return b; }
  if(cw_d_zero(a) && cw_d_zero(b)) { return vec2<u32>(0u,(a.y & b.y) & 2147483648u); }
  if(cw_d_zero(a)) { return b; } if(cw_d_zero(b)) { return a; }
  let pa = cw_d_parts(a); let pb = cw_d_parts(b);
  var x = cw_d_shl(pa.significand,3u); var y = cw_d_shl(pb.significand,3u);
  var e = max(pa.exponent,pb.exponent); var sign = a.y & 2147483648u;
  x = cw_d_jam(x,u32(e-pa.exponent)); y = cw_d_jam(y,u32(e-pb.exponent));
  var sum: vec2<u32>;
  if(((a.y ^ b.y) >> 31u) == 0u) {
    sum = cw_d_uadd(x,y);
    if((sum.y & 16777216u) != 0u) { sum = cw_d_jam(sum,1u); e++; }
  } else {
    if(cw_d_uless(x,y)) { sum = cw_d_usub(y,x); sign = b.y & 2147483648u; }
    else { sum = cw_d_usub(x,y); }
    if(all(sum == vec2<u32>(0u))) { return vec2<u32>(0u); }
    loop { if((sum.y & 8388608u) != 0u) { break; } sum = cw_d_shl(sum,1u); e--; }
  }
  return cw_d_pack(sign,e,sum);
}
fn cw_d_sub(a: cw_f64,b: cw_f64) -> cw_f64 { return cw_d_add(a,cw_d_neg(b)); }
fn cw_d_mul(a: cw_f64,b: cw_f64) -> cw_f64 {
  let sign = (a.y ^ b.y) & 2147483648u;
  if(cw_d_nan(a) || cw_d_nan(b) || (cw_d_inf(a) && cw_d_zero(b)) || (cw_d_inf(b) && cw_d_zero(a))) { return vec2<u32>(0u,2146959360u); }
  if(cw_d_inf(a) || cw_d_inf(b)) { return vec2<u32>(0u,sign | 2146435072u); }
  if(cw_d_zero(a) || cw_d_zero(b)) { return vec2<u32>(0u,sign); }
  let pa = cw_d_parts(a); let pb = cw_d_parts(b);
  if(all(pa.significand == vec2<u32>(0u,1048576u))) { return cw_d_pack(sign,pa.exponent+pb.exponent,cw_d_shl(pb.significand,3u)); }
  if(all(pb.significand == vec2<u32>(0u,1048576u))) { return cw_d_pack(sign,pa.exponent+pb.exponent,cw_d_shl(pa.significand,3u)); }
  var product = vec4<u32>(0u); var term = vec4<u32>(pa.significand,0u,0u); var multiplier = pb.significand;
  for(var i = 0u; i < 53u; i++) {
    if((multiplier.x & 1u) != 0u) {
      var carry = 0u;
      for(var j = 0u; j < 4u; j++) {
        let p = product[j]; let s = p + term[j]; let t = s + carry;
        carry = select(0u,1u,s < p || t < s); product[j] = t;
      }
    }
    term = vec4<u32>(term.x << 1u,(term.y << 1u) | (term.x >> 31u),(term.z << 1u) | (term.y >> 31u),(term.w << 1u) | (term.z >> 31u));
    multiplier = cw_d_shr(multiplier,1u);
  }
  let extra = select(0u,1u,(product.w & 512u) != 0u);
  for(var i = 0u; i < 49u+extra; i++) {
    product = vec4<u32>((product.x >> 1u) | (product.y << 31u) | (product.x & 1u),(product.y >> 1u) | (product.z << 31u),(product.z >> 1u) | (product.w << 31u),product.w >> 1u);
  }
  return cw_d_pack(sign,pa.exponent+pb.exponent+i32(extra),product.xy);
}
fn cw_d_div(a: cw_f64,b: cw_f64) -> cw_f64 {
  let sign = (a.y ^ b.y) & 2147483648u;
  if(cw_d_nan(a) || cw_d_nan(b) || (cw_d_inf(a) && cw_d_inf(b)) || (cw_d_zero(a) && cw_d_zero(b))) { return vec2<u32>(0u,2146959360u); }
  if(cw_d_inf(a) || cw_d_zero(b)) { return vec2<u32>(0u,sign | 2146435072u); }
  if(cw_d_zero(a) || cw_d_inf(b)) { return vec2<u32>(0u,sign); }
  let pa = cw_d_parts(a); let pb = cw_d_parts(b); var e = pa.exponent-pb.exponent;
  var remainder = pa.significand; var q = vec2<u32>(0u);
  if(cw_d_uless(remainder,pb.significand)) { remainder = cw_d_shl(remainder,1u); e--; }
  for(var i = 0u; i < 56u; i++) {
    q = cw_d_shl(q,1u);
    if(!cw_d_uless(remainder,pb.significand)) { remainder = cw_d_usub(remainder,pb.significand); q.x |= 1u; }
    if(i < 55u) { remainder = cw_d_shl(remainder,1u); }
  }
  if(any(remainder != vec2<u32>(0u))) { q.x |= 1u; }
  return cw_d_pack(sign,e,q);
}

// Restoring square root: 56 root bits include guard/round/sticky for binary64.
fn cw_d_sqrt(a: cw_f64) -> cw_f64 {
  if(cw_d_nan(a)) { return vec2<u32>(0u,2146959360u); }
  if(cw_d_zero(a)) { return a; }
  if((a.y & 2147483648u) != 0u) { return vec2<u32>(0u,2146959360u); }
  if(cw_d_inf(a)) { return a; }
  let p = cw_d_parts(a);
  let odd = p.exponent & 1i;
  let shift = 58i + odd;
  var root = vec2<u32>(0u);
  var remainder = vec2<u32>(0u);
  for(var i=55i; i>=0i; i--) {
    var pair=0u;
    for(var j=0i; j<2i; j++) {
      let bit=2i*i+j-shift;
      if(bit>=0i && bit<53i) { pair |= ((p.significand[u32(bit)/32u] >> (u32(bit)%32u)) & 1u) << u32(j); }
    }
    remainder=cw_d_shl(remainder,2u); remainder.x |= pair;
    var trial=cw_d_shl(root,2u); trial.x |= 1u;
    root=cw_d_shl(root,1u);
    if(!cw_d_uless(remainder,trial)) { remainder=cw_d_usub(remainder,trial); root.x |= 1u; }
  }
  if(any(remainder!=vec2<u32>(0u))) { root.x |= 1u; }
  return cw_d_pack(0u,(p.exponent-odd)/2i,root);
}
fn cw_d_fmin(a: cw_f64,b: cw_f64) -> cw_f64 {
  if(cw_d_nan(a)) { return b; } if(cw_d_nan(b)) { return a; }
  if(cw_d_zero(a) && cw_d_zero(b)) { return vec2<u32>(0u,(a.y | b.y) & 2147483648u); }
  return select(b,a,cw_d_lt(a,b));
}
fn cw_d_fmax(a: cw_f64,b: cw_f64) -> cw_f64 {
  if(cw_d_nan(a)) { return b; } if(cw_d_nan(b)) { return a; }
  if(cw_d_zero(a) && cw_d_zero(b)) { return vec2<u32>(0u,(a.y & b.y) & 2147483648u); }
  return select(a,b,cw_d_lt(a,b));
}
`;
