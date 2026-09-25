// MIT compatibility implementation of the XORWOW integer recurrence and
// cuRAND uniform mapping. Scope: 64-bit integer seeds, zero subsequence/offset,
// curand and curand_uniform only. State layout is compiler-owned, not CUDA ABI.
// Reference: installed CUDA 13.3 curand_kernel.h and curand_uniform.h.
export const CURAND_XORWOW_SOURCE = `
struct curandState { unsigned int d; unsigned int v[5]; };
__device__ void curand_init(size_t seed,unsigned int subsequence,unsigned int offset,curandState *state) {
 unsigned int low=(uint(seed)^2865916745u)*1099087573u;
 unsigned int high=(uint(seed>>32u)^4158451677u)*2591861531u;
 state->v[0]=123456789u+low;
 state->v[1]=362436069u^low;
 state->v[2]=521288629u+high;
 state->v[3]=88675123u^high;
 state->v[4]=5783321u+low;
 state->d=6615241u+low+high;
}
__device__ unsigned int curand(curandState *state) {
 unsigned int head=state->v[0];
 unsigned int tail=state->v[4];
 unsigned int shifted=head^(head>>2u);
 unsigned int next=tail^(tail<<4u)^shifted^(shifted<<1u);
 for(int j=0;j<4;j++)state->v[j]=state->v[j+1];
 state->v[4]=next;
 state->d+=362437u;
 return next+state->d;
}
__device__ float curand_uniform(curandState *state) {
 return float(curand(state))*2.3283064365386963e-10f+1.1641532182693481e-10f;
}
`;
