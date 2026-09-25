// SPDX-License-Identifier: MIT
// Stable key/value ordering; preserve raw key bits and use the original position for ties.
const sortBody=String.raw`
__global__ void sortPrepare(const uint*keys,const uint*values,uint2*pairs,uint*order,uint count,uint padded){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<padded){pairs[i]=i<count?make_uint2(keys[i],values[i]):make_uint2(0xffffffffu,0u);order[i]=i;}}
__global__ void sortStage(uint2*pairs,uint*order,uint count,uint size,uint stride){uint i=blockIdx.x*blockDim.x+threadIdx.x,j=i^stride;if(i<count&&j>i&&j<count){uint2 a=pairs[i],b=pairs[j];uint ai=order[i],bi=order[j];uint ak=sortKey(a.x),bk=sortKey(b.x);bool greater=ak>bk||(ak==bk&&ai>bi);bool ascending=(i&size)==0;if(greater==ascending){pairs[i]=b;pairs[j]=a;order[i]=bi;order[j]=ai;}}}
__global__ void sortFinish(const uint2*pairs,uint*keys,uint*values,uint count){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){uint2 p=pairs[i];keys[i]=p.x;values[i]=p.y;}}
`;

export const SORT_SOURCE='__device__ uint sortKey(uint bits){return bits;}\n'+sortBody;
// Numeric ascending order, signed zeros equal, NaNs last and stable. Values and
// original key bits (including NaN payloads and subnormals) are never converted.
export const SORT_FLOAT_SOURCE='__device__ uint sortKey(uint bits){uint magnitude=bits&0x7fffffffu;if(magnitude>0x7f800000u)return 0xffffffffu;if(magnitude==0u)return 0x80000000u;return (bits&0x80000000u)!=0u?~bits:(bits^0x80000000u);}\n'+sortBody;
