// SPDX-License-Identifier: MIT
// Runtime implementation of an exclusive uint scan; compiled by the same CUDA frontend.
export const SCAN_SOURCE=String.raw`
__global__ void scanBlocks(const unsigned int* input,unsigned int* output,unsigned int* totals,unsigned int count){
 __shared__ unsigned int temp[512];unsigned int lane=threadIdx.x,base=blockIdx.x*512;
 temp[lane]=base+lane<count?input[base+lane]:0;
 temp[lane+256]=base+lane+256<count?input[base+lane+256]:0;
 __syncthreads();
 for(unsigned int offset=1;offset<512;offset<<=1){unsigned int index=(lane+1)*offset*2-1;if(index<512)temp[index]+=temp[index-offset];__syncthreads();}
 if(lane==0){totals[blockIdx.x]=temp[511];temp[511]=0;}
 __syncthreads();
 for(unsigned int offset=256;offset>0;offset>>=1){unsigned int index=(lane+1)*offset*2-1;if(index<512){unsigned int value=temp[index-offset];temp[index-offset]=temp[index];temp[index]+=value;}__syncthreads();}
 if(base+lane<count)output[base+lane]=temp[lane];
 if(base+lane+256<count)output[base+lane+256]=temp[lane+256];
}
__global__ void addScanOffsets(unsigned int* output,const unsigned int* offsets,unsigned int count){unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]+=offsets[i/512];}
`;
