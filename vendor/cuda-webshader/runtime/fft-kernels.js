// SPDX-License-Identifier: MIT
// Unnormalized complex FFT. One workgroup handles one row or column.
// Runtime replacement for a library operation, compiled through the CUDA frontend.
const axisSource=(entry,sign)=>String.raw`
__global__ void ${entry}(const float2*input,float2*output,unsigned width,unsigned height,unsigned axis){
 __shared__ float2 values[2048];
 unsigned lane=threadIdx.x,line=blockIdx.x,n=axis==0?width:height;
 for(unsigned i=lane;i<n;i+=blockDim.x){
  unsigned reverse=0,bits=n;for(unsigned v=i;bits>1;bits>>=1){reverse=(reverse<<1)|(v&1);v>>=1;}
  unsigned index=axis==0?line*width+i:i*width+line;values[reverse]=input[index];
 }
 __syncthreads();
 for(unsigned size=2;size<=n;size<<=1){
  unsigned half=size>>1;
  for(unsigned butterfly=lane;butterfly<n/2;butterfly+=blockDim.x){
   unsigned j=butterfly&(half-1),base=(butterfly/half)*size;
   float angle=${sign}6.2831853071795864769f*(float)j/(float)size;
   float2 a=values[base+j],b=values[base+j+half];float c=cosf(angle),s=sinf(angle);
   float2 product=make_float2(b.x*c-b.y*s,b.x*s+b.y*c);
   values[base+j]=make_float2(a.x+product.x,a.y+product.y);
   values[base+j+half]=make_float2(a.x-product.x,a.y-product.y);
  }
  __syncthreads();
 }
 for(unsigned i=lane;i<n;i+=blockDim.x){unsigned index=axis==0?line*width+i:i*width+line;output[index]=values[i];}
}
`;

export const FFT_SOURCE=axisSource('inverseFftAxis','');
export const FORWARD_FFT_SOURCE=axisSource('forwardFftAxis','-');
export const REAL_FFT_SOURCE=String.raw`
__global__ void realToComplex(const float*input,float2*output,unsigned width,unsigned height,unsigned stride){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<width*height){unsigned x=i%width,y=i/width;output[i]=make_float2(input[y*stride+x],0.0f);}
}
__global__ void packSpectrum(const float2*input,float2*output,unsigned width,unsigned height){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x,packed=width/2+1;if(i<packed*height){unsigned x=i%packed,y=i/packed;output[i]=input[y*width+x];}
}
__global__ void unpackSpectrum(const float2*input,float2*output,unsigned width,unsigned height){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x,packed=width/2+1;if(i<width*height){unsigned x=i%width,y=i/width;if(x<packed)output[i]=input[y*packed+x];else{float2 p=input[((height-y)%height)*packed+width-x];output[i]=make_float2(p.x,-p.y);}}
}
__global__ void complexToReal(const float2*input,float*output,unsigned width,unsigned height,unsigned stride){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<width*height){unsigned x=i%width,y=i/width;output[y*stride+x]=input[i].x;}
}
`;
