// SPDX-License-Identifier: MIT
// Native Windows host. The water/optics implementation is shared verbatim with WebGPU.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#define UNICODE
#define _UNICODE
#include <windows.h>
#include <windowsx.h>
#include <commctrl.h>
#include <d3d11.h>
#include <dxgi.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../src/clearwater.cu"

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;
static void check(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void hr(HRESULT h) { if(FAILED(h)) { std::ostringstream s;s<<"Windows graphics error 0x"<<std::hex<<(unsigned)h;throw std::runtime_error(s.str()); } }
template<class T> struct Buffer {
  T* p=nullptr;size_t count=0;
  ~Buffer(){ if(p)cudaFree(p); }
  void alloc(size_t n){if(p)check(cudaFree(p));p=nullptr;count=n;check(cudaMalloc((void**)&p,n*sizeof(T)));check(cudaMemset(p,0,n*sizeof(T)));}
  std::vector<T> read(){std::vector<T> v(count);check(cudaMemcpy(v.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
struct Camera {float x=0,z=0,y=1.55f,yaw=0,pitch=-.4f,speed=2.2f;};
enum {Clear=100,Open,Energy,Depth,Exposure,Speed,Quality,View,Glare,Cruise,Pause,Reset,Capture};
struct App {
 HWND water=nullptr,controls=nullptr,status=nullptr,stats=nullptr;
 HWND energyLabel=nullptr,depthLabel=nullptr,exposureLabel=nullptr,speedLabel=nullptr;
 HFONT font=nullptr,heading=nullptr;HBRUSH background=nullptr;
 Camera cam;float energy=1,depth=1.6f,exposure=1.1f,time=0,cx=0,cz=0,accumulator=0;
 bool playing=true,glare=true,cruise=false,running=true,drag=false,tap=false,capture=false,resizePending=true;
 bool keys[256]={};POINT previous{},start{};float tapX=0,tapY=0;int quality=1152,view=0,width=0,height=0,ripIndex=0,frames=0;
 float frameMs=0;std::string gpuName;fs::path executableDir,assetPath,outputDir;
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;ComPtr<IDXGISwapChain> swap;
 ComPtr<ID3D11Texture2D> texture;cudaGraphicsResource* shared=nullptr;
 Buffer<float2> seed;Buffer<float> rows,scales;
 Buffer<float4> fft[2],surface,rip[2],ripNormals,caustics,pebbles,hdr,bloom[2],lens[2],lensKernel;
 Buffer<unsigned> photons,pixels;
 const dim3 block{8,8,1},wavesGrid{32,32,3},ripGrid{32,32,1};
 ~App(){cudaDeviceSynchronize();if(shared)cudaGraphicsUnregisterResource(shared);if(font)DeleteObject(font);if(heading)DeleteObject(heading);if(background)DeleteObject(background);}
 void paths(){wchar_t p[32768];GetModuleFileNameW(nullptr,p,32768);executableDir=fs::path(p).parent_path();outputDir=executableDir/L"output";fs::create_directories(outputDir);for(auto candidate:{executableDir/L"assets/seabed.jpg",executableDir/L"../../assets/seabed.jpg",fs::current_path()/L"assets/seabed.jpg"})if(fs::exists(candidate)){assetPath=fs::canonical(candidate);break;}if(assetPath.empty())throw std::runtime_error("Missing assets/seabed.jpg beside ClearwaterNative.exe");}
 void graphics(){
  ComPtr<IDXGIFactory> factory;hr(CreateDXGIFactory(__uuidof(IDXGIFactory),(void**)factory.GetAddressOf()));
  ComPtr<IDXGIAdapter> selected;
  for(UINT i=0;;i++){ComPtr<IDXGIAdapter> a;if(factory->EnumAdapters(i,a.GetAddressOf())==DXGI_ERROR_NOT_FOUND)break;int ordinal=-1;auto e=cudaD3D11GetDevice(&ordinal,a.Get());if(e==cudaSuccess){selected=a;check(cudaSetDevice(ordinal));cudaDeviceProp info{};check(cudaGetDeviceProperties(&info,ordinal));gpuName=info.name;break;}cudaGetLastError();}
  if(!selected)throw std::runtime_error("An NVIDIA CUDA-capable display adapter is required.");
  D3D_FEATURE_LEVEL level;hr(D3D11CreateDevice(selected.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,nullptr,0,D3D11_SDK_VERSION,device.GetAddressOf(),&level,context.GetAddressOf()));
  DXGI_SWAP_CHAIN_DESC d{};d.BufferDesc.Width=1152;d.BufferDesc.Height=720;d.BufferDesc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;d.SampleDesc.Count=1;d.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;d.BufferCount=2;d.OutputWindow=water;d.Windowed=TRUE;d.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;
  hr(factory->CreateSwapChain(device.Get(),&d,swap.GetAddressOf()));hr(factory->MakeWindowAssociation(water,DXGI_MWA_NO_ALT_ENTER));
 }
 void decodeAsset(){
  ComPtr<IWICImagingFactory> wic;hr(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(wic.GetAddressOf())));
  ComPtr<IWICBitmapDecoder> decoder;hr(wic->CreateDecoderFromFilename(assetPath.c_str(),nullptr,GENERIC_READ,WICDecodeMetadataCacheOnLoad,decoder.GetAddressOf()));ComPtr<IWICBitmapFrameDecode> frame;hr(decoder->GetFrame(0,frame.GetAddressOf()));UINT w,h;hr(frame->GetSize(&w,&h));if(w!=1024||h!=1024)throw std::runtime_error("Seabed asset must be 1024 x 1024.");
  ComPtr<IWICFormatConverter> convert;hr(wic->CreateFormatConverter(convert.GetAddressOf()));hr(convert->Initialize(frame.Get(),GUID_WICPixelFormat32bppRGBA,WICBitmapDitherTypeNone,nullptr,0,WICBitmapPaletteTypeCustom));std::vector<unsigned char> bytes(w*h*4);hr(convert->CopyPixels(nullptr,w*4,(UINT)bytes.size(),bytes.data()));std::vector<float4> linear(w*h);for(size_t i=0;i<linear.size();i++)linear[i]=make_float4(std::pow(bytes[i*4]/255.f,2.2f),std::pow(bytes[i*4+1]/255.f,2.2f),std::pow(bytes[i*4+2]/255.f,2.2f),1);pebbles.alloc(linear.size());check(cudaMemcpy(pebbles.p,linear.data(),linear.size()*sizeof(float4),cudaMemcpyHostToDevice));
 }
 void transform(float4* a,float4* b,float sign){for(int axis=0;axis<2;axis++)for(int p=1;p<256;p*=2){fft_pass<<<wavesGrid,block>>>(a,b,p,axis,sign);std::swap(a,b);}}
 void initGpu(){
  seed.alloc(3*65536);rows.alloc(768);scales.alloc(3);surface.alloc(3*65536);ripNormals.alloc(65536);photons.alloc(512*512*3);caustics.alloc(512*512);lensKernel.alloc(3*65536);
  for(int i=0;i<2;i++){fft[i].alloc(3*65536);rip[i].alloc(65536);lens[i].alloc(3*65536);}
  decodeAsset();seed_spectrum<<<wavesGrid,block>>>(seed.p,7);spectrum_rows<<<12,64>>>(seed.p,rows.p);spectrum_norm<<<1,64>>>(rows.p,scales.p);
  lens_aperture<<<wavesGrid,block>>>(lens[0].p);transform(lens[0].p,lens[1].p,-1);lens_power<<<wavesGrid,block>>>(lens[0].p,lens[1].p);lens_rows<<<12,64>>>(lens[1].p,rows.p);lens_normalize<<<wavesGrid,block>>>(lens[1].p,rows.p,lens[0].p);transform(lens[0].p,lens[1].p,-1);check(cudaMemcpy(lensKernel.p,lens[0].p,3*65536*sizeof(float4),cudaMemcpyDeviceToDevice));check(cudaGetLastError());check(cudaDeviceSynchronize());
 }
 void resize(){
  if(!resizePending)return;resizePending=false;RECT r;GetClientRect(water,&r);if(r.right<1||r.bottom<1)return;
  check(cudaDeviceSynchronize());if(shared){check(cudaGraphicsUnregisterResource(shared));shared=nullptr;}texture.Reset();context->ClearState();context->Flush();width=quality;height=std::max(8,((int)std::ceil((double)width*r.bottom/r.right)+7)/8*8);
  hr(swap->ResizeBuffers(2,width,height,DXGI_FORMAT_R8G8B8A8_UNORM,0));D3D11_TEXTURE2D_DESC desc{};desc.Width=width;desc.Height=height;desc.MipLevels=1;desc.ArraySize=1;desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;desc.SampleDesc.Count=1;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
  hr(device->CreateTexture2D(&desc,nullptr,texture.GetAddressOf()));check(cudaGraphicsD3D11RegisterResource(&shared,texture.Get(),cudaGraphicsRegisterFlagsNone));check(cudaGraphicsResourceSetMapFlags(shared,cudaGraphicsMapFlagsWriteDiscard));
  pixels.alloc((size_t)width*height);hdr.alloc((size_t)width*height);bloom[0].alloc((size_t)width*height);bloom[1].alloc((size_t)width*height);
 }
 void look(int dx,int dy){cam.yaw+=dx*.003f;cam.pitch=std::clamp(cam.pitch-dy*.003f,-1.55f,1.55f);}
 void wheel(int delta){cam.speed=std::clamp(cam.speed*std::exp(delta*.002f),.1f,200.f);labels();}
 void move(float dt){
  cam.yaw+=(keys[VK_RIGHT]-keys[VK_LEFT])*dt;cam.pitch=std::clamp(cam.pitch+(keys[VK_UP]-keys[VK_DOWN])*dt,-1.55f,1.55f);
  float f=float(keys['W']-keys['S'])+(cruise?1.f:0.f),s=float(keys['D']-keys['A']),u=float(keys['E']-keys['Q']),len=std::max(1.f,std::sqrt(f*f+s*s+u*u));f/=len;s/=len;u/=len;float step=cam.speed*(keys[VK_SHIFT]?6.f:1.f)*dt;
  cam.x+=step*(std::sin(cam.yaw)*std::cos(cam.pitch)*f+std::cos(cam.yaw)*s);cam.z+=step*(-std::cos(cam.yaw)*std::cos(cam.pitch)*f+std::sin(cam.yaw)*s);cam.y=std::max(.65f,cam.y+step*(std::sin(cam.pitch)*f+u));
 }
 void step(float dt){
  move(dt);if(playing)time+=dt;
  evolve_spectrum<<<wavesGrid,block>>>(seed.p,scales.p,fft[0].p,time,energy,depth);transform(fft[0].p,fft[1].p,1);resolve_surface<<<wavesGrid,block>>>(fft[0].p,surface.p);
  if(playing)accumulator=std::min(.1f,accumulator+dt);
  while(accumulator>=1.f/120){float nx=std::round(cam.x*16)/16,nz=std::round(cam.z*16)/16;int sx=(int)std::round((nx-cx)*16),sz=(int)std::round((nz-cz)*16);cx=nx;cz=nz;
   ripple_step<<<ripGrid,block>>>(rip[ripIndex].p,rip[1-ripIndex].p,sx,sz,cx,cz,cam.x,cam.z,cam.y,cam.yaw,cam.pitch,float(width)/height,tapX,tapY,tap?1:0);tap=false;ripIndex=1-ripIndex;accumulator-=1.f/120;
  }ripple_normals<<<ripGrid,block>>>(rip[ripIndex].p,ripNormals.p);
 }
 void draw(){
  dim3 grid(width/8,height/8,1);clear_caustics<<<dim3(64,64,1),block>>>(photons.p);trace_caustics<<<dim3(128,128,1),block>>>(surface.p,photons.p,depth);filter_caustics<<<dim3(64,64,1),block>>>(photons.p,caustics.p);
  render_water<<<grid,block>>>(surface.p,ripNormals.p,caustics.p,pebbles.p,hdr.p,width,height,cam.x,cam.z,cam.y,cam.yaw,cam.pitch,cx,cz,depth,time,view);
  if(glare){glare_source<<<wavesGrid,block>>>(hdr.p,lens[0].p,width,height);transform(lens[0].p,lens[1].p,-1);glare_multiply<<<wavesGrid,block>>>(lens[0].p,lensKernel.p,lens[1].p);transform(lens[1].p,lens[0].p,1);}
  bloom_pass<<<grid,block>>>(hdr.p,bloom[0].p,width,height,0);bloom_pass<<<grid,block>>>(bloom[0].p,bloom[1].p,width,height,1);present<<<grid,block>>>(hdr.p,bloom[1].p,lens[1].p,pixels.p,width,height,exposure,glare?1:0);check(cudaGetLastError());
  check(cudaGraphicsMapResources(1,&shared));cudaArray_t array;check(cudaGraphicsSubResourceGetMappedArray(&array,shared,0,0));check(cudaMemcpy2DToArray(array,0,0,pixels.p,width*4,width*4,height,cudaMemcpyDeviceToDevice));check(cudaGraphicsUnmapResources(1,&shared));ComPtr<ID3D11Texture2D> back;hr(swap->GetBuffer(0,IID_PPV_ARGS(back.GetAddressOf())));context->CopyResource(back.Get(),texture.Get());hr(swap->Present(1,0));
 }
 void png(const fs::path& path){
  auto bytes=pixels.read();for(auto &pixel:bytes)pixel=(pixel&0xff00ff00u)|((pixel&0xffu)<<16)|((pixel>>16)&0xffu);ComPtr<IWICImagingFactory> wic;hr(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(wic.GetAddressOf())));ComPtr<IWICStream> stream;hr(wic->CreateStream(stream.GetAddressOf()));hr(stream->InitializeFromFilename(path.c_str(),GENERIC_WRITE));ComPtr<IWICBitmapEncoder> encoder;hr(wic->CreateEncoder(GUID_ContainerFormatPng,nullptr,encoder.GetAddressOf()));hr(encoder->Initialize(stream.Get(),WICBitmapEncoderNoCache));ComPtr<IWICBitmapFrameEncode> frame;ComPtr<IPropertyBag2> props;hr(encoder->CreateNewFrame(frame.GetAddressOf(),props.GetAddressOf()));hr(frame->Initialize(props.Get()));hr(frame->SetSize(width,height));WICPixelFormatGUID format=GUID_WICPixelFormat32bppBGRA;hr(frame->SetPixelFormat(&format));if(format!=GUID_WICPixelFormat32bppBGRA)throw std::runtime_error("PNG encoder did not accept BGRA.");hr(frame->WritePixels(height,width*4,(UINT)(bytes.size()*4),(BYTE*)bytes.data()));hr(frame->Commit());hr(encoder->Commit());SetWindowTextW(status,(L"Saved "+path.filename().wstring()).c_str());
 }
 HWND control(const wchar_t* cls,const wchar_t* text,int id,int x,int y,int w,int h,DWORD style=0){HWND child=CreateWindowExW(0,cls,text,WS_CHILD|WS_VISIBLE|style,x,y,w,h,controls,(HMENU)(INT_PTR)id,GetModuleHandleW(nullptr),nullptr);if(!child)throw std::runtime_error("Could not create a native control.");SendMessageW(child,WM_SETFONT,(WPARAM)font,TRUE);return child;}
 HWND slider(const wchar_t* title,int id,int y,int low,int high){auto label=control(L"STATIC",title,0,24,y,300,22);auto h=control(TRACKBAR_CLASSW,L"",id,20,y+25,300,28,TBS_HORZ|TBS_NOTICKS|WS_TABSTOP);SendMessageW(h,TBM_SETRANGE,TRUE,MAKELONG(low,high));return label;}
 void labels(){if(!controls)return;wchar_t text[128];swprintf_s(text,L"Wave energy                     %.2f",energy);SetWindowTextW(energyLabel,text);swprintf_s(text,L"Water depth                      %.1f m",depth);SetWindowTextW(depthLabel,text);swprintf_s(text,L"Exposure                           %.2f",exposure);SetWindowTextW(exposureLabel,text);swprintf_s(text,L"Flight speed                      %.1f m/s",cam.speed);SetWindowTextW(speedLabel,text);
  SendDlgItemMessageW(controls,Energy,TBM_SETPOS,TRUE,(LPARAM)std::lround(energy*100));SendDlgItemMessageW(controls,Depth,TBM_SETPOS,TRUE,(LPARAM)std::lround(depth*10));SendDlgItemMessageW(controls,Exposure,TBM_SETPOS,TRUE,(LPARAM)std::lround(exposure*100));SendDlgItemMessageW(controls,Speed,TBM_SETPOS,TRUE,(LPARAM)std::lround(std::log(cam.speed/.1f)/std::log(2000.f)*1000));SetDlgItemTextW(controls,Pause,playing?L"Pause":L"Resume");
 }
 void controlsUi(){
  font=CreateFontW(-15,0,0,0,FW_NORMAL,FALSE,FALSE,FALSE,DEFAULT_CHARSET,OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,L"Segoe UI");heading=CreateFontW(-30,0,0,0,FW_NORMAL,FALSE,FALSE,FALSE,DEFAULT_CHARSET,OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,L"Georgia");background=CreateSolidBrush(RGB(24,51,58));
  control(L"STATIC",L"CLEARWATER  /  NATIVE CUDA",0,24,22,305,20);auto title=control(L"STATIC",L"Room to drift.",0,24,52,305,42);SendMessageW(title,WM_SETFONT,(WPARAM)heading,TRUE);
  control(L"BUTTON",L"Clearwater",Clear,24,107,142,32,WS_TABSTOP);control(L"BUTTON",L"Open water",Open,176,107,142,32,WS_TABSTOP);
  energyLabel=slider(L"Wave energy",Energy,162,15,300);depthLabel=slider(L"Water depth",Depth,227,5,160);exposureLabel=slider(L"Exposure",Exposure,292,40,240);speedLabel=slider(L"Flight speed",Speed,357,0,1000);
  control(L"STATIC",L"Resolution",0,24,430,138,20);control(L"STATIC",L"Surface view",0,176,430,140,20);
  auto q=control(L"COMBOBOX",L"",Quality,24,455,142,180,CBS_DROPDOWNLIST|WS_TABSTOP);for(auto s:{L"768 - Performance",L"1152 - Balanced",L"1536 - High"})SendMessageW(q,CB_ADDSTRING,0,(LPARAM)s);SendMessageW(q,CB_SETCURSEL,1,0);
  auto v=control(L"COMBOBOX",L"",View,176,455,142,180,CBS_DROPDOWNLIST|WS_TABSTOP);for(auto s:{L"Water",L"Caustics",L"Normals"})SendMessageW(v,CB_ADDSTRING,0,(LPARAM)s);SendMessageW(v,CB_SETCURSEL,0,0);
  control(L"BUTTON",L"Lens glare",Glare,24,497,140,24,BS_AUTOCHECKBOX|WS_TABSTOP);SendDlgItemMessageW(controls,Glare,BM_SETCHECK,BST_CHECKED,0);control(L"BUTTON",L"Drift forward",Cruise,176,497,142,24,BS_AUTOCHECKBOX|WS_TABSTOP);
  control(L"BUTTON",L"Pause",Pause,24,537,90,32,WS_TABSTOP);control(L"BUTTON",L"Reset view",Reset,125,537,92,32,WS_TABSTOP);control(L"BUTTON",L"Save PNG",Capture,228,537,90,32,WS_TABSTOP);
  control(L"STATIC",L"Drag / arrows: look   |   WASD: fly\nE / Q: up / down   |   Shift: 6x boost\nScroll: speed   |   Space: pause   |   H: controls",0,24,591,300,62);stats=control(L"STATIC",L"Starting CUDA...",0,24,666,300,52);status=control(L"STATIC",L"Shared CUDA kernels. GPU-only presentation.",0,24,728,300,40);labels();
 }
 void command(int id,int notification){
  if(id==Clear||id==Open){energy=id==Open?2.4f:1;depth=id==Open?12:1.6f;cam.y=id==Open?3:1.55f;cam.pitch=id==Open?-.19f:-.4f;}
  if(id==Pause)playing=!playing;if(id==Reset)cam=Camera{};if(id==Capture)capture=true;
  if(id==Glare)glare=SendDlgItemMessageW(controls,Glare,BM_GETCHECK,0,0)==BST_CHECKED;if(id==Cruise)cruise=SendDlgItemMessageW(controls,Cruise,BM_GETCHECK,0,0)==BST_CHECKED;
  if(id==Quality&&notification==CBN_SELCHANGE){int sel=(int)SendDlgItemMessageW(controls,Quality,CB_GETCURSEL,0,0);quality=sel==0?768:sel==1?1152:1536;resizePending=true;}if(id==View&&notification==CBN_SELCHANGE)view=(int)SendDlgItemMessageW(controls,View,CB_GETCURSEL,0,0);labels();
 }
 void scrollControl(HWND h){int id=GetDlgCtrlID(h),v=(int)SendMessageW(h,TBM_GETPOS,0,0);if(id==Energy)energy=v/100.f;if(id==Depth)depth=v/10.f;if(id==Exposure)exposure=v/100.f;if(id==Speed)cam.speed=.1f*std::pow(2000.f,v/1000.f);labels();}
 void smoke(){
  auto require=[](bool yes,const char* s){if(!yes)throw std::runtime_error(s);};
  Buffer<float4> a,b;a.alloc(3*65536);b.alloc(3*65536);std::vector<float4> input(3*65536,make_float4(0,0,0,0));input[1].x=.5f;input[255].x=.5f;input[65536+7*256+3]=make_float4(.3f,-.2f,0,0);check(cudaMemcpy(a.p,input.data(),input.size()*sizeof(float4),cudaMemcpyHostToDevice));transform(a.p,b.p,1);auto result=a.read();double error=0;for(int z=0;z<256;z++)for(int x=0;x<256;x++){double angle=6.283185307179586*(3*x+7*z)/256;error=std::max(error,std::abs(result[z*256+x].x-std::cos(6.283185307179586*x/256)));error=std::max(error,std::abs(result[65536+z*256+x].x-(.3*std::cos(angle)+.2*std::sin(angle))));}require(error<1e-5,"Native FFT reference mismatch");
  cam=Camera{};look(100,-100);require(cam.yaw>0&&cam.pitch>-.4f,"Camera direction mismatch");look(-100,100);float old=cam.speed;wheel(120);require(cam.speed>old,"Wheel speed mismatch");cam=Camera{};cam.y=4;cam.pitch=0;keys['D']=true;move(.1f);require(cam.x>0,"Strafe mismatch");keys['D']=false;keys['E']=true;move(.1f);require(cam.y>4,"Rise mismatch");keys['E']=false;cam=Camera{};cam.pitch=0;keys['W']=true;move(.1f);float base=cam.z;cam.z=0;keys[VK_SHIFT]=true;move(.1f);float ratio=cam.z/base;require(std::abs(ratio-6)<.001f,"Shift boost mismatch");std::fill(std::begin(keys),std::end(keys),false);cam=Camera{};
  playing=false;time=5;resize();step(0);draw();png(outputDir/L"native-clearwater.png");auto field=surface.read();for(auto f:field)require(std::isfinite(f.x)&&std::isfinite(f.y)&&std::isfinite(f.z),"Non-finite native wave field");
  playing=true;tap=true;tapX=.2f;tapY=-.6f;step(1.f/60);check(cudaDeviceSynchronize());auto ripData=rip[ripIndex].read();float peak=0;for(auto r:ripData)peak=std::max(peak,std::abs(r.x));require(peak>.00001f,"Native ripple was not generated");
  command(Open,BN_CLICKED);cam.x=10000;cam.z=-10000;time=20;playing=false;quality=768;resizePending=true;resize();step(0);draw();png(outputDir/L"native-open-water.png");
  for(int mode=1;mode<=2;mode++){view=mode;draw();}view=0;glare=false;draw();glare=true;draw();check(cudaDeviceSynchronize());require(IsWindow(water)&&IsWindow(controls)&&water!=controls,"Separate windows missing");
  auto image=hdr.read();for(auto c:image)require(std::isfinite(c.x)&&std::isfinite(c.y)&&std::isfinite(c.z),"Non-finite native HDR");
  std::ofstream log(outputDir/L"native-smoke.json");log<<"{\n  \"passed\": true,\n  \"gpu\": \""<<gpuName<<"\",\n  \"fftMaxError\": "<<error<<",\n  \"shiftRatio\": "<<ratio<<",\n  \"ripplePeak\": "<<peak<<",\n  \"separateWindows\": true,\n  \"resizeAndDebugViews\": true,\n  \"finiteAt10km\": true,\n  \"gpuPresentation\": \"CUDA-D3D11 device-to-device\"\n}\n";
 }
};
static App* app=nullptr;
static LRESULT CALLBACK windowProc(HWND w,UINT message,WPARAM wp,LPARAM lp){
 if(!app)return DefWindowProcW(w,message,wp,lp);
 if(message==WM_CLOSE){if(w==app->controls){ShowWindow(w,SW_HIDE);return 0;}app->running=false;DestroyWindow(w);return 0;}
 if(message==WM_DESTROY&&w==app->water){PostQuitMessage(0);return 0;}
 if(message==WM_CTLCOLORSTATIC||message==WM_CTLCOLORBTN){SetTextColor((HDC)wp,RGB(224,239,233));SetBkColor((HDC)wp,RGB(24,51,58));return (LRESULT)app->background;}
 if(message==WM_ERASEBKGND&&w==app->controls){RECT r;GetClientRect(w,&r);FillRect((HDC)wp,&r,app->background);return 1;}
 if(message==WM_COMMAND&&w==app->controls){app->command(LOWORD(wp),HIWORD(wp));return 0;}
 if(message==WM_HSCROLL&&w==app->controls){app->scrollControl((HWND)lp);return 0;}
 if(w==app->water){
  if(message==WM_SIZE){app->resizePending=true;return 0;}
  if(message==WM_KILLFOCUS){std::fill(std::begin(app->keys),std::end(app->keys),false);app->drag=false;ReleaseCapture();return 0;}
  if(message==WM_KEYDOWN){if(wp<256)app->keys[wp]=true;if(!(lp&(1<<30))){if(wp==VK_SPACE)app->command(Pause,0);if(wp=='H'){ShowWindow(app->controls,IsWindowVisible(app->controls)?SW_HIDE:SW_SHOWNOACTIVATE);}if(wp==VK_ESCAPE){app->running=false;}}return 0;}
  if(message==WM_KEYUP){if(wp<256)app->keys[wp]=false;return 0;}
  if(message==WM_LBUTTONDOWN){SetFocus(w);SetCapture(w);app->drag=true;app->previous=app->start={GET_X_LPARAM(lp),GET_Y_LPARAM(lp)};return 0;}
  if(message==WM_MOUSEMOVE&&app->drag){POINT p{GET_X_LPARAM(lp),GET_Y_LPARAM(lp)};app->look(p.x-app->previous.x,p.y-app->previous.y);app->previous=p;return 0;}
  if(message==WM_LBUTTONUP){if(app->drag){int x=GET_X_LPARAM(lp),y=GET_Y_LPARAM(lp);if(std::hypot(float(x-app->start.x),float(y-app->start.y))<6){RECT r;GetClientRect(w,&r);app->tapX=2.f*x/r.right-1;app->tapY=1-2.f*y/r.bottom;app->tap=true;}}app->drag=false;ReleaseCapture();return 0;}
  if(message==WM_MOUSEWHEEL){app->wheel(GET_WHEEL_DELTA_WPARAM(wp));return 0;}
 }return DefWindowProcW(w,message,wp,lp);
}
int WINAPI wWinMain(HINSTANCE instance,HINSTANCE,LPWSTR commandLine,int){
 int result=0;SetProcessDPIAware();HRESULT com=CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED);
 try{
  App instanceApp;app=&instanceApp;app->paths();INITCOMMONCONTROLSEX cc{sizeof(cc),ICC_BAR_CLASSES|ICC_STANDARD_CLASSES};InitCommonControlsEx(&cc);
  WNDCLASSEXW c{sizeof(c)};c.lpfnWndProc=windowProc;c.hInstance=instance;c.hCursor=LoadCursorW(nullptr,IDC_ARROW);c.lpszClassName=L"ClearwaterNativeWindow";c.hbrBackground=(HBRUSH)GetStockObject(BLACK_BRUSH);RegisterClassExW(&c);
  app->water=CreateWindowExW(0,c.lpszClassName,L"Clearwater - Native CUDA",WS_OVERLAPPEDWINDOW,30,45,1040,760,nullptr,nullptr,instance,nullptr);
  app->controls=CreateWindowExW(0,c.lpszClassName,L"Clearwater - Controls",WS_OVERLAPPED|WS_CAPTION|WS_SYSMENU|WS_MINIMIZEBOX,1090,45,360,820,nullptr,nullptr,instance,nullptr);
  if(!app->water||!app->controls)throw std::runtime_error("Could not create water and control windows.");app->controlsUi();ShowWindow(app->water,SW_SHOW);ShowWindow(app->controls,SW_SHOWNOACTIVATE);UpdateWindow(app->controls);
  app->graphics();app->initGpu();app->resize();bool smoke=std::wstring(commandLine).find(L"--smoke")!=std::wstring::npos;
  if(smoke){app->smoke();app->running=false;}
  auto last=std::chrono::steady_clock::now();MSG msg{};
  while(app->running){while(PeekMessageW(&msg,nullptr,0,0,PM_REMOVE)){if(msg.message==WM_QUIT){app->running=false;break;}if(!IsDialogMessageW(app->controls,&msg)){TranslateMessage(&msg);DispatchMessageW(&msg);}}if(!app->running)break;if(IsIconic(app->water)){Sleep(30);last=std::chrono::steady_clock::now();continue;}
   auto now=std::chrono::steady_clock::now();float dt=std::min(.05f,std::chrono::duration<float>(now-last).count());last=now;app->resize();app->step(dt);app->draw();app->frameMs=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-now).count();app->frames++;
   if(app->capture){app->capture=false;app->png(app->outputDir/(L"Clearwater-"+std::to_wstring(GetTickCount64())+L".png"));}
   if(app->frames%15==0){wchar_t text[256];swprintf_s(text,L"%d x %d  |  %.1f ms/frame\nSpeed %.1f m/s  |  Position %.0f, %.0f",app->width,app->height,app->frameMs,app->cam.speed*(app->keys[VK_SHIFT]?6:1),app->cam.x,app->cam.z);SetWindowTextW(app->stats,text);}
  }
  if(IsWindow(app->water))DestroyWindow(app->water);if(IsWindow(app->controls))DestroyWindow(app->controls);app=nullptr;
 }catch(const std::exception& e){result=1;wchar_t exe[32768];GetModuleFileNameW(nullptr,exe,32768);fs::path folder=fs::path(exe).parent_path()/L"output";std::error_code ec;fs::create_directories(folder,ec);std::ofstream(folder/L"error.log")<<e.what()<<"\n";if(std::wstring(commandLine).find(L"--smoke")==std::wstring::npos)MessageBoxA(nullptr,e.what(),"Clearwater native error",MB_OK|MB_ICONERROR);app=nullptr;}
 if(SUCCEEDED(com))CoUninitialize();return result;
}
