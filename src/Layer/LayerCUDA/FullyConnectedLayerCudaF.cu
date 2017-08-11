#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <FLOAT.h>
#include "MkFn.h"
#include "../../Lib/Lib.h"
#include "LibCuda.h"
#include "FullyConnectedLayerCudaF.h"
__constant__ int _BatchSize;
__constant__ float _LearningRate;

void FullyConnectedLayerCudaF::GetFieldName(int field_idx, wchar_t* name){
	switch(field_idx){
		case 0: wcscpy(name, L"X"); break;
		case 1: wcscpy(name, L"Y"); break;
		case 2: wcscpy(name, L"x"); break;
		case 3: wcscpy(name, L"y"); break;
		case 4: wcscpy(name, L"w"); break;
		case 5: wcscpy(name, L"b"); break;
		case 6: wcscpy(name, L"u"); break;
		case 7: wcscpy(name, L"delta_x"); break;
		case 8: wcscpy(name, L"delta_y"); break;
		case 9: wcscpy(name, L"delta_w"); break;
		case 10: wcscpy(name, L"delta_b"); break;
		case 11: wcscpy(name, L"delta_u"); break;
		default: name[0] = 0; break;
	}
}

int FullyConnectedLayerCudaF::GetFieldDimension(int field_idx){
	switch(field_idx){
	case 0: return 0;
	case 1: return 0;
	case 2: return 1;
	case 3: return 1;
	case 4: return 2;
	case 5: return 1;
	case 6: return 1;
	case 7: return 1;
	case 8: return 1;
	case 9: return 2;
	case 10: return 1;
	case 11: return 1;
	default: return -1;
	}
}

int* FullyConnectedLayerCudaF::GetFieldSize(int field_idx){
	switch(field_idx){
	case 0: return 0;
	case 1: return 0;
	case 2:
		x_size_[0] = X;
		return x_size_;
	case 3:
		y_size_[0] = Y;
		return y_size_;
	case 4:
		w_size_[0] = Y;
		w_size_[1] = X;
		return w_size_;
	case 5:
		b_size_[0] = Y;
		return b_size_;
	case 6:
		u_size_[0] = Y;
		return u_size_;
	case 7:
		delta_x_size_[0] = X;
		return delta_x_size_;
	case 8:
		delta_y_size_[0] = Y;
		return delta_y_size_;
	case 9:
		delta_w_size_[0] = Y;
		delta_w_size_[1] = X;
		return delta_w_size_;
	case 10:
		delta_b_size_[0] = Y;
		return delta_b_size_;
	case 11:
		delta_u_size_[0] = Y;
		return delta_u_size_;
	default: return 0;
	}
}

void FullyConnectedLayerCudaF::GetFieldValue(int field_idx, void* dst){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(dst, &X, _cnt * sizeof(int)); break;
	case 1: memcpy(dst, &Y, _cnt * sizeof(int)); break;
	case 2: memcpy(dst, x, _cnt * sizeof(float)); break;
	case 3: memcpy(dst, y, _cnt * sizeof(float)); break;
	case 4: memcpy(dst, w, _cnt * sizeof(float)); break;
	case 5: memcpy(dst, b, _cnt * sizeof(float)); break;
	case 6: memcpy(dst, u, _cnt * sizeof(float)); break;
	case 7: memcpy(dst, delta_x, _cnt * sizeof(float)); break;
	case 8: memcpy(dst, delta_y, _cnt * sizeof(float)); break;
	case 9: memcpy(dst, delta_w, _cnt * sizeof(float)); break;
	case 10: memcpy(dst, delta_b, _cnt * sizeof(float)); break;
	case 11: memcpy(dst, delta_u, _cnt * sizeof(float)); break;
	}
}

void FullyConnectedLayerCudaF::SetFieldValue(int field_idx, void* src){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(&X, src, _cnt * sizeof(int)); break;
	case 1: memcpy(&Y, src, _cnt * sizeof(int)); break;
	case 2: memcpy(x, src, _cnt * sizeof(float)); break;
	case 3: memcpy(y, src, _cnt * sizeof(float)); break;
	case 4: memcpy(w, src, _cnt * sizeof(float)); break;
	case 5: memcpy(b, src, _cnt * sizeof(float)); break;
	case 6: memcpy(u, src, _cnt * sizeof(float)); break;
	case 7: memcpy(delta_x, src, _cnt * sizeof(float)); break;
	case 8: memcpy(delta_y, src, _cnt * sizeof(float)); break;
	case 9: memcpy(delta_w, src, _cnt * sizeof(float)); break;
	case 10: memcpy(delta_b, src, _cnt * sizeof(float)); break;
	case 11: memcpy(delta_u, src, _cnt * sizeof(float)); break;
	}
}

FullyConnectedLayerCudaF::FullyConnectedLayerCudaF(int x_size, int y_size){

    // X = x_size
    X = x_size;


    // Y = y_size
    Y = y_size;

	SetNormalRand(w, Y * X);
	SetNormalRand(b, Y);
	_chk(cudaStreamCreate(&_stream_y));
	_chk(cudaStreamCreate(&_stream_u));
	_chk(cudaStreamCreate(&_stream_delta_x));
	_chk(cudaStreamCreate(&_stream_delta_w));
	_chk(cudaStreamCreate(&_stream_delta_b));
	_chk(cudaStreamCreate(&_stream_delta_u));

	_chk(cudaEventCreate(&_event_y));
	_chk(cudaEventCreate(&_event_u));
	_chk(cudaEventCreate(&_event_delta_x));
	_chk(cudaEventCreate(&_event_delta_w));
	_chk(cudaEventCreate(&_event_delta_b));
	_chk(cudaEventCreate(&_event_delta_u));

}

FullyConnectedLayerCudaF::~FullyConnectedLayerCudaF(){
	Free();
	_chk(cudaStreamDestroy(_stream_y));
	_chk(cudaStreamDestroy(_stream_u));
	_chk(cudaStreamDestroy(_stream_delta_x));
	_chk(cudaStreamDestroy(_stream_delta_w));
	_chk(cudaStreamDestroy(_stream_delta_b));
	_chk(cudaStreamDestroy(_stream_delta_u));

	_chk(cudaEventDestroy(_event_y));
	_chk(cudaEventDestroy(_event_u));
	_chk(cudaEventDestroy(_event_delta_x));
	_chk(cudaEventDestroy(_event_delta_w));
	_chk(cudaEventDestroy(_event_delta_b));
	_chk(cudaEventDestroy(_event_delta_u));

	_Free(w);
	_Free(b);

}

void FullyConnectedLayerCudaF::Allocate(){
	_chk(_Malloc(y, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(u, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(delta_x, BatchSize * X * sizeof(float))); 
	_chk(_Malloc(delta_w, BatchSize * Y * X * sizeof(float))); 
	_chk(_Malloc(delta_b, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(delta_u, BatchSize * Y * sizeof(float))); 
}

void FullyConnectedLayerCudaF::Free(){
	_chk(_Free(y)); 
	_chk(_Free(u)); 
	_chk(_Free(delta_x)); 
	_chk(_Free(delta_w)); 
	_chk(_Free(delta_b)); 
	_chk(_Free(delta_u)); 
}
__global__ static void forward_u(float* u, int X, float* x, float* w, float* b, int Y){
	int _batch_idx = threadIdx.x;
	int i = blockIdx.x;

    // u[i] = (from j in Range(X) select x[j] * w[i, j]).Sum() + b[i]
    float _wk1 = 0;
    for(int j = 0; j < X; j++){
        _wk1 += x[(j) * _BatchSize + _batch_idx] * w[X * i + j];
    }
    u[(i) * _BatchSize + _batch_idx] = _wk1 + b[i];

}
void FullyConnectedLayerCudaF::Start_forward_u(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	forward_u<<<blocksPerGrid, threadsPerBlock, 0, _stream_u>>>(u, X, x, w, b, Y);
	_chk(cudaEventRecord(_event_u, _stream_u));
}
__global__ static void forward_y(float* y, float* u, int Y){
	int _batch_idx = threadIdx.x;
	int i = blockIdx.x;

    // y[i] = sigmoid(u[i])
    y[(i) * _BatchSize + _batch_idx] = sigmoid(u[(i) * _BatchSize + _batch_idx]);

}
void FullyConnectedLayerCudaF::Start_forward_y(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_y, _event_u, 0));
	forward_y<<<blocksPerGrid, threadsPerBlock, 0, _stream_y>>>(y, u, Y);
	_chk(cudaEventRecord(_event_y, _stream_y));
}
void FullyConnectedLayerCudaF::Forward(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	Start_forward_u();
	Start_forward_y();
}
__global__ static void backward_delta_u(float* delta_u, float* delta_y, float* u, int Y){
	int _batch_idx = threadIdx.x;
	int i = blockIdx.x;

    // delta_u[i] = delta_y[i] * sigmoid_prime(u[i])
    delta_u[(i) * _BatchSize + _batch_idx] = delta_y[(i) * _BatchSize + _batch_idx] * sigmoid_prime(u[(i) * _BatchSize + _batch_idx]);

}
void FullyConnectedLayerCudaF::Start_backward_delta_u(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	backward_delta_u<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_u>>>(delta_u, delta_y, u, Y);
	_chk(cudaEventRecord(_event_delta_u, _stream_delta_u));
}
__global__ static void backward_delta_x(float* delta_x, int Y, float* delta_u, float* w, int X){
	int _batch_idx = threadIdx.x;
	int i_j = blockIdx.x;

    // delta_x[i_j] = (from i in Range(Y) select delta_u[i] * w[i, i_j]).Sum()
    float _wk2 = 0;
    for(int i = 0; i < Y; i++){
        _wk2 += delta_u[(i) * _BatchSize + _batch_idx] * w[X * i + i_j];
    }
    delta_x[(i_j) * _BatchSize + _batch_idx] = _wk2;

}
void FullyConnectedLayerCudaF::Start_backward_delta_x(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = X;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_x, _event_delta_u, 0));
	backward_delta_x<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_x>>>(delta_x, Y, delta_u, w, X);
	_chk(cudaEventRecord(_event_delta_x, _stream_delta_x));
}
__global__ static void backward_delta_w(float* delta_w, float* delta_u, float* x, int Y, int X){
	int _batch_idx = threadIdx.x;
	int i = blockIdx.y;
	int i_j = blockIdx.x;

    // delta_w[i, i_j] = delta_u[i] * x[i_j]
    delta_w[(X * i + i_j) * _BatchSize + _batch_idx] = delta_u[(i) * _BatchSize + _batch_idx] * x[(i_j) * _BatchSize + _batch_idx];

}
void FullyConnectedLayerCudaF::Start_backward_delta_w(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_y = Y;
	blocks_x = X;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_w, _event_delta_u, 0));
	backward_delta_w<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_w>>>(delta_w, delta_u, x, Y, X);
	_chk(cudaEventRecord(_event_delta_w, _stream_delta_w));
}
__global__ static void backward_delta_b(float* delta_b, float* delta_u, int Y){
	int _batch_idx = threadIdx.x;
	int i = blockIdx.x;

    // delta_b[i] = delta_u[i]
    delta_b[(i) * _BatchSize + _batch_idx] = delta_u[(i) * _BatchSize + _batch_idx];

}
void FullyConnectedLayerCudaF::Start_backward_delta_b(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_b, _event_delta_u, 0));
	backward_delta_b<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_b>>>(delta_b, delta_u, Y);
	_chk(cudaEventRecord(_event_delta_b, _stream_delta_b));
}
void FullyConnectedLayerCudaF::Backward(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	Start_backward_delta_u();
	Start_backward_delta_x();
	Start_backward_delta_w();
	Start_backward_delta_b();
}
__global__ static void UpdateParameterKernel_0(float* w, float* delta_w){
	int _idx = (blockIdx.x) * blockDim.x + threadIdx.x;
	int offset = _idx * _BatchSize;
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_w[offset + i];
		}
		w[_idx] -= _LearningRate * sum;
	}
}
void FullyConnectedLayerCudaF::UpdateParameter_0(){
	int threads_x = 1;
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	threads_x = X;
	dim3 threadsPerBlock = dim3(threads_x);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	UpdateParameterKernel_0<<<blocksPerGrid, threadsPerBlock>>>(w, delta_w);
}
__global__ static void UpdateParameterKernel_1(float* b, float* delta_b){
	int _idx = threadIdx.x;
	int offset = _idx * _BatchSize;
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_b[offset + i];
		}
		b[_idx] -= _LearningRate * sum;
	}
}
void FullyConnectedLayerCudaF::UpdateParameter_1(){
	int threads_x = 1;
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	threads_x = Y;
	dim3 threadsPerBlock = dim3(threads_x);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	UpdateParameterKernel_1<<<blocksPerGrid, threadsPerBlock>>>(b, delta_b);
}
void FullyConnectedLayerCudaF::UpdateParameter(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	_chk(_MemcpyToSymbol(_LearningRate, LearningRate, sizeof(LearningRate)));
	_chk(cudaDeviceSynchronize());
	UpdateParameter_0();
	UpdateParameter_1();
	_chk(cudaDeviceSynchronize());
}

extern "C" DllExport Layer* MakeFullyConnectedLayerCudaF(int x_size, int y_size){
	return new FullyConnectedLayerCudaF(x_size, y_size);
}
