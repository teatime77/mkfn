#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <FLOAT.h>
#include "MkFn.h"
#include "../../Lib/Lib.h"
#include "LibCuda.h"
#include "RecurrentLayerCudaD.h"
__constant__ int _BatchSize;
__constant__ double _LearningRate;

void RecurrentLayerCudaD::GetFieldName(int field_idx, wchar_t* name){
	switch(field_idx){
		case 0: wcscpy(name, L"T"); break;
		case 1: wcscpy(name, L"X"); break;
		case 2: wcscpy(name, L"Y"); break;
		case 3: wcscpy(name, L"x"); break;
		case 4: wcscpy(name, L"y"); break;
		case 5: wcscpy(name, L"win"); break;
		case 6: wcscpy(name, L"w"); break;
		case 7: wcscpy(name, L"b"); break;
		case 8: wcscpy(name, L"u"); break;
		case 9: wcscpy(name, L"delta_x"); break;
		case 10: wcscpy(name, L"delta_y"); break;
		case 11: wcscpy(name, L"delta_win"); break;
		case 12: wcscpy(name, L"delta_w"); break;
		case 13: wcscpy(name, L"delta_b"); break;
		case 14: wcscpy(name, L"delta_u"); break;
		default: name[0] = 0; break;
	}
}

int RecurrentLayerCudaD::GetFieldDimension(int field_idx){
	switch(field_idx){
	case 0: return 0;
	case 1: return 0;
	case 2: return 0;
	case 3: return 2;
	case 4: return 2;
	case 5: return 2;
	case 6: return 2;
	case 7: return 1;
	case 8: return 2;
	case 9: return 2;
	case 10: return 2;
	case 11: return 2;
	case 12: return 2;
	case 13: return 1;
	case 14: return 2;
	default: return -1;
	}
}

int* RecurrentLayerCudaD::GetFieldSize(int field_idx){
	switch(field_idx){
	case 0: return 0;
	case 1: return 0;
	case 2: return 0;
	case 3:
		x_size_[0] = T;
		x_size_[1] = X;
		return x_size_;
	case 4:
		y_size_[0] = T;
		y_size_[1] = Y;
		return y_size_;
	case 5:
		win_size_[0] = Y;
		win_size_[1] = X;
		return win_size_;
	case 6:
		w_size_[0] = Y;
		w_size_[1] = Y;
		return w_size_;
	case 7:
		b_size_[0] = Y;
		return b_size_;
	case 8:
		u_size_[0] = T;
		u_size_[1] = Y;
		return u_size_;
	case 9:
		delta_x_size_[0] = T;
		delta_x_size_[1] = X;
		return delta_x_size_;
	case 10:
		delta_y_size_[0] = T;
		delta_y_size_[1] = Y;
		return delta_y_size_;
	case 11:
		delta_win_size_[0] = Y;
		delta_win_size_[1] = X;
		return delta_win_size_;
	case 12:
		delta_w_size_[0] = Y;
		delta_w_size_[1] = Y;
		return delta_w_size_;
	case 13:
		delta_b_size_[0] = Y;
		return delta_b_size_;
	case 14:
		delta_u_size_[0] = T;
		delta_u_size_[1] = Y;
		return delta_u_size_;
	default: return 0;
	}
}

void RecurrentLayerCudaD::GetFieldValue(int field_idx, void* dst){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(dst, &T, _cnt * sizeof(int)); break;
	case 1: memcpy(dst, &X, _cnt * sizeof(int)); break;
	case 2: memcpy(dst, &Y, _cnt * sizeof(int)); break;
	case 3: memcpy(dst, x, _cnt * sizeof(double)); break;
	case 4: memcpy(dst, y, _cnt * sizeof(double)); break;
	case 5: memcpy(dst, win, _cnt * sizeof(double)); break;
	case 6: memcpy(dst, w, _cnt * sizeof(double)); break;
	case 7: memcpy(dst, b, _cnt * sizeof(double)); break;
	case 8: memcpy(dst, u, _cnt * sizeof(double)); break;
	case 9: memcpy(dst, delta_x, _cnt * sizeof(double)); break;
	case 10: memcpy(dst, delta_y, _cnt * sizeof(double)); break;
	case 11: memcpy(dst, delta_win, _cnt * sizeof(double)); break;
	case 12: memcpy(dst, delta_w, _cnt * sizeof(double)); break;
	case 13: memcpy(dst, delta_b, _cnt * sizeof(double)); break;
	case 14: memcpy(dst, delta_u, _cnt * sizeof(double)); break;
	}
}

void RecurrentLayerCudaD::SetFieldValue(int field_idx, void* src){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(&T, src, _cnt * sizeof(int)); break;
	case 1: memcpy(&X, src, _cnt * sizeof(int)); break;
	case 2: memcpy(&Y, src, _cnt * sizeof(int)); break;
	case 3: memcpy(x, src, _cnt * sizeof(double)); break;
	case 4: memcpy(y, src, _cnt * sizeof(double)); break;
	case 5: memcpy(win, src, _cnt * sizeof(double)); break;
	case 6: memcpy(w, src, _cnt * sizeof(double)); break;
	case 7: memcpy(b, src, _cnt * sizeof(double)); break;
	case 8: memcpy(u, src, _cnt * sizeof(double)); break;
	case 9: memcpy(delta_x, src, _cnt * sizeof(double)); break;
	case 10: memcpy(delta_y, src, _cnt * sizeof(double)); break;
	case 11: memcpy(delta_win, src, _cnt * sizeof(double)); break;
	case 12: memcpy(delta_w, src, _cnt * sizeof(double)); break;
	case 13: memcpy(delta_b, src, _cnt * sizeof(double)); break;
	case 14: memcpy(delta_u, src, _cnt * sizeof(double)); break;
	}
}

RecurrentLayerCudaD::RecurrentLayerCudaD(int t_size, int x_size, int y_size){

    // T = t_size
    T = t_size;


    // X = x_size
    X = x_size;


    // Y = y_size
    Y = y_size;

	SetNormalRand(win, Y * X);
	SetNormalRand(w, Y * Y);
	SetNormalRand(b, Y);
	_chk(cudaStreamCreate(&_stream_y));
	_chk(cudaStreamCreate(&_stream_u));
	_chk(cudaStreamCreate(&_stream_delta_x));
	_chk(cudaStreamCreate(&_stream_delta_win));
	_chk(cudaStreamCreate(&_stream_delta_w));
	_chk(cudaStreamCreate(&_stream_delta_b));
	_chk(cudaStreamCreate(&_stream_delta_u));

	_chk(cudaEventCreate(&_event_y));
	_chk(cudaEventCreate(&_event_u));
	_chk(cudaEventCreate(&_event_delta_x));
	_chk(cudaEventCreate(&_event_delta_win));
	_chk(cudaEventCreate(&_event_delta_w));
	_chk(cudaEventCreate(&_event_delta_b));
	_chk(cudaEventCreate(&_event_delta_u));

}

RecurrentLayerCudaD::~RecurrentLayerCudaD(){
	Free();
	_chk(cudaStreamDestroy(_stream_y));
	_chk(cudaStreamDestroy(_stream_u));
	_chk(cudaStreamDestroy(_stream_delta_x));
	_chk(cudaStreamDestroy(_stream_delta_win));
	_chk(cudaStreamDestroy(_stream_delta_w));
	_chk(cudaStreamDestroy(_stream_delta_b));
	_chk(cudaStreamDestroy(_stream_delta_u));

	_chk(cudaEventDestroy(_event_y));
	_chk(cudaEventDestroy(_event_u));
	_chk(cudaEventDestroy(_event_delta_x));
	_chk(cudaEventDestroy(_event_delta_win));
	_chk(cudaEventDestroy(_event_delta_w));
	_chk(cudaEventDestroy(_event_delta_b));
	_chk(cudaEventDestroy(_event_delta_u));

	_Free(win);
	_Free(w);
	_Free(b);

}

void RecurrentLayerCudaD::Allocate(){
	_chk(_Malloc(y, BatchSize * T * Y * sizeof(double))); 
	_chk(_Malloc(u, BatchSize * T * Y * sizeof(double))); 
	_chk(_Malloc(delta_x, BatchSize * T * X * sizeof(double))); 
	_chk(_Malloc(delta_win, BatchSize * Y * X * sizeof(double))); 
	_chk(_Malloc(delta_w, BatchSize * Y * Y * sizeof(double))); 
	_chk(_Malloc(delta_b, BatchSize * Y * sizeof(double))); 
	_chk(_Malloc(delta_u, BatchSize * T * Y * sizeof(double))); 
}

void RecurrentLayerCudaD::Free(){
	_chk(_Free(y)); 
	_chk(_Free(u)); 
	_chk(_Free(delta_x)); 
	_chk(_Free(delta_win)); 
	_chk(_Free(delta_w)); 
	_chk(_Free(delta_b)); 
	_chk(_Free(delta_u)); 
}
__global__ static void forward_u(double* u, int X, double* x, double* win, int Y, double* w, double* y, double* b, int T, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // u[t, j] = (from i in Range(X) select x[t, i] * win[j, i]).Sum() + (from i in Range(Y) select w[j, i] * y[t -1, i]).Sum() + b[j]
    double _wk1 = 0;
    for(int i = 0; i < X; i++){
        _wk1 += x[(X * t + i) * _BatchSize + _batch_idx] * win[X * j + i];
    }
    double _wk2 = 0;
    for(int i = 0; i < Y; i++){
        _wk2 += w[Y * j + i] * (0 <= t - 1 ? y[(Y * (t -1) + i) * _BatchSize + _batch_idx] : 0);
    }
    u[(Y * t + j) * _BatchSize + _batch_idx] = _wk1 + _wk2 + b[j];

}
void RecurrentLayerCudaD::Start_forward_u(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	forward_u<<<blocksPerGrid, threadsPerBlock, 0, _stream_u>>>(u, X, x, win, Y, w, y, b, T, t);
	_chk(cudaEventRecord(_event_u, _stream_u));
}
__global__ static void forward_y(double* y, double* u, int T, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // y[t, j] = sigmoid(u[t, j])
    y[(Y * t + j) * _BatchSize + _batch_idx] = sigmoid(u[(Y * t + j) * _BatchSize + _batch_idx]);

}
void RecurrentLayerCudaD::Start_forward_y(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_y, _event_u, 0));
	forward_y<<<blocksPerGrid, threadsPerBlock, 0, _stream_y>>>(y, u, T, Y, t);
	_chk(cudaEventRecord(_event_y, _stream_y));
}
void RecurrentLayerCudaD::Forward(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	Start_forward_u();
	Start_forward_y();
}
__global__ static void backward_delta_y(double* delta_y, int Y, double* delta_u, double* w, int T, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_y[t, j] = (from j in Range(Y) select delta_u[t + 1, j] * w[j, j]).Sum()
    double _wk3 = 0;
    for(int j = 0; j < Y; j++){
        _wk3 += (t + 1 < T ? delta_u[(Y * (t + 1) + j) * _BatchSize + _batch_idx] : 0) * w[Y * j + j];
    }
    delta_y[(Y * t + j) * _BatchSize + _batch_idx] += _wk3;

}
void RecurrentLayerCudaD::Start_backward_delta_y(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	backward_delta_y<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_y>>>(delta_y, Y, delta_u, w, T, t);
	_chk(cudaEventRecord(_event_delta_y, _stream_delta_y));
}
__global__ static void backward_delta_u(double* delta_u, double* delta_y, double* u, int T, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_u[t, j] = delta_y[t, j] * sigmoid_prime(u[t, j])
    delta_u[(Y * t + j) * _BatchSize + _batch_idx] = delta_y[(Y * t + j) * _BatchSize + _batch_idx] * sigmoid_prime(u[(Y * t + j) * _BatchSize + _batch_idx]);

}
void RecurrentLayerCudaD::Start_backward_delta_u(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_u, _event_delta_y, 0));
	backward_delta_u<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_u>>>(delta_u, delta_y, u, T, Y, t);
	_chk(cudaEventRecord(_event_delta_u, _stream_delta_u));
}
__global__ static void backward_delta_x(double* delta_x, int Y, double* delta_u, double* win, int T, int X, int t){
	int _batch_idx = threadIdx.x;
	int i_i = blockIdx.x;

    // delta_x[t, i_i] = (from j in Range(Y) select delta_u[t, j] * win[j, i_i]).Sum()
    double _wk4 = 0;
    for(int j = 0; j < Y; j++){
        _wk4 += delta_u[(Y * t + j) * _BatchSize + _batch_idx] * win[X * j + i_i];
    }
    delta_x[(X * t + i_i) * _BatchSize + _batch_idx] = _wk4;

}
void RecurrentLayerCudaD::Start_backward_delta_x(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = X;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_x, _event_delta_u, 0));
	backward_delta_x<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_x>>>(delta_x, Y, delta_u, win, T, X, t);
	_chk(cudaEventRecord(_event_delta_x, _stream_delta_x));
}
__global__ static void backward_delta_win(double* delta_win, int T, double* delta_u, double* x, int Y, int X, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.y;
	int i_i = blockIdx.x;

    // delta_win[j, i_i] = (from t in Range(T) select delta_u[t, j] * x[t, i_i]).Sum()
    double _wk5 = 0;
    for(int t = 0; t < T; t++){
        _wk5 += delta_u[(Y * t + j) * _BatchSize + _batch_idx] * x[(X * t + i_i) * _BatchSize + _batch_idx];
    }
    delta_win[(X * j + i_i) * _BatchSize + _batch_idx] = _wk5;

}
void RecurrentLayerCudaD::Start_backward_delta_win(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_y = Y;
	blocks_x = X;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_win, _event_delta_u, 0));
	backward_delta_win<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_win>>>(delta_win, T, delta_u, x, Y, X, t);
	_chk(cudaEventRecord(_event_delta_win, _stream_delta_win));
}
__global__ static void backward_delta_w(double* delta_w, int T, double* delta_u, double* y, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.y;
	int i_i = blockIdx.x;

    // delta_w[j, i_i] = (from t in Range(T) select delta_u[t, j] * y[t -1, i_i]).Sum()
    double _wk6 = 0;
    for(int t = 0; t < T; t++){
        _wk6 += delta_u[(Y * t + j) * _BatchSize + _batch_idx] * (0 <= t - 1 ? y[(Y * (t -1) + i_i) * _BatchSize + _batch_idx] : 0);
    }
    delta_w[(Y * j + i_i) * _BatchSize + _batch_idx] = _wk6;

}
void RecurrentLayerCudaD::Start_backward_delta_w(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_y = Y;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_w, _event_delta_u, 0));
	backward_delta_w<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_w>>>(delta_w, T, delta_u, y, Y, t);
	_chk(cudaEventRecord(_event_delta_w, _stream_delta_w));
}
__global__ static void backward_delta_b(double* delta_b, int T, double* delta_u, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_b[j] = (from t in Range(T) select delta_u[t, j]).Sum()
    double _wk7 = 0;
    for(int t = 0; t < T; t++){
        _wk7 += delta_u[(Y * t + j) * _BatchSize + _batch_idx];
    }
    delta_b[(j) * _BatchSize + _batch_idx] = _wk7;

}
void RecurrentLayerCudaD::Start_backward_delta_b(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_b, _event_delta_u, 0));
	backward_delta_b<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_b>>>(delta_b, T, delta_u, Y, t);
	_chk(cudaEventRecord(_event_delta_b, _stream_delta_b));
}
void RecurrentLayerCudaD::Backward(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	Start_backward_delta_y();
	Start_backward_delta_u();
	Start_backward_delta_x();
	Start_backward_delta_win();
	Start_backward_delta_w();
	Start_backward_delta_b();
}
__global__ static void UpdateParameterKernel_0(double* win, double* delta_win){
	int _idx = (blockIdx.x) * blockDim.x + threadIdx.x;
	int offset = _idx * _BatchSize;
	{
		double sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_win[offset + i];
		}
		win[_idx] -= _LearningRate * sum;
	}
}
void RecurrentLayerCudaD::UpdateParameter_0(){
	int threads_x = 1;
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	threads_x = X;
	dim3 threadsPerBlock = dim3(threads_x);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	UpdateParameterKernel_0<<<blocksPerGrid, threadsPerBlock>>>(win, delta_win);
}
__global__ static void UpdateParameterKernel_1(double* w, double* delta_w){
	int _idx = (blockIdx.x) * blockDim.x + threadIdx.x;
	int offset = _idx * _BatchSize;
	{
		double sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_w[offset + i];
		}
		w[_idx] -= _LearningRate * sum;
	}
}
void RecurrentLayerCudaD::UpdateParameter_1(){
	int threads_x = 1;
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	threads_x = Y;
	dim3 threadsPerBlock = dim3(threads_x);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	UpdateParameterKernel_1<<<blocksPerGrid, threadsPerBlock>>>(w, delta_w);
}
__global__ static void UpdateParameterKernel_2(double* b, double* delta_b){
	int _idx = threadIdx.x;
	int offset = _idx * _BatchSize;
	{
		double sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_b[offset + i];
		}
		b[_idx] -= _LearningRate * sum;
	}
}
void RecurrentLayerCudaD::UpdateParameter_2(){
	int threads_x = 1;
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	threads_x = Y;
	dim3 threadsPerBlock = dim3(threads_x);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	UpdateParameterKernel_2<<<blocksPerGrid, threadsPerBlock>>>(b, delta_b);
}
void RecurrentLayerCudaD::UpdateParameter(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	_chk(_MemcpyToSymbol(_LearningRate, LearningRate, sizeof(LearningRate)));
	_chk(cudaDeviceSynchronize());
	UpdateParameter_0();
	UpdateParameter_1();
	UpdateParameter_2();
	_chk(cudaDeviceSynchronize());
}

extern "C" DllExport Layer* MakeRecurrentLayerCudaD(int t_size, int x_size, int y_size){
	return new RecurrentLayerCudaD(t_size, x_size, y_size);
}
