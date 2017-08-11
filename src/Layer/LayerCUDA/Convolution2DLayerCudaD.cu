#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <FLOAT.h>
#include "MkFn.h"
#include "../../Lib/Lib.h"
#include "LibCuda.h"
#include "Convolution2DLayerCudaD.h"
__constant__ int _BatchSize;
__constant__ double _LearningRate;

void Convolution2DLayerCudaD::GetFieldName(int field_idx, wchar_t* name){
	switch(field_idx){
		case 0: wcscpy(name, L"M"); break;
		case 1: wcscpy(name, L"N"); break;
		case 2: wcscpy(name, L"K"); break;
		case 3: wcscpy(name, L"CM"); break;
		case 4: wcscpy(name, L"CN"); break;
		case 5: wcscpy(name, L"x"); break;
		case 6: wcscpy(name, L"y"); break;
		case 7: wcscpy(name, L"u"); break;
		case 8: wcscpy(name, L"h"); break;
		case 9: wcscpy(name, L"b"); break;
		case 10: wcscpy(name, L"delta_x"); break;
		case 11: wcscpy(name, L"delta_y"); break;
		case 12: wcscpy(name, L"delta_u"); break;
		case 13: wcscpy(name, L"delta_h"); break;
		case 14: wcscpy(name, L"delta_b"); break;
		default: name[0] = 0; break;
	}
}

int Convolution2DLayerCudaD::GetFieldDimension(int field_idx){
	switch(field_idx){
	case 0: return 0;
	case 1: return 0;
	case 2: return 0;
	case 3: return 0;
	case 4: return 0;
	case 5: return 2;
	case 6: return 3;
	case 7: return 3;
	case 8: return 3;
	case 9: return 1;
	case 10: return 2;
	case 11: return 3;
	case 12: return 3;
	case 13: return 3;
	case 14: return 1;
	default: return -1;
	}
}

int* Convolution2DLayerCudaD::GetFieldSize(int field_idx){
	switch(field_idx){
	case 0: return 0;
	case 1: return 0;
	case 2: return 0;
	case 3: return 0;
	case 4: return 0;
	case 5:
		x_size_[0] = M;
		x_size_[1] = N;
		return x_size_;
	case 6:
		y_size_[0] = (M - CM) + 1;
		y_size_[1] = (N - CN) + 1;
		y_size_[2] = K;
		return y_size_;
	case 7:
		u_size_[0] = (M - CM) + 1;
		u_size_[1] = (N - CN) + 1;
		u_size_[2] = K;
		return u_size_;
	case 8:
		h_size_[0] = CM;
		h_size_[1] = CN;
		h_size_[2] = K;
		return h_size_;
	case 9:
		b_size_[0] = K;
		return b_size_;
	case 10:
		delta_x_size_[0] = M;
		delta_x_size_[1] = N;
		return delta_x_size_;
	case 11:
		delta_y_size_[0] = (M - CM) + 1;
		delta_y_size_[1] = (N - CN) + 1;
		delta_y_size_[2] = K;
		return delta_y_size_;
	case 12:
		delta_u_size_[0] = (M - CM) + 1;
		delta_u_size_[1] = (N - CN) + 1;
		delta_u_size_[2] = K;
		return delta_u_size_;
	case 13:
		delta_h_size_[0] = CM;
		delta_h_size_[1] = CN;
		delta_h_size_[2] = K;
		return delta_h_size_;
	case 14:
		delta_b_size_[0] = K;
		return delta_b_size_;
	default: return 0;
	}
}

void Convolution2DLayerCudaD::GetFieldValue(int field_idx, void* dst){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(dst, &M, _cnt * sizeof(int)); break;
	case 1: memcpy(dst, &N, _cnt * sizeof(int)); break;
	case 2: memcpy(dst, &K, _cnt * sizeof(int)); break;
	case 3: memcpy(dst, &CM, _cnt * sizeof(int)); break;
	case 4: memcpy(dst, &CN, _cnt * sizeof(int)); break;
	case 5: memcpy(dst, x, _cnt * sizeof(double)); break;
	case 6: memcpy(dst, y, _cnt * sizeof(double)); break;
	case 7: memcpy(dst, u, _cnt * sizeof(double)); break;
	case 8: memcpy(dst, h, _cnt * sizeof(double)); break;
	case 9: memcpy(dst, b, _cnt * sizeof(double)); break;
	case 10: memcpy(dst, delta_x, _cnt * sizeof(double)); break;
	case 11: memcpy(dst, delta_y, _cnt * sizeof(double)); break;
	case 12: memcpy(dst, delta_u, _cnt * sizeof(double)); break;
	case 13: memcpy(dst, delta_h, _cnt * sizeof(double)); break;
	case 14: memcpy(dst, delta_b, _cnt * sizeof(double)); break;
	}
}

void Convolution2DLayerCudaD::SetFieldValue(int field_idx, void* src){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(&M, src, _cnt * sizeof(int)); break;
	case 1: memcpy(&N, src, _cnt * sizeof(int)); break;
	case 2: memcpy(&K, src, _cnt * sizeof(int)); break;
	case 3: memcpy(&CM, src, _cnt * sizeof(int)); break;
	case 4: memcpy(&CN, src, _cnt * sizeof(int)); break;
	case 5: memcpy(x, src, _cnt * sizeof(double)); break;
	case 6: memcpy(y, src, _cnt * sizeof(double)); break;
	case 7: memcpy(u, src, _cnt * sizeof(double)); break;
	case 8: memcpy(h, src, _cnt * sizeof(double)); break;
	case 9: memcpy(b, src, _cnt * sizeof(double)); break;
	case 10: memcpy(delta_x, src, _cnt * sizeof(double)); break;
	case 11: memcpy(delta_y, src, _cnt * sizeof(double)); break;
	case 12: memcpy(delta_u, src, _cnt * sizeof(double)); break;
	case 13: memcpy(delta_h, src, _cnt * sizeof(double)); break;
	case 14: memcpy(delta_b, src, _cnt * sizeof(double)); break;
	}
}

Convolution2DLayerCudaD::Convolution2DLayerCudaD(int m_size, int n_size, int k_size, int cm_size, int cn_size){

    // M = m_size
    M = m_size;


    // N = n_size
    N = n_size;


    // K = k_size
    K = k_size;


    // CM = cm_size
    CM = cm_size;


    // CN = cn_size
    CN = cn_size;

	SetNormalRand(h, CM * CN * K);
	SetNormalRand(b, K);
	_chk(cudaStreamCreate(&_stream_y));
	_chk(cudaStreamCreate(&_stream_u));
	_chk(cudaStreamCreate(&_stream_delta_x));
	_chk(cudaStreamCreate(&_stream_delta_u));
	_chk(cudaStreamCreate(&_stream_delta_h));
	_chk(cudaStreamCreate(&_stream_delta_b));

	_chk(cudaEventCreate(&_event_y));
	_chk(cudaEventCreate(&_event_u));
	_chk(cudaEventCreate(&_event_delta_x));
	_chk(cudaEventCreate(&_event_delta_u));
	_chk(cudaEventCreate(&_event_delta_h));
	_chk(cudaEventCreate(&_event_delta_b));

}

Convolution2DLayerCudaD::~Convolution2DLayerCudaD(){
	Free();
	_chk(cudaStreamDestroy(_stream_y));
	_chk(cudaStreamDestroy(_stream_u));
	_chk(cudaStreamDestroy(_stream_delta_x));
	_chk(cudaStreamDestroy(_stream_delta_u));
	_chk(cudaStreamDestroy(_stream_delta_h));
	_chk(cudaStreamDestroy(_stream_delta_b));

	_chk(cudaEventDestroy(_event_y));
	_chk(cudaEventDestroy(_event_u));
	_chk(cudaEventDestroy(_event_delta_x));
	_chk(cudaEventDestroy(_event_delta_u));
	_chk(cudaEventDestroy(_event_delta_h));
	_chk(cudaEventDestroy(_event_delta_b));

	_Free(h);
	_Free(b);

}

void Convolution2DLayerCudaD::Allocate(){
	_chk(_Malloc(y, BatchSize * ((M - CM) + 1) * ((N - CN) + 1) * K * sizeof(double))); 
	_chk(_Malloc(u, BatchSize * ((M - CM) + 1) * ((N - CN) + 1) * K * sizeof(double))); 
	_chk(_Malloc(delta_x, BatchSize * M * N * sizeof(double))); 
	_chk(_Malloc(delta_u, BatchSize * ((M - CM) + 1) * ((N - CN) + 1) * K * sizeof(double))); 
	_chk(_Malloc(delta_h, BatchSize * CM * CN * K * sizeof(double))); 
	_chk(_Malloc(delta_b, BatchSize * K * sizeof(double))); 
}

void Convolution2DLayerCudaD::Free(){
	_chk(_Free(y)); 
	_chk(_Free(u)); 
	_chk(_Free(delta_x)); 
	_chk(_Free(delta_u)); 
	_chk(_Free(delta_h)); 
	_chk(_Free(delta_b)); 
}
__global__ static void forward_u(double* u, int CM, int CN, double* x, double* h, double* b, int M, int N, int K){
	int _batch_idx = threadIdx.x;
	int i = blockIdx.z;
	int j = blockIdx.y;
	int k = blockIdx.x;

    // u[i, j, k] = (from p in Range(CM) from q in Range(CN) select x[i + p, j + q] * h[p, q, k]).Sum() + b[k]
    double _wk1 = 0;
    for(int p = 0; p < CM; p++){
        for(int q = 0; q < CN; q++){
            _wk1 += x[(N * (i + p) + (j + q)) * _BatchSize + _batch_idx] * h[K * (CN * p + q) + k];
        }
    }
    u[(K * (((N - CN) + 1) * i + j) + k) * _BatchSize + _batch_idx] = _wk1 + b[k];

}
void Convolution2DLayerCudaD::Start_forward_u(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_z = (M - CM) + 1;
	blocks_y = (N - CN) + 1;
	blocks_x = K;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	forward_u<<<blocksPerGrid, threadsPerBlock, 0, _stream_u>>>(u, CM, CN, x, h, b, M, N, K);
	_chk(cudaEventRecord(_event_u, _stream_u));
}
__global__ static void forward_y(double* y, double* u, int M, int CM, int N, int CN, int K){
	int _batch_idx = threadIdx.x;
	int i = blockIdx.z;
	int j = blockIdx.y;
	int k = blockIdx.x;

    // y[i, j, k] = sigmoid(u[i, j, k])
    y[(K * (((N - CN) + 1) * i + j) + k) * _BatchSize + _batch_idx] = sigmoid(u[(K * (((N - CN) + 1) * i + j) + k) * _BatchSize + _batch_idx]);

}
void Convolution2DLayerCudaD::Start_forward_y(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_z = (M - CM) + 1;
	blocks_y = (N - CN) + 1;
	blocks_x = K;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_y, _event_u, 0));
	forward_y<<<blocksPerGrid, threadsPerBlock, 0, _stream_y>>>(y, u, M, CM, N, CN, K);
	_chk(cudaEventRecord(_event_y, _stream_y));
}
void Convolution2DLayerCudaD::Forward(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	Start_forward_u();
	Start_forward_y();
}
__global__ static void backward_delta_u(double* delta_u, double* delta_y, double* u, int M, int CM, int N, int CN, int K){
	int _batch_idx = threadIdx.x;
	int i = blockIdx.z;
	int j = blockIdx.y;
	int k = blockIdx.x;

    // delta_u[i, j, k] = delta_y[i, j, k] * sigmoid_prime(u[i, j, k])
    delta_u[(K * (((N - CN) + 1) * i + j) + k) * _BatchSize + _batch_idx] = delta_y[(K * (((N - CN) + 1) * i + j) + k) * _BatchSize + _batch_idx] * sigmoid_prime(u[(K * (((N - CN) + 1) * i + j) + k) * _BatchSize + _batch_idx]);

}
void Convolution2DLayerCudaD::Start_backward_delta_u(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_z = (M - CM) + 1;
	blocks_y = (N - CN) + 1;
	blocks_x = K;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	backward_delta_u<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_u>>>(delta_u, delta_y, u, M, CM, N, CN, K);
	_chk(cudaEventRecord(_event_delta_u, _stream_delta_u));
}
__global__ static void backward_delta_x(double* delta_x, int K, int M, int CM, int N, int CN, double* delta_u, double* h){
	int _batch_idx = threadIdx.x;
	int i_0 = blockIdx.y;
	int i_1 = blockIdx.x;

    // delta_x[i_0, i_1] = (from k in Range(K) from p in Range(max(0, i_0 + M - CM + 2), min(CM, i_0)) from q in Range(max(0, i_1 + N - CN + 2), min(CN, i_1)) select delta_u[i_0 - p, i_1 - q, k] * h[p, q, k]).Sum()
    double _wk2 = 0;
    for(int k = 0; k < K; k++){
        for(int p = max(0, i_0 + M - CM + 2); p < min(CM, i_0); p++){
            for(int q = max(0, i_1 + N - CN + 2); q < min(CN, i_1); q++){
                _wk2 += delta_u[(K * (((N - CN) + 1) * (i_0 - p) + (i_1 - q)) + k) * _BatchSize + _batch_idx] * h[K * (CN * p + q) + k];
            }
        }
    }
    delta_x[(N * i_0 + i_1) * _BatchSize + _batch_idx] = _wk2;

}
void Convolution2DLayerCudaD::Start_backward_delta_x(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_y = M;
	blocks_x = N;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_x, _event_delta_u, 0));
	backward_delta_x<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_x>>>(delta_x, K, M, CM, N, CN, delta_u, h);
	_chk(cudaEventRecord(_event_delta_x, _stream_delta_x));
}
__global__ static void backward_delta_h(double* delta_h, int M, int CM, int N, int CN, double* delta_u, double* x, int K){
	int _batch_idx = threadIdx.x;
	int i_p = blockIdx.z;
	int i_q = blockIdx.y;
	int k = blockIdx.x;

    // delta_h[i_p, i_q, k] = (from i in Range(M - CM + 1) from j in Range(N - CN + 1) select delta_u[i, j, k] * x[i + i_p, j + i_q]).Sum()
    double _wk3 = 0;
    for(int i = 0; i < M - CM + 1; i++){
        for(int j = 0; j < N - CN + 1; j++){
            _wk3 += delta_u[(K * (((N - CN) + 1) * i + j) + k) * _BatchSize + _batch_idx] * x[(N * (i + i_p) + (j + i_q)) * _BatchSize + _batch_idx];
        }
    }
    delta_h[(K * (CN * i_p + i_q) + k) * _BatchSize + _batch_idx] = _wk3;

}
void Convolution2DLayerCudaD::Start_backward_delta_h(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_z = CM;
	blocks_y = CN;
	blocks_x = K;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_h, _event_delta_u, 0));
	backward_delta_h<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_h>>>(delta_h, M, CM, N, CN, delta_u, x, K);
	_chk(cudaEventRecord(_event_delta_h, _stream_delta_h));
}
__global__ static void backward_delta_b(double* delta_b, int M, int CM, int N, int CN, double* delta_u, int K){
	int _batch_idx = threadIdx.x;
	int k = blockIdx.x;

    // delta_b[k] = (from i in Range(M - CM + 1) from j in Range(N - CN + 1) select delta_u[i, j, k]).Sum()
    double _wk4 = 0;
    for(int i = 0; i < M - CM + 1; i++){
        for(int j = 0; j < N - CN + 1; j++){
            _wk4 += delta_u[(K * (((N - CN) + 1) * i + j) + k) * _BatchSize + _batch_idx];
        }
    }
    delta_b[(k) * _BatchSize + _batch_idx] = _wk4;

}
void Convolution2DLayerCudaD::Start_backward_delta_b(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = K;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_b, _event_delta_u, 0));
	backward_delta_b<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_b>>>(delta_b, M, CM, N, CN, delta_u, K);
	_chk(cudaEventRecord(_event_delta_b, _stream_delta_b));
}
void Convolution2DLayerCudaD::Backward(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	Start_backward_delta_u();
	Start_backward_delta_x();
	Start_backward_delta_h();
	Start_backward_delta_b();
}
__global__ static void UpdateParameterKernel_0(double* h, double* delta_h){
	int _idx = ((blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int offset = _idx * _BatchSize;
	{
		double sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_h[offset + i];
		}
		h[_idx] -= _LearningRate * sum;
	}
}
void Convolution2DLayerCudaD::UpdateParameter_0(){
	int threads_x = 1;
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_y = CM;
	blocks_x = CN;
	threads_x = K;
	dim3 threadsPerBlock = dim3(threads_x);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	UpdateParameterKernel_0<<<blocksPerGrid, threadsPerBlock>>>(h, delta_h);
}
__global__ static void UpdateParameterKernel_1(double* b, double* delta_b){
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
void Convolution2DLayerCudaD::UpdateParameter_1(){
	int threads_x = 1;
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	threads_x = K;
	dim3 threadsPerBlock = dim3(threads_x);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	UpdateParameterKernel_1<<<blocksPerGrid, threadsPerBlock>>>(b, delta_b);
}
void Convolution2DLayerCudaD::UpdateParameter(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	_chk(_MemcpyToSymbol(_LearningRate, LearningRate, sizeof(LearningRate)));
	_chk(cudaDeviceSynchronize());
	UpdateParameter_0();
	UpdateParameter_1();
	_chk(cudaDeviceSynchronize());
}

extern "C" DllExport Layer* MakeConvolution2DLayerCudaD(int m_size, int n_size, int k_size, int cm_size, int cn_size){
	return new Convolution2DLayerCudaD(m_size, n_size, k_size, cm_size, cn_size);
}
