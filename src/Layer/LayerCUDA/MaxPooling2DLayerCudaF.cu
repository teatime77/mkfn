#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <FLOAT.h>
#include "MkFn.h"
#include "../../Lib/Lib.h"
#include "LibCuda.h"
#include "MaxPooling2DLayerCudaF.h"
__constant__ int _BatchSize;
__constant__ float _LearningRate;

void MaxPooling2DLayerCudaF::GetFieldName(int field_idx, wchar_t* name){
	switch(field_idx){
		case 0: wcscpy(name, L"M"); break;
		case 1: wcscpy(name, L"N"); break;
		case 2: wcscpy(name, L"K"); break;
		case 3: wcscpy(name, L"PM"); break;
		case 4: wcscpy(name, L"PN"); break;
		case 5: wcscpy(name, L"MH"); break;
		case 6: wcscpy(name, L"NH"); break;
		case 7: wcscpy(name, L"x"); break;
		case 8: wcscpy(name, L"y"); break;
		case 9: wcscpy(name, L"delta_x"); break;
		case 10: wcscpy(name, L"delta_y"); break;
		case 11: wcscpy(name, L"_idx_x"); break;
		default: name[0] = 0; break;
	}
}

int MaxPooling2DLayerCudaF::GetFieldDimension(int field_idx){
	switch(field_idx){
	case 0: return 0;
	case 1: return 0;
	case 2: return 0;
	case 3: return 0;
	case 4: return 0;
	case 5: return 0;
	case 6: return 0;
	case 7: return 3;
	case 8: return 3;
	case 9: return 3;
	case 10: return 3;
	case 11: return 3;
	default: return -1;
	}
}

int* MaxPooling2DLayerCudaF::GetFieldSize(int field_idx){
	switch(field_idx){
	case 0: return 0;
	case 1: return 0;
	case 2: return 0;
	case 3: return 0;
	case 4: return 0;
	case 5: return 0;
	case 6: return 0;
	case 7:
		x_size_[0] = M;
		x_size_[1] = N;
		x_size_[2] = K;
		return x_size_;
	case 8:
		y_size_[0] = MH;
		y_size_[1] = NH;
		y_size_[2] = K;
		return y_size_;
	case 9:
		delta_x_size_[0] = M;
		delta_x_size_[1] = N;
		delta_x_size_[2] = K;
		return delta_x_size_;
	case 10:
		delta_y_size_[0] = MH;
		delta_y_size_[1] = NH;
		delta_y_size_[2] = K;
		return delta_y_size_;
	case 11:
		_idx_x_size_[0] = MH;
		_idx_x_size_[1] = NH;
		_idx_x_size_[2] = K;
		return _idx_x_size_;
	default: return 0;
	}
}

void MaxPooling2DLayerCudaF::GetFieldValue(int field_idx, void* dst){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(dst, &M, _cnt * sizeof(int)); break;
	case 1: memcpy(dst, &N, _cnt * sizeof(int)); break;
	case 2: memcpy(dst, &K, _cnt * sizeof(int)); break;
	case 3: memcpy(dst, &PM, _cnt * sizeof(int)); break;
	case 4: memcpy(dst, &PN, _cnt * sizeof(int)); break;
	case 5: memcpy(dst, &MH, _cnt * sizeof(int)); break;
	case 6: memcpy(dst, &NH, _cnt * sizeof(int)); break;
	case 7: memcpy(dst, x, _cnt * sizeof(float)); break;
	case 8: memcpy(dst, y, _cnt * sizeof(float)); break;
	case 9: memcpy(dst, delta_x, _cnt * sizeof(float)); break;
	case 10: memcpy(dst, delta_y, _cnt * sizeof(float)); break;
	case 11: memcpy(dst, _idx_x, _cnt * sizeof(int)); break;
	}
}

void MaxPooling2DLayerCudaF::SetFieldValue(int field_idx, void* src){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(&M, src, _cnt * sizeof(int)); break;
	case 1: memcpy(&N, src, _cnt * sizeof(int)); break;
	case 2: memcpy(&K, src, _cnt * sizeof(int)); break;
	case 3: memcpy(&PM, src, _cnt * sizeof(int)); break;
	case 4: memcpy(&PN, src, _cnt * sizeof(int)); break;
	case 5: memcpy(&MH, src, _cnt * sizeof(int)); break;
	case 6: memcpy(&NH, src, _cnt * sizeof(int)); break;
	case 7: memcpy(x, src, _cnt * sizeof(float)); break;
	case 8: memcpy(y, src, _cnt * sizeof(float)); break;
	case 9: memcpy(delta_x, src, _cnt * sizeof(float)); break;
	case 10: memcpy(delta_y, src, _cnt * sizeof(float)); break;
	case 11: memcpy(_idx_x, src, _cnt * sizeof(int)); break;
	}
}

MaxPooling2DLayerCudaF::MaxPooling2DLayerCudaF(int m_size, int n_size, int k_size, int pm_size, int pn_size){

    // M = m_size
    M = m_size;


    // N = n_size
    N = n_size;


    // K = k_size
    K = k_size;


    // PM = pm_size
    PM = pm_size;


    // PN = pn_size
    PN = pn_size;


    // MH = M / PM
    MH = M / PM;


    // NH = N / PN
    NH = N / PN;

	_chk(cudaStreamCreate(&_stream_y));
	_chk(cudaStreamCreate(&_stream_delta_x));
	_chk(cudaStreamCreate(&_stream__idx_x));

	_chk(cudaEventCreate(&_event_y));
	_chk(cudaEventCreate(&_event_delta_x));
	_chk(cudaEventCreate(&_event__idx_x));

}

MaxPooling2DLayerCudaF::~MaxPooling2DLayerCudaF(){
	Free();
	_chk(cudaStreamDestroy(_stream_y));
	_chk(cudaStreamDestroy(_stream_delta_x));
	_chk(cudaStreamDestroy(_stream__idx_x));

	_chk(cudaEventDestroy(_event_y));
	_chk(cudaEventDestroy(_event_delta_x));
	_chk(cudaEventDestroy(_event__idx_x));


}

void MaxPooling2DLayerCudaF::Allocate(){
	_chk(_Malloc(y, BatchSize * MH * NH * K * sizeof(float))); 
	_chk(_Malloc(delta_x, BatchSize * M * N * K * sizeof(float))); 
	_chk(_Malloc(_idx_x, BatchSize * MH * NH * K * sizeof(int))); 
}

void MaxPooling2DLayerCudaF::Free(){
	_chk(_Free(y)); 
	_chk(_Free(delta_x)); 
	_chk(_Free(_idx_x)); 
}
__global__ static void forward_y(float* y, int PM, int PN, float* x, int* _idx_x, int MH, int NH, int K, int M, int N){
	int _batch_idx = threadIdx.x;
	int i = blockIdx.z;
	int j = blockIdx.y;
	int k = blockIdx.x;

    // y[i, j, k] = (from p in Range(PM) from q in Range(PN) select x[i * PM + p, j * PN + q, k]).Max()
    float _wk1 = -FLT_MAX;
    int _wk1_max_idx = 0;
    for(int p = 0; p < PM; p++){
        for(int q = 0; q < PN; q++){
            int _wk1_idx = (K * (N * (i * PM + p) + (j * PN + q)) + k) * _BatchSize + _batch_idx;
            if(_wk1 < x[_wk1_idx]) {
                _wk1 = x[_wk1_idx];
                _wk1_max_idx = _wk1_idx;
            }
        }
    }
    _idx_x[(K * (NH * i + j) + k) * _BatchSize + _batch_idx] = _wk1_max_idx;
    y[(K * (NH * i + j) + k) * _BatchSize + _batch_idx] = _wk1;

}
void MaxPooling2DLayerCudaF::Start_forward_y(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_z = MH;
	blocks_y = NH;
	blocks_x = K;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	forward_y<<<blocksPerGrid, threadsPerBlock, 0, _stream_y>>>(y, PM, PN, x, _idx_x, MH, NH, K, M, N);
	_chk(cudaEventRecord(_event_y, _stream_y));
}
void MaxPooling2DLayerCudaF::Forward(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	Start_forward_y();
}
__global__ static void backward_delta_x(float* delta_x, float* delta_y, int PM, int PN, float* x, float* y, int* _idx_x, int M, int N, int K, int MH, int NH){
	int _batch_idx = threadIdx.x;
	int i_0 = blockIdx.z;
	int i_1 = blockIdx.y;
	int k = blockIdx.x;

    // delta_x[i_0, i_1, k] = delta_y[i_0 / PM, i_1 / PN, k] * (_idx_x[(K * (NH * (i_0 / PM) + i_1 / PN) + k) * BatchSize + _batch_idx] == (K * (N * i_0 + i_1) + k) * BatchSize + _batch_idx ? x[(K * (N * i_0 + i_1) + k) * BatchSize + _batch_idx] : 0)
    delta_x[(K * (N * i_0 + i_1) + k) * _BatchSize + _batch_idx] = delta_y[(K * (NH * (i_0 / PM) + i_1 / PN) + k) * _BatchSize + _batch_idx] * (_idx_x[(K * (NH * (i_0 / PM) + i_1 / PN) + k) * _BatchSize + _batch_idx] == (K * (N * i_0 + i_1) + k) * _BatchSize + _batch_idx ? x[(K * (N * i_0 + i_1) + k) * _BatchSize + _batch_idx] : 0);

}
void MaxPooling2DLayerCudaF::Start_backward_delta_x(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_z = M;
	blocks_y = N;
	blocks_x = K;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	backward_delta_x<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_x>>>(delta_x, delta_y, PM, PN, x, y, _idx_x, M, N, K, MH, NH);
	_chk(cudaEventRecord(_event_delta_x, _stream_delta_x));
}
void MaxPooling2DLayerCudaF::Backward(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	Start_backward_delta_x();
}
void MaxPooling2DLayerCudaF::UpdateParameter(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	_chk(_MemcpyToSymbol(_LearningRate, LearningRate, sizeof(LearningRate)));
	_chk(cudaDeviceSynchronize());
	_chk(cudaDeviceSynchronize());
}

extern "C" DllExport Layer* MakeMaxPooling2DLayerCudaF(int m_size, int n_size, int k_size, int pm_size, int pn_size){
	return new MaxPooling2DLayerCudaF(m_size, n_size, k_size, pm_size, pn_size);
}
