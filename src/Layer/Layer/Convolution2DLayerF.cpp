#include <stdlib.h>
#include <stdio.h>
#include <FLOAT.h>
#include "MkFn.h"
#include "../../Lib/Lib.h"
#include "Convolution2DLayerF.h"

void Convolution2DLayerF::GetFieldName(int field_idx, wchar_t* name){
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

int Convolution2DLayerF::GetFieldDimension(int field_idx){
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

int* Convolution2DLayerF::GetFieldSize(int field_idx){
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

void Convolution2DLayerF::GetFieldValue(int field_idx, void* dst){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(dst, &M, _cnt * sizeof(int)); break;
	case 1: memcpy(dst, &N, _cnt * sizeof(int)); break;
	case 2: memcpy(dst, &K, _cnt * sizeof(int)); break;
	case 3: memcpy(dst, &CM, _cnt * sizeof(int)); break;
	case 4: memcpy(dst, &CN, _cnt * sizeof(int)); break;
	case 5: memcpy(dst, x, _cnt * sizeof(float)); break;
	case 6: memcpy(dst, y, _cnt * sizeof(float)); break;
	case 7: memcpy(dst, u, _cnt * sizeof(float)); break;
	case 8: memcpy(dst, h, _cnt * sizeof(float)); break;
	case 9: memcpy(dst, b, _cnt * sizeof(float)); break;
	case 10: memcpy(dst, delta_x, _cnt * sizeof(float)); break;
	case 11: memcpy(dst, delta_y, _cnt * sizeof(float)); break;
	case 12: memcpy(dst, delta_u, _cnt * sizeof(float)); break;
	case 13: memcpy(dst, delta_h, _cnt * sizeof(float)); break;
	case 14: memcpy(dst, delta_b, _cnt * sizeof(float)); break;
	}
}

void Convolution2DLayerF::SetFieldValue(int field_idx, void* src){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(&M, src, _cnt * sizeof(int)); break;
	case 1: memcpy(&N, src, _cnt * sizeof(int)); break;
	case 2: memcpy(&K, src, _cnt * sizeof(int)); break;
	case 3: memcpy(&CM, src, _cnt * sizeof(int)); break;
	case 4: memcpy(&CN, src, _cnt * sizeof(int)); break;
	case 5: memcpy(x, src, _cnt * sizeof(float)); break;
	case 6: memcpy(y, src, _cnt * sizeof(float)); break;
	case 7: memcpy(u, src, _cnt * sizeof(float)); break;
	case 8: memcpy(h, src, _cnt * sizeof(float)); break;
	case 9: memcpy(b, src, _cnt * sizeof(float)); break;
	case 10: memcpy(delta_x, src, _cnt * sizeof(float)); break;
	case 11: memcpy(delta_y, src, _cnt * sizeof(float)); break;
	case 12: memcpy(delta_u, src, _cnt * sizeof(float)); break;
	case 13: memcpy(delta_h, src, _cnt * sizeof(float)); break;
	case 14: memcpy(delta_b, src, _cnt * sizeof(float)); break;
	}
}

Convolution2DLayerF::Convolution2DLayerF(int m_size, int n_size, int k_size, int cm_size, int cn_size){

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
}

Convolution2DLayerF::~Convolution2DLayerF(){
	Free();
	_Free(h);
	_Free(b);

}

void Convolution2DLayerF::Allocate(){
	_chk(_Malloc(y, BatchSize * ((M - CM) + 1) * ((N - CN) + 1) * K * sizeof(float))); 
	_chk(_Malloc(u, BatchSize * ((M - CM) + 1) * ((N - CN) + 1) * K * sizeof(float))); 
	_chk(_Malloc(delta_x, BatchSize * M * N * sizeof(float))); 
	_chk(_Malloc(delta_u, BatchSize * ((M - CM) + 1) * ((N - CN) + 1) * K * sizeof(float))); 
	_chk(_Malloc(delta_h, BatchSize * CM * CN * K * sizeof(float))); 
	_chk(_Malloc(delta_b, BatchSize * K * sizeof(float))); 
}

void Convolution2DLayerF::Free(){
	_chk(_Free(y)); 
	_chk(_Free(u)); 
	_chk(_Free(delta_x)); 
	_chk(_Free(delta_u)); 
	_chk(_Free(delta_h)); 
	_chk(_Free(delta_b)); 
}
void Convolution2DLayerF::Forward(){
#pragma omp parallel for
    for (int i = 0; i < (M - CM) + 1; i++) {
        for (int j = 0; j < (N - CN) + 1; j++) {
            for (int k = 0; k < K; k++) {
                for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                    // u[i, j, k] = (from p in Range(CM) from q in Range(CN) select x[i + p, j + q] * h[p, q, k]).Sum() + b[k]
                    float _wk1 = 0;
                    for(int p = 0; p < CM; p++){
                        for(int q = 0; q < CN; q++){
                            _wk1 += x[(N * (i + p) + (j + q)) * BatchSize + _batch_idx] * h[K * (CN * p + q) + k];
                        }
                    }
                    u[(K * (((N - CN) + 1) * i + j) + k) * BatchSize + _batch_idx] = _wk1 + b[k];

                }
            }
        }
    }
#pragma omp parallel for
    for (int i = 0; i < (M - CM) + 1; i++) {
        for (int j = 0; j < (N - CN) + 1; j++) {
            for (int k = 0; k < K; k++) {
                for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                    // y[i, j, k] = sigmoid(u[i, j, k])
                    y[(K * (((N - CN) + 1) * i + j) + k) * BatchSize + _batch_idx] = sigmoid(u[(K * (((N - CN) + 1) * i + j) + k) * BatchSize + _batch_idx]);

                }
            }
        }
    }
}
void Convolution2DLayerF::Backward(){
#pragma omp parallel for
    for (int i = 0; i < (M - CM) + 1; i++) {
        for (int j = 0; j < (N - CN) + 1; j++) {
            for (int k = 0; k < K; k++) {
                for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                    // delta_u[i, j, k] = delta_y[i, j, k] * sigmoid_prime(u[i, j, k])
                    delta_u[(K * (((N - CN) + 1) * i + j) + k) * BatchSize + _batch_idx] = delta_y[(K * (((N - CN) + 1) * i + j) + k) * BatchSize + _batch_idx] * sigmoid_prime(u[(K * (((N - CN) + 1) * i + j) + k) * BatchSize + _batch_idx]);

                }
            }
        }
    }
#pragma omp parallel for
    for (int i_0 = 0; i_0 < M; i_0++) {
        for (int i_1 = 0; i_1 < N; i_1++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_x[i_0, i_1] = (from k in Range(K) from p in Range(max(0, i_0 + M - CM + 2), min(CM, i_0)) from q in Range(max(0, i_1 + N - CN + 2), min(CN, i_1)) select delta_u[i_0 - p, i_1 - q, k] * h[p, q, k]).Sum()
                float _wk2 = 0;
                for(int k = 0; k < K; k++){
                    for(int p = max(0, i_0 + M - CM + 2); p < min(CM, i_0); p++){
                        for(int q = max(0, i_1 + N - CN + 2); q < min(CN, i_1); q++){
                            _wk2 += delta_u[(K * (((N - CN) + 1) * (i_0 - p) + (i_1 - q)) + k) * BatchSize + _batch_idx] * h[K * (CN * p + q) + k];
                        }
                    }
                }
                delta_x[(N * i_0 + i_1) * BatchSize + _batch_idx] = _wk2;

            }
        }
    }
#pragma omp parallel for
    for (int i_p = 0; i_p < CM; i_p++) {
        for (int i_q = 0; i_q < CN; i_q++) {
            for (int k = 0; k < K; k++) {
                for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                    // delta_h[i_p, i_q, k] = (from i in Range(M - CM + 1) from j in Range(N - CN + 1) select delta_u[i, j, k] * x[i + i_p, j + i_q]).Sum()
                    float _wk3 = 0;
                    for(int i = 0; i < M - CM + 1; i++){
                        for(int j = 0; j < N - CN + 1; j++){
                            _wk3 += delta_u[(K * (((N - CN) + 1) * i + j) + k) * BatchSize + _batch_idx] * x[(N * (i + i_p) + (j + i_q)) * BatchSize + _batch_idx];
                        }
                    }
                    delta_h[(K * (CN * i_p + i_q) + k) * BatchSize + _batch_idx] = _wk3;

                }
            }
        }
    }
#pragma omp parallel for
    for (int k = 0; k < K; k++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_b[k] = (from i in Range(M - CM + 1) from j in Range(N - CN + 1) select delta_u[i, j, k]).Sum()
            float _wk4 = 0;
            for(int i = 0; i < M - CM + 1; i++){
                for(int j = 0; j < N - CN + 1; j++){
                    _wk4 += delta_u[(K * (((N - CN) + 1) * i + j) + k) * BatchSize + _batch_idx];
                }
            }
            delta_b[(k) * BatchSize + _batch_idx] = _wk4;

        }
    }
}

void Convolution2DLayerF::UpdateParameter_0(){
	int _count = CM * CN * K;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_h[offset + i];
			}
			h[_idx] -= LearningRate * sum;
		}
	}
}

void Convolution2DLayerF::UpdateParameter_1(){
	int _count = K;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_b[offset + i];
			}
			b[_idx] -= LearningRate * sum;
		}
	}
}
void Convolution2DLayerF::UpdateParameter(){
	UpdateParameter_0();
	UpdateParameter_1();
}

extern "C" DllExport Layer* MakeConvolution2DLayerF(int m_size, int n_size, int k_size, int cm_size, int cn_size){
	return new Convolution2DLayerF(m_size, n_size, k_size, cm_size, cn_size);
}
