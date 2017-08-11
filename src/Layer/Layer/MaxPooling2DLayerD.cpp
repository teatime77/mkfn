#include <stdlib.h>
#include <stdio.h>
#include <FLOAT.h>
#include "MkFn.h"
#include "../../Lib/Lib.h"
#include "MaxPooling2DLayerD.h"

void MaxPooling2DLayerD::GetFieldName(int field_idx, wchar_t* name){
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

int MaxPooling2DLayerD::GetFieldDimension(int field_idx){
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

int* MaxPooling2DLayerD::GetFieldSize(int field_idx){
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

void MaxPooling2DLayerD::GetFieldValue(int field_idx, void* dst){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(dst, &M, _cnt * sizeof(int)); break;
	case 1: memcpy(dst, &N, _cnt * sizeof(int)); break;
	case 2: memcpy(dst, &K, _cnt * sizeof(int)); break;
	case 3: memcpy(dst, &PM, _cnt * sizeof(int)); break;
	case 4: memcpy(dst, &PN, _cnt * sizeof(int)); break;
	case 5: memcpy(dst, &MH, _cnt * sizeof(int)); break;
	case 6: memcpy(dst, &NH, _cnt * sizeof(int)); break;
	case 7: memcpy(dst, x, _cnt * sizeof(double)); break;
	case 8: memcpy(dst, y, _cnt * sizeof(double)); break;
	case 9: memcpy(dst, delta_x, _cnt * sizeof(double)); break;
	case 10: memcpy(dst, delta_y, _cnt * sizeof(double)); break;
	case 11: memcpy(dst, _idx_x, _cnt * sizeof(int)); break;
	}
}

void MaxPooling2DLayerD::SetFieldValue(int field_idx, void* src){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(&M, src, _cnt * sizeof(int)); break;
	case 1: memcpy(&N, src, _cnt * sizeof(int)); break;
	case 2: memcpy(&K, src, _cnt * sizeof(int)); break;
	case 3: memcpy(&PM, src, _cnt * sizeof(int)); break;
	case 4: memcpy(&PN, src, _cnt * sizeof(int)); break;
	case 5: memcpy(&MH, src, _cnt * sizeof(int)); break;
	case 6: memcpy(&NH, src, _cnt * sizeof(int)); break;
	case 7: memcpy(x, src, _cnt * sizeof(double)); break;
	case 8: memcpy(y, src, _cnt * sizeof(double)); break;
	case 9: memcpy(delta_x, src, _cnt * sizeof(double)); break;
	case 10: memcpy(delta_y, src, _cnt * sizeof(double)); break;
	case 11: memcpy(_idx_x, src, _cnt * sizeof(int)); break;
	}
}

MaxPooling2DLayerD::MaxPooling2DLayerD(int m_size, int n_size, int k_size, int pm_size, int pn_size){

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

}

MaxPooling2DLayerD::~MaxPooling2DLayerD(){
	Free();

}

void MaxPooling2DLayerD::Allocate(){
	_chk(_Malloc(y, BatchSize * MH * NH * K * sizeof(double))); 
	_chk(_Malloc(delta_x, BatchSize * M * N * K * sizeof(double))); 
	_chk(_Malloc(_idx_x, BatchSize * MH * NH * K * sizeof(int))); 
}

void MaxPooling2DLayerD::Free(){
	_chk(_Free(y)); 
	_chk(_Free(delta_x)); 
	_chk(_Free(_idx_x)); 
}
void MaxPooling2DLayerD::Forward(){
#pragma omp parallel for
    for (int i = 0; i < MH; i++) {
        for (int j = 0; j < NH; j++) {
            for (int k = 0; k < K; k++) {
                for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                    // y[i, j, k] = (from p in Range(PM) from q in Range(PN) select x[i * PM + p, j * PN + q, k]).Max()
                    double _wk1 = -FLT_MAX;
                    int _wk1_max_idx = 0;
                    for(int p = 0; p < PM; p++){
                        for(int q = 0; q < PN; q++){
                            int _wk1_idx = (K * (N * (i * PM + p) + (j * PN + q)) + k) * BatchSize + _batch_idx;
                            if(_wk1 < x[_wk1_idx]) {
                                _wk1 = x[_wk1_idx];
                                _wk1_max_idx = _wk1_idx;
                            }
                        }
                    }
                    _idx_x[(K * (NH * i + j) + k) * BatchSize + _batch_idx] = _wk1_max_idx;
                    y[(K * (NH * i + j) + k) * BatchSize + _batch_idx] = _wk1;

                }
            }
        }
    }
}
void MaxPooling2DLayerD::Backward(){
#pragma omp parallel for
    for (int i_0 = 0; i_0 < M; i_0++) {
        for (int i_1 = 0; i_1 < N; i_1++) {
            for (int k = 0; k < K; k++) {
                for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                    // delta_x[i_0, i_1, k] = delta_y[i_0 / PM, i_1 / PN, k] * (_idx_x[(K * (NH * (i_0 / PM) + i_1 / PN) + k) * BatchSize + _batch_idx] == (K * (N * i_0 + i_1) + k) * BatchSize + _batch_idx ? x[(K * (N * i_0 + i_1) + k) * BatchSize + _batch_idx] : 0)
                    delta_x[(K * (N * i_0 + i_1) + k) * BatchSize + _batch_idx] = delta_y[(K * (NH * (i_0 / PM) + i_1 / PN) + k) * BatchSize + _batch_idx] * (_idx_x[(K * (NH * (i_0 / PM) + i_1 / PN) + k) * BatchSize + _batch_idx] == (K * (N * i_0 + i_1) + k) * BatchSize + _batch_idx ? x[(K * (N * i_0 + i_1) + k) * BatchSize + _batch_idx] : 0);

                }
            }
        }
    }
}
void MaxPooling2DLayerD::UpdateParameter(){
}

extern "C" DllExport Layer* MakeMaxPooling2DLayerD(int m_size, int n_size, int k_size, int pm_size, int pn_size){
	return new MaxPooling2DLayerD(m_size, n_size, k_size, pm_size, pn_size);
}
