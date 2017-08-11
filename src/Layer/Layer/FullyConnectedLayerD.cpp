#include <stdlib.h>
#include <stdio.h>
#include <FLOAT.h>
#include "MkFn.h"
#include "../../Lib/Lib.h"
#include "FullyConnectedLayerD.h"

void FullyConnectedLayerD::GetFieldName(int field_idx, wchar_t* name){
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

int FullyConnectedLayerD::GetFieldDimension(int field_idx){
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

int* FullyConnectedLayerD::GetFieldSize(int field_idx){
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

void FullyConnectedLayerD::GetFieldValue(int field_idx, void* dst){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(dst, &X, _cnt * sizeof(int)); break;
	case 1: memcpy(dst, &Y, _cnt * sizeof(int)); break;
	case 2: memcpy(dst, x, _cnt * sizeof(double)); break;
	case 3: memcpy(dst, y, _cnt * sizeof(double)); break;
	case 4: memcpy(dst, w, _cnt * sizeof(double)); break;
	case 5: memcpy(dst, b, _cnt * sizeof(double)); break;
	case 6: memcpy(dst, u, _cnt * sizeof(double)); break;
	case 7: memcpy(dst, delta_x, _cnt * sizeof(double)); break;
	case 8: memcpy(dst, delta_y, _cnt * sizeof(double)); break;
	case 9: memcpy(dst, delta_w, _cnt * sizeof(double)); break;
	case 10: memcpy(dst, delta_b, _cnt * sizeof(double)); break;
	case 11: memcpy(dst, delta_u, _cnt * sizeof(double)); break;
	}
}

void FullyConnectedLayerD::SetFieldValue(int field_idx, void* src){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(&X, src, _cnt * sizeof(int)); break;
	case 1: memcpy(&Y, src, _cnt * sizeof(int)); break;
	case 2: memcpy(x, src, _cnt * sizeof(double)); break;
	case 3: memcpy(y, src, _cnt * sizeof(double)); break;
	case 4: memcpy(w, src, _cnt * sizeof(double)); break;
	case 5: memcpy(b, src, _cnt * sizeof(double)); break;
	case 6: memcpy(u, src, _cnt * sizeof(double)); break;
	case 7: memcpy(delta_x, src, _cnt * sizeof(double)); break;
	case 8: memcpy(delta_y, src, _cnt * sizeof(double)); break;
	case 9: memcpy(delta_w, src, _cnt * sizeof(double)); break;
	case 10: memcpy(delta_b, src, _cnt * sizeof(double)); break;
	case 11: memcpy(delta_u, src, _cnt * sizeof(double)); break;
	}
}

FullyConnectedLayerD::FullyConnectedLayerD(int x_size, int y_size){

    // X = x_size
    X = x_size;


    // Y = y_size
    Y = y_size;

	SetNormalRand(w, Y * X);
	SetNormalRand(b, Y);
}

FullyConnectedLayerD::~FullyConnectedLayerD(){
	Free();
	_Free(w);
	_Free(b);

}

void FullyConnectedLayerD::Allocate(){
	_chk(_Malloc(y, BatchSize * Y * sizeof(double))); 
	_chk(_Malloc(u, BatchSize * Y * sizeof(double))); 
	_chk(_Malloc(delta_x, BatchSize * X * sizeof(double))); 
	_chk(_Malloc(delta_w, BatchSize * Y * X * sizeof(double))); 
	_chk(_Malloc(delta_b, BatchSize * Y * sizeof(double))); 
	_chk(_Malloc(delta_u, BatchSize * Y * sizeof(double))); 
}

void FullyConnectedLayerD::Free(){
	_chk(_Free(y)); 
	_chk(_Free(u)); 
	_chk(_Free(delta_x)); 
	_chk(_Free(delta_w)); 
	_chk(_Free(delta_b)); 
	_chk(_Free(delta_u)); 
}
void FullyConnectedLayerD::Forward(){
#pragma omp parallel for
    for (int i = 0; i < Y; i++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // u[i] = (from j in Range(X) select x[j] * w[i, j]).Sum() + b[i]
            double _wk1 = 0;
            for(int j = 0; j < X; j++){
                _wk1 += x[(j) * BatchSize + _batch_idx] * w[X * i + j];
            }
            u[(i) * BatchSize + _batch_idx] = _wk1 + b[i];

        }
    }
#pragma omp parallel for
    for (int i = 0; i < Y; i++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // y[i] = sigmoid(u[i])
            y[(i) * BatchSize + _batch_idx] = sigmoid(u[(i) * BatchSize + _batch_idx]);

        }
    }
}
void FullyConnectedLayerD::Backward(){
#pragma omp parallel for
    for (int i = 0; i < Y; i++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_u[i] = delta_y[i] * sigmoid_prime(u[i])
            delta_u[(i) * BatchSize + _batch_idx] = delta_y[(i) * BatchSize + _batch_idx] * sigmoid_prime(u[(i) * BatchSize + _batch_idx]);

        }
    }
#pragma omp parallel for
    for (int i_j = 0; i_j < X; i_j++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_x[i_j] = (from i in Range(Y) select delta_u[i] * w[i, i_j]).Sum()
            double _wk2 = 0;
            for(int i = 0; i < Y; i++){
                _wk2 += delta_u[(i) * BatchSize + _batch_idx] * w[X * i + i_j];
            }
            delta_x[(i_j) * BatchSize + _batch_idx] = _wk2;

        }
    }
#pragma omp parallel for
    for (int i = 0; i < Y; i++) {
        for (int i_j = 0; i_j < X; i_j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_w[i, i_j] = delta_u[i] * x[i_j]
                delta_w[(X * i + i_j) * BatchSize + _batch_idx] = delta_u[(i) * BatchSize + _batch_idx] * x[(i_j) * BatchSize + _batch_idx];

            }
        }
    }
#pragma omp parallel for
    for (int i = 0; i < Y; i++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_b[i] = delta_u[i]
            delta_b[(i) * BatchSize + _batch_idx] = delta_u[(i) * BatchSize + _batch_idx];

        }
    }
}

void FullyConnectedLayerD::UpdateParameter_0(){
	int _count = Y * X;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			double sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_w[offset + i];
			}
			w[_idx] -= LearningRate * sum;
		}
	}
}

void FullyConnectedLayerD::UpdateParameter_1(){
	int _count = Y;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			double sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_b[offset + i];
			}
			b[_idx] -= LearningRate * sum;
		}
	}
}
void FullyConnectedLayerD::UpdateParameter(){
	UpdateParameter_0();
	UpdateParameter_1();
}

extern "C" DllExport Layer* MakeFullyConnectedLayerD(int x_size, int y_size){
	return new FullyConnectedLayerD(x_size, y_size);
}
