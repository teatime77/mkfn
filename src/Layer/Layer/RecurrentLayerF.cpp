#include <stdlib.h>
#include <stdio.h>
#include <FLOAT.h>
#include "MkFn.h"
#include "../../Lib/Lib.h"
#include "RecurrentLayerF.h"

void RecurrentLayerF::GetFieldName(int field_idx, wchar_t* name){
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

int RecurrentLayerF::GetFieldDimension(int field_idx){
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

int* RecurrentLayerF::GetFieldSize(int field_idx){
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

void RecurrentLayerF::GetFieldValue(int field_idx, void* dst){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(dst, &T, _cnt * sizeof(int)); break;
	case 1: memcpy(dst, &X, _cnt * sizeof(int)); break;
	case 2: memcpy(dst, &Y, _cnt * sizeof(int)); break;
	case 3: memcpy(dst, x, _cnt * sizeof(float)); break;
	case 4: memcpy(dst, y, _cnt * sizeof(float)); break;
	case 5: memcpy(dst, win, _cnt * sizeof(float)); break;
	case 6: memcpy(dst, w, _cnt * sizeof(float)); break;
	case 7: memcpy(dst, b, _cnt * sizeof(float)); break;
	case 8: memcpy(dst, u, _cnt * sizeof(float)); break;
	case 9: memcpy(dst, delta_x, _cnt * sizeof(float)); break;
	case 10: memcpy(dst, delta_y, _cnt * sizeof(float)); break;
	case 11: memcpy(dst, delta_win, _cnt * sizeof(float)); break;
	case 12: memcpy(dst, delta_w, _cnt * sizeof(float)); break;
	case 13: memcpy(dst, delta_b, _cnt * sizeof(float)); break;
	case 14: memcpy(dst, delta_u, _cnt * sizeof(float)); break;
	}
}

void RecurrentLayerF::SetFieldValue(int field_idx, void* src){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(&T, src, _cnt * sizeof(int)); break;
	case 1: memcpy(&X, src, _cnt * sizeof(int)); break;
	case 2: memcpy(&Y, src, _cnt * sizeof(int)); break;
	case 3: memcpy(x, src, _cnt * sizeof(float)); break;
	case 4: memcpy(y, src, _cnt * sizeof(float)); break;
	case 5: memcpy(win, src, _cnt * sizeof(float)); break;
	case 6: memcpy(w, src, _cnt * sizeof(float)); break;
	case 7: memcpy(b, src, _cnt * sizeof(float)); break;
	case 8: memcpy(u, src, _cnt * sizeof(float)); break;
	case 9: memcpy(delta_x, src, _cnt * sizeof(float)); break;
	case 10: memcpy(delta_y, src, _cnt * sizeof(float)); break;
	case 11: memcpy(delta_win, src, _cnt * sizeof(float)); break;
	case 12: memcpy(delta_w, src, _cnt * sizeof(float)); break;
	case 13: memcpy(delta_b, src, _cnt * sizeof(float)); break;
	case 14: memcpy(delta_u, src, _cnt * sizeof(float)); break;
	}
}

RecurrentLayerF::RecurrentLayerF(int t_size, int x_size, int y_size){

    // T = t_size
    T = t_size;


    // X = x_size
    X = x_size;


    // Y = y_size
    Y = y_size;

	SetNormalRand(win, Y * X);
	SetNormalRand(w, Y * Y);
	SetNormalRand(b, Y);
}

RecurrentLayerF::~RecurrentLayerF(){
	Free();
	_Free(win);
	_Free(w);
	_Free(b);

}

void RecurrentLayerF::Allocate(){
	_chk(_Malloc(y, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(u, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(delta_x, BatchSize * T * X * sizeof(float))); 
	_chk(_Malloc(delta_win, BatchSize * Y * X * sizeof(float))); 
	_chk(_Malloc(delta_w, BatchSize * Y * Y * sizeof(float))); 
	_chk(_Malloc(delta_b, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(delta_u, BatchSize * T * Y * sizeof(float))); 
}

void RecurrentLayerF::Free(){
	_chk(_Free(y)); 
	_chk(_Free(u)); 
	_chk(_Free(delta_x)); 
	_chk(_Free(delta_win)); 
	_chk(_Free(delta_w)); 
	_chk(_Free(delta_b)); 
	_chk(_Free(delta_u)); 
}
void RecurrentLayerF::Forward(){
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // u[t, j] = (from i in Range(X) select x[t, i] * win[j, i]).Sum() + (from i in Range(Y) select w[j, i] * y[t -1, i]).Sum() + b[j]
                float _wk1 = 0;
                for(int i = 0; i < X; i++){
                    _wk1 += x[(X * t + i) * BatchSize + _batch_idx] * win[X * j + i];
                }
                float _wk2 = 0;
                for(int i = 0; i < Y; i++){
                    _wk2 += w[Y * j + i] * (0 <= t - 1 ? y[(Y * (t -1) + i) * BatchSize + _batch_idx] : 0);
                }
                u[(Y * t + j) * BatchSize + _batch_idx] = _wk1 + _wk2 + b[j];

            }
        }
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // y[t, j] = sigmoid(u[t, j])
                y[(Y * t + j) * BatchSize + _batch_idx] = sigmoid(u[(Y * t + j) * BatchSize + _batch_idx]);

            }
        }
}
void RecurrentLayerF::Backward(){
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_y[t, j] = (from j in Range(Y) select delta_u[t + 1, j] * w[j, j]).Sum()
                float _wk3 = 0;
                for(int j = 0; j < Y; j++){
                    _wk3 += (t + 1 < T ? delta_u[(Y * (t + 1) + j) * BatchSize + _batch_idx] : 0) * w[Y * j + j];
                }
                delta_y[(Y * t + j) * BatchSize + _batch_idx] += _wk3;

            }
        }
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_u[t, j] = delta_y[t, j] * sigmoid_prime(u[t, j])
                delta_u[(Y * t + j) * BatchSize + _batch_idx] = delta_y[(Y * t + j) * BatchSize + _batch_idx] * sigmoid_prime(u[(Y * t + j) * BatchSize + _batch_idx]);

            }
        }
#pragma omp parallel for
        for (int i_i = 0; i_i < X; i_i++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_x[t, i_i] = (from j in Range(Y) select delta_u[t, j] * win[j, i_i]).Sum()
                float _wk4 = 0;
                for(int j = 0; j < Y; j++){
                    _wk4 += delta_u[(Y * t + j) * BatchSize + _batch_idx] * win[X * j + i_i];
                }
                delta_x[(X * t + i_i) * BatchSize + _batch_idx] = _wk4;

            }
        }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int i_i = 0; i_i < X; i_i++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_win[j, i_i] = (from t in Range(T) select delta_u[t, j] * x[t, i_i]).Sum()
                float _wk5 = 0;
                for(int t = 0; t < T; t++){
                    _wk5 += delta_u[(Y * t + j) * BatchSize + _batch_idx] * x[(X * t + i_i) * BatchSize + _batch_idx];
                }
                delta_win[(X * j + i_i) * BatchSize + _batch_idx] = _wk5;

            }
        }
    }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int i_i = 0; i_i < Y; i_i++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_w[j, i_i] = (from t in Range(T) select delta_u[t, j] * y[t -1, i_i]).Sum()
                float _wk6 = 0;
                for(int t = 0; t < T; t++){
                    _wk6 += delta_u[(Y * t + j) * BatchSize + _batch_idx] * (0 <= t - 1 ? y[(Y * (t -1) + i_i) * BatchSize + _batch_idx] : 0);
                }
                delta_w[(Y * j + i_i) * BatchSize + _batch_idx] = _wk6;

            }
        }
    }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_b[j] = (from t in Range(T) select delta_u[t, j]).Sum()
            float _wk7 = 0;
            for(int t = 0; t < T; t++){
                _wk7 += delta_u[(Y * t + j) * BatchSize + _batch_idx];
            }
            delta_b[(j) * BatchSize + _batch_idx] = _wk7;

        }
    }
}

void RecurrentLayerF::UpdateParameter_0(){
	int _count = Y * X;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_win[offset + i];
			}
			win[_idx] -= LearningRate * sum;
		}
	}
}

void RecurrentLayerF::UpdateParameter_1(){
	int _count = Y * Y;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_w[offset + i];
			}
			w[_idx] -= LearningRate * sum;
		}
	}
}

void RecurrentLayerF::UpdateParameter_2(){
	int _count = Y;
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
void RecurrentLayerF::UpdateParameter(){
	UpdateParameter_0();
	UpdateParameter_1();
	UpdateParameter_2();
}

extern "C" DllExport Layer* MakeRecurrentLayerF(int t_size, int x_size, int y_size){
	return new RecurrentLayerF(t_size, x_size, y_size);
}
