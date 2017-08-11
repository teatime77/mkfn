#include <stdlib.h>
#include <stdio.h>
#include <FLOAT.h>
#include "MkFn.h"
#include "../../Lib/Lib.h"
#include "LSTMLayerF.h"

void LSTMLayerF::GetFieldName(int field_idx, wchar_t* name){
	switch(field_idx){
		case 0: wcscpy(name, L"T"); break;
		case 1: wcscpy(name, L"X"); break;
		case 2: wcscpy(name, L"Y"); break;
		case 3: wcscpy(name, L"x"); break;
		case 4: wcscpy(name, L"y"); break;
		case 5: wcscpy(name, L"wIin"); break;
		case 6: wcscpy(name, L"wFin"); break;
		case 7: wcscpy(name, L"wOin"); break;
		case 8: wcscpy(name, L"win"); break;
		case 9: wcscpy(name, L"wIr"); break;
		case 10: wcscpy(name, L"wFr"); break;
		case 11: wcscpy(name, L"wOr"); break;
		case 12: wcscpy(name, L"wr"); break;
		case 13: wcscpy(name, L"wI"); break;
		case 14: wcscpy(name, L"wF"); break;
		case 15: wcscpy(name, L"wO"); break;
		case 16: wcscpy(name, L"bO"); break;
		case 17: wcscpy(name, L"bF"); break;
		case 18: wcscpy(name, L"bI"); break;
		case 19: wcscpy(name, L"b"); break;
		case 20: wcscpy(name, L"u"); break;
		case 21: wcscpy(name, L"s"); break;
		case 22: wcscpy(name, L"uI"); break;
		case 23: wcscpy(name, L"uF"); break;
		case 24: wcscpy(name, L"uO"); break;
		case 25: wcscpy(name, L"delta_x"); break;
		case 26: wcscpy(name, L"delta_y"); break;
		case 27: wcscpy(name, L"delta_wIin"); break;
		case 28: wcscpy(name, L"delta_wFin"); break;
		case 29: wcscpy(name, L"delta_wOin"); break;
		case 30: wcscpy(name, L"delta_win"); break;
		case 31: wcscpy(name, L"delta_wIr"); break;
		case 32: wcscpy(name, L"delta_wFr"); break;
		case 33: wcscpy(name, L"delta_wOr"); break;
		case 34: wcscpy(name, L"delta_wr"); break;
		case 35: wcscpy(name, L"delta_wI"); break;
		case 36: wcscpy(name, L"delta_wF"); break;
		case 37: wcscpy(name, L"delta_wO"); break;
		case 38: wcscpy(name, L"delta_bO"); break;
		case 39: wcscpy(name, L"delta_bF"); break;
		case 40: wcscpy(name, L"delta_bI"); break;
		case 41: wcscpy(name, L"delta_b"); break;
		case 42: wcscpy(name, L"delta_u"); break;
		case 43: wcscpy(name, L"delta_s"); break;
		case 44: wcscpy(name, L"delta_uI"); break;
		case 45: wcscpy(name, L"delta_uF"); break;
		case 46: wcscpy(name, L"delta_uO"); break;
		default: name[0] = 0; break;
	}
}

int LSTMLayerF::GetFieldDimension(int field_idx){
	switch(field_idx){
	case 0: return 0;
	case 1: return 0;
	case 2: return 0;
	case 3: return 2;
	case 4: return 2;
	case 5: return 2;
	case 6: return 2;
	case 7: return 2;
	case 8: return 2;
	case 9: return 2;
	case 10: return 2;
	case 11: return 2;
	case 12: return 2;
	case 13: return 1;
	case 14: return 1;
	case 15: return 1;
	case 16: return 1;
	case 17: return 1;
	case 18: return 1;
	case 19: return 1;
	case 20: return 2;
	case 21: return 2;
	case 22: return 2;
	case 23: return 2;
	case 24: return 2;
	case 25: return 2;
	case 26: return 2;
	case 27: return 2;
	case 28: return 2;
	case 29: return 2;
	case 30: return 2;
	case 31: return 2;
	case 32: return 2;
	case 33: return 2;
	case 34: return 2;
	case 35: return 1;
	case 36: return 1;
	case 37: return 1;
	case 38: return 1;
	case 39: return 1;
	case 40: return 1;
	case 41: return 1;
	case 42: return 2;
	case 43: return 2;
	case 44: return 2;
	case 45: return 2;
	case 46: return 2;
	default: return -1;
	}
}

int* LSTMLayerF::GetFieldSize(int field_idx){
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
		wIin_size_[0] = Y;
		wIin_size_[1] = X;
		return wIin_size_;
	case 6:
		wFin_size_[0] = Y;
		wFin_size_[1] = X;
		return wFin_size_;
	case 7:
		wOin_size_[0] = Y;
		wOin_size_[1] = X;
		return wOin_size_;
	case 8:
		win_size_[0] = Y;
		win_size_[1] = X;
		return win_size_;
	case 9:
		wIr_size_[0] = Y;
		wIr_size_[1] = Y;
		return wIr_size_;
	case 10:
		wFr_size_[0] = Y;
		wFr_size_[1] = Y;
		return wFr_size_;
	case 11:
		wOr_size_[0] = Y;
		wOr_size_[1] = Y;
		return wOr_size_;
	case 12:
		wr_size_[0] = Y;
		wr_size_[1] = Y;
		return wr_size_;
	case 13:
		wI_size_[0] = Y;
		return wI_size_;
	case 14:
		wF_size_[0] = Y;
		return wF_size_;
	case 15:
		wO_size_[0] = Y;
		return wO_size_;
	case 16:
		bO_size_[0] = Y;
		return bO_size_;
	case 17:
		bF_size_[0] = Y;
		return bF_size_;
	case 18:
		bI_size_[0] = Y;
		return bI_size_;
	case 19:
		b_size_[0] = Y;
		return b_size_;
	case 20:
		u_size_[0] = T;
		u_size_[1] = Y;
		return u_size_;
	case 21:
		s_size_[0] = T;
		s_size_[1] = Y;
		return s_size_;
	case 22:
		uI_size_[0] = T;
		uI_size_[1] = Y;
		return uI_size_;
	case 23:
		uF_size_[0] = T;
		uF_size_[1] = Y;
		return uF_size_;
	case 24:
		uO_size_[0] = T;
		uO_size_[1] = Y;
		return uO_size_;
	case 25:
		delta_x_size_[0] = T;
		delta_x_size_[1] = X;
		return delta_x_size_;
	case 26:
		delta_y_size_[0] = T;
		delta_y_size_[1] = Y;
		return delta_y_size_;
	case 27:
		delta_wIin_size_[0] = Y;
		delta_wIin_size_[1] = X;
		return delta_wIin_size_;
	case 28:
		delta_wFin_size_[0] = Y;
		delta_wFin_size_[1] = X;
		return delta_wFin_size_;
	case 29:
		delta_wOin_size_[0] = Y;
		delta_wOin_size_[1] = X;
		return delta_wOin_size_;
	case 30:
		delta_win_size_[0] = Y;
		delta_win_size_[1] = X;
		return delta_win_size_;
	case 31:
		delta_wIr_size_[0] = Y;
		delta_wIr_size_[1] = Y;
		return delta_wIr_size_;
	case 32:
		delta_wFr_size_[0] = Y;
		delta_wFr_size_[1] = Y;
		return delta_wFr_size_;
	case 33:
		delta_wOr_size_[0] = Y;
		delta_wOr_size_[1] = Y;
		return delta_wOr_size_;
	case 34:
		delta_wr_size_[0] = Y;
		delta_wr_size_[1] = Y;
		return delta_wr_size_;
	case 35:
		delta_wI_size_[0] = Y;
		return delta_wI_size_;
	case 36:
		delta_wF_size_[0] = Y;
		return delta_wF_size_;
	case 37:
		delta_wO_size_[0] = Y;
		return delta_wO_size_;
	case 38:
		delta_bO_size_[0] = Y;
		return delta_bO_size_;
	case 39:
		delta_bF_size_[0] = Y;
		return delta_bF_size_;
	case 40:
		delta_bI_size_[0] = Y;
		return delta_bI_size_;
	case 41:
		delta_b_size_[0] = Y;
		return delta_b_size_;
	case 42:
		delta_u_size_[0] = T;
		delta_u_size_[1] = Y;
		return delta_u_size_;
	case 43:
		delta_s_size_[0] = T;
		delta_s_size_[1] = Y;
		return delta_s_size_;
	case 44:
		delta_uI_size_[0] = T;
		delta_uI_size_[1] = Y;
		return delta_uI_size_;
	case 45:
		delta_uF_size_[0] = T;
		delta_uF_size_[1] = Y;
		return delta_uF_size_;
	case 46:
		delta_uO_size_[0] = T;
		delta_uO_size_[1] = Y;
		return delta_uO_size_;
	default: return 0;
	}
}

void LSTMLayerF::GetFieldValue(int field_idx, void* dst){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(dst, &T, _cnt * sizeof(int)); break;
	case 1: memcpy(dst, &X, _cnt * sizeof(int)); break;
	case 2: memcpy(dst, &Y, _cnt * sizeof(int)); break;
	case 3: memcpy(dst, x, _cnt * sizeof(float)); break;
	case 4: memcpy(dst, y, _cnt * sizeof(float)); break;
	case 5: memcpy(dst, wIin, _cnt * sizeof(float)); break;
	case 6: memcpy(dst, wFin, _cnt * sizeof(float)); break;
	case 7: memcpy(dst, wOin, _cnt * sizeof(float)); break;
	case 8: memcpy(dst, win, _cnt * sizeof(float)); break;
	case 9: memcpy(dst, wIr, _cnt * sizeof(float)); break;
	case 10: memcpy(dst, wFr, _cnt * sizeof(float)); break;
	case 11: memcpy(dst, wOr, _cnt * sizeof(float)); break;
	case 12: memcpy(dst, wr, _cnt * sizeof(float)); break;
	case 13: memcpy(dst, wI, _cnt * sizeof(float)); break;
	case 14: memcpy(dst, wF, _cnt * sizeof(float)); break;
	case 15: memcpy(dst, wO, _cnt * sizeof(float)); break;
	case 16: memcpy(dst, bO, _cnt * sizeof(float)); break;
	case 17: memcpy(dst, bF, _cnt * sizeof(float)); break;
	case 18: memcpy(dst, bI, _cnt * sizeof(float)); break;
	case 19: memcpy(dst, b, _cnt * sizeof(float)); break;
	case 20: memcpy(dst, u, _cnt * sizeof(float)); break;
	case 21: memcpy(dst, s, _cnt * sizeof(float)); break;
	case 22: memcpy(dst, uI, _cnt * sizeof(float)); break;
	case 23: memcpy(dst, uF, _cnt * sizeof(float)); break;
	case 24: memcpy(dst, uO, _cnt * sizeof(float)); break;
	case 25: memcpy(dst, delta_x, _cnt * sizeof(float)); break;
	case 26: memcpy(dst, delta_y, _cnt * sizeof(float)); break;
	case 27: memcpy(dst, delta_wIin, _cnt * sizeof(float)); break;
	case 28: memcpy(dst, delta_wFin, _cnt * sizeof(float)); break;
	case 29: memcpy(dst, delta_wOin, _cnt * sizeof(float)); break;
	case 30: memcpy(dst, delta_win, _cnt * sizeof(float)); break;
	case 31: memcpy(dst, delta_wIr, _cnt * sizeof(float)); break;
	case 32: memcpy(dst, delta_wFr, _cnt * sizeof(float)); break;
	case 33: memcpy(dst, delta_wOr, _cnt * sizeof(float)); break;
	case 34: memcpy(dst, delta_wr, _cnt * sizeof(float)); break;
	case 35: memcpy(dst, delta_wI, _cnt * sizeof(float)); break;
	case 36: memcpy(dst, delta_wF, _cnt * sizeof(float)); break;
	case 37: memcpy(dst, delta_wO, _cnt * sizeof(float)); break;
	case 38: memcpy(dst, delta_bO, _cnt * sizeof(float)); break;
	case 39: memcpy(dst, delta_bF, _cnt * sizeof(float)); break;
	case 40: memcpy(dst, delta_bI, _cnt * sizeof(float)); break;
	case 41: memcpy(dst, delta_b, _cnt * sizeof(float)); break;
	case 42: memcpy(dst, delta_u, _cnt * sizeof(float)); break;
	case 43: memcpy(dst, delta_s, _cnt * sizeof(float)); break;
	case 44: memcpy(dst, delta_uI, _cnt * sizeof(float)); break;
	case 45: memcpy(dst, delta_uF, _cnt * sizeof(float)); break;
	case 46: memcpy(dst, delta_uO, _cnt * sizeof(float)); break;
	}
}

void LSTMLayerF::SetFieldValue(int field_idx, void* src){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(&T, src, _cnt * sizeof(int)); break;
	case 1: memcpy(&X, src, _cnt * sizeof(int)); break;
	case 2: memcpy(&Y, src, _cnt * sizeof(int)); break;
	case 3: memcpy(x, src, _cnt * sizeof(float)); break;
	case 4: memcpy(y, src, _cnt * sizeof(float)); break;
	case 5: memcpy(wIin, src, _cnt * sizeof(float)); break;
	case 6: memcpy(wFin, src, _cnt * sizeof(float)); break;
	case 7: memcpy(wOin, src, _cnt * sizeof(float)); break;
	case 8: memcpy(win, src, _cnt * sizeof(float)); break;
	case 9: memcpy(wIr, src, _cnt * sizeof(float)); break;
	case 10: memcpy(wFr, src, _cnt * sizeof(float)); break;
	case 11: memcpy(wOr, src, _cnt * sizeof(float)); break;
	case 12: memcpy(wr, src, _cnt * sizeof(float)); break;
	case 13: memcpy(wI, src, _cnt * sizeof(float)); break;
	case 14: memcpy(wF, src, _cnt * sizeof(float)); break;
	case 15: memcpy(wO, src, _cnt * sizeof(float)); break;
	case 16: memcpy(bO, src, _cnt * sizeof(float)); break;
	case 17: memcpy(bF, src, _cnt * sizeof(float)); break;
	case 18: memcpy(bI, src, _cnt * sizeof(float)); break;
	case 19: memcpy(b, src, _cnt * sizeof(float)); break;
	case 20: memcpy(u, src, _cnt * sizeof(float)); break;
	case 21: memcpy(s, src, _cnt * sizeof(float)); break;
	case 22: memcpy(uI, src, _cnt * sizeof(float)); break;
	case 23: memcpy(uF, src, _cnt * sizeof(float)); break;
	case 24: memcpy(uO, src, _cnt * sizeof(float)); break;
	case 25: memcpy(delta_x, src, _cnt * sizeof(float)); break;
	case 26: memcpy(delta_y, src, _cnt * sizeof(float)); break;
	case 27: memcpy(delta_wIin, src, _cnt * sizeof(float)); break;
	case 28: memcpy(delta_wFin, src, _cnt * sizeof(float)); break;
	case 29: memcpy(delta_wOin, src, _cnt * sizeof(float)); break;
	case 30: memcpy(delta_win, src, _cnt * sizeof(float)); break;
	case 31: memcpy(delta_wIr, src, _cnt * sizeof(float)); break;
	case 32: memcpy(delta_wFr, src, _cnt * sizeof(float)); break;
	case 33: memcpy(delta_wOr, src, _cnt * sizeof(float)); break;
	case 34: memcpy(delta_wr, src, _cnt * sizeof(float)); break;
	case 35: memcpy(delta_wI, src, _cnt * sizeof(float)); break;
	case 36: memcpy(delta_wF, src, _cnt * sizeof(float)); break;
	case 37: memcpy(delta_wO, src, _cnt * sizeof(float)); break;
	case 38: memcpy(delta_bO, src, _cnt * sizeof(float)); break;
	case 39: memcpy(delta_bF, src, _cnt * sizeof(float)); break;
	case 40: memcpy(delta_bI, src, _cnt * sizeof(float)); break;
	case 41: memcpy(delta_b, src, _cnt * sizeof(float)); break;
	case 42: memcpy(delta_u, src, _cnt * sizeof(float)); break;
	case 43: memcpy(delta_s, src, _cnt * sizeof(float)); break;
	case 44: memcpy(delta_uI, src, _cnt * sizeof(float)); break;
	case 45: memcpy(delta_uF, src, _cnt * sizeof(float)); break;
	case 46: memcpy(delta_uO, src, _cnt * sizeof(float)); break;
	}
}

LSTMLayerF::LSTMLayerF(int t_size, int x_size, int y_size){

    // T = t_size
    T = t_size;


    // X = x_size
    X = x_size;


    // Y = y_size
    Y = y_size;

	SetNormalRand(wIin, Y * X);
	SetNormalRand(wFin, Y * X);
	SetNormalRand(wOin, Y * X);
	SetNormalRand(win, Y * X);
	SetNormalRand(wIr, Y * Y);
	SetNormalRand(wFr, Y * Y);
	SetNormalRand(wOr, Y * Y);
	SetNormalRand(wr, Y * Y);
	SetNormalRand(wI, Y);
	SetNormalRand(wF, Y);
	SetNormalRand(wO, Y);
	SetNormalRand(bO, Y);
	SetNormalRand(bF, Y);
	SetNormalRand(bI, Y);
	SetNormalRand(b, Y);
}

LSTMLayerF::~LSTMLayerF(){
	Free();
	_Free(wIin);
	_Free(wFin);
	_Free(wOin);
	_Free(win);
	_Free(wIr);
	_Free(wFr);
	_Free(wOr);
	_Free(wr);
	_Free(wI);
	_Free(wF);
	_Free(wO);
	_Free(bO);
	_Free(bF);
	_Free(bI);
	_Free(b);

}

void LSTMLayerF::Allocate(){
	_chk(_Malloc(y, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(u, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(s, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(uI, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(uF, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(uO, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(delta_x, BatchSize * T * X * sizeof(float))); 
	_chk(_Malloc(delta_wIin, BatchSize * Y * X * sizeof(float))); 
	_chk(_Malloc(delta_wFin, BatchSize * Y * X * sizeof(float))); 
	_chk(_Malloc(delta_wOin, BatchSize * Y * X * sizeof(float))); 
	_chk(_Malloc(delta_win, BatchSize * Y * X * sizeof(float))); 
	_chk(_Malloc(delta_wIr, BatchSize * Y * Y * sizeof(float))); 
	_chk(_Malloc(delta_wFr, BatchSize * Y * Y * sizeof(float))); 
	_chk(_Malloc(delta_wOr, BatchSize * Y * Y * sizeof(float))); 
	_chk(_Malloc(delta_wr, BatchSize * Y * Y * sizeof(float))); 
	_chk(_Malloc(delta_wI, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(delta_wF, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(delta_wO, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(delta_bO, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(delta_bF, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(delta_bI, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(delta_b, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(delta_u, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(delta_s, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(delta_uI, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(delta_uF, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(delta_uO, BatchSize * T * Y * sizeof(float))); 
}

void LSTMLayerF::Free(){
	_chk(_Free(y)); 
	_chk(_Free(u)); 
	_chk(_Free(s)); 
	_chk(_Free(uI)); 
	_chk(_Free(uF)); 
	_chk(_Free(uO)); 
	_chk(_Free(delta_x)); 
	_chk(_Free(delta_wIin)); 
	_chk(_Free(delta_wFin)); 
	_chk(_Free(delta_wOin)); 
	_chk(_Free(delta_win)); 
	_chk(_Free(delta_wIr)); 
	_chk(_Free(delta_wFr)); 
	_chk(_Free(delta_wOr)); 
	_chk(_Free(delta_wr)); 
	_chk(_Free(delta_wI)); 
	_chk(_Free(delta_wF)); 
	_chk(_Free(delta_wO)); 
	_chk(_Free(delta_bO)); 
	_chk(_Free(delta_bF)); 
	_chk(_Free(delta_bI)); 
	_chk(_Free(delta_b)); 
	_chk(_Free(delta_u)); 
	_chk(_Free(delta_s)); 
	_chk(_Free(delta_uI)); 
	_chk(_Free(delta_uF)); 
	_chk(_Free(delta_uO)); 
}
void LSTMLayerF::Forward(){
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // uF[t, j] = (from i in Range(X) select wFin[j, i] * x[t, i]).Sum() + (from i in Range(Y) select wFr[j, i] * y[t -1, i]).Sum() + wF[j] * s[t -1, j] + bF[j]
                float _wk1 = 0;
                for(int i = 0; i < X; i++){
                    _wk1 += wFin[X * j + i] * x[(X * t + i) * BatchSize + _batch_idx];
                }
                float _wk2 = 0;
                for(int i = 0; i < Y; i++){
                    _wk2 += wFr[Y * j + i] * (0 <= t - 1 ? y[(Y * (t -1) + i) * BatchSize + _batch_idx] : 0);
                }
                uF[(Y * t + j) * BatchSize + _batch_idx] = _wk1 + _wk2 + wF[j] * (0 <= t - 1 ? s[(Y * (t -1) + j) * BatchSize + _batch_idx] : 0) + bF[j];

            }
        }
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // uI[t, j] = (from i in Range(X) select wIin[j, i] * x[t, i]).Sum() + (from i in Range(Y) select wIr[j, i] * y[t -1, i]).Sum() + wI[j] * s[t -1, j] + bI[j]
                float _wk3 = 0;
                for(int i = 0; i < X; i++){
                    _wk3 += wIin[X * j + i] * x[(X * t + i) * BatchSize + _batch_idx];
                }
                float _wk4 = 0;
                for(int i = 0; i < Y; i++){
                    _wk4 += wIr[Y * j + i] * (0 <= t - 1 ? y[(Y * (t -1) + i) * BatchSize + _batch_idx] : 0);
                }
                uI[(Y * t + j) * BatchSize + _batch_idx] = _wk3 + _wk4 + wI[j] * (0 <= t - 1 ? s[(Y * (t -1) + j) * BatchSize + _batch_idx] : 0) + bI[j];

            }
        }
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // u[t, j] = (from i in Range(X) select win[j, i] * x[t, i]).Sum() + (from i in Range(Y) select wr[j, i] * y[t -1, i]).Sum() + b[j]
                float _wk5 = 0;
                for(int i = 0; i < X; i++){
                    _wk5 += win[X * j + i] * x[(X * t + i) * BatchSize + _batch_idx];
                }
                float _wk6 = 0;
                for(int i = 0; i < Y; i++){
                    _wk6 += wr[Y * j + i] * (0 <= t - 1 ? y[(Y * (t -1) + i) * BatchSize + _batch_idx] : 0);
                }
                u[(Y * t + j) * BatchSize + _batch_idx] = _wk5 + _wk6 + b[j];

            }
        }
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // s[t, j] = sigmoid(uF[t, j]) * s[t -1, j] + sigmoid(uI[t, j]) * sigmoid(u[t, j])
                s[(Y * t + j) * BatchSize + _batch_idx] = sigmoid(uF[(Y * t + j) * BatchSize + _batch_idx]) * (0 <= t - 1 ? s[(Y * (t -1) + j) * BatchSize + _batch_idx] : 0) + sigmoid(uI[(Y * t + j) * BatchSize + _batch_idx]) * sigmoid(u[(Y * t + j) * BatchSize + _batch_idx]);

            }
        }
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // uO[t, j] = (from i in Range(X) select wOin[j, i] * x[t, i]).Sum() + (from i in Range(Y) select wOr[j, i] * y[t -1, i]).Sum() + wO[j] * s[t, j] + bO[j]
                float _wk7 = 0;
                for(int i = 0; i < X; i++){
                    _wk7 += wOin[X * j + i] * x[(X * t + i) * BatchSize + _batch_idx];
                }
                float _wk8 = 0;
                for(int i = 0; i < Y; i++){
                    _wk8 += wOr[Y * j + i] * (0 <= t - 1 ? y[(Y * (t -1) + i) * BatchSize + _batch_idx] : 0);
                }
                uO[(Y * t + j) * BatchSize + _batch_idx] = _wk7 + _wk8 + wO[j] * s[(Y * t + j) * BatchSize + _batch_idx] + bO[j];

            }
        }
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // y[t, j] = sigmoid(uO[t, j]) * sigmoid(s[t, j])
                y[(Y * t + j) * BatchSize + _batch_idx] = sigmoid(uO[(Y * t + j) * BatchSize + _batch_idx]) * sigmoid(s[(Y * t + j) * BatchSize + _batch_idx]);

            }
        }
}
void LSTMLayerF::Backward(){
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_y[t, j] = (from j in Range(Y) select delta_uO[t + 1, j] * wOr[j, j]).Sum() + (from j in Range(Y) select delta_uF[t + 1, j] * wFr[j, j]).Sum() + (from j in Range(Y) select delta_uI[t + 1, j] * wIr[j, j]).Sum() + (from j in Range(Y) select delta_u[t + 1, j] * wr[j, j]).Sum()
                float _wk9 = 0;
                for(int j = 0; j < Y; j++){
                    _wk9 += (t + 1 < T ? delta_uO[(Y * (t + 1) + j) * BatchSize + _batch_idx] : 0) * wOr[Y * j + j];
                }
                float _wk10 = 0;
                for(int j = 0; j < Y; j++){
                    _wk10 += (t + 1 < T ? delta_uF[(Y * (t + 1) + j) * BatchSize + _batch_idx] : 0) * wFr[Y * j + j];
                }
                float _wk11 = 0;
                for(int j = 0; j < Y; j++){
                    _wk11 += (t + 1 < T ? delta_uI[(Y * (t + 1) + j) * BatchSize + _batch_idx] : 0) * wIr[Y * j + j];
                }
                float _wk12 = 0;
                for(int j = 0; j < Y; j++){
                    _wk12 += (t + 1 < T ? delta_u[(Y * (t + 1) + j) * BatchSize + _batch_idx] : 0) * wr[Y * j + j];
                }
                delta_y[(Y * t + j) * BatchSize + _batch_idx] += _wk9 + _wk10 + _wk11 + _wk12;

            }
        }
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_uO[t, j] = delta_y[t, j] * sigmoid_prime(uO[t, j]) * sigmoid(s[t, j])
                delta_uO[(Y * t + j) * BatchSize + _batch_idx] = delta_y[(Y * t + j) * BatchSize + _batch_idx] * sigmoid_prime(uO[(Y * t + j) * BatchSize + _batch_idx]) * sigmoid(s[(Y * t + j) * BatchSize + _batch_idx]);

            }
        }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int i_i = 0; i_i < X; i_i++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_wOin[j, i_i] = (from t in Range(T) select delta_uO[t, j] * x[t, i_i]).Sum()
                float _wk13 = 0;
                for(int t = 0; t < T; t++){
                    _wk13 += delta_uO[(Y * t + j) * BatchSize + _batch_idx] * x[(X * t + i_i) * BatchSize + _batch_idx];
                }
                delta_wOin[(X * j + i_i) * BatchSize + _batch_idx] = _wk13;

            }
        }
    }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int i_i = 0; i_i < Y; i_i++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_wOr[j, i_i] = (from t in Range(T) select delta_uO[t, j] * y[t -1, i_i]).Sum()
                float _wk14 = 0;
                for(int t = 0; t < T; t++){
                    _wk14 += delta_uO[(Y * t + j) * BatchSize + _batch_idx] * (0 <= t - 1 ? y[(Y * (t -1) + i_i) * BatchSize + _batch_idx] : 0);
                }
                delta_wOr[(Y * j + i_i) * BatchSize + _batch_idx] = _wk14;

            }
        }
    }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_wO[j] = (from t in Range(T) select delta_uO[t, j] * s[t, j]).Sum()
            float _wk15 = 0;
            for(int t = 0; t < T; t++){
                _wk15 += delta_uO[(Y * t + j) * BatchSize + _batch_idx] * s[(Y * t + j) * BatchSize + _batch_idx];
            }
            delta_wO[(j) * BatchSize + _batch_idx] = _wk15;

        }
    }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_bO[j] = (from t in Range(T) select delta_uO[t, j]).Sum()
            float _wk16 = 0;
            for(int t = 0; t < T; t++){
                _wk16 += delta_uO[(Y * t + j) * BatchSize + _batch_idx];
            }
            delta_bO[(j) * BatchSize + _batch_idx] = _wk16;

        }
    }
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_s[t, j] = delta_y[t, j] * sigmoid(uO[t, j]) * sigmoid_prime(s[t, j]) + delta_s[t + 1, j] * sigmoid(uF[t + 1, j]) + delta_uO[t, j] * wO[j] + delta_uF[t + 1, j] * wF[j] + delta_uI[t + 1, j] * wI[j]
                delta_s[(Y * t + j) * BatchSize + _batch_idx] = delta_y[(Y * t + j) * BatchSize + _batch_idx] * sigmoid(uO[(Y * t + j) * BatchSize + _batch_idx]) * sigmoid_prime(s[(Y * t + j) * BatchSize + _batch_idx]) + (t + 1 < T ? delta_s[(Y * (t + 1) + j) * BatchSize + _batch_idx] : 0) * sigmoid((t + 1 < T ? uF[(Y * (t + 1) + j) * BatchSize + _batch_idx] : 0)) + delta_uO[(Y * t + j) * BatchSize + _batch_idx] * wO[j] + (t + 1 < T ? delta_uF[(Y * (t + 1) + j) * BatchSize + _batch_idx] : 0) * wF[j] + (t + 1 < T ? delta_uI[(Y * (t + 1) + j) * BatchSize + _batch_idx] : 0) * wI[j];

            }
        }
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_u[t, j] = delta_s[t, j] * sigmoid(uI[t, j]) * sigmoid_prime(u[t, j])
                delta_u[(Y * t + j) * BatchSize + _batch_idx] = delta_s[(Y * t + j) * BatchSize + _batch_idx] * sigmoid(uI[(Y * t + j) * BatchSize + _batch_idx]) * sigmoid_prime(u[(Y * t + j) * BatchSize + _batch_idx]);

            }
        }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int i_i = 0; i_i < X; i_i++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_win[j, i_i] = (from t in Range(T) select delta_u[t, j] * x[t, i_i]).Sum()
                float _wk17 = 0;
                for(int t = 0; t < T; t++){
                    _wk17 += delta_u[(Y * t + j) * BatchSize + _batch_idx] * x[(X * t + i_i) * BatchSize + _batch_idx];
                }
                delta_win[(X * j + i_i) * BatchSize + _batch_idx] = _wk17;

            }
        }
    }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int i_i = 0; i_i < Y; i_i++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_wr[j, i_i] = (from t in Range(T) select delta_u[t, j] * y[t -1, i_i]).Sum()
                float _wk18 = 0;
                for(int t = 0; t < T; t++){
                    _wk18 += delta_u[(Y * t + j) * BatchSize + _batch_idx] * (0 <= t - 1 ? y[(Y * (t -1) + i_i) * BatchSize + _batch_idx] : 0);
                }
                delta_wr[(Y * j + i_i) * BatchSize + _batch_idx] = _wk18;

            }
        }
    }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_b[j] = (from t in Range(T) select delta_u[t, j]).Sum()
            float _wk19 = 0;
            for(int t = 0; t < T; t++){
                _wk19 += delta_u[(Y * t + j) * BatchSize + _batch_idx];
            }
            delta_b[(j) * BatchSize + _batch_idx] = _wk19;

        }
    }
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_uI[t, j] = delta_s[t, j] * sigmoid_prime(uI[t, j]) * sigmoid(u[t, j])
                delta_uI[(Y * t + j) * BatchSize + _batch_idx] = delta_s[(Y * t + j) * BatchSize + _batch_idx] * sigmoid_prime(uI[(Y * t + j) * BatchSize + _batch_idx]) * sigmoid(u[(Y * t + j) * BatchSize + _batch_idx]);

            }
        }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int i_i = 0; i_i < X; i_i++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_wIin[j, i_i] = (from t in Range(T) select delta_uI[t, j] * x[t, i_i]).Sum()
                float _wk20 = 0;
                for(int t = 0; t < T; t++){
                    _wk20 += delta_uI[(Y * t + j) * BatchSize + _batch_idx] * x[(X * t + i_i) * BatchSize + _batch_idx];
                }
                delta_wIin[(X * j + i_i) * BatchSize + _batch_idx] = _wk20;

            }
        }
    }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int i_i = 0; i_i < Y; i_i++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_wIr[j, i_i] = (from t in Range(T) select delta_uI[t, j] * y[t -1, i_i]).Sum()
                float _wk21 = 0;
                for(int t = 0; t < T; t++){
                    _wk21 += delta_uI[(Y * t + j) * BatchSize + _batch_idx] * (0 <= t - 1 ? y[(Y * (t -1) + i_i) * BatchSize + _batch_idx] : 0);
                }
                delta_wIr[(Y * j + i_i) * BatchSize + _batch_idx] = _wk21;

            }
        }
    }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_wI[j] = (from t in Range(T) select delta_uI[t, j] * s[t -1, j]).Sum()
            float _wk22 = 0;
            for(int t = 0; t < T; t++){
                _wk22 += delta_uI[(Y * t + j) * BatchSize + _batch_idx] * (0 <= t - 1 ? s[(Y * (t -1) + j) * BatchSize + _batch_idx] : 0);
            }
            delta_wI[(j) * BatchSize + _batch_idx] = _wk22;

        }
    }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_bI[j] = (from t in Range(T) select delta_uI[t, j]).Sum()
            float _wk23 = 0;
            for(int t = 0; t < T; t++){
                _wk23 += delta_uI[(Y * t + j) * BatchSize + _batch_idx];
            }
            delta_bI[(j) * BatchSize + _batch_idx] = _wk23;

        }
    }
#pragma omp parallel for
        for (int j = 0; j < Y; j++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_uF[t, j] = delta_s[t, j] * sigmoid_prime(uF[t, j]) * s[t -1, j]
                delta_uF[(Y * t + j) * BatchSize + _batch_idx] = delta_s[(Y * t + j) * BatchSize + _batch_idx] * sigmoid_prime(uF[(Y * t + j) * BatchSize + _batch_idx]) * (0 <= t - 1 ? s[(Y * (t -1) + j) * BatchSize + _batch_idx] : 0);

            }
        }
#pragma omp parallel for
        for (int i_i = 0; i_i < X; i_i++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_x[t, i_i] = (from j in Range(Y) select delta_uO[t, j] * wOin[j, i_i]).Sum() + (from j in Range(Y) select delta_uF[t, j] * wFin[j, i_i]).Sum() + (from j in Range(Y) select delta_uI[t, j] * wIin[j, i_i]).Sum() + (from j in Range(Y) select delta_u[t, j] * win[j, i_i]).Sum()
                float _wk24 = 0;
                for(int j = 0; j < Y; j++){
                    _wk24 += delta_uO[(Y * t + j) * BatchSize + _batch_idx] * wOin[X * j + i_i];
                }
                float _wk25 = 0;
                for(int j = 0; j < Y; j++){
                    _wk25 += delta_uF[(Y * t + j) * BatchSize + _batch_idx] * wFin[X * j + i_i];
                }
                float _wk26 = 0;
                for(int j = 0; j < Y; j++){
                    _wk26 += delta_uI[(Y * t + j) * BatchSize + _batch_idx] * wIin[X * j + i_i];
                }
                float _wk27 = 0;
                for(int j = 0; j < Y; j++){
                    _wk27 += delta_u[(Y * t + j) * BatchSize + _batch_idx] * win[X * j + i_i];
                }
                delta_x[(X * t + i_i) * BatchSize + _batch_idx] = _wk24 + _wk25 + _wk26 + _wk27;

            }
        }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int i_i = 0; i_i < X; i_i++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_wFin[j, i_i] = (from t in Range(T) select delta_uF[t, j] * x[t, i_i]).Sum()
                float _wk28 = 0;
                for(int t = 0; t < T; t++){
                    _wk28 += delta_uF[(Y * t + j) * BatchSize + _batch_idx] * x[(X * t + i_i) * BatchSize + _batch_idx];
                }
                delta_wFin[(X * j + i_i) * BatchSize + _batch_idx] = _wk28;

            }
        }
    }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int i_i = 0; i_i < Y; i_i++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_wFr[j, i_i] = (from t in Range(T) select delta_uF[t, j] * y[t -1, i_i]).Sum()
                float _wk29 = 0;
                for(int t = 0; t < T; t++){
                    _wk29 += delta_uF[(Y * t + j) * BatchSize + _batch_idx] * (0 <= t - 1 ? y[(Y * (t -1) + i_i) * BatchSize + _batch_idx] : 0);
                }
                delta_wFr[(Y * j + i_i) * BatchSize + _batch_idx] = _wk29;

            }
        }
    }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_wF[j] = (from t in Range(T) select delta_uF[t, j] * s[t -1, j]).Sum()
            float _wk30 = 0;
            for(int t = 0; t < T; t++){
                _wk30 += delta_uF[(Y * t + j) * BatchSize + _batch_idx] * (0 <= t - 1 ? s[(Y * (t -1) + j) * BatchSize + _batch_idx] : 0);
            }
            delta_wF[(j) * BatchSize + _batch_idx] = _wk30;

        }
    }
#pragma omp parallel for
    for (int j = 0; j < Y; j++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_bF[j] = (from t in Range(T) select delta_uF[t, j]).Sum()
            float _wk31 = 0;
            for(int t = 0; t < T; t++){
                _wk31 += delta_uF[(Y * t + j) * BatchSize + _batch_idx];
            }
            delta_bF[(j) * BatchSize + _batch_idx] = _wk31;

        }
    }
}

void LSTMLayerF::UpdateParameter_0(){
	int _count = Y * X;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_wIin[offset + i];
			}
			wIin[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_wFin[offset + i];
			}
			wFin[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_wOin[offset + i];
			}
			wOin[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_win[offset + i];
			}
			win[_idx] -= LearningRate * sum;
		}
	}
}

void LSTMLayerF::UpdateParameter_1(){
	int _count = Y * Y;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_wIr[offset + i];
			}
			wIr[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_wFr[offset + i];
			}
			wFr[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_wOr[offset + i];
			}
			wOr[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_wr[offset + i];
			}
			wr[_idx] -= LearningRate * sum;
		}
	}
}

void LSTMLayerF::UpdateParameter_2(){
	int _count = Y;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_wI[offset + i];
			}
			wI[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_wF[offset + i];
			}
			wF[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_wO[offset + i];
			}
			wO[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_bO[offset + i];
			}
			bO[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_bF[offset + i];
			}
			bF[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_bI[offset + i];
			}
			bI[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_b[offset + i];
			}
			b[_idx] -= LearningRate * sum;
		}
	}
}
void LSTMLayerF::UpdateParameter(){
	UpdateParameter_0();
	UpdateParameter_1();
	UpdateParameter_2();
}

extern "C" DllExport Layer* MakeLSTMLayerF(int t_size, int x_size, int y_size){
	return new LSTMLayerF(t_size, x_size, y_size);
}
