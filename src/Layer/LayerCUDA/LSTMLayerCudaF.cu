#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <FLOAT.h>
#include "MkFn.h"
#include "../../Lib/Lib.h"
#include "LibCuda.h"
#include "LSTMLayerCudaF.h"
__constant__ int _BatchSize;
__constant__ float _LearningRate;

void LSTMLayerCudaF::GetFieldName(int field_idx, wchar_t* name){
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

int LSTMLayerCudaF::GetFieldDimension(int field_idx){
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

int* LSTMLayerCudaF::GetFieldSize(int field_idx){
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

void LSTMLayerCudaF::GetFieldValue(int field_idx, void* dst){
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

void LSTMLayerCudaF::SetFieldValue(int field_idx, void* src){
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

LSTMLayerCudaF::LSTMLayerCudaF(int t_size, int x_size, int y_size){

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
	_chk(cudaStreamCreate(&_stream_y));
	_chk(cudaStreamCreate(&_stream_u));
	_chk(cudaStreamCreate(&_stream_s));
	_chk(cudaStreamCreate(&_stream_uI));
	_chk(cudaStreamCreate(&_stream_uF));
	_chk(cudaStreamCreate(&_stream_uO));
	_chk(cudaStreamCreate(&_stream_delta_x));
	_chk(cudaStreamCreate(&_stream_delta_wIin));
	_chk(cudaStreamCreate(&_stream_delta_wFin));
	_chk(cudaStreamCreate(&_stream_delta_wOin));
	_chk(cudaStreamCreate(&_stream_delta_win));
	_chk(cudaStreamCreate(&_stream_delta_wIr));
	_chk(cudaStreamCreate(&_stream_delta_wFr));
	_chk(cudaStreamCreate(&_stream_delta_wOr));
	_chk(cudaStreamCreate(&_stream_delta_wr));
	_chk(cudaStreamCreate(&_stream_delta_wI));
	_chk(cudaStreamCreate(&_stream_delta_wF));
	_chk(cudaStreamCreate(&_stream_delta_wO));
	_chk(cudaStreamCreate(&_stream_delta_bO));
	_chk(cudaStreamCreate(&_stream_delta_bF));
	_chk(cudaStreamCreate(&_stream_delta_bI));
	_chk(cudaStreamCreate(&_stream_delta_b));
	_chk(cudaStreamCreate(&_stream_delta_u));
	_chk(cudaStreamCreate(&_stream_delta_s));
	_chk(cudaStreamCreate(&_stream_delta_uI));
	_chk(cudaStreamCreate(&_stream_delta_uF));
	_chk(cudaStreamCreate(&_stream_delta_uO));

	_chk(cudaEventCreate(&_event_y));
	_chk(cudaEventCreate(&_event_u));
	_chk(cudaEventCreate(&_event_s));
	_chk(cudaEventCreate(&_event_uI));
	_chk(cudaEventCreate(&_event_uF));
	_chk(cudaEventCreate(&_event_uO));
	_chk(cudaEventCreate(&_event_delta_x));
	_chk(cudaEventCreate(&_event_delta_wIin));
	_chk(cudaEventCreate(&_event_delta_wFin));
	_chk(cudaEventCreate(&_event_delta_wOin));
	_chk(cudaEventCreate(&_event_delta_win));
	_chk(cudaEventCreate(&_event_delta_wIr));
	_chk(cudaEventCreate(&_event_delta_wFr));
	_chk(cudaEventCreate(&_event_delta_wOr));
	_chk(cudaEventCreate(&_event_delta_wr));
	_chk(cudaEventCreate(&_event_delta_wI));
	_chk(cudaEventCreate(&_event_delta_wF));
	_chk(cudaEventCreate(&_event_delta_wO));
	_chk(cudaEventCreate(&_event_delta_bO));
	_chk(cudaEventCreate(&_event_delta_bF));
	_chk(cudaEventCreate(&_event_delta_bI));
	_chk(cudaEventCreate(&_event_delta_b));
	_chk(cudaEventCreate(&_event_delta_u));
	_chk(cudaEventCreate(&_event_delta_s));
	_chk(cudaEventCreate(&_event_delta_uI));
	_chk(cudaEventCreate(&_event_delta_uF));
	_chk(cudaEventCreate(&_event_delta_uO));

}

LSTMLayerCudaF::~LSTMLayerCudaF(){
	Free();
	_chk(cudaStreamDestroy(_stream_y));
	_chk(cudaStreamDestroy(_stream_u));
	_chk(cudaStreamDestroy(_stream_s));
	_chk(cudaStreamDestroy(_stream_uI));
	_chk(cudaStreamDestroy(_stream_uF));
	_chk(cudaStreamDestroy(_stream_uO));
	_chk(cudaStreamDestroy(_stream_delta_x));
	_chk(cudaStreamDestroy(_stream_delta_wIin));
	_chk(cudaStreamDestroy(_stream_delta_wFin));
	_chk(cudaStreamDestroy(_stream_delta_wOin));
	_chk(cudaStreamDestroy(_stream_delta_win));
	_chk(cudaStreamDestroy(_stream_delta_wIr));
	_chk(cudaStreamDestroy(_stream_delta_wFr));
	_chk(cudaStreamDestroy(_stream_delta_wOr));
	_chk(cudaStreamDestroy(_stream_delta_wr));
	_chk(cudaStreamDestroy(_stream_delta_wI));
	_chk(cudaStreamDestroy(_stream_delta_wF));
	_chk(cudaStreamDestroy(_stream_delta_wO));
	_chk(cudaStreamDestroy(_stream_delta_bO));
	_chk(cudaStreamDestroy(_stream_delta_bF));
	_chk(cudaStreamDestroy(_stream_delta_bI));
	_chk(cudaStreamDestroy(_stream_delta_b));
	_chk(cudaStreamDestroy(_stream_delta_u));
	_chk(cudaStreamDestroy(_stream_delta_s));
	_chk(cudaStreamDestroy(_stream_delta_uI));
	_chk(cudaStreamDestroy(_stream_delta_uF));
	_chk(cudaStreamDestroy(_stream_delta_uO));

	_chk(cudaEventDestroy(_event_y));
	_chk(cudaEventDestroy(_event_u));
	_chk(cudaEventDestroy(_event_s));
	_chk(cudaEventDestroy(_event_uI));
	_chk(cudaEventDestroy(_event_uF));
	_chk(cudaEventDestroy(_event_uO));
	_chk(cudaEventDestroy(_event_delta_x));
	_chk(cudaEventDestroy(_event_delta_wIin));
	_chk(cudaEventDestroy(_event_delta_wFin));
	_chk(cudaEventDestroy(_event_delta_wOin));
	_chk(cudaEventDestroy(_event_delta_win));
	_chk(cudaEventDestroy(_event_delta_wIr));
	_chk(cudaEventDestroy(_event_delta_wFr));
	_chk(cudaEventDestroy(_event_delta_wOr));
	_chk(cudaEventDestroy(_event_delta_wr));
	_chk(cudaEventDestroy(_event_delta_wI));
	_chk(cudaEventDestroy(_event_delta_wF));
	_chk(cudaEventDestroy(_event_delta_wO));
	_chk(cudaEventDestroy(_event_delta_bO));
	_chk(cudaEventDestroy(_event_delta_bF));
	_chk(cudaEventDestroy(_event_delta_bI));
	_chk(cudaEventDestroy(_event_delta_b));
	_chk(cudaEventDestroy(_event_delta_u));
	_chk(cudaEventDestroy(_event_delta_s));
	_chk(cudaEventDestroy(_event_delta_uI));
	_chk(cudaEventDestroy(_event_delta_uF));
	_chk(cudaEventDestroy(_event_delta_uO));

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

void LSTMLayerCudaF::Allocate(){
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

void LSTMLayerCudaF::Free(){
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
__global__ static void forward_uF(float* uF, int X, float* wFin, float* x, int Y, float* wFr, float* y, float* wF, float* s, float* bF, int T, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // uF[t, j] = (from i in Range(X) select wFin[j, i] * x[t, i]).Sum() + (from i in Range(Y) select wFr[j, i] * y[t -1, i]).Sum() + wF[j] * s[t -1, j] + bF[j]
    float _wk1 = 0;
    for(int i = 0; i < X; i++){
        _wk1 += wFin[X * j + i] * x[(X * t + i) * _BatchSize + _batch_idx];
    }
    float _wk2 = 0;
    for(int i = 0; i < Y; i++){
        _wk2 += wFr[Y * j + i] * (0 <= t - 1 ? y[(Y * (t -1) + i) * _BatchSize + _batch_idx] : 0);
    }
    uF[(Y * t + j) * _BatchSize + _batch_idx] = _wk1 + _wk2 + wF[j] * (0 <= t - 1 ? s[(Y * (t -1) + j) * _BatchSize + _batch_idx] : 0) + bF[j];

}
void LSTMLayerCudaF::Start_forward_uF(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	forward_uF<<<blocksPerGrid, threadsPerBlock, 0, _stream_uF>>>(uF, X, wFin, x, Y, wFr, y, wF, s, bF, T, t);
	_chk(cudaEventRecord(_event_uF, _stream_uF));
}
__global__ static void forward_uI(float* uI, int X, float* wIin, float* x, int Y, float* wIr, float* y, float* wI, float* s, float* bI, int T, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // uI[t, j] = (from i in Range(X) select wIin[j, i] * x[t, i]).Sum() + (from i in Range(Y) select wIr[j, i] * y[t -1, i]).Sum() + wI[j] * s[t -1, j] + bI[j]
    float _wk3 = 0;
    for(int i = 0; i < X; i++){
        _wk3 += wIin[X * j + i] * x[(X * t + i) * _BatchSize + _batch_idx];
    }
    float _wk4 = 0;
    for(int i = 0; i < Y; i++){
        _wk4 += wIr[Y * j + i] * (0 <= t - 1 ? y[(Y * (t -1) + i) * _BatchSize + _batch_idx] : 0);
    }
    uI[(Y * t + j) * _BatchSize + _batch_idx] = _wk3 + _wk4 + wI[j] * (0 <= t - 1 ? s[(Y * (t -1) + j) * _BatchSize + _batch_idx] : 0) + bI[j];

}
void LSTMLayerCudaF::Start_forward_uI(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	forward_uI<<<blocksPerGrid, threadsPerBlock, 0, _stream_uI>>>(uI, X, wIin, x, Y, wIr, y, wI, s, bI, T, t);
	_chk(cudaEventRecord(_event_uI, _stream_uI));
}
__global__ static void forward_u(float* u, int X, float* win, float* x, int Y, float* wr, float* y, float* b, int T, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // u[t, j] = (from i in Range(X) select win[j, i] * x[t, i]).Sum() + (from i in Range(Y) select wr[j, i] * y[t -1, i]).Sum() + b[j]
    float _wk5 = 0;
    for(int i = 0; i < X; i++){
        _wk5 += win[X * j + i] * x[(X * t + i) * _BatchSize + _batch_idx];
    }
    float _wk6 = 0;
    for(int i = 0; i < Y; i++){
        _wk6 += wr[Y * j + i] * (0 <= t - 1 ? y[(Y * (t -1) + i) * _BatchSize + _batch_idx] : 0);
    }
    u[(Y * t + j) * _BatchSize + _batch_idx] = _wk5 + _wk6 + b[j];

}
void LSTMLayerCudaF::Start_forward_u(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	forward_u<<<blocksPerGrid, threadsPerBlock, 0, _stream_u>>>(u, X, win, x, Y, wr, y, b, T, t);
	_chk(cudaEventRecord(_event_u, _stream_u));
}
__global__ static void forward_s(float* s, float* uF, float* uI, float* u, int T, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // s[t, j] = sigmoid(uF[t, j]) * s[t -1, j] + sigmoid(uI[t, j]) * sigmoid(u[t, j])
    s[(Y * t + j) * _BatchSize + _batch_idx] = sigmoid(uF[(Y * t + j) * _BatchSize + _batch_idx]) * (0 <= t - 1 ? s[(Y * (t -1) + j) * _BatchSize + _batch_idx] : 0) + sigmoid(uI[(Y * t + j) * _BatchSize + _batch_idx]) * sigmoid(u[(Y * t + j) * _BatchSize + _batch_idx]);

}
void LSTMLayerCudaF::Start_forward_s(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_s, _event_uF, 0));
	_chk(cudaStreamWaitEvent(_stream_s, _event_uI, 0));
	_chk(cudaStreamWaitEvent(_stream_s, _event_u, 0));
	forward_s<<<blocksPerGrid, threadsPerBlock, 0, _stream_s>>>(s, uF, uI, u, T, Y, t);
	_chk(cudaEventRecord(_event_s, _stream_s));
}
__global__ static void forward_uO(float* uO, int X, float* wOin, float* x, int Y, float* wOr, float* y, float* wO, float* s, float* bO, int T, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // uO[t, j] = (from i in Range(X) select wOin[j, i] * x[t, i]).Sum() + (from i in Range(Y) select wOr[j, i] * y[t -1, i]).Sum() + wO[j] * s[t, j] + bO[j]
    float _wk7 = 0;
    for(int i = 0; i < X; i++){
        _wk7 += wOin[X * j + i] * x[(X * t + i) * _BatchSize + _batch_idx];
    }
    float _wk8 = 0;
    for(int i = 0; i < Y; i++){
        _wk8 += wOr[Y * j + i] * (0 <= t - 1 ? y[(Y * (t -1) + i) * _BatchSize + _batch_idx] : 0);
    }
    uO[(Y * t + j) * _BatchSize + _batch_idx] = _wk7 + _wk8 + wO[j] * s[(Y * t + j) * _BatchSize + _batch_idx] + bO[j];

}
void LSTMLayerCudaF::Start_forward_uO(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_uO, _event_s, 0));
	forward_uO<<<blocksPerGrid, threadsPerBlock, 0, _stream_uO>>>(uO, X, wOin, x, Y, wOr, y, wO, s, bO, T, t);
	_chk(cudaEventRecord(_event_uO, _stream_uO));
}
__global__ static void forward_y(float* y, float* uO, float* s, int T, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // y[t, j] = sigmoid(uO[t, j]) * sigmoid(s[t, j])
    y[(Y * t + j) * _BatchSize + _batch_idx] = sigmoid(uO[(Y * t + j) * _BatchSize + _batch_idx]) * sigmoid(s[(Y * t + j) * _BatchSize + _batch_idx]);

}
void LSTMLayerCudaF::Start_forward_y(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_y, _event_uO, 0));
	_chk(cudaStreamWaitEvent(_stream_y, _event_s, 0));
	forward_y<<<blocksPerGrid, threadsPerBlock, 0, _stream_y>>>(y, uO, s, T, Y, t);
	_chk(cudaEventRecord(_event_y, _stream_y));
}
void LSTMLayerCudaF::Forward(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	Start_forward_uF();
	Start_forward_uI();
	Start_forward_u();
	Start_forward_s();
	Start_forward_uO();
	Start_forward_y();
}
__global__ static void backward_delta_y(float* delta_y, int Y, float* delta_uO, float* wOr, float* delta_uF, float* wFr, float* delta_uI, float* wIr, float* delta_u, float* wr, int T, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_y[t, j] = (from j in Range(Y) select delta_uO[t + 1, j] * wOr[j, j]).Sum() + (from j in Range(Y) select delta_uF[t + 1, j] * wFr[j, j]).Sum() + (from j in Range(Y) select delta_uI[t + 1, j] * wIr[j, j]).Sum() + (from j in Range(Y) select delta_u[t + 1, j] * wr[j, j]).Sum()
    float _wk9 = 0;
    for(int j = 0; j < Y; j++){
        _wk9 += (t + 1 < T ? delta_uO[(Y * (t + 1) + j) * _BatchSize + _batch_idx] : 0) * wOr[Y * j + j];
    }
    float _wk10 = 0;
    for(int j = 0; j < Y; j++){
        _wk10 += (t + 1 < T ? delta_uF[(Y * (t + 1) + j) * _BatchSize + _batch_idx] : 0) * wFr[Y * j + j];
    }
    float _wk11 = 0;
    for(int j = 0; j < Y; j++){
        _wk11 += (t + 1 < T ? delta_uI[(Y * (t + 1) + j) * _BatchSize + _batch_idx] : 0) * wIr[Y * j + j];
    }
    float _wk12 = 0;
    for(int j = 0; j < Y; j++){
        _wk12 += (t + 1 < T ? delta_u[(Y * (t + 1) + j) * _BatchSize + _batch_idx] : 0) * wr[Y * j + j];
    }
    delta_y[(Y * t + j) * _BatchSize + _batch_idx] += _wk9 + _wk10 + _wk11 + _wk12;

}
void LSTMLayerCudaF::Start_backward_delta_y(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	backward_delta_y<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_y>>>(delta_y, Y, delta_uO, wOr, delta_uF, wFr, delta_uI, wIr, delta_u, wr, T, t);
	_chk(cudaEventRecord(_event_delta_y, _stream_delta_y));
}
__global__ static void backward_delta_uO(float* delta_uO, float* delta_y, float* uO, float* s, int T, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_uO[t, j] = delta_y[t, j] * sigmoid_prime(uO[t, j]) * sigmoid(s[t, j])
    delta_uO[(Y * t + j) * _BatchSize + _batch_idx] = delta_y[(Y * t + j) * _BatchSize + _batch_idx] * sigmoid_prime(uO[(Y * t + j) * _BatchSize + _batch_idx]) * sigmoid(s[(Y * t + j) * _BatchSize + _batch_idx]);

}
void LSTMLayerCudaF::Start_backward_delta_uO(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_uO, _event_delta_y, 0));
	backward_delta_uO<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_uO>>>(delta_uO, delta_y, uO, s, T, Y, t);
	_chk(cudaEventRecord(_event_delta_uO, _stream_delta_uO));
}
__global__ static void backward_delta_wOin(float* delta_wOin, int T, float* delta_uO, float* x, int Y, int X, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.y;
	int i_i = blockIdx.x;

    // delta_wOin[j, i_i] = (from t in Range(T) select delta_uO[t, j] * x[t, i_i]).Sum()
    float _wk13 = 0;
    for(int t = 0; t < T; t++){
        _wk13 += delta_uO[(Y * t + j) * _BatchSize + _batch_idx] * x[(X * t + i_i) * _BatchSize + _batch_idx];
    }
    delta_wOin[(X * j + i_i) * _BatchSize + _batch_idx] = _wk13;

}
void LSTMLayerCudaF::Start_backward_delta_wOin(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_y = Y;
	blocks_x = X;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_wOin, _event_delta_uO, 0));
	backward_delta_wOin<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_wOin>>>(delta_wOin, T, delta_uO, x, Y, X, t);
	_chk(cudaEventRecord(_event_delta_wOin, _stream_delta_wOin));
}
__global__ static void backward_delta_wOr(float* delta_wOr, int T, float* delta_uO, float* y, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.y;
	int i_i = blockIdx.x;

    // delta_wOr[j, i_i] = (from t in Range(T) select delta_uO[t, j] * y[t -1, i_i]).Sum()
    float _wk14 = 0;
    for(int t = 0; t < T; t++){
        _wk14 += delta_uO[(Y * t + j) * _BatchSize + _batch_idx] * (0 <= t - 1 ? y[(Y * (t -1) + i_i) * _BatchSize + _batch_idx] : 0);
    }
    delta_wOr[(Y * j + i_i) * _BatchSize + _batch_idx] = _wk14;

}
void LSTMLayerCudaF::Start_backward_delta_wOr(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_y = Y;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_wOr, _event_delta_uO, 0));
	backward_delta_wOr<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_wOr>>>(delta_wOr, T, delta_uO, y, Y, t);
	_chk(cudaEventRecord(_event_delta_wOr, _stream_delta_wOr));
}
__global__ static void backward_delta_wO(float* delta_wO, int T, float* delta_uO, float* s, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_wO[j] = (from t in Range(T) select delta_uO[t, j] * s[t, j]).Sum()
    float _wk15 = 0;
    for(int t = 0; t < T; t++){
        _wk15 += delta_uO[(Y * t + j) * _BatchSize + _batch_idx] * s[(Y * t + j) * _BatchSize + _batch_idx];
    }
    delta_wO[(j) * _BatchSize + _batch_idx] = _wk15;

}
void LSTMLayerCudaF::Start_backward_delta_wO(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_wO, _event_delta_uO, 0));
	backward_delta_wO<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_wO>>>(delta_wO, T, delta_uO, s, Y, t);
	_chk(cudaEventRecord(_event_delta_wO, _stream_delta_wO));
}
__global__ static void backward_delta_bO(float* delta_bO, int T, float* delta_uO, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_bO[j] = (from t in Range(T) select delta_uO[t, j]).Sum()
    float _wk16 = 0;
    for(int t = 0; t < T; t++){
        _wk16 += delta_uO[(Y * t + j) * _BatchSize + _batch_idx];
    }
    delta_bO[(j) * _BatchSize + _batch_idx] = _wk16;

}
void LSTMLayerCudaF::Start_backward_delta_bO(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_bO, _event_delta_uO, 0));
	backward_delta_bO<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_bO>>>(delta_bO, T, delta_uO, Y, t);
	_chk(cudaEventRecord(_event_delta_bO, _stream_delta_bO));
}
__global__ static void backward_delta_s(float* delta_s, float* delta_y, float* uO, float* s, float* uF, float* delta_uO, float* wO, float* delta_uF, float* wF, float* delta_uI, float* wI, int T, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_s[t, j] = delta_y[t, j] * sigmoid(uO[t, j]) * sigmoid_prime(s[t, j]) + delta_s[t + 1, j] * sigmoid(uF[t + 1, j]) + delta_uO[t, j] * wO[j] + delta_uF[t + 1, j] * wF[j] + delta_uI[t + 1, j] * wI[j]
    delta_s[(Y * t + j) * _BatchSize + _batch_idx] = delta_y[(Y * t + j) * _BatchSize + _batch_idx] * sigmoid(uO[(Y * t + j) * _BatchSize + _batch_idx]) * sigmoid_prime(s[(Y * t + j) * _BatchSize + _batch_idx]) + (t + 1 < T ? delta_s[(Y * (t + 1) + j) * _BatchSize + _batch_idx] : 0) * sigmoid((t + 1 < T ? uF[(Y * (t + 1) + j) * _BatchSize + _batch_idx] : 0)) + delta_uO[(Y * t + j) * _BatchSize + _batch_idx] * wO[j] + (t + 1 < T ? delta_uF[(Y * (t + 1) + j) * _BatchSize + _batch_idx] : 0) * wF[j] + (t + 1 < T ? delta_uI[(Y * (t + 1) + j) * _BatchSize + _batch_idx] : 0) * wI[j];

}
void LSTMLayerCudaF::Start_backward_delta_s(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_s, _event_delta_y, 0));
	_chk(cudaStreamWaitEvent(_stream_delta_s, _event_delta_uO, 0));
	backward_delta_s<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_s>>>(delta_s, delta_y, uO, s, uF, delta_uO, wO, delta_uF, wF, delta_uI, wI, T, Y, t);
	_chk(cudaEventRecord(_event_delta_s, _stream_delta_s));
}
__global__ static void backward_delta_u(float* delta_u, float* delta_s, float* uI, float* u, int T, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_u[t, j] = delta_s[t, j] * sigmoid(uI[t, j]) * sigmoid_prime(u[t, j])
    delta_u[(Y * t + j) * _BatchSize + _batch_idx] = delta_s[(Y * t + j) * _BatchSize + _batch_idx] * sigmoid(uI[(Y * t + j) * _BatchSize + _batch_idx]) * sigmoid_prime(u[(Y * t + j) * _BatchSize + _batch_idx]);

}
void LSTMLayerCudaF::Start_backward_delta_u(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_u, _event_delta_s, 0));
	backward_delta_u<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_u>>>(delta_u, delta_s, uI, u, T, Y, t);
	_chk(cudaEventRecord(_event_delta_u, _stream_delta_u));
}
__global__ static void backward_delta_win(float* delta_win, int T, float* delta_u, float* x, int Y, int X, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.y;
	int i_i = blockIdx.x;

    // delta_win[j, i_i] = (from t in Range(T) select delta_u[t, j] * x[t, i_i]).Sum()
    float _wk17 = 0;
    for(int t = 0; t < T; t++){
        _wk17 += delta_u[(Y * t + j) * _BatchSize + _batch_idx] * x[(X * t + i_i) * _BatchSize + _batch_idx];
    }
    delta_win[(X * j + i_i) * _BatchSize + _batch_idx] = _wk17;

}
void LSTMLayerCudaF::Start_backward_delta_win(){
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
__global__ static void backward_delta_wr(float* delta_wr, int T, float* delta_u, float* y, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.y;
	int i_i = blockIdx.x;

    // delta_wr[j, i_i] = (from t in Range(T) select delta_u[t, j] * y[t -1, i_i]).Sum()
    float _wk18 = 0;
    for(int t = 0; t < T; t++){
        _wk18 += delta_u[(Y * t + j) * _BatchSize + _batch_idx] * (0 <= t - 1 ? y[(Y * (t -1) + i_i) * _BatchSize + _batch_idx] : 0);
    }
    delta_wr[(Y * j + i_i) * _BatchSize + _batch_idx] = _wk18;

}
void LSTMLayerCudaF::Start_backward_delta_wr(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_y = Y;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_wr, _event_delta_u, 0));
	backward_delta_wr<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_wr>>>(delta_wr, T, delta_u, y, Y, t);
	_chk(cudaEventRecord(_event_delta_wr, _stream_delta_wr));
}
__global__ static void backward_delta_b(float* delta_b, int T, float* delta_u, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_b[j] = (from t in Range(T) select delta_u[t, j]).Sum()
    float _wk19 = 0;
    for(int t = 0; t < T; t++){
        _wk19 += delta_u[(Y * t + j) * _BatchSize + _batch_idx];
    }
    delta_b[(j) * _BatchSize + _batch_idx] = _wk19;

}
void LSTMLayerCudaF::Start_backward_delta_b(){
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
__global__ static void backward_delta_uI(float* delta_uI, float* delta_s, float* uI, float* u, int T, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_uI[t, j] = delta_s[t, j] * sigmoid_prime(uI[t, j]) * sigmoid(u[t, j])
    delta_uI[(Y * t + j) * _BatchSize + _batch_idx] = delta_s[(Y * t + j) * _BatchSize + _batch_idx] * sigmoid_prime(uI[(Y * t + j) * _BatchSize + _batch_idx]) * sigmoid(u[(Y * t + j) * _BatchSize + _batch_idx]);

}
void LSTMLayerCudaF::Start_backward_delta_uI(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_uI, _event_delta_s, 0));
	backward_delta_uI<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_uI>>>(delta_uI, delta_s, uI, u, T, Y, t);
	_chk(cudaEventRecord(_event_delta_uI, _stream_delta_uI));
}
__global__ static void backward_delta_wIin(float* delta_wIin, int T, float* delta_uI, float* x, int Y, int X, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.y;
	int i_i = blockIdx.x;

    // delta_wIin[j, i_i] = (from t in Range(T) select delta_uI[t, j] * x[t, i_i]).Sum()
    float _wk20 = 0;
    for(int t = 0; t < T; t++){
        _wk20 += delta_uI[(Y * t + j) * _BatchSize + _batch_idx] * x[(X * t + i_i) * _BatchSize + _batch_idx];
    }
    delta_wIin[(X * j + i_i) * _BatchSize + _batch_idx] = _wk20;

}
void LSTMLayerCudaF::Start_backward_delta_wIin(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_y = Y;
	blocks_x = X;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_wIin, _event_delta_uI, 0));
	backward_delta_wIin<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_wIin>>>(delta_wIin, T, delta_uI, x, Y, X, t);
	_chk(cudaEventRecord(_event_delta_wIin, _stream_delta_wIin));
}
__global__ static void backward_delta_wIr(float* delta_wIr, int T, float* delta_uI, float* y, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.y;
	int i_i = blockIdx.x;

    // delta_wIr[j, i_i] = (from t in Range(T) select delta_uI[t, j] * y[t -1, i_i]).Sum()
    float _wk21 = 0;
    for(int t = 0; t < T; t++){
        _wk21 += delta_uI[(Y * t + j) * _BatchSize + _batch_idx] * (0 <= t - 1 ? y[(Y * (t -1) + i_i) * _BatchSize + _batch_idx] : 0);
    }
    delta_wIr[(Y * j + i_i) * _BatchSize + _batch_idx] = _wk21;

}
void LSTMLayerCudaF::Start_backward_delta_wIr(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_y = Y;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_wIr, _event_delta_uI, 0));
	backward_delta_wIr<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_wIr>>>(delta_wIr, T, delta_uI, y, Y, t);
	_chk(cudaEventRecord(_event_delta_wIr, _stream_delta_wIr));
}
__global__ static void backward_delta_wI(float* delta_wI, int T, float* delta_uI, float* s, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_wI[j] = (from t in Range(T) select delta_uI[t, j] * s[t -1, j]).Sum()
    float _wk22 = 0;
    for(int t = 0; t < T; t++){
        _wk22 += delta_uI[(Y * t + j) * _BatchSize + _batch_idx] * (0 <= t - 1 ? s[(Y * (t -1) + j) * _BatchSize + _batch_idx] : 0);
    }
    delta_wI[(j) * _BatchSize + _batch_idx] = _wk22;

}
void LSTMLayerCudaF::Start_backward_delta_wI(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_wI, _event_delta_uI, 0));
	backward_delta_wI<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_wI>>>(delta_wI, T, delta_uI, s, Y, t);
	_chk(cudaEventRecord(_event_delta_wI, _stream_delta_wI));
}
__global__ static void backward_delta_bI(float* delta_bI, int T, float* delta_uI, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_bI[j] = (from t in Range(T) select delta_uI[t, j]).Sum()
    float _wk23 = 0;
    for(int t = 0; t < T; t++){
        _wk23 += delta_uI[(Y * t + j) * _BatchSize + _batch_idx];
    }
    delta_bI[(j) * _BatchSize + _batch_idx] = _wk23;

}
void LSTMLayerCudaF::Start_backward_delta_bI(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_bI, _event_delta_uI, 0));
	backward_delta_bI<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_bI>>>(delta_bI, T, delta_uI, Y, t);
	_chk(cudaEventRecord(_event_delta_bI, _stream_delta_bI));
}
__global__ static void backward_delta_uF(float* delta_uF, float* delta_s, float* uF, float* s, int T, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_uF[t, j] = delta_s[t, j] * sigmoid_prime(uF[t, j]) * s[t -1, j]
    delta_uF[(Y * t + j) * _BatchSize + _batch_idx] = delta_s[(Y * t + j) * _BatchSize + _batch_idx] * sigmoid_prime(uF[(Y * t + j) * _BatchSize + _batch_idx]) * (0 <= t - 1 ? s[(Y * (t -1) + j) * _BatchSize + _batch_idx] : 0);

}
void LSTMLayerCudaF::Start_backward_delta_uF(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_uF, _event_delta_s, 0));
	backward_delta_uF<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_uF>>>(delta_uF, delta_s, uF, s, T, Y, t);
	_chk(cudaEventRecord(_event_delta_uF, _stream_delta_uF));
}
__global__ static void backward_delta_x(float* delta_x, int Y, float* delta_uO, float* wOin, float* delta_uF, float* wFin, float* delta_uI, float* wIin, float* delta_u, float* win, int T, int X, int t){
	int _batch_idx = threadIdx.x;
	int i_i = blockIdx.x;

    // delta_x[t, i_i] = (from j in Range(Y) select delta_uO[t, j] * wOin[j, i_i]).Sum() + (from j in Range(Y) select delta_uF[t, j] * wFin[j, i_i]).Sum() + (from j in Range(Y) select delta_uI[t, j] * wIin[j, i_i]).Sum() + (from j in Range(Y) select delta_u[t, j] * win[j, i_i]).Sum()
    float _wk24 = 0;
    for(int j = 0; j < Y; j++){
        _wk24 += delta_uO[(Y * t + j) * _BatchSize + _batch_idx] * wOin[X * j + i_i];
    }
    float _wk25 = 0;
    for(int j = 0; j < Y; j++){
        _wk25 += delta_uF[(Y * t + j) * _BatchSize + _batch_idx] * wFin[X * j + i_i];
    }
    float _wk26 = 0;
    for(int j = 0; j < Y; j++){
        _wk26 += delta_uI[(Y * t + j) * _BatchSize + _batch_idx] * wIin[X * j + i_i];
    }
    float _wk27 = 0;
    for(int j = 0; j < Y; j++){
        _wk27 += delta_u[(Y * t + j) * _BatchSize + _batch_idx] * win[X * j + i_i];
    }
    delta_x[(X * t + i_i) * _BatchSize + _batch_idx] = _wk24 + _wk25 + _wk26 + _wk27;

}
void LSTMLayerCudaF::Start_backward_delta_x(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = X;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_x, _event_delta_uO, 0));
	_chk(cudaStreamWaitEvent(_stream_delta_x, _event_delta_uF, 0));
	_chk(cudaStreamWaitEvent(_stream_delta_x, _event_delta_uI, 0));
	_chk(cudaStreamWaitEvent(_stream_delta_x, _event_delta_u, 0));
	backward_delta_x<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_x>>>(delta_x, Y, delta_uO, wOin, delta_uF, wFin, delta_uI, wIin, delta_u, win, T, X, t);
	_chk(cudaEventRecord(_event_delta_x, _stream_delta_x));
}
__global__ static void backward_delta_wFin(float* delta_wFin, int T, float* delta_uF, float* x, int Y, int X, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.y;
	int i_i = blockIdx.x;

    // delta_wFin[j, i_i] = (from t in Range(T) select delta_uF[t, j] * x[t, i_i]).Sum()
    float _wk28 = 0;
    for(int t = 0; t < T; t++){
        _wk28 += delta_uF[(Y * t + j) * _BatchSize + _batch_idx] * x[(X * t + i_i) * _BatchSize + _batch_idx];
    }
    delta_wFin[(X * j + i_i) * _BatchSize + _batch_idx] = _wk28;

}
void LSTMLayerCudaF::Start_backward_delta_wFin(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_y = Y;
	blocks_x = X;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_wFin, _event_delta_uF, 0));
	backward_delta_wFin<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_wFin>>>(delta_wFin, T, delta_uF, x, Y, X, t);
	_chk(cudaEventRecord(_event_delta_wFin, _stream_delta_wFin));
}
__global__ static void backward_delta_wFr(float* delta_wFr, int T, float* delta_uF, float* y, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.y;
	int i_i = blockIdx.x;

    // delta_wFr[j, i_i] = (from t in Range(T) select delta_uF[t, j] * y[t -1, i_i]).Sum()
    float _wk29 = 0;
    for(int t = 0; t < T; t++){
        _wk29 += delta_uF[(Y * t + j) * _BatchSize + _batch_idx] * (0 <= t - 1 ? y[(Y * (t -1) + i_i) * _BatchSize + _batch_idx] : 0);
    }
    delta_wFr[(Y * j + i_i) * _BatchSize + _batch_idx] = _wk29;

}
void LSTMLayerCudaF::Start_backward_delta_wFr(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_y = Y;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_wFr, _event_delta_uF, 0));
	backward_delta_wFr<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_wFr>>>(delta_wFr, T, delta_uF, y, Y, t);
	_chk(cudaEventRecord(_event_delta_wFr, _stream_delta_wFr));
}
__global__ static void backward_delta_wF(float* delta_wF, int T, float* delta_uF, float* s, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_wF[j] = (from t in Range(T) select delta_uF[t, j] * s[t -1, j]).Sum()
    float _wk30 = 0;
    for(int t = 0; t < T; t++){
        _wk30 += delta_uF[(Y * t + j) * _BatchSize + _batch_idx] * (0 <= t - 1 ? s[(Y * (t -1) + j) * _BatchSize + _batch_idx] : 0);
    }
    delta_wF[(j) * _BatchSize + _batch_idx] = _wk30;

}
void LSTMLayerCudaF::Start_backward_delta_wF(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_wF, _event_delta_uF, 0));
	backward_delta_wF<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_wF>>>(delta_wF, T, delta_uF, s, Y, t);
	_chk(cudaEventRecord(_event_delta_wF, _stream_delta_wF));
}
__global__ static void backward_delta_bF(float* delta_bF, int T, float* delta_uF, int Y, int t){
	int _batch_idx = threadIdx.x;
	int j = blockIdx.x;

    // delta_bF[j] = (from t in Range(T) select delta_uF[t, j]).Sum()
    float _wk31 = 0;
    for(int t = 0; t < T; t++){
        _wk31 += delta_uF[(Y * t + j) * _BatchSize + _batch_idx];
    }
    delta_bF[(j) * _BatchSize + _batch_idx] = _wk31;

}
void LSTMLayerCudaF::Start_backward_delta_bF(){
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	dim3 threadsPerBlock = dim3(BatchSize);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	_chk(cudaStreamWaitEvent(_stream_delta_bF, _event_delta_uF, 0));
	backward_delta_bF<<<blocksPerGrid, threadsPerBlock, 0, _stream_delta_bF>>>(delta_bF, T, delta_uF, Y, t);
	_chk(cudaEventRecord(_event_delta_bF, _stream_delta_bF));
}
void LSTMLayerCudaF::Backward(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	Start_backward_delta_y();
	Start_backward_delta_uO();
	Start_backward_delta_wOin();
	Start_backward_delta_wOr();
	Start_backward_delta_wO();
	Start_backward_delta_bO();
	Start_backward_delta_s();
	Start_backward_delta_u();
	Start_backward_delta_win();
	Start_backward_delta_wr();
	Start_backward_delta_b();
	Start_backward_delta_uI();
	Start_backward_delta_wIin();
	Start_backward_delta_wIr();
	Start_backward_delta_wI();
	Start_backward_delta_bI();
	Start_backward_delta_uF();
	Start_backward_delta_x();
	Start_backward_delta_wFin();
	Start_backward_delta_wFr();
	Start_backward_delta_wF();
	Start_backward_delta_bF();
}
__global__ static void UpdateParameterKernel_0(float* wIin, float* delta_wIin, float* wFin, float* delta_wFin, float* wOin, float* delta_wOin, float* win, float* delta_win){
	int _idx = (blockIdx.x) * blockDim.x + threadIdx.x;
	int offset = _idx * _BatchSize;
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_wIin[offset + i];
		}
		wIin[_idx] -= _LearningRate * sum;
	}
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_wFin[offset + i];
		}
		wFin[_idx] -= _LearningRate * sum;
	}
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_wOin[offset + i];
		}
		wOin[_idx] -= _LearningRate * sum;
	}
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_win[offset + i];
		}
		win[_idx] -= _LearningRate * sum;
	}
}
void LSTMLayerCudaF::UpdateParameter_0(){
	int threads_x = 1;
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	threads_x = X;
	dim3 threadsPerBlock = dim3(threads_x);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	UpdateParameterKernel_0<<<blocksPerGrid, threadsPerBlock>>>(wIin, delta_wIin, wFin, delta_wFin, wOin, delta_wOin, win, delta_win);
}
__global__ static void UpdateParameterKernel_1(float* wIr, float* delta_wIr, float* wFr, float* delta_wFr, float* wOr, float* delta_wOr, float* wr, float* delta_wr){
	int _idx = (blockIdx.x) * blockDim.x + threadIdx.x;
	int offset = _idx * _BatchSize;
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_wIr[offset + i];
		}
		wIr[_idx] -= _LearningRate * sum;
	}
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_wFr[offset + i];
		}
		wFr[_idx] -= _LearningRate * sum;
	}
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_wOr[offset + i];
		}
		wOr[_idx] -= _LearningRate * sum;
	}
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_wr[offset + i];
		}
		wr[_idx] -= _LearningRate * sum;
	}
}
void LSTMLayerCudaF::UpdateParameter_1(){
	int threads_x = 1;
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	blocks_x = Y;
	threads_x = Y;
	dim3 threadsPerBlock = dim3(threads_x);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	UpdateParameterKernel_1<<<blocksPerGrid, threadsPerBlock>>>(wIr, delta_wIr, wFr, delta_wFr, wOr, delta_wOr, wr, delta_wr);
}
__global__ static void UpdateParameterKernel_2(float* wI, float* delta_wI, float* wF, float* delta_wF, float* wO, float* delta_wO, float* bO, float* delta_bO, float* bF, float* delta_bF, float* bI, float* delta_bI, float* b, float* delta_b){
	int _idx = threadIdx.x;
	int offset = _idx * _BatchSize;
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_wI[offset + i];
		}
		wI[_idx] -= _LearningRate * sum;
	}
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_wF[offset + i];
		}
		wF[_idx] -= _LearningRate * sum;
	}
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_wO[offset + i];
		}
		wO[_idx] -= _LearningRate * sum;
	}
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_bO[offset + i];
		}
		bO[_idx] -= _LearningRate * sum;
	}
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_bF[offset + i];
		}
		bF[_idx] -= _LearningRate * sum;
	}
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_bI[offset + i];
		}
		bI[_idx] -= _LearningRate * sum;
	}
	{
		float sum = 0;
		for (int i = 0; i < _BatchSize; i++) {
			sum += delta_b[offset + i];
		}
		b[_idx] -= _LearningRate * sum;
	}
}
void LSTMLayerCudaF::UpdateParameter_2(){
	int threads_x = 1;
	int blocks_x = 1;
	int blocks_y = 1;
	int blocks_z = 1;
	threads_x = Y;
	dim3 threadsPerBlock = dim3(threads_x);
	dim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);
	UpdateParameterKernel_2<<<blocksPerGrid, threadsPerBlock>>>(wI, delta_wI, wF, delta_wF, wO, delta_wO, bO, delta_bO, bF, delta_bF, bI, delta_bI, b, delta_b);
}
void LSTMLayerCudaF::UpdateParameter(){
	_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));
	_chk(_MemcpyToSymbol(_LearningRate, LearningRate, sizeof(LearningRate)));
	_chk(cudaDeviceSynchronize());
	UpdateParameter_0();
	UpdateParameter_1();
	UpdateParameter_2();
	_chk(cudaDeviceSynchronize());
}

extern "C" DllExport Layer* MakeLSTMLayerCudaF(int t_size, int x_size, int y_size){
	return new LSTMLayerCudaF(t_size, x_size, y_size);
}
