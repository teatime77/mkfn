// Layer.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <FLOAT.h>
#include "MkFn.h"
#include "../../Lib/Lib.h"

extern "C" DllExport void DeviceSynchronize() {
}

extern "C" DllExport void DeviceInit() {
}

extern "C" DllExport void DeviceEnd() {
}

extern "C" DllExport void* DeviceMalloc(size_t size) {
	return malloc(size);
}


extern "C" DllExport void DeviceFree(void* p) {
	free(p);
}

#if _WIN64
#define	PLayer	long long
#else
#define	PLayer	int
#endif


extern "C" DllExport  int GetFieldCount(PLayer layer) {
	return ((Layer*)layer)->GetFieldCount();
}

extern "C" DllExport  void GetFieldName(PLayer layer, int field_idx, wchar_t* name) {
	((Layer*)layer)->GetFieldName(field_idx, name);
}

extern "C" DllExport  int GetFieldDimension(PLayer layer, int field_idx) {
	return ((Layer*)layer)->GetFieldDimension(field_idx);
}

extern "C" DllExport  int* GetFieldSize(PLayer layer, int field_idx) {
	return ((Layer*)layer)->GetFieldSize(field_idx);
}

extern "C" DllExport  void GetFieldValue(PLayer layer, int field_idx, void* dst) {
	return ((Layer*)layer)->GetFieldValue(field_idx, dst);
}

extern "C" DllExport  void SetFieldValue(PLayer layer, int field_idx, void* src) {
	return ((Layer*)layer)->SetFieldValue(field_idx, src);
}

extern "C" DllExport  int GetFieldIndexByName(PLayer layer, wchar_t* name) {
	return ((Layer*)layer)->GetFieldIndexByName( name);
}

extern "C" DllExport  int GetFieldElementCount(PLayer layer, int field_idx) {
	return ((Layer*)layer)->GetFieldElementCount(field_idx);
}

extern "C" DllExport int GetBatchSize(PLayer layer) {
	return ((Layer*)layer)->BatchSize;
}

extern "C" DllExport void SetBatchSize(PLayer layer, int batch_size) {
	((Layer*)layer)->BatchSize = batch_size;
}

extern "C" DllExport float GetLearningRate(PLayer layer) {
	return ((Layer*)layer)->LearningRate;
}

extern "C" DllExport void SetLearningRate(PLayer layer, float learning_rate) {
	((Layer*)layer)->LearningRate = learning_rate;
}

extern "C" DllExport int Get_t(PLayer layer) {
	return ((Layer*)layer)->t;
}

extern "C" DllExport void Set_t(PLayer layer, int t) {
	((Layer*)layer)->t = t;
}


extern "C" DllExport void Destroy(PLayer layer) {
	delete (Layer*)layer;
}

extern "C" DllExport void Forward(PLayer layer) {
	((Layer*)layer)->Forward();
}

extern "C" DllExport void Backward(PLayer layer) {
	((Layer*)layer)->Backward();
}

extern "C" DllExport void Allocate(PLayer layer) {
	((Layer*)layer)->Allocate();
}

extern "C" DllExport void Free(PLayer layer) {
	((Layer*)layer)->Free();
}

extern "C" DllExport void UpdateParameter(PLayer layer) {
	((Layer*)layer)->UpdateParameter();
}

extern "C" DllExport void SetInput(PLayer layer, void* src) {
	((Layer*)layer)->SetInput(src);
}

extern "C" DllExport void* GetInput(PLayer layer, int t = 0) {
	return ((Layer*)layer)->GetInput(t);
}

extern "C" DllExport void* GetOutput(PLayer layer, int t = 0) {
	return ((Layer*)layer)->GetOutput(t);
}

extern "C" DllExport void SetIputDelta(PLayer layer, void* src) {
	((Layer*)layer)->SetIputDelta(src);
}

extern "C" DllExport void SetOutputDelta(PLayer layer, void* src) {
	((Layer*)layer)->SetOutputDelta(src);
}

extern "C" DllExport void* GetOutputDelta(PLayer layer, int t = 0) {
	return ((Layer*)layer)->GetOutputDelta(t);
}

extern "C" DllExport void* GetInputDelta(PLayer layer, int t = 0) {
	return ((Layer*)layer)->GetInputDelta(t);
}


extern "C" DllExport void SetInputData(PLayer layer, void* src, int size) {
	((Layer*)layer)->SetInputData(src, size);
}

extern "C" DllExport void SetOutputDeltaData(PLayer layer, void* src, int size) {
	((Layer*)layer)->SetOutputDeltaData(src, size);
}

extern "C" DllExport void GetOutputData(PLayer layer, void* dst, int size) {
	((Layer*)layer)->GetOutputData(dst, size);
}

extern "C" DllExport void ConnectLayer(PLayer layer, PLayer next_layer) {
	((Layer*)layer)->ConnectLayer((Layer*)next_layer);
}

extern "C" DllExport int GetInputCount(PLayer layer) {
	return ((Layer*)layer)->GetInputCount();
}

extern "C" DllExport int GetOutputCount(PLayer layer) {
	return ((Layer*)layer)->GetOutputCount();
}

extern "C" DllExport int GetTimeCount(PLayer layer) {
	return ((Layer*)layer)->GetTimeCount();
}

extern "C" DllExport int GetTimeInputCount(PLayer layer) {
	return ((Layer*)layer)->GetTimeInputCount();
}

extern "C" DllExport int GetTimeOutputCount(PLayer layer) {
	return ((Layer*)layer)->GetTimeOutputCount();
}

extern "C" DllExport void SetTimeCount(PLayer layer, int time_count) {
	((Layer*)layer)->SetTimeCount(time_count);
}

extern "C" DllExport BOOL IsGPU(PLayer layer) {
	//return ((Layer*)layer)->IsGPU();
	bool b = ((Layer*)layer)->IsGPU();
	return b;
}

extern "C" DllExport void Add(float* a, int n, float* b) {
	for (int i = 0; i < 10; i++) {
		b[i] = a[i] + n;
	}
}


extern "C" DllExport int two(int n) {
	return n * 2;
}
