#pragma once

class Layer {
public:
	int BatchSize;
	float LearningRate;

	Layer() {
	}

	virtual void Forward() = 0;
	virtual void Backward() = 0;
	virtual void Allocate() = 0;
	virtual void Free() = 0;
	virtual void UpdateParameter() = 0;
	virtual void SetInput(void* src) = 0;
	virtual void* GetOutput() = 0;
	virtual void SetOutputDelta(void* src) = 0;
	virtual void* GetInputDelta() = 0;
};

void InitLog();
void Log(wchar_t *szFormat, ...);
float NormalRand();

template <class T> inline void SetNormalRand(T* &x, int size) {
	x = new T[size];

	for (int i = 0; i < size; i++) {
		x[i] = NormalRand();
	}
}