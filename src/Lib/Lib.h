#pragma once

class Layer {
public:
	int BatchSize;
	float LearningRate;
	int t;

	Layer() {
	}

	virtual ~Layer() {};

	virtual void Forward() = 0;
	virtual void Backward() = 0;
	virtual void Allocate() = 0;
	virtual void Free() = 0;
	virtual void UpdateParameter() = 0;
	virtual void SetInput(void* src) = 0;
	virtual void* GetInput(int t = 0) = 0;
	virtual void* GetOutput(int t = 0) = 0;
	virtual void SetIputDelta(void* src) = 0;
	virtual void SetOutputDelta(void* src) = 0;
	virtual void* GetOutputDelta(int t = 0) = 0;
	virtual void** GetOutputDeltaPtr() = 0;
	virtual void* GetInputDelta(int t = 0) = 0;

	virtual int GetInputCount() = 0;
	virtual int GetOutputCount() = 0;
	virtual int GetTimeCount() = 0;
	virtual int GetTimeInputCount() = 0;
	virtual int GetTimeOutputCount() = 0;

#ifdef __CUDACC__
	virtual void SetInputStream(cudaStream_t src) = 0;
	virtual cudaStream_t GetOutputStream() = 0;

	virtual void SetOutputDeltaStream(cudaStream_t src) = 0;
	virtual cudaStream_t GetInputDeltaStream() = 0;

	virtual void SetInputEvent(cudaEvent_t src) = 0;
	virtual cudaEvent_t GetOutputEvent() = 0;

	virtual void SetOutputDeltaEvent(cudaEvent_t src) = 0;
	virtual cudaEvent_t GetInputDeltaEvent() = 0;
#endif
};

void InitLog();
void Log(wchar_t *szFormat, ...);
void LogA(char *szFormat, ...);
double UniformRandom();
float NormalRand();

template <class T> inline void SetNormalRand(T* &x, int size) {
	T* wk = (T*)malloc(size * sizeof(T));

	for (int i = 0; i < size; i++) {
		wk[i] = NormalRand();
	}

#ifdef __CUDACC__

	_chk(cudaMalloc(&x, size * sizeof(T)));
	_chk(cudaMemcpy(x, wk, size * sizeof(T), cudaMemcpyHostToDevice));
	free(wk);
#else
	x = wk;
#endif
}