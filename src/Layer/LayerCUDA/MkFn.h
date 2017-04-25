#pragma once

#define _chk(ans) { gpuAssert((ans), #ans, __FILE__, __LINE__); }

#define _Memcpy(dst,src, size)	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)
#define _Malloc(x, size)		cudaMalloc(&x, size)

template <class T> inline cudaError_t _Free(T* &x) {
	cudaError_t sts = cudaSuccess;

	if (x != 0) {
		sts = cudaFree(x);
		x = 0;
	}
	return sts;
}

#define _MemcpyToSymbol(dst, src, size) cudaMemcpyToSymbol(dst, &src, size)

void LogA(char *szFormat, ...);

inline void gpuAssert(cudaError_t code, char*s, const char *file, int line, bool abort = true){
	if (code != cudaSuccess){
		LogA("GPUassert: %s %s %s %d\n", cudaGetErrorString(code), s, file, line);
		fprintf(stderr, "GPUassert: %s %s %s %d\n", cudaGetErrorString(code), s, file, line);
		if (abort) {
			exit(code);
		}
	}
}

template <class T> inline T max(T x, T y) {
	return x < y ? y : x;
}

template <class T> inline T min(T x, T y) {
	return x < y ? y : x;
}

__device__ inline double sigmoid(double z) {
	return 1.0 / (1.0 + exp(-z));
}

__device__ inline double sigmoid_prime(double z) {
	double f = sigmoid(z);
	return f * (1 - f);
}

template <class T> inline void SetNormalRand(T* &x, int size) {
	T* wk = (T*)malloc(size * sizeof(T));

	for (int i = 0; i < size; i++) {
		wk[i] = NormalRand();
	}

	_chk(cudaMalloc(&x, size * sizeof(T)));
	_chk(cudaMemcpy(x, wk, size * sizeof(T), cudaMemcpyHostToDevice));
	free(wk);
}
