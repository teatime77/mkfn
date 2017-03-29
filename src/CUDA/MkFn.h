#pragma once

#define _chk(ans) { gpuAssert((ans), #ans, __FILE__, __LINE__); }

#define _Memcpy(dst,src, size)	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)
#define _Malloc(x, size)		cudaMalloc(&x, size)
#define _Free(x)				cudaFree(x)
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

__device__ inline double MaxPoolPrime(double) {
	return 0;
}


//__device__ extern double sigmoid(double);
//__device__ extern double sigmoid_prime(double);
//__device__ extern double MaxPoolPrime(double);
