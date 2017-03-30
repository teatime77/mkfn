#pragma once

#define _chk(ans) ans

#define _Memcpy(dst,src, size) memcpy(dst, src, size)
#define _Malloc(x, size)		MyMalloc((void**)&x, size)
#define _Free(x)				if(x != 0){ free(x); x = 0; }
#define _MemcpyToSymbol(dst, src, size) memcpy(&dst, &src, size)

#define __constant__ static
#define __global__

inline void MyMalloc(void** x, size_t size) {
	*x = malloc(size);
}

template <class T> inline T max(T x, T y) {
	return x < y ? y : x;
}

template <class T> inline T min(T x, T y) {
	return x < y ? y : x;
}

double sigmoid(double);
double sigmoid_prime(double);
