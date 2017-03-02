#pragma once

template <class T> inline T max(T x, T y) {
	return x < y ? y : x;
}

template <class T> inline T min(T x, T y) {
	return x < y ? y : x;
}

__device__ inline double sigmoid(double) {
	return 0;
}

__device__ inline double sigmoid_prime(double) {
	return 0;
}

__device__ inline double MaxPoolPrime(double) {
	return 0;
}


//__device__ extern double sigmoid(double);
//__device__ extern double sigmoid_prime(double);
//__device__ extern double MaxPoolPrime(double);
