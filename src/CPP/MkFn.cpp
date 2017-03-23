#include <math.h>

double sigmoid(double z) {
	return 1.0 / (1.0 + exp(-z));
}

double sigmoid_prime(double z) {
	double f = sigmoid(z);
	return f * (1 - f);
}

double MaxPoolPrime(double) {
	return 0;
}
