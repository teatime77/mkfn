#include "stdafx.h"
#include "direct.h"
#include "stdio.h"
#include "stdlib.h"
#include <stdarg.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <random>
#include <vector>
#include <assert.h> 
#include "windows.h"

FILE* fpLog;

UCHAR* ReadBinaryFile(wchar_t* mnist_dir, wchar_t* file_name) {
	FILE* fp;
	wchar_t path[_MAX_PATH];

	swprintf(path, _MAX_PATH, L"%ls\\%ls", mnist_dir, file_name);
	fp = _wfopen(path, L"rb");
	if (fp != NULL) {

		fpos_t fsize;
		fseek(fp, 0, SEEK_END);
		fgetpos(fp, &fsize);

		UCHAR * buf = (UCHAR*)malloc(fsize);

		fseek(fp, 0, SEEK_SET);
		size_t sz = fread(buf, 1, fsize, fp);

		fclose(fp);

		if (sz == fsize) {

			return buf;
		}
	}

	return NULL;
}

int BytesToInt(UCHAR* v, int offset) {
	return v[offset] * 0x1000000 + v[offset + 1] * 0x10000 + v[offset + 2] * 0x100 + v[offset + 3];
}


// 非決定的な乱数生成器を生成します。
std::random_device rnd;     

//  メルセンヌ・ツイスタの32ビット版、引数は初期シード値
std::mt19937 mt(rnd());     

#define MT_RNG		(1000 * 1000)

// [1, MT_RNG] 範囲の一様乱数
std::uniform_int_distribution<> rand10000(1, MT_RNG);        

/*
  [0, all_count-1]の範囲から、sample_count個の整数を抜き出して、その配列を返します。
*/
int* RandomSampling(int all_count, int sample_count) {
	int* ret = new int[sample_count];

	int* numbers = new int[all_count];
	for (int i = 0; i < all_count; i++) {
		numbers[i] = i;
	}
	
	for (int i = 0; i < sample_count; i++) {
		int n = mt() % (all_count - i);

		ret[i] = numbers[n];
		numbers[n] = numbers[all_count - i - 1];
	}
	
	//for (int i = 0; i < sample_count; i++) {
	//	for (int j = i + 1; j < sample_count; j++) {
	//		assert(ret[i] != ret[j]);
	//	}
	//}

	return ret;
}

/*
	(0, 1]の範囲の一様乱数を返します。
*/
double UniformRandom() {
	return rand10000(mt) / (double)MT_RNG;
}

// 正規乱数
class NormalRandom {
	bool Flag = false;
	double C;
	double Theta;

public:
	float Next() {
		Flag = !Flag;
		if (Flag) {
			double d1 = UniformRandom();
			assert(0 < d1);

			double d2 = -2 * log(d1);
			assert(!isinf(d2));

			C = sqrt(d2);
			Theta = UniformRandom() * M_PI * 2;

			return (float)(C * sin(Theta));
		}
		else {
			return (float)(C * cos(Theta));
		}
	}
};

/*
	正規乱数を返します。
*/
float NormalRand() {
	static NormalRandom* Rn;

	if (Rn == NULL) {
		//srand((unsigned)time(NULL));
		srand(0);

		Rn = new NormalRandom();
	}

	return Rn->Next();
}

/*
	正規乱数のテスト
*/
static void NormalRandomTest() {
	int n = 200;

	int* v = (int*)calloc(n, sizeof(int));
	for (int i = 0; i < 100 * 1000 * 1000; i++) {
		float r = NormalRand();
		int k = (int)(r * 25 + n / 2);
		if (k < 0) {
			v[0]++;
		}
		else if (n <= k) {
			v[n - 1]++;
		}
		else {
			v[k]++;
		}
	}

	for (int k = 0; k < n; k++) {
		fprintf(fpLog, "%d\r\n", v[k]);
	}

	free(v);
	fflush(fpLog);
}

/*
	乱数のテスト
*/
void SimpleRandDemo(int n){
	// 正規乱数のテスト
	NormalRandomTest();

	// 一様乱数のテスト
	int i;
	double min1 = 100, max1 = -100, sum1 = 0;
	for (i = 0; i < n; i++) {
		double f = UniformRandom();// rand() / (double)RAND_MAX;

		sum1 += f;
		min1 = __min(min1, f);
		max1 = __max(max1, f);
	}

	printf("min:%f max:%f avg:%f\n", min1, max1, sum1 / n);
}


/*
	ログファイルを初期化します。
*/
void InitLog() {
	wchar_t log_path[_MAX_PATH];

	_wgetcwd(log_path, _MAX_PATH);
	wcscat(log_path, L"\\LOG.txt");

	fpLog = _wfopen(log_path, L"wb");
}


/*
	デバッグの出力をします。
*/
void Log(wchar_t *szFormat, ...) {
#define NUMCHARS	1024
	wchar_t buf[NUMCHARS];  // Large buffer for long filenames or URLs
	const int LASTCHAR = NUMCHARS - 1;

	// Format the input string
	va_list pArgs;
	va_start(pArgs, szFormat);

	// Use a bounded buffer size to prevent buffer overruns.  Limit count to
	// character size minus one to allow for a NULL terminating character.
	_vsnwprintf_s(buf, NUMCHARS - 1, szFormat, pArgs);
	va_end(pArgs);

	// Ensure that the formatted string is NULL-terminated
	buf[LASTCHAR] = L'\0';
	wcscat_s(buf, L"\r\n");

	OutputDebugStringW(buf);
}
