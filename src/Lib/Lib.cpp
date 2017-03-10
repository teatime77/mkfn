#include "stdafx.h"
#include "direct.h"
#include "stdio.h"
#include "stdlib.h"
#include <random>
#include <vector>
#include <assert.h> 
#include "Lib.h"
#include "Network.h"

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

int* RandomSampling(int all_count, int sample_count) {
	int* ret = new int[sample_count];

	int* numbers = new int[all_count];
	for (int i = 0; i < all_count; i++) {
		numbers[i] = i;
	}
	
	std::mt19937 mt;
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
