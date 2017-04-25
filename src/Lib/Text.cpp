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

#include "../Lib/Lib.h"

#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4996)

void ReadCharTable(wchar_t* char_tbl, wchar_t* char_tbl_inv) {
	wchar_t char_path[_MAX_PATH];
	FILE	*fp;
	wchar_t	buf[CHAR_COUNT];

	swprintf(char_path, _MAX_PATH, L"%ls\\aozorabunko\\char.txt", DataDir);

	fp = _wfopen(char_path, L"rb");
	size_t sz = fread(buf, 2, CHAR_COUNT, fp);
	fclose(fp);

	// ファイルの先頭のユニコード マーカーを除きます。
	sz--;

	// 文字テーブルを-1でクリアします。
	for (int i = 0; i < CHAR_COUNT; i++) {
		char_tbl[i] = INVALID_CHAR;
	}

	memset(char_tbl_inv, 0, CHAR_COUNT * sizeof(wchar_t));

	for (int i = 0; i < sz; i++) {
		char_tbl[buf[1 + i]] = i;
		char_tbl_inv[i] = buf[1 + i];
	}
}


static int LineIdx;
static wchar_t*	CharBuf;

void InitText(int batch_size, int line_len, int& train_cnt, wchar_t* char_tbl, wchar_t* char_tbl_inv) {
	wchar_t text_path[_MAX_PATH];
	int line_NL_len = line_len + 1;

	FILE	*fp;

	swprintf(text_path, _MAX_PATH, L"%ls\\aozorabunko\\Line-%d.txt", DataDir, line_len);


	struct _stat	file_stat;

	if (_wstat(text_path, &file_stat) != 0) {
		// ファイルが存在しない場合

		train_cnt = 0;
		return;
	}

	long char_len = (file_stat.st_size - 2) / 2;

	int line_count = char_len / line_NL_len;

	train_cnt = (line_count / batch_size) * batch_size;

	CharBuf = new wchar_t[file_stat.st_size / sizeof(wchar_t) + 1];
	wchar_t* buf = new wchar_t[file_stat.st_size / sizeof(wchar_t) + 1];

	fp = _wfopen(text_path, L"rb");
	size_t sz = fread(buf, 1, file_stat.st_size, fp);
	fclose(fp);

	assert(sz == file_stat.st_size);

	int* idxes = RandomSampling(train_cnt, train_cnt);
	for (int i = 0; i < train_cnt; i++) {
		memcpy(CharBuf + 1 + i * line_NL_len, buf + 1 + idxes[i] * line_NL_len, line_NL_len * sizeof(wchar_t));
	}
	CharBuf[1 + train_cnt * line_NL_len] = 0;

	delete[] buf;

	delete[] idxes;

	if (char_tbl == 0) {
		return;
	}

	memset(char_tbl    , 0, CHAR_COUNT * sizeof(wchar_t));
	memset(char_tbl_inv, 0, CHAR_COUNT * sizeof(wchar_t));

	for (int i = 1; CharBuf[i]; i++) {
		wchar_t ch = CharBuf[i];
		wchar_t* p = wcschr(char_tbl_inv, ch);
		if (p == 0) {
			size_t k = wcslen(char_tbl_inv);
			char_tbl[ch] = (wchar_t)k;
			char_tbl_inv[k] = ch;
		}
	}
	Log(L"文字数:%d", wcslen(char_tbl_inv));

	// 文字テーブルを-1でクリアします。
	for (int i = 0; i < CHAR_COUNT; i++) {
		if (char_tbl[i] == 0) {
			char_tbl[i] = INVALID_CHAR;
		}
	}
}

wchar_t* ReadText(int batch_size, int line_len, int mini_batch_idx) {
	return CharBuf + 1 + mini_batch_idx * (line_len + 1) * batch_size;
}

void ClearText() {
	delete[] CharBuf;
}
