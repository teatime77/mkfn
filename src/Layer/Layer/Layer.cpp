// Layer.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <FLOAT.h>
#include "MkFn.h"
#include "../../Lib/Lib.h"

extern "C" DllExport void DeviceSynchronize() {
}

extern "C" DllExport void DeviceInit() {
}

extern "C" DllExport void* DeviceMalloc(size_t size) {
	return malloc(size);
}


extern "C" DllExport void DeviceFree(void* p) {
	free(p);
}
