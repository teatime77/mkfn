// CTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <vector>
#include <typeinfo.h> 
#include "../Lib/Lib.h"
#include "MkFn.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "MaxPoolingLayer.h"
#include "RecurrentLayer.h"
#include "LSTMLayer.h"
#include "../Lib/Network.h"

void SimpleRandDemo(int n);

int main(){
	NetworkTest();

	return 0;
}

