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

//#include "../Layer/Layer/FullyConnectedLayerF.h"
//#include "../Layer/Layer/ConvolutionalLayerF.h"
//#include "../Layer/Layer/MaxPoolingLayerF.h"
//#include "../Layer/Layer/RecurrentLayerF.h"
//#include "../Layer/Layer/LSTMLayerF.h"
//
//#include "../Layer/Layer/FullyConnectedLayerD.h"
//#include "../Layer/Layer/ConvolutionalLayerD.h"
//#include "../Layer/Layer/MaxPoolingLayerD.h"
//#include "../Layer/Layer/RecurrentLayerD.h"
//#include "../Layer/Layer/LSTMLayerD.h"

extern "C" DllExport Layer* MakeFullyConnectedLayerF(int x_size, int y_size);
extern "C" DllExport Layer* MakeConvolutionalLayerF(int m_size, int n_size, int k_size, int h_size);
extern "C" DllExport Layer* MakeMaxPoolingLayerF(int m_size, int n_size, int k_size, int h_size);
extern "C" DllExport Layer* MakeRecurrentLayerF(int t_size, int x_size, int y_size);
extern "C" DllExport Layer* MakeLSTMLayerF(int t_size, int x_size, int y_size);

extern "C" DllExport Layer* MakeFullyConnectedLayerD(int x_size, int y_size);
extern "C" DllExport Layer* MakeConvolutionalLayerD(int m_size, int n_size, int k_size, int h_size);
extern "C" DllExport Layer* MakeMaxPoolingLayerD(int m_size, int n_size, int k_size, int h_size);
extern "C" DllExport Layer* MakeRecurrentLayerD(int t_size, int x_size, int y_size);
extern "C" DllExport Layer* MakeLSTMLayerD(int t_size, int x_size, int y_size);

#define MakeFullyConnectedLayer	MakeFullyConnectedLayerF
#define MakeConvolutionalLayer	MakeConvolutionalLayerF
#define MakeMaxPoolingLayer		MakeMaxPoolingLayerF
#define MakeRecurrentLayer		MakeRecurrentLayerF
#define MakeLSTMLayer			MakeLSTMLayerF

#include "../Lib/Network.h"


int main(){
	NetworkTest();
	return 0;
}

