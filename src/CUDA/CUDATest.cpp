// CUDATest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <vector>
#include <typeinfo.h> 

#include "../Lib/Lib.h"
#include "MkFn.h"

//#include "../Layer/LayerCUDA/FullyConnectedLayerCudaF.h"
//#include "../Layer/LayerCUDA/ConvolutionalLayerCudaF.h"
//#include "../Layer/LayerCUDA/MaxPoolingLayerCudaF.h"
//#include "../Layer/LayerCUDA/RecurrentLayerCudaF.h"
//#include "../Layer/LayerCUDA/LSTMLayerCudaF.h"
//
//#include "../Layer/LayerCUDA/FullyConnectedLayerCudaD.h"
//#include "../Layer/LayerCUDA/ConvolutionalLayerCudaD.h"
//#include "../Layer/LayerCUDA/MaxPoolingLayerCudaD.h"
//#include "../Layer/LayerCUDA/RecurrentLayerCudaD.h"
//#include "../Layer/LayerCUDA/LSTMLayerCudaD.h"

extern "C" DllExport Layer* MakeFullyConnectedLayerCudaF(int x_size, int y_size);
extern "C" DllExport Layer* MakeConvolutionalLayerCudaF(int m_size, int n_size, int k_size, int h_size);
extern "C" DllExport Layer* MakeMaxPoolingLayerCudaF(int m_size, int n_size, int k_size, int h_size);
extern "C" DllExport Layer* MakeRecurrentLayerCudaF(int t_size, int x_size, int y_size);
extern "C" DllExport Layer* MakeLSTMLayerCudaF(int t_size, int x_size, int y_size);

extern "C" DllExport Layer* MakeFullyConnectedLayerCudaD(int x_size, int y_size);
extern "C" DllExport Layer* MakeConvolutionalLayerCudaD(int m_size, int n_size, int k_size, int h_size);
extern "C" DllExport Layer* MakeMaxPoolingLayerCudaD(int m_size, int n_size, int k_size, int h_size);
extern "C" DllExport Layer* MakeRecurrentLayerCudaD(int t_size, int x_size, int y_size);
extern "C" DllExport Layer* MakeLSTMLayerCudaD(int t_size, int x_size, int y_size);

#define MakeFullyConnectedLayer	MakeFullyConnectedLayerCudaF
#define MakeConvolutionalLayer	MakeConvolutionalLayerCudaF
#define MakeMaxPoolingLayer		MakeMaxPoolingLayerCudaF
#define MakeRecurrentLayer		MakeRecurrentLayerCudaF
#define MakeLSTMLayer			MakeLSTMLayerCudaF


#include "../Lib/Network.h"


int main()
{
	NetworkTest();
	return 0;
}

