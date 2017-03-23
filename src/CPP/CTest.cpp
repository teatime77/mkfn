// CTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include <vector>
#include "../Lib/Lib.h"
#include "MkFn.h"
#include "../Lib/Network.h"
#include "FullyConnectedLayer.h"

void SimpleRandDemo(int n);

int main(){
	// ログファイルを初期化します。
	InitLog();

	Network<double> *net = new Network<double>();
	net->EpochSize = 100;
	net->TrainBatchSize = 10;
	net->TestBatchSize = 20;
	net->Layers = std::vector<Layer*>{
		new FullyConnectedLayer(28 * 28, 30),
		new FullyConnectedLayer(30, 10)
	};

	for (size_t i = 0; i < net->Layers.size(); i++) {
		net->Layers[i]->LearningRate = 3.0f / net->TrainBatchSize;
	}

	net->DeepLearning();

	return 0;
}

