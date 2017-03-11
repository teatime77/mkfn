// CTest.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include <vector>
#include "../Lib/Lib.h"
#include "../Lib/Network.h"


int main(){
	Network<double> *net = new Network<double>();
	net->EpochSize = 100;
	net->TrainBatchSize = 10;
	net->TestBatchSize = 20;
	net->Layers = std::vector<Layer*>{
		new Layer(),
		new Layer()
	};

	net->DeepLearning();

	return 0;
}

