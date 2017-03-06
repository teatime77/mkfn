// LibTest.cpp : �R���\�[�� �A�v���P�[�V�����̃G���g�� �|�C���g���`���܂��B
//

#include "stdafx.h"
#include <vector>
#include "Lib.h"

int main(){
	Network<double> *net = new Network<double>();
	net->EpochSize = 100;
	net->TrainBatchSize = 10;
	net->Layers = std::vector<Layer*>{
		new Layer(),
		new Layer()
	};

	net->DeepLearning();

    return 0;
}

