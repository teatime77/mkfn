#pragma once

#ifndef UCHAR
#define UCHAR unsigned char
#endif // !UCHAR

#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4996)


UCHAR* ReadBinaryFile(wchar_t* mnist_dir, wchar_t* file_name);
int BytesToInt(UCHAR* v, int offset);
int* RandomSampling(int all_count, int sample_count);

template<class T>
class Network {
public:
	int EpochSize;
	int TrainBatchSize;
	int TestBatchSize;
	int TrainCnt;
	int TestCnt;
	int DomainLen;
	int RangeLen;
	T* TrainX;
	T* TestX;
	UCHAR* TrainLabel;
	UCHAR* TestLabel;

	std::vector<Layer*> Layers;
	Layer* FirstLayer;
	Layer* LastLayer;

	void ReadMNIST() {
		wchar_t mnist_dir[_MAX_PATH];

		_wgetcwd(mnist_dir, _MAX_PATH);

		*wcsrchr(mnist_dir, L'\\') = 0;
		wcscat(mnist_dir, L"\\Lib\\MNIST");
		wprintf(mnist_dir);

		UCHAR	*buf;

		buf = ReadBinaryFile(mnist_dir, L"train-images.idx3-ubyte");

		TrainCnt = BytesToInt(buf, 4);
		int img_h = BytesToInt(buf, 8);
		int img_w = BytesToInt(buf, 12);

		DomainLen = img_h * img_w;
		RangeLen = 10;

		int sz = TrainCnt * DomainLen;
		TrainX = new T[sz];
		for (int i = 0; i < sz; i++) {
			TrainX[i] = buf[16 + i] / 256.0f;
		}

		free(buf);

		buf = ReadBinaryFile(mnist_dir, L"train-labels.idx1-ubyte");
		TrainLabel = new UCHAR[TrainCnt];
		memcpy(TrainLabel, buf + 8, TrainCnt);

		free(buf);

		buf = ReadBinaryFile(mnist_dir, L"t10k-images.idx3-ubyte");
		TestCnt = BytesToInt(buf, 4);
		int test_sz = TestCnt * img_h * img_w;
		TestX = new T[test_sz];
		for (int i = 0; i < test_sz; i++) {
			TestX[i] = buf[16 + i] / 256.0f;
		}

		free(buf);

		buf = ReadBinaryFile(mnist_dir, L"t10k-labels.idx1-ubyte");
		TestLabel = new UCHAR[TestCnt];
		memcpy(TestLabel, buf + 8, TestCnt);

		free(buf);
	}

	void SetBatchData(T* X, T* batch_X, T* batch_Y, UCHAR* label, int batch_size, int* idxes, int mini_batch_idx) {
		memset(batch_Y, 0, RangeLen * batch_size * sizeof(T));

		for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
			int idx = mini_batch_idx * batch_size + batch_idx;

			if (idxes != NULL) {

				idx = idxes[idx];
			}

			for (int ix = 0; ix < DomainLen; ix++) {
				batch_X[ix * batch_size + batch_idx] = X[idx * DomainLen + ix];
			}

			batch_Y[label[idx] * batch_size + batch_idx] = 1;
		}
	}

	void UpdateMiniBatch(void* batch_X, void* batch_Y) {
		FirstLayer->SetX(batch_X);

		for (size_t i = 0; i < Layers.size(); i++) {
			Layers[i]->Forward();
		}

		LastLayer->SetY(batch_Y);
		for (size_t i = Layers.size() - 1; 1 <= i; i--) {
			Layers[i]->Backward();
		}

		for (size_t i = Layers.size() - 1; 1 <= i; i--) {
			Layers[i]->UpdateParameter();
		}
	}

	int ArgMax(T* result_Y, int batch_size, int mini_batch_idx, UCHAR* arg_max, UCHAR* label) {
		int eq_cnt = 0;

		for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
			int idx = mini_batch_idx * batch_size + batch_idx;

			T max_val = 0;
			int max_idx = 0;
			for (int i = 0; i < RangeLen; i++) {
				int idx = batch_idx * RangeLen + i;
				T val = result_Y[idx];
				if (max_val < val) {

					max_val = val;
					max_idx = i;
				}
			}

			arg_max[batch_idx] = max_idx;
			if (max_idx == label[idx]) {
				eq_cnt++;
			}
		}

		return eq_cnt;
	}

	int Evaluate(T* batch_X, T* batch_Y, int batch_size, int mini_batch_idx, UCHAR* arg_max, UCHAR* label) {
		FirstLayer->SetX(batch_X);

		for (size_t i = 0; i < Layers.size(); i++) {
			Layers[i]->Forward();
		}

		T* result_Y = (T*)LastLayer->GetY();

		int eq_cnt = ArgMax(result_Y, batch_size, mini_batch_idx, arg_max, label);

		return eq_cnt;
	}

	void AllocateLayers(int batch_size) {
		for (size_t i = 0; i < Layers.size(); i++) {
			Layers[i]->BatchSize = batch_size;
			Layers[i]->Allocate();
		}
	}

	void FreeLayers() {
		for (size_t i = 0; i < Layers.size(); i++) {
			Layers[i]->Free();
		}
	}

	void SGD() {

		int train_batch_cnt = TrainCnt / TrainBatchSize;
		int test_batch_cnt = TestCnt / TestBatchSize;

		T* train_batch_X = new T[TrainBatchSize * DomainLen];
		T* train_batch_Y = new T[TrainBatchSize * RangeLen];

		T* test_batch_X = new T[TestBatchSize * DomainLen];
		T* test_batch_Y = new T[TestBatchSize * RangeLen];

		UCHAR* test_arg_max = new UCHAR[TestBatchSize];

		for (int epoch_idx = 0; epoch_idx < EpochSize; epoch_idx++) {

			int* idxes = RandomSampling(TrainCnt, TrainCnt);

			AllocateLayers(TrainBatchSize);

			for (int mini_batch_idx = 0; mini_batch_idx < train_batch_cnt; mini_batch_idx++) {

				SetBatchData(TrainX, train_batch_X, train_batch_Y, TrainLabel, TrainBatchSize, idxes, mini_batch_idx);

				UpdateMiniBatch(train_batch_X, train_batch_Y);
			}

			FreeLayers();
			AllocateLayers(TestBatchSize);

			for (int mini_batch_idx = 0; mini_batch_idx < test_batch_cnt; mini_batch_idx++) {

				SetBatchData(TestX, test_batch_X, test_batch_Y, TestLabel, TestBatchSize, NULL, mini_batch_idx);

				int eq_cnt = Evaluate(test_batch_X, test_batch_Y, TestBatchSize, mini_batch_idx, test_arg_max, TestLabel);
			}

			FreeLayers();
		}
	}

	void DeepLearning() {
		FirstLayer = Layers[0];
		LastLayer = Layers[Layers.size() - 1];

		ReadMNIST();
		SGD();
	}
};
