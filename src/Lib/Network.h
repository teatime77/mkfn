#pragma once

#include <assert.h>

#ifndef UCHAR
#define UCHAR unsigned char
#endif // !UCHAR

#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4996)

extern "C" DllExport void DeviceSynchronize();
extern "C" DllExport void DeviceInit();
extern "C" DllExport void* DeviceMalloc(size_t size);
extern "C" DllExport void DeviceFree(void* p);

UCHAR* ReadBinaryFile(wchar_t* mnist_dir, wchar_t* file_name);
int BytesToInt(UCHAR* v, int offset);

extern FILE* fpLog;

enum NetworkType {
	Simple,
	CNN,
	RNN,
	LSTM,
};

enum LossFunction {
	SquareError,
	SoftMax,
};

template<class T>
class Network {
public:
	NetworkType Type;
	int EpochSize;
	int TrainBatchSize;
	int TestBatchSize;
	int TrainCnt;
	int TestCnt;
	int DomainLen;
	int RangeLen;
	T* TrainX;
	T* TrainY;
	T* TestX;
	T* TestY;
	UCHAR* TrainLabel;
	UCHAR* TestLabel;
	int EpochIdx;
	int MiniBatchIdx;
	T CostSum;
	int CostCount;
	int UpdateMiniBatchCount;

	std::vector<Layer*> Layers;
	Layer* FirstLayer;
	Layer* LastLayer;

	void ReadMNIST() {
		wchar_t mnist_dir[_MAX_PATH];

		swprintf(mnist_dir, _MAX_PATH, L"%ls\\MNIST", DataDir);

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

	void CharToOneHotX(wchar_t* char_tbl, T* batch_X, int Time, int one_hot_size, int batch_size, wchar_t* text) {
		for (int t = 0; t < Time; t++) {
			for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
				wchar_t ch1 = text[batch_idx * (Time + 1 + 1) + t];
				int  idx1 = char_tbl[ch1];
				if (idx1 < one_hot_size) {
					batch_X[(t * one_hot_size + idx1) * batch_size + batch_idx] = 1;
				}
			}
		}
	}

	void CharToOneHotY(wchar_t* char_tbl, T* batch_Y, int Time, int one_hot_size, int batch_size, wchar_t* text, int t) {
		for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
			wchar_t ch2 = text[batch_idx * (Time + 1 + 1) + t + 1];
			int  idx2 = char_tbl[ch2];
			if (idx2 < one_hot_size) {
				batch_Y[idx2 * batch_size + batch_idx] = 1;
			}
		}
	}

	void SetBatchData(T* X, T* batch_X, T* batch_Y, UCHAR* label, int batch_size, int* idxes) {
		memset(batch_Y, 0, RangeLen * batch_size * sizeof(T));

		for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
			int idx = MiniBatchIdx * batch_size + batch_idx;

			if (idxes != NULL) {

				idx = idxes[idx];
			}

			for (int ix = 0; ix < DomainLen; ix++) {
				batch_X[ix * batch_size + batch_idx] = X[idx * DomainLen + ix];
			}

			batch_Y[label[idx] * batch_size + batch_idx] = 1;
		}
	}

	/*
		損失関数の微分	
	*/
	void CostDerivative(T* cost_derivative, T* last_y, T* batch_Y, int size) {
		for (int i = 0; i < size; i++) {
			cost_derivative[i] = last_y[i] - batch_Y[i];
		}
	}

	/*
	損失関数
	*/
	T Cost(T* cost_derivative, int size) {
		double sum = 0;
		for (int i = 0; i < size; i++) {
			T cd = cost_derivative[i];
			sum += cd * cd;
		}

		return (T)(sum / size);
	}


	/*
	損失関数の微分
	*/
	T SoftMax(T* cost_derivative, T* last_y, T* batch_Y, T* exp_work, int range_len, int batch_size, int batch_idx) {
		T max_val = -10000;
		for (int i = 0; i < range_len; i++) {
			int k = i * batch_size + batch_idx;

			if (max_val < last_y[k]) {
				max_val = last_y[k];
			}
		}

		T sum = 0;
		for (int i = 0; i < range_len; i++) {
			int k = i * batch_size + batch_idx;

			T  d = exp(last_y[k] - max_val);
			sum += d;
			exp_work[k] = d;
		}

		T cost_sum = 0;
		for (int i = 0; i < range_len; i++) {
			int k = i * batch_size + batch_idx;

			T y = exp_work[k] / sum;
			cost_derivative[k] = y - batch_Y[k];

			cost_sum += batch_Y[k] * log(y);
		}

		return - cost_sum;
	}


	/*
		C++版とCUDA版の計算結果の違いをダンプファイルに書いて比較します。	
	void Dmp(wchar_t* msg, T* p, int sz) {
		if (0 <= sz) return;
		wchar_t base_dir[_MAX_PATH];
		wchar_t dmp_path[_MAX_PATH];

		_wgetcwd(base_dir, _MAX_PATH);
		*wcsrchr(base_dir, L'\\') = 0;

		swprintf(dmp_path, L"%ls\\data\\%ls", base_dir, msg);

		FILE *fp;
#ifdef __CUDACC__
		T* c = new T[sz];
		T* f = new T[sz];

		_chk(cudaDeviceSynchronize());

		_chk(cudaMemcpy(c, p, sz * sizeof(T), cudaMemcpyDeviceToHost));
		_chk(cudaDeviceSynchronize());

		fp = _wfopen(dmp_path, L"rb");

		fread(f, sizeof(T), sz, fp);
		T sum = 0;
		for (int i = 0; i < sz; i++) {
//			if (c[i] != f[i]) {
			T diff = fabs(c[i] - f[i]);
			sum += diff;
			if(1.0e-14 < diff){
//				Log(L"%ls %d : %.16f != %.16f %g", msg, i, c[i], f[i], diff);
			}
		}
		Log(L"%ls %g", msg, sum / sz);
		delete[] c;
		delete[] f;
#else

		fp = _wfopen(dmp_path, L"wb");
		fwrite(p, sizeof(T), sz, fp);
#endif

		fclose(fp);
	}
	*/

	/*
		ミニバッチごとにパラメータを更新します。
	*/
	void UpdateMiniBatch(T* batch_X, T* batch_Y, T* last_y, T* cost_derivative) {
		//-------------------------------------------------- 入力をセットします。

		FirstLayer->SetInputData(batch_X, DomainLen * TrainBatchSize * sizeof(T));

		for (size_t i = 0; i < Layers.size(); i++) {
			Layers[i]->Forward();
		}

		//-------------------------------------------------- 出力を得ます。
		int last_y_len = TrainBatchSize * RangeLen;
		LastLayer->GetOutputData(last_y, last_y_len * sizeof(T));

		//-------------------------------------------------- 損失関数を計算します。
		CostDerivative(cost_derivative, last_y, batch_Y, last_y_len);

		T cost = Cost(cost_derivative, last_y_len);

		//-------------------------------------------------- δyをセットします。
		LastLayer->SetOutputDeltaData(cost_derivative, last_y_len * sizeof(T));

		for (int i = (int)Layers.size() - 1; 0 <= i; i--) {
			Layers[i]->Backward();
		}
		DeviceSynchronize();

		for (int i = (int)Layers.size() - 1; 0 <= i; i--) {
			Layers[i]->UpdateParameter();
		}

/*
		Dmp(L"y0", (T*)Layers[0]->GetOutput(), TrainBatchSize * Layers[0]->GetOutputCount());
		ConvolutionalLayer* cl = (ConvolutionalLayer*)FirstLayer;
		Dmp(L"h", cl->h, cl->H * cl->H * cl->K);
		FullyConnectedLayerF* fc = (FullyConnectedLayerF*)Layers[3];
		Dmp(L"fc3-w", fc->w, fc->Y * fc->X);
*/

		DeviceSynchronize();
	}

	/*
	ミニバッチごとにパラメータを更新します。
	*/
	void RNNUpdateMiniBatch(T* batch_X, T* batch_Y, T* last_y, T* cost_derivative, T* exp_work, wchar_t* char_tbl, wchar_t* char_tbl_inv, wchar_t* text) {
		wchar_t	input[100];
//		wchar_t	input2[100];
		wchar_t	output[100];

		int Time = FirstLayer->GetTimeCount();

/*
		FullyConnectedLayerCudaF* fc = (FullyConnectedLayerCudaF*)Layers[1];

		//-------------------------------------------------- テストデータを計算します。
		T* px = batch_X;
		for (int t = 0; t < Time; t++) {
			for (int ix = 0; ix < X; ix++) {
				for (int batch_idx = 0; batch_idx < TrainBatchSize; batch_idx++) {
					*px = sin(ix * (M_PI / Time));// +0.1f * UniformRandom();

					px++;
				}
			}
		}

		T* py = batch_Y;
		for (int iy = 0; iy < fc->Y; iy++) {
			for (int batch_idx = 0; batch_idx < TrainBatchSize; batch_idx++) {
				*py = sin(iy * (M_PI / Time));// +0.1f * UniformRandom();

				py++;
			}
		}
*/

		//-------------------------------------------------- 入力をセットします。
		FirstLayer->SetInputData(batch_X, DomainLen * TrainBatchSize * sizeof(T));

		// 順方向の時刻
		for (int t = 0; t < Time; t++) {

			memset(batch_Y, 0, RangeLen  * TrainBatchSize * sizeof(T));
			CharToOneHotY(char_tbl, batch_Y, Time, RangeLen, TrainBatchSize, text, t);

			FirstLayer->t = t;

			//-------------------------------------------------- レイヤーの入出力を結合します。
			for (size_t i = 0; i + 1 < Layers.size(); i++) {
				assert(Layers[i]->GetTimeOutputCount() == Layers[i + 1]->GetInputCount());

				// 次のレイヤーの入力は、現在のレイヤーの出力にします。(順伝播)
				Layers[i + 1]->SetInput(Layers[i]->GetOutput(t));
			}

			//-------------------------------------------------- 順伝播
			for (size_t i = 0; i < Layers.size(); i++) {
				Layers[i]->Forward();
			}

			//T* p = (T*)fc->GetOutput(t);
			//for (int i = 0; i < fc->Y * TrainBatchSize; i++) {
			//	Log(L"%f", p[i]);
			//}

			//-------------------------------------------------- 出力を得ます。
			int last_y_len = TrainBatchSize * RangeLen;
			LastLayer->GetOutputData(last_y, last_y_len * sizeof(T));

#if 0
			CostDerivative(cost_derivative, last_y, batch_Y, last_y_len);

			T cost = Cost(cost_derivative, last_y_len);
#else
			T cost = 0;
			for (int batch_idx = 0; batch_idx < TrainBatchSize; batch_idx++) {
				T cost2 = SoftMax(cost_derivative, last_y, batch_Y, exp_work, RangeLen, TrainBatchSize, batch_idx);
/*
				if (MiniBatchIdx % 100 == 0 && batch_idx == TrainBatchSize - 1) {
					for (int i = 0; i < RangeLen; i++) {
						int k = i * TrainBatchSize + batch_idx;
						T sv = last_y[k];

						T dy = 0.0001;
						last_y[k] += dy;

						T cost3 = SoftMax(cost_derivative, last_y, batch_Y, exp_work, RangeLen, TrainBatchSize, batch_idx);
						Log(L"diff: %.16e  δ:%.16e", cost2, cost3, cost3 - cost2, dy * cost_derivative[k]);

						last_y[k] = sv;
					}
				}
*/


				cost += cost2;
			}
			cost /= TrainBatchSize;

#endif
			//-------------------------------------------------- 損失関数を計算します。
			CostSum += cost;
			CostCount++;


			if (UpdateMiniBatchCount % 141 == 0) {//MiniBatchIdx
				int batch_idx = UpdateMiniBatchCount % TrainBatchSize;
				T max_val = -10000;
				int max_idx = 0;
				for (int i = 0; i < RangeLen; i++) {
					T val = last_y[i * TrainBatchSize + batch_idx];
					if (max_val < val) {

						max_val = val;
						max_idx = i;
					}
				}
				output[t] = char_tbl_inv[max_idx];

				max_val = -10000;
				max_idx = 0;
				for (int i = 0; i < RangeLen; i++) {
					T val = batch_Y[i * TrainBatchSize + batch_idx];
					if (max_val < val) {

						max_val = val;
						max_idx = i;
					}
				}
//				input2[t] = char_tbl_inv[max_idx];


				input[t] = text[batch_idx * (Time + 1 + 1) + t];

				if (t == Time - 1) {
					input[t + 1] = 0;
//					input2[t + 1] = 0;
					output[t + 1] = 0;

					//Log(L"IN2: %ls", input2);
					Log(L"IN : %ls", input);
					Log(L"OUT: %ls", output);

					Log(L"epock : %d  cost : %f", EpochIdx, CostSum / CostCount);
				}
			}

			//-------------------------------------------------- δyをセットします。
			LastLayer->SetOutputDeltaData(cost_derivative, last_y_len * sizeof(T));

			//-------------------------------------------------- 逆伝播
			// RNN以外のレイヤーの逆伝播をします。
			for (int i = (int)Layers.size() - 1; 1 <= i; i--) {
				void* input_delta_save = 0;

				// 現在のレイヤーの出力のデルタは、次のレイヤーの入力のデルタにします。(逆伝播)
				if (Layers[i - 1]->GetTimeCount() != 0) {

					input_delta_save = Layers[i]->GetInputDelta(t);
					Layers[i]->SetIputDelta(Layers[i - 1]->GetOutputDelta(t));
				}
				else {

					Layers[i - 1]->SetOutputDelta(Layers[i]->GetInputDelta(t));
				}

				Layers[i]->Backward();

				if (input_delta_save != 0) {

					Layers[i]->SetIputDelta(input_delta_save);
				}
			}
		}


		// 時刻の逆方向に、RNNのレイヤーの逆伝播をします。
		for (int t = Time - 1; 0 <= t; t--) {
			FirstLayer->t = t;
			FirstLayer->Backward();
		}

		// パラメータを更新します。
		for (int i = (int)Layers.size() - 1; 0 <= i; i--) {
			Layers[i]->UpdateParameter();
		}

		/*
		Dmp(L"y0", (T*)Layers[0]->GetOutput(), TrainBatchSize * Layers[0]->GetOutputCount());
		ConvolutionalLayer* cl = (ConvolutionalLayer*)FirstLayer;
		Dmp(L"h", cl->h, cl->H * cl->H * cl->K);
		FullyConnectedLayerF* fc = (FullyConnectedLayerF*)Layers[3];
		Dmp(L"fc3-w", fc->w, fc->Y * fc->X);
		*/

		DeviceSynchronize();
	}


	/*
		RNN用SGD
	*/
	void RNNSGD() {
		wchar_t	char_tbl[CHAR_COUNT];
		wchar_t	char_tbl_inv[CHAR_COUNT];
		ReadCharTable(char_tbl, char_tbl_inv);

		CostSum = 0;
		CostCount = 0;
		UpdateMiniBatchCount = 0;
		while (true){

			for (EpochIdx = 0; EpochIdx < EpochSize; EpochIdx++) {
				int time_len = EpochIdx + 5;
				int line_len = time_len + 1;

				InitText(TrainBatchSize, line_len, TrainCnt, char_tbl, char_tbl_inv);
				if (TrainCnt == 0) {
					break;
				}

				FirstLayer->SetTimeCount(time_len);
				int X = FirstLayer->GetTimeInputCount();
				int Y = FirstLayer->GetTimeOutputCount();
				int Time = FirstLayer->GetTimeCount();

				DomainLen = FirstLayer->GetInputCount();
				RangeLen = LastLayer->GetOutputCount();

				int train_batch_cnt = TrainCnt / TrainBatchSize;

				T* train_batch_X = new T[DomainLen * TrainBatchSize];
				T* train_batch_Y = new T[RangeLen * TrainBatchSize];
				T* train_last_Y = new T[RangeLen * TrainBatchSize];

				T* cost_derivative = new T[RangeLen * TrainBatchSize];
				T* exp_work = new T[RangeLen * TrainBatchSize];

				// すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
				AllocateConnectLayers(TrainBatchSize);

				size_t delta_y_sz = TrainBatchSize * Time * Y * sizeof(double);
				void* out_delta_y = DeviceMalloc(delta_y_sz);

				FirstLayer->SetOutputDelta(out_delta_y);

				for (MiniBatchIdx = 0; MiniBatchIdx < train_batch_cnt; ) {//MiniBatchIdx++

					wchar_t* text = ReadText(TrainBatchSize, line_len, MiniBatchIdx);

					memset(train_batch_X, 0, DomainLen * TrainBatchSize * sizeof(T));
					CharToOneHotX(char_tbl, train_batch_X, Time, X, TrainBatchSize, text);

					RNNUpdateMiniBatch(train_batch_X, train_batch_Y, train_last_Y, cost_derivative, exp_work, char_tbl, char_tbl_inv, text);
					UpdateMiniBatchCount++;

					if (MiniBatchIdx % 100 == 0) {
						//Log(L"epock : %d   mini batch: %d  cost : %f", EpochIdx, MiniBatchIdx, CostSum / CostCount);
					}
				}

				FreeLayers();
				DeviceFree(out_delta_y);

				//Log(L"epock : %d  cost : %f", EpochIdx, CostSum / CostCount);

				delete[] train_batch_X;
				delete[] train_batch_Y;
				delete[] train_last_Y;
				delete[] cost_derivative;
				delete[] exp_work;
			}
		}
	}


	int ArgMax(T* result_Y, int batch_size, UCHAR* arg_max, UCHAR* label) {
		int eq_cnt = 0;

		for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {

			T max_val = -10000;
			int max_idx = 0;
			for (int i = 0; i < RangeLen; i++) {
				T val = result_Y[ i * batch_size + batch_idx ];
				if (max_val < val) {

					max_val = val;
					max_idx = i;
				}
			}

			arg_max[batch_idx] = max_idx;

			if (max_idx == label[ MiniBatchIdx * batch_size + batch_idx ]) {
				eq_cnt++;
			}
		}

		return eq_cnt;
	}

	int Evaluate(T* batch_X, T* batch_Y, T* last_y, int batch_size, UCHAR* arg_max, UCHAR* label) {
		FirstLayer->SetInputData(batch_X, batch_size * DomainLen * sizeof(T));

		for (size_t i = 0; i < Layers.size(); i++) {
			Layers[i]->Forward();
		}

		LastLayer->GetOutputData(last_y, batch_size * RangeLen * sizeof(T));

		int eq_cnt = ArgMax(last_y, batch_size, arg_max, label);

		return eq_cnt;
	}

	/*
	すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
	*/
	void AllocateConnectLayers(int batch_size) {
		void* p;

		p = DeviceMalloc(batch_size * DomainLen * sizeof(T));
		FirstLayer->SetInput(p);

		p = DeviceMalloc(batch_size * RangeLen * sizeof(T));
		LastLayer->SetOutputDelta(p);

		for (size_t i = 0; i < Layers.size(); i++) {
			Layers[i]->BatchSize = batch_size;
			Layers[i]->Allocate();
		}

		// レイヤーの入出力を結合します。
		for (size_t i = 0; i + 1 < Layers.size(); i++) {

			if (Layers[i]->GetTimeCount() == 0) {
				assert(Layers[i]->GetOutputCount() == Layers[i + 1]->GetInputCount());

				// 次のレイヤーの入力は、現在のレイヤーの出力にします。(順伝播)
				Layers[i + 1]->SetInput(Layers[i]->GetOutput());

				// 現在のレイヤーの出力のデルタは、次のレイヤーの入力のデルタにします。(逆伝播)
				Layers[i]->SetOutputDelta(Layers[i + 1]->GetInputDelta());
			}

			Layers[i]->ConnectLayer(Layers[i + 1]);
		}
	}

	/*
	すべてのレイヤーのメモリを解放します。
	*/
	void FreeLayers() {
		for (size_t i = 0; i < Layers.size(); i++) {
			Layers[i]->Free();
		}

		DeviceFree(FirstLayer->GetInput());
		DeviceFree(LastLayer->GetOutputDelta());
	}

	/*
	確率的勾配降下法 (stochastic gradient descent, SGD)
	*/
	void SGD() {

		int train_batch_cnt = TrainCnt / TrainBatchSize;
		int test_batch_cnt = TestCnt / TestBatchSize;

		T* train_batch_X   = new T[TrainBatchSize * DomainLen];
		T* train_batch_Y   = new T[TrainBatchSize * RangeLen];
		T* train_last_Y	   = new T[TrainBatchSize * RangeLen];

		T* cost_derivative = new T[TrainBatchSize * RangeLen];

		T* test_batch_X = new T[TestBatchSize * DomainLen];
		T* test_batch_Y = new T[TestBatchSize * RangeLen];
		T* test_last_Y  = new T[TestBatchSize * RangeLen];

		UCHAR* test_arg_max = new UCHAR[TestBatchSize];

		for (EpochIdx = 0; EpochIdx < EpochSize; EpochIdx++) {

			int* idxes = RandomSampling(TrainCnt, TrainCnt);

			// すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
			AllocateConnectLayers(TrainBatchSize);

			for (MiniBatchIdx = 0; MiniBatchIdx < train_batch_cnt; MiniBatchIdx++) {

				SetBatchData(TrainX, train_batch_X, train_batch_Y, TrainLabel, TrainBatchSize, idxes);

				UpdateMiniBatch(train_batch_X, train_batch_Y, train_last_Y, cost_derivative);
			}

			FreeLayers();

			// すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
			AllocateConnectLayers(TestBatchSize);

			int eq_cnt_sum = 0;
			for (MiniBatchIdx = 0; MiniBatchIdx < test_batch_cnt; MiniBatchIdx++) {

				SetBatchData(TestX, test_batch_X, test_batch_Y, TestLabel, TestBatchSize, NULL);

				int eq_cnt = Evaluate(test_batch_X, test_batch_Y, test_last_Y, TestBatchSize, test_arg_max, TestLabel);
				eq_cnt_sum += eq_cnt;
			}
			Log(L"epoch %d : %d / %d", EpochIdx, eq_cnt_sum, TestCnt);

			FreeLayers();

			delete[] idxes;
		}

		delete[] train_batch_X;
		delete[] train_batch_Y;
		delete[] train_last_Y;
		delete[] cost_derivative;
		delete[] test_batch_X;
		delete[] test_batch_Y;
		delete[] test_last_Y;
		delete[] test_arg_max;
	}

	void DeepLearning() {
		FirstLayer = Layers[0];
		LastLayer = Layers[Layers.size() - 1];

		switch (Type){
		case NetworkType::Simple:
		case NetworkType::CNN:
			SGD();
			break;

		case NetworkType::RNN:
		case NetworkType::LSTM:
			RNNSGD();
			break;
		}

		for (size_t i = 0; i < Layers.size(); i++) {
			delete Layers[i];
		}
		Layers.clear();
	}
};


void NetworkTest() {
	// 初期処理をします。
	Init();

	DeviceInit();

	Network<float> *net = new Network<float>();
	net->EpochSize = 100;
	net->TestBatchSize = 20;

	float learning_rate = 1.0f;
	for (int run_idx = 0; ; run_idx++) {
		net->Type = NetworkType::Simple;
		net->Type = NetworkType::RNN;
		net->Type = NetworkType::CNN;
		net->Type = NetworkType::LSTM;
		switch (net->Type) {
		case NetworkType::Simple:
			net->TrainBatchSize = 10;
			net->ReadMNIST();
			net->Layers = std::vector<Layer*>{
				MakeFullyConnectedLayer(28 * 28, 30),
				MakeFullyConnectedLayer(30, 10)
			};
			break;

		case NetworkType::CNN:
			net->TrainBatchSize = 10;
			net->ReadMNIST();
			net->Layers = std::vector<Layer*>{
				//new ConvolutionalLayer(28, 28, 20, 5),
				//new MaxPoolingLayer(24, 24, 20, 2),
				//new FullyConnectedLayerF(12 * 12 * 20, 100),
				MakeConvolutionalLayer(28, 28, 5, 5),
				MakeMaxPoolingLayer(24, 24, 5, 2),
				MakeFullyConnectedLayer(12 * 12 * 5, 100),
				MakeFullyConnectedLayer(100, 10)
			};
			break;

		case NetworkType::RNN:
			learning_rate = 0.1f;
			net->TrainBatchSize = 7;
			net->Layers = std::vector<Layer*>{
				//new RecurrentLayer(5, 2, 10),
				//new FullyConnectedLayerF(10, 2)
				MakeRecurrentLayer(20, 28, 100),
				MakeFullyConnectedLayer(10, 28)
			};
			break;

		case NetworkType::LSTM:
			learning_rate = 0.1f;
			net->TrainBatchSize = 7;
			net->Layers = std::vector<Layer*>{
				//new LSTMLayer(50, 2000, 100),
				//new LSTMLayer(20, 1000, 100),
				MakeLSTMLayer(20, 28, 100),
				MakeFullyConnectedLayer(100, 28)
			};
			break;
		}

		for (size_t i = 0; i < net->Layers.size(); i++) {
			net->Layers[i]->LearningRate = learning_rate / net->TrainBatchSize;//3.0f
		}

		net->DeepLearning();
	}
}
