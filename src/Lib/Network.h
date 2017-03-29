#pragma once

#ifndef UCHAR
#define UCHAR unsigned char
#endif // !UCHAR

#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4996)


UCHAR* ReadBinaryFile(wchar_t* mnist_dir, wchar_t* file_name);
int BytesToInt(UCHAR* v, int offset);
int* RandomSampling(int all_count, int sample_count);

extern FILE* fpLog;

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
	int EpochIdx;
	int MiniBatchIdx;

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

		return (T)(sum / 2);
	}

	/*
		C++版とCUDA版の計算結果の違いをダンプファイルに書いて比較します。	
	*/
	void Dmp(wchar_t* msg, T* p, int sz) {
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
		for (int i = 0; i < sz; i++) {
			if (c[i] != f[i]) {
				Log(L"%ls %d : %.16f != %.16f %g", msg, i, c[i], f[i], max(abs(c[i]), abs(f[i])) / (c[i] - f[i]));
			}
		}
		delete[] c;
		delete[] f;
#else

		fp = _wfopen(dmp_path, L"wb");
		fwrite(p, sizeof(T), sz, fp);
#endif

		fclose(fp);
	}

	/*
		ミニバッチごとにパラメータを更新します。
	*/
	void UpdateMiniBatch(T* batch_X, T* batch_Y, T* last_y, T* cost_derivative) {
		for (int i = 0; i < TrainBatchSize * DomainLen; i++) {
			//fprintf(fpLog, "x:%f\r\n", batch_X[i]);
		}

#ifdef __CUDACC__
		_chk(cudaMemcpy(FirstLayer->GetInput(), batch_X, TrainBatchSize * DomainLen * sizeof(T), cudaMemcpyHostToDevice));
		_chk(cudaDeviceSynchronize());
#else
		FirstLayer->SetInput(batch_X);
#endif

		for (size_t i = 0; i < Layers.size(); i++) {
			Layers[i]->Forward();
		}

#ifdef __CUDACC__
		_chk(cudaDeviceSynchronize());
		_chk(cudaMemcpy(last_y, LastLayer->GetOutput(), TrainBatchSize * RangeLen * sizeof(T), cudaMemcpyDeviceToHost));
		T* last_y_ptr = last_y;
#else
		T* last_y_ptr = (T*)LastLayer->GetOutput();
#endif

		int size = TrainBatchSize * RangeLen;

		// 損失関数の微分
		CostDerivative(cost_derivative, last_y_ptr, batch_Y, size);

		T cost = Cost(cost_derivative, size);

/*
		// 損失関数をログに書きます。
		fprintf(fpLog, "%f\r\n", cost);
		Log(L"%d:%d:%.16f", EpochIdx, MiniBatchIdx, cost);

		static int log_idx;
		log_idx++;
		if (log_idx % 1000 == 0) {
			// 計算の途中でログを見れるように、1000回ごとにフラッシュします。

			fflush(fpLog);
		}
*/

#ifdef __CUDACC__
		_chk(cudaMemcpy(LastLayer->GetOutputDelta(), cost_derivative, TrainBatchSize * RangeLen * sizeof(T), cudaMemcpyHostToDevice));
		_chk(cudaDeviceSynchronize());
#else
		LastLayer->SetOutputDelta(cost_derivative);
#endif

		for (int i = (int)Layers.size() - 1; 0 <= i; i--) {
			Layers[i]->Backward();
		}
#ifdef __CUDACC__
		_chk(cudaDeviceSynchronize());
#endif

		for (int i = (int)Layers.size() - 1; 0 <= i; i--) {
			Layers[i]->UpdateParameter();
		}
/*
		FullyConnectedLayer* fl = (FullyConnectedLayer*)FirstLayer;
		Dmp(L"b", fl->b, fl->Y);
		Dmp(L"w", fl->w, fl->Y * fl->X);
*/

#ifdef __CUDACC__
		_chk(cudaDeviceSynchronize());
#endif
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
#ifdef __CUDACC__
		_chk(cudaMemcpy(FirstLayer->GetInput(), batch_X, batch_size * DomainLen * sizeof(T), cudaMemcpyHostToDevice));
		_chk(cudaDeviceSynchronize());
#else
		FirstLayer->SetInput(batch_X);
#endif

		for (size_t i = 0; i < Layers.size(); i++) {
			Layers[i]->Forward();
		}

#ifdef __CUDACC__
		_chk(cudaDeviceSynchronize());
		_chk(cudaMemcpy(last_y, LastLayer->GetOutput(), batch_size * RangeLen * sizeof(T), cudaMemcpyDeviceToHost));
		_chk(cudaDeviceSynchronize());
		T* last_y_ptr = last_y;
#else
		T* last_y_ptr = (T*)LastLayer->GetOutput();
#endif

		int eq_cnt = ArgMax(last_y_ptr, batch_size, arg_max, label);

		return eq_cnt;
	}

	/*
	すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
	*/
	void AllocateConnectLayers(int batch_size) {
#ifdef __CUDACC__
		void* p;
		
		_chk(cudaMalloc(&p, batch_size * DomainLen * sizeof(T)));
		FirstLayer->SetInput(p);

		_chk(cudaMalloc(&p, batch_size * RangeLen * sizeof(T)));
		LastLayer->SetOutputDelta(p);
#endif

		for (size_t i = 0; i < Layers.size(); i++) {
			Layers[i]->BatchSize = batch_size;
			Layers[i]->Allocate();
		}

		// レイヤーの入出力を結合します。
		for (size_t i = 0; i + 1 < Layers.size(); i++) {
			// 次のレイヤーの入力は、現在のレイヤーの出力にします。(順伝播)
			Layers[i + 1]->SetInput(Layers[i]->GetOutput());

			// 現在のレイヤーの出力のデルタは、次のレイヤーの入力のデルタにします。(逆伝播)
			Layers[i]->SetOutputDelta(Layers[i + 1]->GetInputDelta());

#ifdef __CUDACC__
			// 次のレイヤーの入力のストリームは、現在のレイヤーの出力のストリームにします。(順伝播)
			Layers[i + 1]->SetInputStream(Layers[i]->GetOutputStream());

			// 現在のレイヤーの出力のデルタのストリームは、次のレイヤーの入力のデルタのストリームにします。(逆伝播)
			Layers[i]->SetOutputDeltaStream(Layers[i + 1]->GetInputDeltaStream());

			// 次のレイヤーの入力のイベントは、現在のレイヤーの出力のイベントにします。(順伝播)
			Layers[i + 1]->SetInputEvent(Layers[i]->GetOutputEvent());

			// 現在のレイヤーの出力のデルタのイベントは、次のレイヤーの入力のデルタのイベントにします。(逆伝播)
			Layers[i]->SetOutputDeltaEvent(Layers[i + 1]->GetInputDeltaEvent());

#endif
		}
	}

	/*
	すべてのレイヤーのメモリを解放します。
	*/
	void FreeLayers() {
		for (size_t i = 0; i < Layers.size(); i++) {
			Layers[i]->Free();
		}

#ifdef __CUDACC__
		_chk(cudaFree(FirstLayer->GetInput()));
		_chk(cudaFree(LastLayer->GetOutputDelta()));
#endif
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
	}

	void DeepLearning() {
		FirstLayer = Layers[0];
		LastLayer = Layers[Layers.size() - 1];

		ReadMNIST();
		SGD();
	}
};
