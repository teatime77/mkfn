#pragma once

#include <assert.h>

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
		�����֐��̔���	
	*/
	void CostDerivative(T* cost_derivative, T* last_y, T* batch_Y, int size) {
		for (int i = 0; i < size; i++) {
			cost_derivative[i] = last_y[i] - batch_Y[i];
		}
	}

	/*
	�����֐�
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
		C++�ł�CUDA�ł̌v�Z���ʂ̈Ⴂ���_���v�t�@�C���ɏ����Ĕ�r���܂��B	
	*/
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

	/*
		�~�j�o�b�`���ƂɃp�����[�^���X�V���܂��B
	*/
	void UpdateMiniBatch(T* batch_X, T* batch_Y, T* last_y, T* cost_derivative) {
#ifdef __CUDACC__
		_chk(cudaMemcpy(FirstLayer->GetInput(), batch_X, TrainBatchSize * DomainLen * sizeof(T), cudaMemcpyHostToDevice));
		_chk(cudaDeviceSynchronize());
#else
		FirstLayer->SetInput(batch_X);
#endif

		for (size_t i = 0; i < Layers.size(); i++) {
			Layers[i]->Forward();
		}

		ConvolutionalLayer* cl = (ConvolutionalLayer*)FirstLayer;
		Dmp(L"y0", (T*)Layers[0]->GetOutput(), TrainBatchSize * Layers[0]->GetOutputCount());
		Dmp(L"y1", (T*)Layers[1]->GetOutput(), TrainBatchSize * Layers[1]->GetOutputCount());
		Dmp(L"y2", (T*)Layers[2]->GetOutput(), TrainBatchSize * Layers[2]->GetOutputCount());
		Dmp(L"y3", (T*)Layers[3]->GetOutput(), TrainBatchSize * Layers[3]->GetOutputCount());

#ifdef __CUDACC__
		_chk(cudaDeviceSynchronize());
		_chk(cudaMemcpy(last_y, LastLayer->GetOutput(), TrainBatchSize * RangeLen * sizeof(T), cudaMemcpyDeviceToHost));
		T* last_y_ptr = last_y;
#else
		T* last_y_ptr = (T*)LastLayer->GetOutput();
#endif

		int size = TrainBatchSize * RangeLen;

		// �����֐��̔���
		CostDerivative(cost_derivative, last_y_ptr, batch_Y, size);

		T cost = Cost(cost_derivative, size);

/*
		// �����֐������O�ɏ����܂��B
		fprintf(fpLog, "%f\r\n", cost);
		Log(L"%d:%d:%.16f", EpochIdx, MiniBatchIdx, cost);

		static int log_idx;
		log_idx++;
		if (log_idx % 1000 == 0) {
			// �v�Z�̓r���Ń��O�������悤�ɁA1000�񂲂ƂɃt���b�V�����܂��B

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
		Dmp(L"b", fl->b, fl->Y);
		Dmp(L"w", fl->w, fl->Y * fl->X);
		if (MiniBatchIdx == 0) {
		}
*/

		FullyConnectedLayer* fc = (FullyConnectedLayer*)Layers[3];
		Dmp(L"fc3-w", fc->w, fc->Y * fc->X);
		Dmp(L"fc3-b", fc->b, fc->Y);

		fc = (FullyConnectedLayer*)Layers[2];
		Dmp(L"fc2-delta_w", fc->delta_w, TrainBatchSize * fc->Y * fc->X);
		Dmp(L"fc2-delta_b", fc->delta_b, TrainBatchSize * fc->Y);
		Dmp(L"fc2-w", fc->w, fc->Y * fc->X);
		Dmp(L"fc2-b", fc->b, fc->Y);

		Dmp(L"h", cl->h, cl->H * cl->H * cl->K);
		Dmp(L"b", cl->b, cl->K);


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
	���ׂẴ��C���[�̃����������蓖�āA���C���[�̓��o�͂��������܂��B
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

		// ���C���[�̓��o�͂��������܂��B
		for (size_t i = 0; i + 1 < Layers.size(); i++) {

			assert(Layers[i]->GetOutputCount() == Layers[i + 1]->GetInputCount());

			// ���̃��C���[�̓��͂́A���݂̃��C���[�̏o�͂ɂ��܂��B(���`�d)
			Layers[i + 1]->SetInput(Layers[i]->GetOutput());

			// ���݂̃��C���[�̏o�͂̃f���^�́A���̃��C���[�̓��͂̃f���^�ɂ��܂��B(�t�`�d)
			Layers[i]->SetOutputDelta(Layers[i + 1]->GetInputDelta());

#ifdef __CUDACC__
			// ���̃��C���[�̓��͂̃X�g���[���́A���݂̃��C���[�̏o�͂̃X�g���[���ɂ��܂��B(���`�d)
			Layers[i + 1]->SetInputStream(Layers[i]->GetOutputStream());

			// ���݂̃��C���[�̏o�͂̃f���^�̃X�g���[���́A���̃��C���[�̓��͂̃f���^�̃X�g���[���ɂ��܂��B(�t�`�d)
			Layers[i]->SetOutputDeltaStream(Layers[i + 1]->GetInputDeltaStream());

			// ���̃��C���[�̓��͂̃C�x���g�́A���݂̃��C���[�̏o�͂̃C�x���g�ɂ��܂��B(���`�d)
			Layers[i + 1]->SetInputEvent(Layers[i]->GetOutputEvent());

			// ���݂̃��C���[�̏o�͂̃f���^�̃C�x���g�́A���̃��C���[�̓��͂̃f���^�̃C�x���g�ɂ��܂��B(�t�`�d)
			Layers[i]->SetOutputDeltaEvent(Layers[i + 1]->GetInputDeltaEvent());

#endif
		}
	}

	/*
	���ׂẴ��C���[�̃�������������܂��B
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
	�m���I���z�~���@ (stochastic gradient descent, SGD)
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

			// ���ׂẴ��C���[�̃����������蓖�āA���C���[�̓��o�͂��������܂��B
			AllocateConnectLayers(TrainBatchSize);

			for (MiniBatchIdx = 0; MiniBatchIdx < train_batch_cnt; MiniBatchIdx++) {

				SetBatchData(TrainX, train_batch_X, train_batch_Y, TrainLabel, TrainBatchSize, idxes);

				UpdateMiniBatch(train_batch_X, train_batch_Y, train_last_Y, cost_derivative);
			}

			FreeLayers();

			// ���ׂẴ��C���[�̃����������蓖�āA���C���[�̓��o�͂��������܂��B
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

		SGD();

		for (size_t i = 0; i < Layers.size(); i++) {
			delete Layers[i];
		}
		Layers.clear();
	}
};


void NetworkTest() {
	// ���O�t�@�C�������������܂��B
	InitLog();

#ifdef __CUDACC__
	_chk(cudaSetDevice(0));
#endif

	Network<double> *net = new Network<double>();
	net->EpochSize = 100;
	net->TrainBatchSize = 10;
	net->TestBatchSize = 20;
	net->ReadMNIST();

	for (int run_idx = 0; ; run_idx++) {
		int type = 1;
		switch (type) {
		case 0:
			net->Layers = std::vector<Layer*>{
				new FullyConnectedLayer(28 * 28, 30),
				new FullyConnectedLayer(30, 10)
			};
			break;

		case 1:
			net->Layers = std::vector<Layer*>{
				//new ConvolutionalLayer(28, 28, 20, 5),
				//new MaxPoolingLayer(24, 24, 20, 2),
				//new FullyConnectedLayer(12 * 12 * 20, 100),
				new ConvolutionalLayer(28, 28, 5, 5),
				new MaxPoolingLayer(24, 24, 5, 2),
				new FullyConnectedLayer(12 * 12 * 5, 100),
				new FullyConnectedLayer(100, 10)
			};
			break;
		}

		for (size_t i = 0; i < net->Layers.size(); i++) {
			net->Layers[i]->LearningRate = 3.0f / net->TrainBatchSize;
		}

		net->DeepLearning();
	}
}
