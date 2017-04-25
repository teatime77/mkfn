#pragma once

class DllExport LayerCuda : public Layer {
public:
	virtual void SetInputData(void* src, int size) {
		_chk(cudaMemcpy(GetInput(), src, size, cudaMemcpyHostToDevice));
		_chk(cudaDeviceSynchronize());
	}

	virtual void SetOutputDeltaData(void* src, int size) {
		_chk(cudaMemcpy(GetOutputDelta(), src, size, cudaMemcpyHostToDevice));
		_chk(cudaDeviceSynchronize());
	}

	virtual void* GetOutputData(void* dst, int size) {
		_chk(cudaDeviceSynchronize());
		_chk(cudaMemcpy(dst, GetOutput(), size, cudaMemcpyDeviceToHost));

		return dst;
	}

	virtual void SetInputStream(cudaStream_t src) = 0;
	virtual cudaStream_t GetOutputStream() = 0;

	virtual void SetOutputDeltaStream(cudaStream_t src) = 0;
	virtual cudaStream_t GetInputDeltaStream() = 0;

	virtual void SetInputEvent(cudaEvent_t src) = 0;
	virtual cudaEvent_t GetOutputEvent() = 0;

	virtual void SetOutputDeltaEvent(cudaEvent_t src) = 0;
	virtual cudaEvent_t GetInputDeltaEvent() = 0;

	virtual void ConnectLayer(Layer* next_layer) {
		LayerCuda* next_cuda_layer = (LayerCuda*)next_layer;

		// 次のレイヤーの入力のストリームは、現在のレイヤーの出力のストリームにします。(順伝播)
		next_cuda_layer->SetInputStream(GetOutputStream());

		// 現在のレイヤーの出力のデルタのストリームは、次のレイヤーの入力のデルタのストリームにします。(逆伝播)
		SetOutputDeltaStream(next_cuda_layer->GetInputDeltaStream());

		// 次のレイヤーの入力のイベントは、現在のレイヤーの出力のイベントにします。(順伝播)
		next_cuda_layer->SetInputEvent(GetOutputEvent());

		// 現在のレイヤーの出力のデルタのイベントは、次のレイヤーの入力のデルタのイベントにします。(逆伝播)
		SetOutputDeltaEvent(next_cuda_layer->GetInputDeltaEvent());
	}

	virtual bool IsGPU() override { return true; }
};
