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

		// ���̃��C���[�̓��͂̃X�g���[���́A���݂̃��C���[�̏o�͂̃X�g���[���ɂ��܂��B(���`�d)
		next_cuda_layer->SetInputStream(GetOutputStream());

		// ���݂̃��C���[�̏o�͂̃f���^�̃X�g���[���́A���̃��C���[�̓��͂̃f���^�̃X�g���[���ɂ��܂��B(�t�`�d)
		SetOutputDeltaStream(next_cuda_layer->GetInputDeltaStream());

		// ���̃��C���[�̓��͂̃C�x���g�́A���݂̃��C���[�̏o�͂̃C�x���g�ɂ��܂��B(���`�d)
		next_cuda_layer->SetInputEvent(GetOutputEvent());

		// ���݂̃��C���[�̏o�͂̃f���^�̃C�x���g�́A���̃��C���[�̓��͂̃f���^�̃C�x���g�ɂ��܂��B(�t�`�d)
		SetOutputDeltaEvent(next_cuda_layer->GetInputDeltaEvent());
	}

	virtual bool IsGPU() override { return true; }
};
