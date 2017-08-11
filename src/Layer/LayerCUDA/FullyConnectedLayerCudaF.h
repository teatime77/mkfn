class DllExport FullyConnectedLayerCudaF : public LayerCuda {
public:
    int X;
    int Y;
    float* x;
    float* y;
    float* w;
    float* b;
    float* u;
    float* delta_x;
    float* delta_y;
    float* delta_w;
    float* delta_b;
    float* delta_u;
	cudaStream_t _stream_x;
	cudaStream_t _stream_y;
	cudaStream_t _stream_u;
	cudaStream_t _stream_delta_x;
	cudaStream_t _stream_delta_y;
	cudaStream_t _stream_delta_w;
	cudaStream_t _stream_delta_b;
	cudaStream_t _stream_delta_u;
	cudaEvent_t _event_x;
	cudaEvent_t _event_y;
	cudaEvent_t _event_u;
	cudaEvent_t _event_delta_x;
	cudaEvent_t _event_delta_y;
	cudaEvent_t _event_delta_w;
	cudaEvent_t _event_delta_b;
	cudaEvent_t _event_delta_u;

	int x_size_[1];
	int y_size_[1];
	int w_size_[2];
	int b_size_[1];
	int u_size_[1];
	int delta_x_size_[1];
	int delta_y_size_[1];
	int delta_w_size_[2];
	int delta_b_size_[1];
	int delta_u_size_[1];

	virtual int GetFieldCount() override { return 12; }
	virtual void GetFieldName(int field_idx, wchar_t* name) override;
	virtual int GetFieldDimension(int field_idx) override;
	virtual int* GetFieldSize(int field_idx) override;
	virtual void GetFieldValue(int field_idx, void* dst) override;
	virtual void SetFieldValue(int field_idx, void* src) override;

	FullyConnectedLayerCudaF(int x_size, int y_size);
	virtual ~FullyConnectedLayerCudaF();
	virtual void Forward() override;
	virtual void Backward() override;
	virtual void Allocate() override;
	virtual void Free() override;

	virtual void SetInput (void* src) override { x = (float*)src; }
	virtual void* GetInput(int t = 0)  override { return x; }
	virtual void* GetOutput(int t = 0) override { return y; }
	virtual void SetIputDelta   (void* src) override { delta_x = (float*)src; }
	virtual void SetOutputDelta (void* src) override { delta_y = (float*)src; }
	virtual void* GetOutputDelta(int t = 0) override { return delta_y; }
	virtual void* GetInputDelta (int t = 0) override { return delta_x; }

	virtual int GetInputCount()  override { return X; }
	virtual int GetOutputCount() override { return Y; }
	virtual int GetTimeCount()       override { return 0; }
	virtual int GetTimeInputCount()  override { return X; }
	virtual int GetTimeOutputCount() override { return Y; }

	virtual void SetInputStream (cudaStream_t src) override { _stream_x = src; }
	virtual cudaStream_t GetOutputStream() override { return _stream_y; }
	virtual void SetOutputDeltaStream (cudaStream_t src) override { _stream_delta_y = src; }
	virtual cudaStream_t GetInputDeltaStream() override { return _stream_delta_x; }

	virtual void SetInputEvent (cudaEvent_t src) override { _event_x = src; }
	virtual cudaEvent_t GetOutputEvent() override { return _event_y; }
	virtual void SetOutputDeltaEvent (cudaEvent_t src) override { _event_delta_y = src; }
	virtual cudaEvent_t GetInputDeltaEvent() override { return _event_delta_x; }

	void Start_forward_u();
	void Start_forward_y();

	void Start_backward_delta_u();
	void Start_backward_delta_x();
	void Start_backward_delta_w();
	void Start_backward_delta_b();

	virtual void UpdateParameter() override;
	void UpdateParameter_0();
	void UpdateParameter_1();
};
