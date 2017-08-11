class DllExport LSTMLayerCudaF : public LayerCuda {
public:
    int T;
    int X;
    int Y;
    float* x;
    float* y;
    float* wIin;
    float* wFin;
    float* wOin;
    float* win;
    float* wIr;
    float* wFr;
    float* wOr;
    float* wr;
    float* wI;
    float* wF;
    float* wO;
    float* bO;
    float* bF;
    float* bI;
    float* b;
    float* u;
    float* s;
    float* uI;
    float* uF;
    float* uO;
    float* delta_x;
    float* delta_y;
    float* delta_wIin;
    float* delta_wFin;
    float* delta_wOin;
    float* delta_win;
    float* delta_wIr;
    float* delta_wFr;
    float* delta_wOr;
    float* delta_wr;
    float* delta_wI;
    float* delta_wF;
    float* delta_wO;
    float* delta_bO;
    float* delta_bF;
    float* delta_bI;
    float* delta_b;
    float* delta_u;
    float* delta_s;
    float* delta_uI;
    float* delta_uF;
    float* delta_uO;
	cudaStream_t _stream_x;
	cudaStream_t _stream_y;
	cudaStream_t _stream_u;
	cudaStream_t _stream_s;
	cudaStream_t _stream_uI;
	cudaStream_t _stream_uF;
	cudaStream_t _stream_uO;
	cudaStream_t _stream_delta_x;
	cudaStream_t _stream_delta_y;
	cudaStream_t _stream_delta_wIin;
	cudaStream_t _stream_delta_wFin;
	cudaStream_t _stream_delta_wOin;
	cudaStream_t _stream_delta_win;
	cudaStream_t _stream_delta_wIr;
	cudaStream_t _stream_delta_wFr;
	cudaStream_t _stream_delta_wOr;
	cudaStream_t _stream_delta_wr;
	cudaStream_t _stream_delta_wI;
	cudaStream_t _stream_delta_wF;
	cudaStream_t _stream_delta_wO;
	cudaStream_t _stream_delta_bO;
	cudaStream_t _stream_delta_bF;
	cudaStream_t _stream_delta_bI;
	cudaStream_t _stream_delta_b;
	cudaStream_t _stream_delta_u;
	cudaStream_t _stream_delta_s;
	cudaStream_t _stream_delta_uI;
	cudaStream_t _stream_delta_uF;
	cudaStream_t _stream_delta_uO;
	cudaEvent_t _event_x;
	cudaEvent_t _event_y;
	cudaEvent_t _event_u;
	cudaEvent_t _event_s;
	cudaEvent_t _event_uI;
	cudaEvent_t _event_uF;
	cudaEvent_t _event_uO;
	cudaEvent_t _event_delta_x;
	cudaEvent_t _event_delta_y;
	cudaEvent_t _event_delta_wIin;
	cudaEvent_t _event_delta_wFin;
	cudaEvent_t _event_delta_wOin;
	cudaEvent_t _event_delta_win;
	cudaEvent_t _event_delta_wIr;
	cudaEvent_t _event_delta_wFr;
	cudaEvent_t _event_delta_wOr;
	cudaEvent_t _event_delta_wr;
	cudaEvent_t _event_delta_wI;
	cudaEvent_t _event_delta_wF;
	cudaEvent_t _event_delta_wO;
	cudaEvent_t _event_delta_bO;
	cudaEvent_t _event_delta_bF;
	cudaEvent_t _event_delta_bI;
	cudaEvent_t _event_delta_b;
	cudaEvent_t _event_delta_u;
	cudaEvent_t _event_delta_s;
	cudaEvent_t _event_delta_uI;
	cudaEvent_t _event_delta_uF;
	cudaEvent_t _event_delta_uO;

	int x_size_[2];
	int y_size_[2];
	int wIin_size_[2];
	int wFin_size_[2];
	int wOin_size_[2];
	int win_size_[2];
	int wIr_size_[2];
	int wFr_size_[2];
	int wOr_size_[2];
	int wr_size_[2];
	int wI_size_[1];
	int wF_size_[1];
	int wO_size_[1];
	int bO_size_[1];
	int bF_size_[1];
	int bI_size_[1];
	int b_size_[1];
	int u_size_[2];
	int s_size_[2];
	int uI_size_[2];
	int uF_size_[2];
	int uO_size_[2];
	int delta_x_size_[2];
	int delta_y_size_[2];
	int delta_wIin_size_[2];
	int delta_wFin_size_[2];
	int delta_wOin_size_[2];
	int delta_win_size_[2];
	int delta_wIr_size_[2];
	int delta_wFr_size_[2];
	int delta_wOr_size_[2];
	int delta_wr_size_[2];
	int delta_wI_size_[1];
	int delta_wF_size_[1];
	int delta_wO_size_[1];
	int delta_bO_size_[1];
	int delta_bF_size_[1];
	int delta_bI_size_[1];
	int delta_b_size_[1];
	int delta_u_size_[2];
	int delta_s_size_[2];
	int delta_uI_size_[2];
	int delta_uF_size_[2];
	int delta_uO_size_[2];

	virtual int GetFieldCount() override { return 47; }
	virtual void GetFieldName(int field_idx, wchar_t* name) override;
	virtual int GetFieldDimension(int field_idx) override;
	virtual int* GetFieldSize(int field_idx) override;
	virtual void GetFieldValue(int field_idx, void* dst) override;
	virtual void SetFieldValue(int field_idx, void* src) override;

	LSTMLayerCudaF(int t_size, int x_size, int y_size);
	virtual ~LSTMLayerCudaF();
	virtual void Forward() override;
	virtual void Backward() override;
	virtual void Allocate() override;
	virtual void Free() override;

	virtual void SetInput (void* src) override { x = (float*)src; }
	virtual void* GetInput(int t = 0)  override { return x + t * X * BatchSize; }
	virtual void* GetOutput(int t = 0) override { return y + t * Y * BatchSize; }
	virtual void SetIputDelta   (void* src) override { delta_x = (float*)src; }
	virtual void SetOutputDelta (void* src) override { delta_y = (float*)src; }
	virtual void* GetOutputDelta(int t = 0) override { return delta_y + t * Y * BatchSize; }
	virtual void* GetInputDelta (int t = 0) override { return delta_x + t * X * BatchSize; }

	virtual int GetInputCount()  override { return T * X; }
	virtual int GetOutputCount() override { return T * Y; }
	virtual int GetTimeCount()       override { return T; }
	virtual int GetTimeInputCount()  override { return X; }
	virtual int GetTimeOutputCount() override { return Y; }
	virtual void SetTimeCount(int time_count) override { T = time_count; }

	virtual void SetInputStream (cudaStream_t src) override { _stream_x = src; }
	virtual cudaStream_t GetOutputStream() override { return _stream_y; }
	virtual void SetOutputDeltaStream (cudaStream_t src) override { _stream_delta_y = src; }
	virtual cudaStream_t GetInputDeltaStream() override { return _stream_delta_x; }

	virtual void SetInputEvent (cudaEvent_t src) override { _event_x = src; }
	virtual cudaEvent_t GetOutputEvent() override { return _event_y; }
	virtual void SetOutputDeltaEvent (cudaEvent_t src) override { _event_delta_y = src; }
	virtual cudaEvent_t GetInputDeltaEvent() override { return _event_delta_x; }

	void Start_forward_uF();
	void Start_forward_uI();
	void Start_forward_u();
	void Start_forward_s();
	void Start_forward_uO();
	void Start_forward_y();

	void Start_backward_delta_y();
	void Start_backward_delta_uO();
	void Start_backward_delta_wOin();
	void Start_backward_delta_wOr();
	void Start_backward_delta_wO();
	void Start_backward_delta_bO();
	void Start_backward_delta_s();
	void Start_backward_delta_u();
	void Start_backward_delta_win();
	void Start_backward_delta_wr();
	void Start_backward_delta_b();
	void Start_backward_delta_uI();
	void Start_backward_delta_wIin();
	void Start_backward_delta_wIr();
	void Start_backward_delta_wI();
	void Start_backward_delta_bI();
	void Start_backward_delta_uF();
	void Start_backward_delta_x();
	void Start_backward_delta_wFin();
	void Start_backward_delta_wFr();
	void Start_backward_delta_wF();
	void Start_backward_delta_bF();

	virtual void UpdateParameter() override;
	void UpdateParameter_0();
	void UpdateParameter_1();
	void UpdateParameter_2();
};
