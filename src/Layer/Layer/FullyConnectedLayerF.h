class DllExport FullyConnectedLayerF : public Layer {
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

	FullyConnectedLayerF(int x_size, int y_size);
	virtual ~FullyConnectedLayerF();
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
