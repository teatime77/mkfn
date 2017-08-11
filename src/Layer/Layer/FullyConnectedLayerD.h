class DllExport FullyConnectedLayerD : public Layer {
public:
    int X;
    int Y;
    double* x;
    double* y;
    double* w;
    double* b;
    double* u;
    double* delta_x;
    double* delta_y;
    double* delta_w;
    double* delta_b;
    double* delta_u;

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

	FullyConnectedLayerD(int x_size, int y_size);
	virtual ~FullyConnectedLayerD();
	virtual void Forward() override;
	virtual void Backward() override;
	virtual void Allocate() override;
	virtual void Free() override;

	virtual void SetInput (void* src) override { x = (double*)src; }
	virtual void* GetInput(int t = 0)  override { return x; }
	virtual void* GetOutput(int t = 0) override { return y; }
	virtual void SetIputDelta   (void* src) override { delta_x = (double*)src; }
	virtual void SetOutputDelta (void* src) override { delta_y = (double*)src; }
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
