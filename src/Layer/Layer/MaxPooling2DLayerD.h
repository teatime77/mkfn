class DllExport MaxPooling2DLayerD : public Layer {
public:
    int M;
    int N;
    int K;
    int PM;
    int PN;
    int MH;
    int NH;
    double* x;
    double* y;
    double* delta_x;
    double* delta_y;
    int* _idx_x;

	int x_size_[3];
	int y_size_[3];
	int delta_x_size_[3];
	int delta_y_size_[3];
	int _idx_x_size_[3];

	virtual int GetFieldCount() override { return 12; }
	virtual void GetFieldName(int field_idx, wchar_t* name) override;
	virtual int GetFieldDimension(int field_idx) override;
	virtual int* GetFieldSize(int field_idx) override;
	virtual void GetFieldValue(int field_idx, void* dst) override;
	virtual void SetFieldValue(int field_idx, void* src) override;

	MaxPooling2DLayerD(int m_size, int n_size, int k_size, int pm_size, int pn_size);
	virtual ~MaxPooling2DLayerD();
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

	virtual int GetInputCount()  override { return M * N * K; }
	virtual int GetOutputCount() override { return MH * NH * K; }
	virtual int GetTimeCount()       override { return 0; }
	virtual int GetTimeInputCount()  override { return M * N * K; }
	virtual int GetTimeOutputCount() override { return MH * NH * K; }

	void Start_forward_y();

	void Start_backward_delta_x();

	virtual void UpdateParameter() override;
};
