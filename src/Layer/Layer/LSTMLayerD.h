class DllExport LSTMLayerD : public Layer {
public:
    int T;
    int X;
    int Y;
    double* x;
    double* y;
    double* wIin;
    double* wFin;
    double* wOin;
    double* win;
    double* wIr;
    double* wFr;
    double* wOr;
    double* wr;
    double* wI;
    double* wF;
    double* wO;
    double* bO;
    double* bF;
    double* bI;
    double* b;
    double* u;
    double* s;
    double* uI;
    double* uF;
    double* uO;
    double* delta_x;
    double* delta_y;
    double* delta_wIin;
    double* delta_wFin;
    double* delta_wOin;
    double* delta_win;
    double* delta_wIr;
    double* delta_wFr;
    double* delta_wOr;
    double* delta_wr;
    double* delta_wI;
    double* delta_wF;
    double* delta_wO;
    double* delta_bO;
    double* delta_bF;
    double* delta_bI;
    double* delta_b;
    double* delta_u;
    double* delta_s;
    double* delta_uI;
    double* delta_uF;
    double* delta_uO;

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

	LSTMLayerD(int t_size, int x_size, int y_size);
	virtual ~LSTMLayerD();
	virtual void Forward() override;
	virtual void Backward() override;
	virtual void Allocate() override;
	virtual void Free() override;

	virtual void SetInput (void* src) override { x = (double*)src; }
	virtual void* GetInput(int t = 0)  override { return x + t * X * BatchSize; }
	virtual void* GetOutput(int t = 0) override { return y + t * Y * BatchSize; }
	virtual void SetIputDelta   (void* src) override { delta_x = (double*)src; }
	virtual void SetOutputDelta (void* src) override { delta_y = (double*)src; }
	virtual void* GetOutputDelta(int t = 0) override { return delta_y + t * Y * BatchSize; }
	virtual void* GetInputDelta (int t = 0) override { return delta_x + t * X * BatchSize; }

	virtual int GetInputCount()  override { return T * X; }
	virtual int GetOutputCount() override { return T * Y; }
	virtual int GetTimeCount()       override { return T; }
	virtual int GetTimeInputCount()  override { return X; }
	virtual int GetTimeOutputCount() override { return Y; }
	virtual void SetTimeCount(int time_count) override { T = time_count; }

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
