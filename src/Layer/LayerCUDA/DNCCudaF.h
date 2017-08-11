﻿class DllExport DNCCudaF : public LayerCuda {
public:
    int T;
    int N;
    int W;
    int R;
    int X;
    int Y;
    int χl;
    int χ2hl;
    float* χ;
    float* χ2h;
    float* gin;
    float* gfo;
    float* s;
    float* o;
    float* h;
    float* Wi;
    float* Wf;
    float* Ws;
    float* Wo;
    float* bi;
    float* bf;
    float* bs;
    float* bo;
    float* x;
    float* y;
    float* v;
    float* z;
    float* M;
    float* kr;
    float* r;
    float* βr;
    float* kw;
    float* βw;
    float* e;
    float* ν;
    float* gf;
    float* ga;
    float* gw;
    float* ψ;
    float* u;
    int* φ;
    float* a;
    float* cw;
    float* ww;
    float* p;
    float* L;
    float* f;
    float* b;
    float* cr;
    float* wr;
    float* π1;
    float* π2;
    float* π3;
    float* Wr;
    float* delta_χ;
    float* delta_χ2h;
    float* delta_gin;
    float* delta_gfo;
    float* delta_s;
    float* delta_o;
    float* delta_h;
    float* delta_Wi;
    float* delta_Wf;
    float* delta_Ws;
    float* delta_Wo;
    float* delta_bi;
    float* delta_bf;
    float* delta_bs;
    float* delta_bo;
    float* delta_x;
    float* delta_y;
    float* delta_v;
    float* delta_z;
    float* delta_M;
    float* delta_kr;
    float* delta_r;
    float* delta_βr;
    float* delta_kw;
    float* delta_βw;
    float* delta_e;
    float* delta_ν;
    float* delta_gf;
    float* delta_ga;
    float* delta_gw;
    float* delta_ψ;
    float* delta_u;
    int* delta_φ;
    float* delta_a;
    float* delta_cw;
    float* delta_ww;
    float* delta_p;
    float* delta_L;
    float* delta_f;
    float* delta_b;
    float* delta_cr;
    float* delta_wr;
    float* delta_π1;
    float* delta_π2;
    float* delta_π3;
    float* delta_Wr;
	cudaStream_t _stream_gin;
	cudaStream_t _stream_gfo;
	cudaStream_t _stream_s;
	cudaStream_t _stream_o;
	cudaStream_t _stream_h;
	cudaStream_t _stream_x;
	cudaStream_t _stream_y;
	cudaStream_t _stream_M;
	cudaStream_t _stream_r;
	cudaStream_t _stream_ψ;
	cudaStream_t _stream_u;
	cudaStream_t _stream_φ;
	cudaStream_t _stream_cw;
	cudaStream_t _stream_ww;
	cudaStream_t _stream_p;
	cudaStream_t _stream_L;
	cudaStream_t _stream_f;
	cudaStream_t _stream_b;
	cudaStream_t _stream_cr;
	cudaStream_t _stream_wr;
	cudaStream_t _stream_delta_χ;
	cudaStream_t _stream_delta_χ2h;
	cudaStream_t _stream_delta_gin;
	cudaStream_t _stream_delta_gfo;
	cudaStream_t _stream_delta_s;
	cudaStream_t _stream_delta_o;
	cudaStream_t _stream_delta_h;
	cudaStream_t _stream_delta_Wi;
	cudaStream_t _stream_delta_Wf;
	cudaStream_t _stream_delta_Ws;
	cudaStream_t _stream_delta_Wo;
	cudaStream_t _stream_delta_bi;
	cudaStream_t _stream_delta_bf;
	cudaStream_t _stream_delta_bs;
	cudaStream_t _stream_delta_bo;
	cudaStream_t _stream_delta_x;
	cudaStream_t _stream_delta_y;
	cudaStream_t _stream_delta_v;
	cudaStream_t _stream_delta_z;
	cudaStream_t _stream_delta_M;
	cudaStream_t _stream_delta_kr;
	cudaStream_t _stream_delta_r;
	cudaStream_t _stream_delta_βr;
	cudaStream_t _stream_delta_kw;
	cudaStream_t _stream_delta_βw;
	cudaStream_t _stream_delta_e;
	cudaStream_t _stream_delta_ν;
	cudaStream_t _stream_delta_gf;
	cudaStream_t _stream_delta_ga;
	cudaStream_t _stream_delta_gw;
	cudaStream_t _stream_delta_ψ;
	cudaStream_t _stream_delta_u;
	cudaStream_t _stream_delta_φ;
	cudaStream_t _stream_delta_a;
	cudaStream_t _stream_delta_cw;
	cudaStream_t _stream_delta_ww;
	cudaStream_t _stream_delta_p;
	cudaStream_t _stream_delta_L;
	cudaStream_t _stream_delta_f;
	cudaStream_t _stream_delta_b;
	cudaStream_t _stream_delta_cr;
	cudaStream_t _stream_delta_wr;
	cudaStream_t _stream_delta_π1;
	cudaStream_t _stream_delta_π2;
	cudaStream_t _stream_delta_π3;
	cudaStream_t _stream_delta_Wr;
	cudaEvent_t _event_gin;
	cudaEvent_t _event_gfo;
	cudaEvent_t _event_s;
	cudaEvent_t _event_o;
	cudaEvent_t _event_h;
	cudaEvent_t _event_x;
	cudaEvent_t _event_y;
	cudaEvent_t _event_M;
	cudaEvent_t _event_r;
	cudaEvent_t _event_ψ;
	cudaEvent_t _event_u;
	cudaEvent_t _event_φ;
	cudaEvent_t _event_cw;
	cudaEvent_t _event_ww;
	cudaEvent_t _event_p;
	cudaEvent_t _event_L;
	cudaEvent_t _event_f;
	cudaEvent_t _event_b;
	cudaEvent_t _event_cr;
	cudaEvent_t _event_wr;
	cudaEvent_t _event_delta_χ;
	cudaEvent_t _event_delta_χ2h;
	cudaEvent_t _event_delta_gin;
	cudaEvent_t _event_delta_gfo;
	cudaEvent_t _event_delta_s;
	cudaEvent_t _event_delta_o;
	cudaEvent_t _event_delta_h;
	cudaEvent_t _event_delta_Wi;
	cudaEvent_t _event_delta_Wf;
	cudaEvent_t _event_delta_Ws;
	cudaEvent_t _event_delta_Wo;
	cudaEvent_t _event_delta_bi;
	cudaEvent_t _event_delta_bf;
	cudaEvent_t _event_delta_bs;
	cudaEvent_t _event_delta_bo;
	cudaEvent_t _event_delta_x;
	cudaEvent_t _event_delta_y;
	cudaEvent_t _event_delta_v;
	cudaEvent_t _event_delta_z;
	cudaEvent_t _event_delta_M;
	cudaEvent_t _event_delta_kr;
	cudaEvent_t _event_delta_r;
	cudaEvent_t _event_delta_βr;
	cudaEvent_t _event_delta_kw;
	cudaEvent_t _event_delta_βw;
	cudaEvent_t _event_delta_e;
	cudaEvent_t _event_delta_ν;
	cudaEvent_t _event_delta_gf;
	cudaEvent_t _event_delta_ga;
	cudaEvent_t _event_delta_gw;
	cudaEvent_t _event_delta_ψ;
	cudaEvent_t _event_delta_u;
	cudaEvent_t _event_delta_φ;
	cudaEvent_t _event_delta_a;
	cudaEvent_t _event_delta_cw;
	cudaEvent_t _event_delta_ww;
	cudaEvent_t _event_delta_p;
	cudaEvent_t _event_delta_L;
	cudaEvent_t _event_delta_f;
	cudaEvent_t _event_delta_b;
	cudaEvent_t _event_delta_cr;
	cudaEvent_t _event_delta_wr;
	cudaEvent_t _event_delta_π1;
	cudaEvent_t _event_delta_π2;
	cudaEvent_t _event_delta_π3;
	cudaEvent_t _event_delta_Wr;

	int χ_size_[2];
	int χ2h_size_[2];
	int gin_size_[2];
	int gfo_size_[2];
	int s_size_[2];
	int o_size_[2];
	int h_size_[2];
	int Wi_size_[2];
	int Wf_size_[2];
	int Ws_size_[2];
	int Wo_size_[2];
	int bi_size_[1];
	int bf_size_[1];
	int bs_size_[1];
	int bo_size_[1];
	int x_size_[2];
	int y_size_[2];
	int v_size_[2];
	int z_size_[2];
	int M_size_[3];
	int kr_size_[3];
	int r_size_[3];
	int βr_size_[2];
	int kw_size_[2];
	int βw_size_[1];
	int e_size_[2];
	int ν_size_[2];
	int gf_size_[2];
	int ga_size_[1];
	int gw_size_[1];
	int ψ_size_[2];
	int u_size_[2];
	int φ_size_[2];
	int a_size_[2];
	int cw_size_[2];
	int ww_size_[2];
	int p_size_[2];
	int L_size_[3];
	int f_size_[3];
	int b_size_[3];
	int cr_size_[3];
	int wr_size_[3];
	int π1_size_[2];
	int π2_size_[2];
	int π3_size_[2];
	int Wr_size_[3];
	int delta_χ_size_[2];
	int delta_χ2h_size_[2];
	int delta_gin_size_[2];
	int delta_gfo_size_[2];
	int delta_s_size_[2];
	int delta_o_size_[2];
	int delta_h_size_[2];
	int delta_Wi_size_[2];
	int delta_Wf_size_[2];
	int delta_Ws_size_[2];
	int delta_Wo_size_[2];
	int delta_bi_size_[1];
	int delta_bf_size_[1];
	int delta_bs_size_[1];
	int delta_bo_size_[1];
	int delta_x_size_[2];
	int delta_y_size_[2];
	int delta_v_size_[2];
	int delta_z_size_[2];
	int delta_M_size_[3];
	int delta_kr_size_[3];
	int delta_r_size_[3];
	int delta_βr_size_[2];
	int delta_kw_size_[2];
	int delta_βw_size_[1];
	int delta_e_size_[2];
	int delta_ν_size_[2];
	int delta_gf_size_[2];
	int delta_ga_size_[1];
	int delta_gw_size_[1];
	int delta_ψ_size_[2];
	int delta_u_size_[2];
	int delta_φ_size_[2];
	int delta_a_size_[2];
	int delta_cw_size_[2];
	int delta_ww_size_[2];
	int delta_p_size_[2];
	int delta_L_size_[3];
	int delta_f_size_[3];
	int delta_b_size_[3];
	int delta_cr_size_[3];
	int delta_wr_size_[3];
	int delta_π1_size_[2];
	int delta_π2_size_[2];
	int delta_π3_size_[2];
	int delta_Wr_size_[3];

	virtual int GetFieldCount() override { return 100; }
	virtual void GetFieldName(int field_idx, wchar_t* name) override;
	virtual int GetFieldDimension(int field_idx) override;
	virtual int* GetFieldSize(int field_idx) override;
	virtual void GetFieldValue(int field_idx, void* dst) override;
	virtual void SetFieldValue(int field_idx, void* src) override;

	DNCCudaF();
	virtual ~DNCCudaF();
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

	void Start_forward_gin();
	void Start_forward_gfo();
	void Start_forward_s();
	void Start_forward_o();
	void Start_forward_h();
	void Start_forward_ψ();
	void Start_forward_u();
	void Start_forward_φ();
	void Start_forward_y();

	void Start_backward_delta_s();
	void Start_backward_delta_gin();
	void Start_backward_delta_gfo();
	void Start_backward_delta_o();
	void Start_backward_delta_χ2h();
	void Start_backward_delta_Wi();
	void Start_backward_delta_Wf();
	void Start_backward_delta_Ws();
	void Start_backward_delta_Wo();
	void Start_backward_delta_bi();
	void Start_backward_delta_bf();
	void Start_backward_delta_bs();
	void Start_backward_delta_bo();
	void Start_backward_delta_M();
	void Start_backward_delta_βr();
	void Start_backward_delta_βw();
	void Start_backward_delta_e();
	void Start_backward_delta_ν();
	void Start_backward_delta_u();
	void Start_backward_delta_ψ();
	void Start_backward_delta_gf();
	void Start_backward_delta_p();
	void Start_backward_delta_wr();
	void Start_backward_delta_f();
	void Start_backward_delta_b();
	void Start_backward_delta_L();
	void Start_backward_delta_ww();
	void Start_backward_delta_ga();
	void Start_backward_delta_gw();
	void Start_backward_delta_a();
	void Start_backward_delta_cw();
	void Start_backward_delta_cr();
	void Start_backward_delta_π1();
	void Start_backward_delta_π2();
	void Start_backward_delta_π3();

	virtual void UpdateParameter() override;
	void UpdateParameter_0();
	void UpdateParameter_1();
	void UpdateParameter_2();
	void UpdateParameter_3();
	void UpdateParameter_4();
	void UpdateParameter_5();
	void UpdateParameter_6();
	void UpdateParameter_7();
	void UpdateParameter_8();
	void UpdateParameter_9();
	void UpdateParameter_10();
};
