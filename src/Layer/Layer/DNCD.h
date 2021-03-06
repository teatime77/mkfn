﻿class DllExport DNCD : public Layer {
public:
    int T;
    int N;
    int W;
    int R;
    int X;
    int Y;
    int χl;
    int χ2hl;
    double* χ;
    double* χ2h;
    double* gin;
    double* gfo;
    double* s;
    double* o;
    double* h;
    double* Wi;
    double* Wf;
    double* Ws;
    double* Wo;
    double* bi;
    double* bf;
    double* bs;
    double* bo;
    double* x;
    double* y;
    double* v;
    double* z;
    double* M;
    double* kr;
    double* r;
    double* βr;
    double* kw;
    double* βw;
    double* e;
    double* ν;
    double* gf;
    double* ga;
    double* gw;
    double* ψ;
    double* u;
    int* φ;
    double* a;
    double* cw;
    double* ww;
    double* p;
    double* L;
    double* f;
    double* b;
    double* cr;
    double* wr;
    double* π1;
    double* π2;
    double* π3;
    double* Wr;
    double* delta_χ;
    double* delta_χ2h;
    double* delta_gin;
    double* delta_gfo;
    double* delta_s;
    double* delta_o;
    double* delta_h;
    double* delta_Wi;
    double* delta_Wf;
    double* delta_Ws;
    double* delta_Wo;
    double* delta_bi;
    double* delta_bf;
    double* delta_bs;
    double* delta_bo;
    double* delta_x;
    double* delta_y;
    double* delta_v;
    double* delta_z;
    double* delta_M;
    double* delta_kr;
    double* delta_r;
    double* delta_βr;
    double* delta_kw;
    double* delta_βw;
    double* delta_e;
    double* delta_ν;
    double* delta_gf;
    double* delta_ga;
    double* delta_gw;
    double* delta_ψ;
    double* delta_u;
    int* delta_φ;
    double* delta_a;
    double* delta_cw;
    double* delta_ww;
    double* delta_p;
    double* delta_L;
    double* delta_f;
    double* delta_b;
    double* delta_cr;
    double* delta_wr;
    double* delta_π1;
    double* delta_π2;
    double* delta_π3;
    double* delta_Wr;

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

	DNCD();
	virtual ~DNCD();
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
