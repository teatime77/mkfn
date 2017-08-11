#include <stdlib.h>
#include <stdio.h>
#include <FLOAT.h>
#include "MkFn.h"
#include "../../Lib/Lib.h"
#include "DNCF.h"

void DNCF::GetFieldName(int field_idx, wchar_t* name){
	switch(field_idx){
		case 0: wcscpy(name, L"T"); break;
		case 1: wcscpy(name, L"N"); break;
		case 2: wcscpy(name, L"W"); break;
		case 3: wcscpy(name, L"R"); break;
		case 4: wcscpy(name, L"X"); break;
		case 5: wcscpy(name, L"Y"); break;
		case 6: wcscpy(name, L"χl"); break;
		case 7: wcscpy(name, L"χ2hl"); break;
		case 8: wcscpy(name, L"χ"); break;
		case 9: wcscpy(name, L"χ2h"); break;
		case 10: wcscpy(name, L"gin"); break;
		case 11: wcscpy(name, L"gfo"); break;
		case 12: wcscpy(name, L"s"); break;
		case 13: wcscpy(name, L"o"); break;
		case 14: wcscpy(name, L"h"); break;
		case 15: wcscpy(name, L"Wi"); break;
		case 16: wcscpy(name, L"Wf"); break;
		case 17: wcscpy(name, L"Ws"); break;
		case 18: wcscpy(name, L"Wo"); break;
		case 19: wcscpy(name, L"bi"); break;
		case 20: wcscpy(name, L"bf"); break;
		case 21: wcscpy(name, L"bs"); break;
		case 22: wcscpy(name, L"bo"); break;
		case 23: wcscpy(name, L"x"); break;
		case 24: wcscpy(name, L"y"); break;
		case 25: wcscpy(name, L"v"); break;
		case 26: wcscpy(name, L"z"); break;
		case 27: wcscpy(name, L"M"); break;
		case 28: wcscpy(name, L"kr"); break;
		case 29: wcscpy(name, L"r"); break;
		case 30: wcscpy(name, L"βr"); break;
		case 31: wcscpy(name, L"kw"); break;
		case 32: wcscpy(name, L"βw"); break;
		case 33: wcscpy(name, L"e"); break;
		case 34: wcscpy(name, L"ν"); break;
		case 35: wcscpy(name, L"gf"); break;
		case 36: wcscpy(name, L"ga"); break;
		case 37: wcscpy(name, L"gw"); break;
		case 38: wcscpy(name, L"ψ"); break;
		case 39: wcscpy(name, L"u"); break;
		case 40: wcscpy(name, L"φ"); break;
		case 41: wcscpy(name, L"a"); break;
		case 42: wcscpy(name, L"cw"); break;
		case 43: wcscpy(name, L"ww"); break;
		case 44: wcscpy(name, L"p"); break;
		case 45: wcscpy(name, L"L"); break;
		case 46: wcscpy(name, L"f"); break;
		case 47: wcscpy(name, L"b"); break;
		case 48: wcscpy(name, L"cr"); break;
		case 49: wcscpy(name, L"wr"); break;
		case 50: wcscpy(name, L"π1"); break;
		case 51: wcscpy(name, L"π2"); break;
		case 52: wcscpy(name, L"π3"); break;
		case 53: wcscpy(name, L"Wr"); break;
		case 54: wcscpy(name, L"delta_χ"); break;
		case 55: wcscpy(name, L"delta_χ2h"); break;
		case 56: wcscpy(name, L"delta_gin"); break;
		case 57: wcscpy(name, L"delta_gfo"); break;
		case 58: wcscpy(name, L"delta_s"); break;
		case 59: wcscpy(name, L"delta_o"); break;
		case 60: wcscpy(name, L"delta_h"); break;
		case 61: wcscpy(name, L"delta_Wi"); break;
		case 62: wcscpy(name, L"delta_Wf"); break;
		case 63: wcscpy(name, L"delta_Ws"); break;
		case 64: wcscpy(name, L"delta_Wo"); break;
		case 65: wcscpy(name, L"delta_bi"); break;
		case 66: wcscpy(name, L"delta_bf"); break;
		case 67: wcscpy(name, L"delta_bs"); break;
		case 68: wcscpy(name, L"delta_bo"); break;
		case 69: wcscpy(name, L"delta_x"); break;
		case 70: wcscpy(name, L"delta_y"); break;
		case 71: wcscpy(name, L"delta_v"); break;
		case 72: wcscpy(name, L"delta_z"); break;
		case 73: wcscpy(name, L"delta_M"); break;
		case 74: wcscpy(name, L"delta_kr"); break;
		case 75: wcscpy(name, L"delta_r"); break;
		case 76: wcscpy(name, L"delta_βr"); break;
		case 77: wcscpy(name, L"delta_kw"); break;
		case 78: wcscpy(name, L"delta_βw"); break;
		case 79: wcscpy(name, L"delta_e"); break;
		case 80: wcscpy(name, L"delta_ν"); break;
		case 81: wcscpy(name, L"delta_gf"); break;
		case 82: wcscpy(name, L"delta_ga"); break;
		case 83: wcscpy(name, L"delta_gw"); break;
		case 84: wcscpy(name, L"delta_ψ"); break;
		case 85: wcscpy(name, L"delta_u"); break;
		case 86: wcscpy(name, L"delta_φ"); break;
		case 87: wcscpy(name, L"delta_a"); break;
		case 88: wcscpy(name, L"delta_cw"); break;
		case 89: wcscpy(name, L"delta_ww"); break;
		case 90: wcscpy(name, L"delta_p"); break;
		case 91: wcscpy(name, L"delta_L"); break;
		case 92: wcscpy(name, L"delta_f"); break;
		case 93: wcscpy(name, L"delta_b"); break;
		case 94: wcscpy(name, L"delta_cr"); break;
		case 95: wcscpy(name, L"delta_wr"); break;
		case 96: wcscpy(name, L"delta_π1"); break;
		case 97: wcscpy(name, L"delta_π2"); break;
		case 98: wcscpy(name, L"delta_π3"); break;
		case 99: wcscpy(name, L"delta_Wr"); break;
		default: name[0] = 0; break;
	}
}

int DNCF::GetFieldDimension(int field_idx){
	switch(field_idx){
	case 0: return 0;
	case 1: return 0;
	case 2: return 0;
	case 3: return 0;
	case 4: return 0;
	case 5: return 0;
	case 6: return 0;
	case 7: return 0;
	case 8: return 2;
	case 9: return 2;
	case 10: return 2;
	case 11: return 2;
	case 12: return 2;
	case 13: return 2;
	case 14: return 2;
	case 15: return 2;
	case 16: return 2;
	case 17: return 2;
	case 18: return 2;
	case 19: return 1;
	case 20: return 1;
	case 21: return 1;
	case 22: return 1;
	case 23: return 2;
	case 24: return 2;
	case 25: return 2;
	case 26: return 2;
	case 27: return 3;
	case 28: return 3;
	case 29: return 3;
	case 30: return 2;
	case 31: return 2;
	case 32: return 1;
	case 33: return 2;
	case 34: return 2;
	case 35: return 2;
	case 36: return 1;
	case 37: return 1;
	case 38: return 2;
	case 39: return 2;
	case 40: return 2;
	case 41: return 2;
	case 42: return 2;
	case 43: return 2;
	case 44: return 2;
	case 45: return 3;
	case 46: return 3;
	case 47: return 3;
	case 48: return 3;
	case 49: return 3;
	case 50: return 2;
	case 51: return 2;
	case 52: return 2;
	case 53: return 3;
	case 54: return 2;
	case 55: return 2;
	case 56: return 2;
	case 57: return 2;
	case 58: return 2;
	case 59: return 2;
	case 60: return 2;
	case 61: return 2;
	case 62: return 2;
	case 63: return 2;
	case 64: return 2;
	case 65: return 1;
	case 66: return 1;
	case 67: return 1;
	case 68: return 1;
	case 69: return 2;
	case 70: return 2;
	case 71: return 2;
	case 72: return 2;
	case 73: return 3;
	case 74: return 3;
	case 75: return 3;
	case 76: return 2;
	case 77: return 2;
	case 78: return 1;
	case 79: return 2;
	case 80: return 2;
	case 81: return 2;
	case 82: return 1;
	case 83: return 1;
	case 84: return 2;
	case 85: return 2;
	case 86: return 2;
	case 87: return 2;
	case 88: return 2;
	case 89: return 2;
	case 90: return 2;
	case 91: return 3;
	case 92: return 3;
	case 93: return 3;
	case 94: return 3;
	case 95: return 3;
	case 96: return 2;
	case 97: return 2;
	case 98: return 2;
	case 99: return 3;
	default: return -1;
	}
}

int* DNCF::GetFieldSize(int field_idx){
	switch(field_idx){
	case 0: return 0;
	case 1: return 0;
	case 2: return 0;
	case 3: return 0;
	case 4: return 0;
	case 5: return 0;
	case 6: return 0;
	case 7: return 0;
	case 8:
		χ_size_[0] = T;
		χ_size_[1] = χl;
		return χ_size_;
	case 9:
		χ2h_size_[0] = T;
		χ2h_size_[1] = χ2hl;
		return χ2h_size_;
	case 10:
		gin_size_[0] = T;
		gin_size_[1] = Y;
		return gin_size_;
	case 11:
		gfo_size_[0] = T;
		gfo_size_[1] = Y;
		return gfo_size_;
	case 12:
		s_size_[0] = T;
		s_size_[1] = Y;
		return s_size_;
	case 13:
		o_size_[0] = T;
		o_size_[1] = Y;
		return o_size_;
	case 14:
		h_size_[0] = T;
		h_size_[1] = Y;
		return h_size_;
	case 15:
		Wi_size_[0] = Y;
		Wi_size_[1] = χ2hl;
		return Wi_size_;
	case 16:
		Wf_size_[0] = Y;
		Wf_size_[1] = χ2hl;
		return Wf_size_;
	case 17:
		Ws_size_[0] = Y;
		Ws_size_[1] = χ2hl;
		return Ws_size_;
	case 18:
		Wo_size_[0] = Y;
		Wo_size_[1] = χ2hl;
		return Wo_size_;
	case 19:
		bi_size_[0] = Y;
		return bi_size_;
	case 20:
		bf_size_[0] = Y;
		return bf_size_;
	case 21:
		bs_size_[0] = Y;
		return bs_size_;
	case 22:
		bo_size_[0] = Y;
		return bo_size_;
	case 23:
		x_size_[0] = T;
		x_size_[1] = X;
		return x_size_;
	case 24:
		y_size_[0] = T;
		y_size_[1] = Y;
		return y_size_;
	case 25:
		v_size_[0] = T;
		v_size_[1] = Y;
		return v_size_;
	case 26:
		z_size_[0] = T;
		z_size_[1] = Y;
		return z_size_;
	case 27:
		M_size_[0] = T;
		M_size_[1] = N;
		M_size_[2] = W;
		return M_size_;
	case 28:
		kr_size_[0] = T;
		kr_size_[1] = R;
		kr_size_[2] = W;
		return kr_size_;
	case 29:
		r_size_[0] = T;
		r_size_[1] = R;
		r_size_[2] = W;
		return r_size_;
	case 30:
		βr_size_[0] = T;
		βr_size_[1] = R;
		return βr_size_;
	case 31:
		kw_size_[0] = T;
		kw_size_[1] = W;
		return kw_size_;
	case 32:
		βw_size_[0] = T;
		return βw_size_;
	case 33:
		e_size_[0] = T;
		e_size_[1] = W;
		return e_size_;
	case 34:
		ν_size_[0] = T;
		ν_size_[1] = W;
		return ν_size_;
	case 35:
		gf_size_[0] = T;
		gf_size_[1] = R;
		return gf_size_;
	case 36:
		ga_size_[0] = T;
		return ga_size_;
	case 37:
		gw_size_[0] = T;
		return gw_size_;
	case 38:
		ψ_size_[0] = T;
		ψ_size_[1] = N;
		return ψ_size_;
	case 39:
		u_size_[0] = T;
		u_size_[1] = N;
		return u_size_;
	case 40:
		φ_size_[0] = T;
		φ_size_[1] = N;
		return φ_size_;
	case 41:
		a_size_[0] = T;
		a_size_[1] = N;
		return a_size_;
	case 42:
		cw_size_[0] = T;
		cw_size_[1] = N;
		return cw_size_;
	case 43:
		ww_size_[0] = T;
		ww_size_[1] = N;
		return ww_size_;
	case 44:
		p_size_[0] = T;
		p_size_[1] = N;
		return p_size_;
	case 45:
		L_size_[0] = T;
		L_size_[1] = N;
		L_size_[2] = N;
		return L_size_;
	case 46:
		f_size_[0] = T;
		f_size_[1] = R;
		f_size_[2] = N;
		return f_size_;
	case 47:
		b_size_[0] = T;
		b_size_[1] = R;
		b_size_[2] = N;
		return b_size_;
	case 48:
		cr_size_[0] = T;
		cr_size_[1] = R;
		cr_size_[2] = N;
		return cr_size_;
	case 49:
		wr_size_[0] = T;
		wr_size_[1] = R;
		wr_size_[2] = N;
		return wr_size_;
	case 50:
		π1_size_[0] = T;
		π1_size_[1] = R;
		return π1_size_;
	case 51:
		π2_size_[0] = T;
		π2_size_[1] = R;
		return π2_size_;
	case 52:
		π3_size_[0] = T;
		π3_size_[1] = R;
		return π3_size_;
	case 53:
		Wr_size_[0] = R;
		Wr_size_[1] = W;
		Wr_size_[2] = Y;
		return Wr_size_;
	case 54:
		delta_χ_size_[0] = T;
		delta_χ_size_[1] = χl;
		return delta_χ_size_;
	case 55:
		delta_χ2h_size_[0] = T;
		delta_χ2h_size_[1] = χ2hl;
		return delta_χ2h_size_;
	case 56:
		delta_gin_size_[0] = T;
		delta_gin_size_[1] = Y;
		return delta_gin_size_;
	case 57:
		delta_gfo_size_[0] = T;
		delta_gfo_size_[1] = Y;
		return delta_gfo_size_;
	case 58:
		delta_s_size_[0] = T;
		delta_s_size_[1] = Y;
		return delta_s_size_;
	case 59:
		delta_o_size_[0] = T;
		delta_o_size_[1] = Y;
		return delta_o_size_;
	case 60:
		delta_h_size_[0] = T;
		delta_h_size_[1] = Y;
		return delta_h_size_;
	case 61:
		delta_Wi_size_[0] = Y;
		delta_Wi_size_[1] = χ2hl;
		return delta_Wi_size_;
	case 62:
		delta_Wf_size_[0] = Y;
		delta_Wf_size_[1] = χ2hl;
		return delta_Wf_size_;
	case 63:
		delta_Ws_size_[0] = Y;
		delta_Ws_size_[1] = χ2hl;
		return delta_Ws_size_;
	case 64:
		delta_Wo_size_[0] = Y;
		delta_Wo_size_[1] = χ2hl;
		return delta_Wo_size_;
	case 65:
		delta_bi_size_[0] = Y;
		return delta_bi_size_;
	case 66:
		delta_bf_size_[0] = Y;
		return delta_bf_size_;
	case 67:
		delta_bs_size_[0] = Y;
		return delta_bs_size_;
	case 68:
		delta_bo_size_[0] = Y;
		return delta_bo_size_;
	case 69:
		delta_x_size_[0] = T;
		delta_x_size_[1] = X;
		return delta_x_size_;
	case 70:
		delta_y_size_[0] = T;
		delta_y_size_[1] = Y;
		return delta_y_size_;
	case 71:
		delta_v_size_[0] = T;
		delta_v_size_[1] = Y;
		return delta_v_size_;
	case 72:
		delta_z_size_[0] = T;
		delta_z_size_[1] = Y;
		return delta_z_size_;
	case 73:
		delta_M_size_[0] = T;
		delta_M_size_[1] = N;
		delta_M_size_[2] = W;
		return delta_M_size_;
	case 74:
		delta_kr_size_[0] = T;
		delta_kr_size_[1] = R;
		delta_kr_size_[2] = W;
		return delta_kr_size_;
	case 75:
		delta_r_size_[0] = T;
		delta_r_size_[1] = R;
		delta_r_size_[2] = W;
		return delta_r_size_;
	case 76:
		delta_βr_size_[0] = T;
		delta_βr_size_[1] = R;
		return delta_βr_size_;
	case 77:
		delta_kw_size_[0] = T;
		delta_kw_size_[1] = W;
		return delta_kw_size_;
	case 78:
		delta_βw_size_[0] = T;
		return delta_βw_size_;
	case 79:
		delta_e_size_[0] = T;
		delta_e_size_[1] = W;
		return delta_e_size_;
	case 80:
		delta_ν_size_[0] = T;
		delta_ν_size_[1] = W;
		return delta_ν_size_;
	case 81:
		delta_gf_size_[0] = T;
		delta_gf_size_[1] = R;
		return delta_gf_size_;
	case 82:
		delta_ga_size_[0] = T;
		return delta_ga_size_;
	case 83:
		delta_gw_size_[0] = T;
		return delta_gw_size_;
	case 84:
		delta_ψ_size_[0] = T;
		delta_ψ_size_[1] = N;
		return delta_ψ_size_;
	case 85:
		delta_u_size_[0] = T;
		delta_u_size_[1] = N;
		return delta_u_size_;
	case 86:
		delta_φ_size_[0] = T;
		delta_φ_size_[1] = N;
		return delta_φ_size_;
	case 87:
		delta_a_size_[0] = T;
		delta_a_size_[1] = N;
		return delta_a_size_;
	case 88:
		delta_cw_size_[0] = T;
		delta_cw_size_[1] = N;
		return delta_cw_size_;
	case 89:
		delta_ww_size_[0] = T;
		delta_ww_size_[1] = N;
		return delta_ww_size_;
	case 90:
		delta_p_size_[0] = T;
		delta_p_size_[1] = N;
		return delta_p_size_;
	case 91:
		delta_L_size_[0] = T;
		delta_L_size_[1] = N;
		delta_L_size_[2] = N;
		return delta_L_size_;
	case 92:
		delta_f_size_[0] = T;
		delta_f_size_[1] = R;
		delta_f_size_[2] = N;
		return delta_f_size_;
	case 93:
		delta_b_size_[0] = T;
		delta_b_size_[1] = R;
		delta_b_size_[2] = N;
		return delta_b_size_;
	case 94:
		delta_cr_size_[0] = T;
		delta_cr_size_[1] = R;
		delta_cr_size_[2] = N;
		return delta_cr_size_;
	case 95:
		delta_wr_size_[0] = T;
		delta_wr_size_[1] = R;
		delta_wr_size_[2] = N;
		return delta_wr_size_;
	case 96:
		delta_π1_size_[0] = T;
		delta_π1_size_[1] = R;
		return delta_π1_size_;
	case 97:
		delta_π2_size_[0] = T;
		delta_π2_size_[1] = R;
		return delta_π2_size_;
	case 98:
		delta_π3_size_[0] = T;
		delta_π3_size_[1] = R;
		return delta_π3_size_;
	case 99:
		delta_Wr_size_[0] = R;
		delta_Wr_size_[1] = W;
		delta_Wr_size_[2] = Y;
		return delta_Wr_size_;
	default: return 0;
	}
}

void DNCF::GetFieldValue(int field_idx, void* dst){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(dst, &T, _cnt * sizeof(int)); break;
	case 1: memcpy(dst, &N, _cnt * sizeof(int)); break;
	case 2: memcpy(dst, &W, _cnt * sizeof(int)); break;
	case 3: memcpy(dst, &R, _cnt * sizeof(int)); break;
	case 4: memcpy(dst, &X, _cnt * sizeof(int)); break;
	case 5: memcpy(dst, &Y, _cnt * sizeof(int)); break;
	case 6: memcpy(dst, &χl, _cnt * sizeof(int)); break;
	case 7: memcpy(dst, &χ2hl, _cnt * sizeof(int)); break;
	case 8: memcpy(dst, χ, _cnt * sizeof(float)); break;
	case 9: memcpy(dst, χ2h, _cnt * sizeof(float)); break;
	case 10: memcpy(dst, gin, _cnt * sizeof(float)); break;
	case 11: memcpy(dst, gfo, _cnt * sizeof(float)); break;
	case 12: memcpy(dst, s, _cnt * sizeof(float)); break;
	case 13: memcpy(dst, o, _cnt * sizeof(float)); break;
	case 14: memcpy(dst, h, _cnt * sizeof(float)); break;
	case 15: memcpy(dst, Wi, _cnt * sizeof(float)); break;
	case 16: memcpy(dst, Wf, _cnt * sizeof(float)); break;
	case 17: memcpy(dst, Ws, _cnt * sizeof(float)); break;
	case 18: memcpy(dst, Wo, _cnt * sizeof(float)); break;
	case 19: memcpy(dst, bi, _cnt * sizeof(float)); break;
	case 20: memcpy(dst, bf, _cnt * sizeof(float)); break;
	case 21: memcpy(dst, bs, _cnt * sizeof(float)); break;
	case 22: memcpy(dst, bo, _cnt * sizeof(float)); break;
	case 23: memcpy(dst, x, _cnt * sizeof(float)); break;
	case 24: memcpy(dst, y, _cnt * sizeof(float)); break;
	case 25: memcpy(dst, v, _cnt * sizeof(float)); break;
	case 26: memcpy(dst, z, _cnt * sizeof(float)); break;
	case 27: memcpy(dst, M, _cnt * sizeof(float)); break;
	case 28: memcpy(dst, kr, _cnt * sizeof(float)); break;
	case 29: memcpy(dst, r, _cnt * sizeof(float)); break;
	case 30: memcpy(dst, βr, _cnt * sizeof(float)); break;
	case 31: memcpy(dst, kw, _cnt * sizeof(float)); break;
	case 32: memcpy(dst, βw, _cnt * sizeof(float)); break;
	case 33: memcpy(dst, e, _cnt * sizeof(float)); break;
	case 34: memcpy(dst, ν, _cnt * sizeof(float)); break;
	case 35: memcpy(dst, gf, _cnt * sizeof(float)); break;
	case 36: memcpy(dst, ga, _cnt * sizeof(float)); break;
	case 37: memcpy(dst, gw, _cnt * sizeof(float)); break;
	case 38: memcpy(dst, ψ, _cnt * sizeof(float)); break;
	case 39: memcpy(dst, u, _cnt * sizeof(float)); break;
	case 40: memcpy(dst, φ, _cnt * sizeof(int)); break;
	case 41: memcpy(dst, a, _cnt * sizeof(float)); break;
	case 42: memcpy(dst, cw, _cnt * sizeof(float)); break;
	case 43: memcpy(dst, ww, _cnt * sizeof(float)); break;
	case 44: memcpy(dst, p, _cnt * sizeof(float)); break;
	case 45: memcpy(dst, L, _cnt * sizeof(float)); break;
	case 46: memcpy(dst, f, _cnt * sizeof(float)); break;
	case 47: memcpy(dst, b, _cnt * sizeof(float)); break;
	case 48: memcpy(dst, cr, _cnt * sizeof(float)); break;
	case 49: memcpy(dst, wr, _cnt * sizeof(float)); break;
	case 50: memcpy(dst, π1, _cnt * sizeof(float)); break;
	case 51: memcpy(dst, π2, _cnt * sizeof(float)); break;
	case 52: memcpy(dst, π3, _cnt * sizeof(float)); break;
	case 53: memcpy(dst, Wr, _cnt * sizeof(float)); break;
	case 54: memcpy(dst, delta_χ, _cnt * sizeof(float)); break;
	case 55: memcpy(dst, delta_χ2h, _cnt * sizeof(float)); break;
	case 56: memcpy(dst, delta_gin, _cnt * sizeof(float)); break;
	case 57: memcpy(dst, delta_gfo, _cnt * sizeof(float)); break;
	case 58: memcpy(dst, delta_s, _cnt * sizeof(float)); break;
	case 59: memcpy(dst, delta_o, _cnt * sizeof(float)); break;
	case 60: memcpy(dst, delta_h, _cnt * sizeof(float)); break;
	case 61: memcpy(dst, delta_Wi, _cnt * sizeof(float)); break;
	case 62: memcpy(dst, delta_Wf, _cnt * sizeof(float)); break;
	case 63: memcpy(dst, delta_Ws, _cnt * sizeof(float)); break;
	case 64: memcpy(dst, delta_Wo, _cnt * sizeof(float)); break;
	case 65: memcpy(dst, delta_bi, _cnt * sizeof(float)); break;
	case 66: memcpy(dst, delta_bf, _cnt * sizeof(float)); break;
	case 67: memcpy(dst, delta_bs, _cnt * sizeof(float)); break;
	case 68: memcpy(dst, delta_bo, _cnt * sizeof(float)); break;
	case 69: memcpy(dst, delta_x, _cnt * sizeof(float)); break;
	case 70: memcpy(dst, delta_y, _cnt * sizeof(float)); break;
	case 71: memcpy(dst, delta_v, _cnt * sizeof(float)); break;
	case 72: memcpy(dst, delta_z, _cnt * sizeof(float)); break;
	case 73: memcpy(dst, delta_M, _cnt * sizeof(float)); break;
	case 74: memcpy(dst, delta_kr, _cnt * sizeof(float)); break;
	case 75: memcpy(dst, delta_r, _cnt * sizeof(float)); break;
	case 76: memcpy(dst, delta_βr, _cnt * sizeof(float)); break;
	case 77: memcpy(dst, delta_kw, _cnt * sizeof(float)); break;
	case 78: memcpy(dst, delta_βw, _cnt * sizeof(float)); break;
	case 79: memcpy(dst, delta_e, _cnt * sizeof(float)); break;
	case 80: memcpy(dst, delta_ν, _cnt * sizeof(float)); break;
	case 81: memcpy(dst, delta_gf, _cnt * sizeof(float)); break;
	case 82: memcpy(dst, delta_ga, _cnt * sizeof(float)); break;
	case 83: memcpy(dst, delta_gw, _cnt * sizeof(float)); break;
	case 84: memcpy(dst, delta_ψ, _cnt * sizeof(float)); break;
	case 85: memcpy(dst, delta_u, _cnt * sizeof(float)); break;
	case 86: memcpy(dst, delta_φ, _cnt * sizeof(int)); break;
	case 87: memcpy(dst, delta_a, _cnt * sizeof(float)); break;
	case 88: memcpy(dst, delta_cw, _cnt * sizeof(float)); break;
	case 89: memcpy(dst, delta_ww, _cnt * sizeof(float)); break;
	case 90: memcpy(dst, delta_p, _cnt * sizeof(float)); break;
	case 91: memcpy(dst, delta_L, _cnt * sizeof(float)); break;
	case 92: memcpy(dst, delta_f, _cnt * sizeof(float)); break;
	case 93: memcpy(dst, delta_b, _cnt * sizeof(float)); break;
	case 94: memcpy(dst, delta_cr, _cnt * sizeof(float)); break;
	case 95: memcpy(dst, delta_wr, _cnt * sizeof(float)); break;
	case 96: memcpy(dst, delta_π1, _cnt * sizeof(float)); break;
	case 97: memcpy(dst, delta_π2, _cnt * sizeof(float)); break;
	case 98: memcpy(dst, delta_π3, _cnt * sizeof(float)); break;
	case 99: memcpy(dst, delta_Wr, _cnt * sizeof(float)); break;
	}
}

void DNCF::SetFieldValue(int field_idx, void* src){
	int _cnt = GetFieldElementCount(field_idx);
	switch(field_idx){
	case 0: memcpy(&T, src, _cnt * sizeof(int)); break;
	case 1: memcpy(&N, src, _cnt * sizeof(int)); break;
	case 2: memcpy(&W, src, _cnt * sizeof(int)); break;
	case 3: memcpy(&R, src, _cnt * sizeof(int)); break;
	case 4: memcpy(&X, src, _cnt * sizeof(int)); break;
	case 5: memcpy(&Y, src, _cnt * sizeof(int)); break;
	case 6: memcpy(&χl, src, _cnt * sizeof(int)); break;
	case 7: memcpy(&χ2hl, src, _cnt * sizeof(int)); break;
	case 8: memcpy(χ, src, _cnt * sizeof(float)); break;
	case 9: memcpy(χ2h, src, _cnt * sizeof(float)); break;
	case 10: memcpy(gin, src, _cnt * sizeof(float)); break;
	case 11: memcpy(gfo, src, _cnt * sizeof(float)); break;
	case 12: memcpy(s, src, _cnt * sizeof(float)); break;
	case 13: memcpy(o, src, _cnt * sizeof(float)); break;
	case 14: memcpy(h, src, _cnt * sizeof(float)); break;
	case 15: memcpy(Wi, src, _cnt * sizeof(float)); break;
	case 16: memcpy(Wf, src, _cnt * sizeof(float)); break;
	case 17: memcpy(Ws, src, _cnt * sizeof(float)); break;
	case 18: memcpy(Wo, src, _cnt * sizeof(float)); break;
	case 19: memcpy(bi, src, _cnt * sizeof(float)); break;
	case 20: memcpy(bf, src, _cnt * sizeof(float)); break;
	case 21: memcpy(bs, src, _cnt * sizeof(float)); break;
	case 22: memcpy(bo, src, _cnt * sizeof(float)); break;
	case 23: memcpy(x, src, _cnt * sizeof(float)); break;
	case 24: memcpy(y, src, _cnt * sizeof(float)); break;
	case 25: memcpy(v, src, _cnt * sizeof(float)); break;
	case 26: memcpy(z, src, _cnt * sizeof(float)); break;
	case 27: memcpy(M, src, _cnt * sizeof(float)); break;
	case 28: memcpy(kr, src, _cnt * sizeof(float)); break;
	case 29: memcpy(r, src, _cnt * sizeof(float)); break;
	case 30: memcpy(βr, src, _cnt * sizeof(float)); break;
	case 31: memcpy(kw, src, _cnt * sizeof(float)); break;
	case 32: memcpy(βw, src, _cnt * sizeof(float)); break;
	case 33: memcpy(e, src, _cnt * sizeof(float)); break;
	case 34: memcpy(ν, src, _cnt * sizeof(float)); break;
	case 35: memcpy(gf, src, _cnt * sizeof(float)); break;
	case 36: memcpy(ga, src, _cnt * sizeof(float)); break;
	case 37: memcpy(gw, src, _cnt * sizeof(float)); break;
	case 38: memcpy(ψ, src, _cnt * sizeof(float)); break;
	case 39: memcpy(u, src, _cnt * sizeof(float)); break;
	case 40: memcpy(φ, src, _cnt * sizeof(int)); break;
	case 41: memcpy(a, src, _cnt * sizeof(float)); break;
	case 42: memcpy(cw, src, _cnt * sizeof(float)); break;
	case 43: memcpy(ww, src, _cnt * sizeof(float)); break;
	case 44: memcpy(p, src, _cnt * sizeof(float)); break;
	case 45: memcpy(L, src, _cnt * sizeof(float)); break;
	case 46: memcpy(f, src, _cnt * sizeof(float)); break;
	case 47: memcpy(b, src, _cnt * sizeof(float)); break;
	case 48: memcpy(cr, src, _cnt * sizeof(float)); break;
	case 49: memcpy(wr, src, _cnt * sizeof(float)); break;
	case 50: memcpy(π1, src, _cnt * sizeof(float)); break;
	case 51: memcpy(π2, src, _cnt * sizeof(float)); break;
	case 52: memcpy(π3, src, _cnt * sizeof(float)); break;
	case 53: memcpy(Wr, src, _cnt * sizeof(float)); break;
	case 54: memcpy(delta_χ, src, _cnt * sizeof(float)); break;
	case 55: memcpy(delta_χ2h, src, _cnt * sizeof(float)); break;
	case 56: memcpy(delta_gin, src, _cnt * sizeof(float)); break;
	case 57: memcpy(delta_gfo, src, _cnt * sizeof(float)); break;
	case 58: memcpy(delta_s, src, _cnt * sizeof(float)); break;
	case 59: memcpy(delta_o, src, _cnt * sizeof(float)); break;
	case 60: memcpy(delta_h, src, _cnt * sizeof(float)); break;
	case 61: memcpy(delta_Wi, src, _cnt * sizeof(float)); break;
	case 62: memcpy(delta_Wf, src, _cnt * sizeof(float)); break;
	case 63: memcpy(delta_Ws, src, _cnt * sizeof(float)); break;
	case 64: memcpy(delta_Wo, src, _cnt * sizeof(float)); break;
	case 65: memcpy(delta_bi, src, _cnt * sizeof(float)); break;
	case 66: memcpy(delta_bf, src, _cnt * sizeof(float)); break;
	case 67: memcpy(delta_bs, src, _cnt * sizeof(float)); break;
	case 68: memcpy(delta_bo, src, _cnt * sizeof(float)); break;
	case 69: memcpy(delta_x, src, _cnt * sizeof(float)); break;
	case 70: memcpy(delta_y, src, _cnt * sizeof(float)); break;
	case 71: memcpy(delta_v, src, _cnt * sizeof(float)); break;
	case 72: memcpy(delta_z, src, _cnt * sizeof(float)); break;
	case 73: memcpy(delta_M, src, _cnt * sizeof(float)); break;
	case 74: memcpy(delta_kr, src, _cnt * sizeof(float)); break;
	case 75: memcpy(delta_r, src, _cnt * sizeof(float)); break;
	case 76: memcpy(delta_βr, src, _cnt * sizeof(float)); break;
	case 77: memcpy(delta_kw, src, _cnt * sizeof(float)); break;
	case 78: memcpy(delta_βw, src, _cnt * sizeof(float)); break;
	case 79: memcpy(delta_e, src, _cnt * sizeof(float)); break;
	case 80: memcpy(delta_ν, src, _cnt * sizeof(float)); break;
	case 81: memcpy(delta_gf, src, _cnt * sizeof(float)); break;
	case 82: memcpy(delta_ga, src, _cnt * sizeof(float)); break;
	case 83: memcpy(delta_gw, src, _cnt * sizeof(float)); break;
	case 84: memcpy(delta_ψ, src, _cnt * sizeof(float)); break;
	case 85: memcpy(delta_u, src, _cnt * sizeof(float)); break;
	case 86: memcpy(delta_φ, src, _cnt * sizeof(int)); break;
	case 87: memcpy(delta_a, src, _cnt * sizeof(float)); break;
	case 88: memcpy(delta_cw, src, _cnt * sizeof(float)); break;
	case 89: memcpy(delta_ww, src, _cnt * sizeof(float)); break;
	case 90: memcpy(delta_p, src, _cnt * sizeof(float)); break;
	case 91: memcpy(delta_L, src, _cnt * sizeof(float)); break;
	case 92: memcpy(delta_f, src, _cnt * sizeof(float)); break;
	case 93: memcpy(delta_b, src, _cnt * sizeof(float)); break;
	case 94: memcpy(delta_cr, src, _cnt * sizeof(float)); break;
	case 95: memcpy(delta_wr, src, _cnt * sizeof(float)); break;
	case 96: memcpy(delta_π1, src, _cnt * sizeof(float)); break;
	case 97: memcpy(delta_π2, src, _cnt * sizeof(float)); break;
	case 98: memcpy(delta_π3, src, _cnt * sizeof(float)); break;
	case 99: memcpy(delta_Wr, src, _cnt * sizeof(float)); break;
	}
}

DNCF::DNCF(){

    // χl = X + R * W
    χl = X + R * W;

	SetNormalRand(χ, T * χl);

    // χ2hl = χl + Y + Y
    χ2hl = χl + Y + Y;

	SetNormalRand(χ2h, T * χ2hl);
	SetNormalRand(Wi, Y * χ2hl);
	SetNormalRand(Wf, Y * χ2hl);
	SetNormalRand(Ws, Y * χ2hl);
	SetNormalRand(Wo, Y * χ2hl);
	SetNormalRand(bi, Y);
	SetNormalRand(bf, Y);
	SetNormalRand(bs, Y);
	SetNormalRand(bo, Y);
	SetNormalRand(v, T * Y);
	SetNormalRand(z, T * Y);
	SetNormalRand(kr, T * R * W);
	SetNormalRand(βr, T * R);
	SetNormalRand(kw, T * W);
	SetNormalRand(βw, T);
	SetNormalRand(e, T * W);
	SetNormalRand(ν, T * W);
	SetNormalRand(gf, T * R);
	SetNormalRand(ga, T);
	SetNormalRand(gw, T);
	SetNormalRand(a, T * N);
	SetNormalRand(π1, T * R);
	SetNormalRand(π2, T * R);
	SetNormalRand(π3, T * R);
	SetNormalRand(Wr, R * W * Y);
}

DNCF::~DNCF(){
	Free();
	_Free(χ);
	_Free(χ2h);
	_Free(Wi);
	_Free(Wf);
	_Free(Ws);
	_Free(Wo);
	_Free(bi);
	_Free(bf);
	_Free(bs);
	_Free(bo);
	_Free(v);
	_Free(z);
	_Free(kr);
	_Free(βr);
	_Free(kw);
	_Free(βw);
	_Free(e);
	_Free(ν);
	_Free(gf);
	_Free(ga);
	_Free(gw);
	_Free(a);
	_Free(π1);
	_Free(π2);
	_Free(π3);
	_Free(Wr);

}

void DNCF::Allocate(){
	_chk(_Malloc(gin, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(gfo, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(s, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(o, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(h, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(y, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(M, BatchSize * T * N * W * sizeof(float))); 
	_chk(_Malloc(r, BatchSize * T * R * W * sizeof(float))); 
	_chk(_Malloc(ψ, BatchSize * T * N * sizeof(float))); 
	_chk(_Malloc(u, BatchSize * T * N * sizeof(float))); 
	_chk(_Malloc(φ, BatchSize * T * N * sizeof(int))); 
	_chk(_Malloc(cw, BatchSize * T * N * sizeof(float))); 
	_chk(_Malloc(ww, BatchSize * T * N * sizeof(float))); 
	_chk(_Malloc(p, BatchSize * T * N * sizeof(float))); 
	_chk(_Malloc(L, BatchSize * T * N * N * sizeof(float))); 
	_chk(_Malloc(f, BatchSize * T * R * N * sizeof(float))); 
	_chk(_Malloc(b, BatchSize * T * R * N * sizeof(float))); 
	_chk(_Malloc(cr, BatchSize * T * R * N * sizeof(float))); 
	_chk(_Malloc(wr, BatchSize * T * R * N * sizeof(float))); 
	_chk(_Malloc(delta_χ, BatchSize * T * χl * sizeof(float))); 
	_chk(_Malloc(delta_χ2h, BatchSize * T * χ2hl * sizeof(float))); 
	_chk(_Malloc(delta_gin, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(delta_gfo, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(delta_s, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(delta_o, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(delta_h, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(delta_Wi, BatchSize * Y * χ2hl * sizeof(float))); 
	_chk(_Malloc(delta_Wf, BatchSize * Y * χ2hl * sizeof(float))); 
	_chk(_Malloc(delta_Ws, BatchSize * Y * χ2hl * sizeof(float))); 
	_chk(_Malloc(delta_Wo, BatchSize * Y * χ2hl * sizeof(float))); 
	_chk(_Malloc(delta_bi, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(delta_bf, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(delta_bs, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(delta_bo, BatchSize * Y * sizeof(float))); 
	_chk(_Malloc(delta_x, BatchSize * T * X * sizeof(float))); 
	_chk(_Malloc(delta_v, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(delta_z, BatchSize * T * Y * sizeof(float))); 
	_chk(_Malloc(delta_M, BatchSize * T * N * W * sizeof(float))); 
	_chk(_Malloc(delta_kr, BatchSize * T * R * W * sizeof(float))); 
	_chk(_Malloc(delta_r, BatchSize * T * R * W * sizeof(float))); 
	_chk(_Malloc(delta_βr, BatchSize * T * R * sizeof(float))); 
	_chk(_Malloc(delta_kw, BatchSize * T * W * sizeof(float))); 
	_chk(_Malloc(delta_βw, BatchSize * T * sizeof(float))); 
	_chk(_Malloc(delta_e, BatchSize * T * W * sizeof(float))); 
	_chk(_Malloc(delta_ν, BatchSize * T * W * sizeof(float))); 
	_chk(_Malloc(delta_gf, BatchSize * T * R * sizeof(float))); 
	_chk(_Malloc(delta_ga, BatchSize * T * sizeof(float))); 
	_chk(_Malloc(delta_gw, BatchSize * T * sizeof(float))); 
	_chk(_Malloc(delta_ψ, BatchSize * T * N * sizeof(float))); 
	_chk(_Malloc(delta_u, BatchSize * T * N * sizeof(float))); 
	_chk(_Malloc(delta_φ, BatchSize * T * N * sizeof(int))); 
	_chk(_Malloc(delta_a, BatchSize * T * N * sizeof(float))); 
	_chk(_Malloc(delta_cw, BatchSize * T * N * sizeof(float))); 
	_chk(_Malloc(delta_ww, BatchSize * T * N * sizeof(float))); 
	_chk(_Malloc(delta_p, BatchSize * T * N * sizeof(float))); 
	_chk(_Malloc(delta_L, BatchSize * T * N * N * sizeof(float))); 
	_chk(_Malloc(delta_f, BatchSize * T * R * N * sizeof(float))); 
	_chk(_Malloc(delta_b, BatchSize * T * R * N * sizeof(float))); 
	_chk(_Malloc(delta_cr, BatchSize * T * R * N * sizeof(float))); 
	_chk(_Malloc(delta_wr, BatchSize * T * R * N * sizeof(float))); 
	_chk(_Malloc(delta_π1, BatchSize * T * R * sizeof(float))); 
	_chk(_Malloc(delta_π2, BatchSize * T * R * sizeof(float))); 
	_chk(_Malloc(delta_π3, BatchSize * T * R * sizeof(float))); 
	_chk(_Malloc(delta_Wr, BatchSize * R * W * Y * sizeof(float))); 
}

void DNCF::Free(){
	_chk(_Free(gin)); 
	_chk(_Free(gfo)); 
	_chk(_Free(s)); 
	_chk(_Free(o)); 
	_chk(_Free(h)); 
	_chk(_Free(y)); 
	_chk(_Free(M)); 
	_chk(_Free(r)); 
	_chk(_Free(ψ)); 
	_chk(_Free(u)); 
	_chk(_Free(φ)); 
	_chk(_Free(cw)); 
	_chk(_Free(ww)); 
	_chk(_Free(p)); 
	_chk(_Free(L)); 
	_chk(_Free(f)); 
	_chk(_Free(b)); 
	_chk(_Free(cr)); 
	_chk(_Free(wr)); 
	_chk(_Free(delta_χ)); 
	_chk(_Free(delta_χ2h)); 
	_chk(_Free(delta_gin)); 
	_chk(_Free(delta_gfo)); 
	_chk(_Free(delta_s)); 
	_chk(_Free(delta_o)); 
	_chk(_Free(delta_h)); 
	_chk(_Free(delta_Wi)); 
	_chk(_Free(delta_Wf)); 
	_chk(_Free(delta_Ws)); 
	_chk(_Free(delta_Wo)); 
	_chk(_Free(delta_bi)); 
	_chk(_Free(delta_bf)); 
	_chk(_Free(delta_bs)); 
	_chk(_Free(delta_bo)); 
	_chk(_Free(delta_x)); 
	_chk(_Free(delta_v)); 
	_chk(_Free(delta_z)); 
	_chk(_Free(delta_M)); 
	_chk(_Free(delta_kr)); 
	_chk(_Free(delta_r)); 
	_chk(_Free(delta_βr)); 
	_chk(_Free(delta_kw)); 
	_chk(_Free(delta_βw)); 
	_chk(_Free(delta_e)); 
	_chk(_Free(delta_ν)); 
	_chk(_Free(delta_gf)); 
	_chk(_Free(delta_ga)); 
	_chk(_Free(delta_gw)); 
	_chk(_Free(delta_ψ)); 
	_chk(_Free(delta_u)); 
	_chk(_Free(delta_φ)); 
	_chk(_Free(delta_a)); 
	_chk(_Free(delta_cw)); 
	_chk(_Free(delta_ww)); 
	_chk(_Free(delta_p)); 
	_chk(_Free(delta_L)); 
	_chk(_Free(delta_f)); 
	_chk(_Free(delta_b)); 
	_chk(_Free(delta_cr)); 
	_chk(_Free(delta_wr)); 
	_chk(_Free(delta_π1)); 
	_chk(_Free(delta_π2)); 
	_chk(_Free(delta_π3)); 
	_chk(_Free(delta_Wr)); 
}
void DNCF::Forward(){
#pragma omp parallel for
        for (int iy = 0; iy < Y; iy++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // gin[t, iy] = sigmoid((from ix in Range(χ2hl) select Wi[iy, ix] * χ2h[t, ix]).Sum() + bi[iy])
                float _wk1 = 0;
                for(int ix = 0; ix < χ2hl; ix++){
                    _wk1 += Wi[χ2hl * iy + ix] * χ2h[χ2hl * t + ix];
                }
                gin[(Y * t + iy) * BatchSize + _batch_idx] = sigmoid(_wk1 + bi[iy]);

            }
        }
#pragma omp parallel for
        for (int iy = 0; iy < Y; iy++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // gfo[t, iy] = sigmoid((from ix in Range(χ2hl) select Wf[iy, ix] * χ2h[t, ix]).Sum() + bf[iy])
                float _wk2 = 0;
                for(int ix = 0; ix < χ2hl; ix++){
                    _wk2 += Wf[χ2hl * iy + ix] * χ2h[χ2hl * t + ix];
                }
                gfo[(Y * t + iy) * BatchSize + _batch_idx] = sigmoid(_wk2 + bf[iy]);

            }
        }
#pragma omp parallel for
        for (int iy = 0; iy < Y; iy++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // s[t, iy] = gfo[t, iy] * s[t -1, iy] + gin[t, iy] * tanh((from ix in Range(χ2hl) select Ws[iy, ix] * χ2h[t, ix]).Sum() + bs[iy])
                float _wk3 = 0;
                for(int ix = 0; ix < χ2hl; ix++){
                    _wk3 += Ws[χ2hl * iy + ix] * χ2h[χ2hl * t + ix];
                }
                s[(Y * t + iy) * BatchSize + _batch_idx] = gfo[(Y * t + iy) * BatchSize + _batch_idx] * (0 <= t - 1 ? s[(Y * (t -1) + iy) * BatchSize + _batch_idx] : 0) + gin[(Y * t + iy) * BatchSize + _batch_idx] * tanh(_wk3 + bs[iy]);

            }
        }
#pragma omp parallel for
        for (int iy = 0; iy < Y; iy++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // o[t, iy] = sigmoid((from ix in Range(χ2hl) select Wo[iy, ix] * χ2h[t, ix]).Sum() + bo[iy])
                float _wk4 = 0;
                for(int ix = 0; ix < χ2hl; ix++){
                    _wk4 += Wo[χ2hl * iy + ix] * χ2h[χ2hl * t + ix];
                }
                o[(Y * t + iy) * BatchSize + _batch_idx] = sigmoid(_wk4 + bo[iy]);

            }
        }
#pragma omp parallel for
        for (int iy = 0; iy < Y; iy++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // h[t, iy] = o[t, iy] * tanh(s[t, iy])
                h[(Y * t + iy) * BatchSize + _batch_idx] = o[(Y * t + iy) * BatchSize + _batch_idx] * tanh(s[(Y * t + iy) * BatchSize + _batch_idx]);

            }
        }
#pragma omp parallel for
        for (int n = 0; n < N; n++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // ψ[t, n] = (from ir in Range(R) select 1 - gf[t, ir] * wr[t -1, ir, n]).Prod()
                double _wk5 = 1;
                for(int ir = 0; ir < R; ir++){
                    _wk5 *= 1 - gf[R * t + ir] * (0 <= t - 1 ? wr[(N * (R * (t -1) + ir) + n) * BatchSize + _batch_idx] : 0);
                }
                ψ[(N * t + n) * BatchSize + _batch_idx] = _wk5;

            }
        }
#pragma omp parallel for
        for (int n = 0; n < N; n++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // u[t, n] = ((u[t -1, n] + ww[t -1, n]) - u[t -1, n] * ww[t -1, n]) * ψ[t, n]
                u[(N * t + n) * BatchSize + _batch_idx] = (((0 <= t - 1 ? u[(N * (t -1) + n) * BatchSize + _batch_idx] : 0) + (0 <= t - 1 ? ww[(N * (t -1) + n) * BatchSize + _batch_idx] : 0)) - (0 <= t - 1 ? u[(N * (t -1) + n) * BatchSize + _batch_idx] : 0) * (0 <= t - 1 ? ww[(N * (t -1) + n) * BatchSize + _batch_idx] : 0)) * ψ[(N * t + n) * BatchSize + _batch_idx];

            }
        }
#pragma omp parallel for
        for (int n = 0; n < N; n++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // φ[t, n] = 0
                φ[(N * t + n) * BatchSize + _batch_idx] = 0;

            }
        }
#pragma omp parallel for
        for (int iy = 0; iy < Y; iy++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // y[t, iy] = 0
                y[(Y * t + iy) * BatchSize + _batch_idx] = 0;

            }
        }
}
void DNCF::Backward(){
#pragma omp parallel for
        for (int iy = 0; iy < Y; iy++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_s[t, iy] = delta_s[t + 1, iy] * gfo[t + 1, iy] + delta_h[t, iy] * o[t, iy] * tanh_prime(s[t, iy])
                delta_s[(Y * t + iy) * BatchSize + _batch_idx] = (t + 1 < T ? delta_s[(Y * (t + 1) + iy) * BatchSize + _batch_idx] : 0) * (t + 1 < T ? gfo[(Y * (t + 1) + iy) * BatchSize + _batch_idx] : 0) + delta_h[(Y * t + iy) * BatchSize + _batch_idx] * o[(Y * t + iy) * BatchSize + _batch_idx] * tanh_prime(s[(Y * t + iy) * BatchSize + _batch_idx]);

            }
        }
#pragma omp parallel for
        for (int iy = 0; iy < Y; iy++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_gin[t, iy] = delta_s[t, iy] * tanh((from ix in Range(χ2hl) select Ws[iy, ix] * χ2h[t, ix]).Sum() + bs[iy])
                float _wk6 = 0;
                for(int ix = 0; ix < χ2hl; ix++){
                    _wk6 += Ws[χ2hl * iy + ix] * χ2h[χ2hl * t + ix];
                }
                delta_gin[(Y * t + iy) * BatchSize + _batch_idx] = delta_s[(Y * t + iy) * BatchSize + _batch_idx] * tanh(_wk6 + bs[iy]);

            }
        }
#pragma omp parallel for
        for (int iy = 0; iy < Y; iy++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_gfo[t, iy] = delta_s[t, iy] * s[t -1, iy]
                delta_gfo[(Y * t + iy) * BatchSize + _batch_idx] = delta_s[(Y * t + iy) * BatchSize + _batch_idx] * (0 <= t - 1 ? s[(Y * (t -1) + iy) * BatchSize + _batch_idx] : 0);

            }
        }
#pragma omp parallel for
        for (int iy = 0; iy < Y; iy++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_o[t, iy] = delta_h[t, iy] * tanh(s[t, iy])
                delta_o[(Y * t + iy) * BatchSize + _batch_idx] = delta_h[(Y * t + iy) * BatchSize + _batch_idx] * tanh(s[(Y * t + iy) * BatchSize + _batch_idx]);

            }
        }
#pragma omp parallel for
        for (int i_ix = 0; i_ix < χ2hl; i_ix++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_χ2h[t, i_ix] = (from iy in Range(Y) select delta_gin[t, iy] * sigmoid_prime(Wi[iy, i_ix] * χ2h[t, i_ix] + bi[iy]) * Wi[iy, i_ix]).Sum() + (from iy in Range(Y) select delta_gfo[t, iy] * sigmoid_prime(Wf[iy, i_ix] * χ2h[t, i_ix] + bf[iy]) * Wf[iy, i_ix]).Sum() + (from iy in Range(Y) select delta_s[t, iy] * gin[t, iy] * tanh_prime(Ws[iy, i_ix] * χ2h[t, i_ix] + bs[iy]) * Ws[iy, i_ix]).Sum() + (from iy in Range(Y) select delta_o[t, iy] * sigmoid_prime(Wo[iy, i_ix] * χ2h[t, i_ix] + bo[iy]) * Wo[iy, i_ix]).Sum()
                float _wk7 = 0;
                for(int iy = 0; iy < Y; iy++){
                    _wk7 += delta_gin[(Y * t + iy) * BatchSize + _batch_idx] * sigmoid_prime(Wi[χ2hl * iy + i_ix] * χ2h[χ2hl * t + i_ix] + bi[iy]) * Wi[χ2hl * iy + i_ix];
                }
                float _wk8 = 0;
                for(int iy = 0; iy < Y; iy++){
                    _wk8 += delta_gfo[(Y * t + iy) * BatchSize + _batch_idx] * sigmoid_prime(Wf[χ2hl * iy + i_ix] * χ2h[χ2hl * t + i_ix] + bf[iy]) * Wf[χ2hl * iy + i_ix];
                }
                float _wk9 = 0;
                for(int iy = 0; iy < Y; iy++){
                    _wk9 += delta_s[(Y * t + iy) * BatchSize + _batch_idx] * gin[(Y * t + iy) * BatchSize + _batch_idx] * tanh_prime(Ws[χ2hl * iy + i_ix] * χ2h[χ2hl * t + i_ix] + bs[iy]) * Ws[χ2hl * iy + i_ix];
                }
                float _wk10 = 0;
                for(int iy = 0; iy < Y; iy++){
                    _wk10 += delta_o[(Y * t + iy) * BatchSize + _batch_idx] * sigmoid_prime(Wo[χ2hl * iy + i_ix] * χ2h[χ2hl * t + i_ix] + bo[iy]) * Wo[χ2hl * iy + i_ix];
                }
                delta_χ2h[(χ2hl * t + i_ix) * BatchSize + _batch_idx] = _wk7 + _wk8 + _wk9 + _wk10;

            }
        }
#pragma omp parallel for
    for (int iy = 0; iy < Y; iy++) {
        for (int i_ix = 0; i_ix < χ2hl; i_ix++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_Wi[iy, i_ix] = (from t in Range(T) select delta_gin[t, iy] * sigmoid_prime(Wi[iy, i_ix] * χ2h[t, i_ix] + bi[iy]) * χ2h[t, i_ix]).Sum()
                float _wk11 = 0;
                for(int t = 0; t < T; t++){
                    _wk11 += delta_gin[(Y * t + iy) * BatchSize + _batch_idx] * sigmoid_prime(Wi[χ2hl * iy + i_ix] * χ2h[χ2hl * t + i_ix] + bi[iy]) * χ2h[χ2hl * t + i_ix];
                }
                delta_Wi[(χ2hl * iy + i_ix) * BatchSize + _batch_idx] = _wk11;

            }
        }
    }
#pragma omp parallel for
    for (int iy = 0; iy < Y; iy++) {
        for (int i_ix = 0; i_ix < χ2hl; i_ix++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_Wf[iy, i_ix] = (from t in Range(T) select delta_gfo[t, iy] * sigmoid_prime(Wf[iy, i_ix] * χ2h[t, i_ix] + bf[iy]) * χ2h[t, i_ix]).Sum()
                float _wk12 = 0;
                for(int t = 0; t < T; t++){
                    _wk12 += delta_gfo[(Y * t + iy) * BatchSize + _batch_idx] * sigmoid_prime(Wf[χ2hl * iy + i_ix] * χ2h[χ2hl * t + i_ix] + bf[iy]) * χ2h[χ2hl * t + i_ix];
                }
                delta_Wf[(χ2hl * iy + i_ix) * BatchSize + _batch_idx] = _wk12;

            }
        }
    }
#pragma omp parallel for
    for (int iy = 0; iy < Y; iy++) {
        for (int i_ix = 0; i_ix < χ2hl; i_ix++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_Ws[iy, i_ix] = (from t in Range(T) select delta_s[t, iy] * gin[t, iy] * tanh_prime(Ws[iy, i_ix] * χ2h[t, i_ix] + bs[iy]) * χ2h[t, i_ix]).Sum()
                float _wk13 = 0;
                for(int t = 0; t < T; t++){
                    _wk13 += delta_s[(Y * t + iy) * BatchSize + _batch_idx] * gin[(Y * t + iy) * BatchSize + _batch_idx] * tanh_prime(Ws[χ2hl * iy + i_ix] * χ2h[χ2hl * t + i_ix] + bs[iy]) * χ2h[χ2hl * t + i_ix];
                }
                delta_Ws[(χ2hl * iy + i_ix) * BatchSize + _batch_idx] = _wk13;

            }
        }
    }
#pragma omp parallel for
    for (int iy = 0; iy < Y; iy++) {
        for (int i_ix = 0; i_ix < χ2hl; i_ix++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_Wo[iy, i_ix] = (from t in Range(T) select delta_o[t, iy] * sigmoid_prime(Wo[iy, i_ix] * χ2h[t, i_ix] + bo[iy]) * χ2h[t, i_ix]).Sum()
                float _wk14 = 0;
                for(int t = 0; t < T; t++){
                    _wk14 += delta_o[(Y * t + iy) * BatchSize + _batch_idx] * sigmoid_prime(Wo[χ2hl * iy + i_ix] * χ2h[χ2hl * t + i_ix] + bo[iy]) * χ2h[χ2hl * t + i_ix];
                }
                delta_Wo[(χ2hl * iy + i_ix) * BatchSize + _batch_idx] = _wk14;

            }
        }
    }
#pragma omp parallel for
    for (int iy = 0; iy < Y; iy++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_bi[iy] = (from t in Range(T) select delta_gin[t, iy] * sigmoid_prime((from ix in Range(χ2hl) select Wi[iy, ix] * χ2h[t, ix]).Sum() + bi[iy])).Sum()
            float _wk15 = 0;
            for(int t = 0; t < T; t++){
                _wk15 += delta_gin[(Y * t + iy) * BatchSize + _batch_idx] * sigmoid_prime(_wk16 + bi[iy]);
            }
            float _wk16 = 0;
            for(int ix = 0; ix < χ2hl; ix++){
                _wk16 += Wi[χ2hl * iy + ix] * χ2h[χ2hl * t + ix];
            }
            delta_bi[(iy) * BatchSize + _batch_idx] = _wk15;

        }
    }
#pragma omp parallel for
    for (int iy = 0; iy < Y; iy++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_bf[iy] = (from t in Range(T) select delta_gfo[t, iy] * sigmoid_prime((from ix in Range(χ2hl) select Wf[iy, ix] * χ2h[t, ix]).Sum() + bf[iy])).Sum()
            float _wk17 = 0;
            for(int t = 0; t < T; t++){
                _wk17 += delta_gfo[(Y * t + iy) * BatchSize + _batch_idx] * sigmoid_prime(_wk18 + bf[iy]);
            }
            float _wk18 = 0;
            for(int ix = 0; ix < χ2hl; ix++){
                _wk18 += Wf[χ2hl * iy + ix] * χ2h[χ2hl * t + ix];
            }
            delta_bf[(iy) * BatchSize + _batch_idx] = _wk17;

        }
    }
#pragma omp parallel for
    for (int iy = 0; iy < Y; iy++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_bs[iy] = (from t in Range(T) select delta_s[t, iy] * gin[t, iy] * tanh_prime((from ix in Range(χ2hl) select Ws[iy, ix] * χ2h[t, ix]).Sum() + bs[iy])).Sum()
            float _wk19 = 0;
            for(int t = 0; t < T; t++){
                _wk19 += delta_s[(Y * t + iy) * BatchSize + _batch_idx] * gin[(Y * t + iy) * BatchSize + _batch_idx] * tanh_prime(_wk20 + bs[iy]);
            }
            float _wk20 = 0;
            for(int ix = 0; ix < χ2hl; ix++){
                _wk20 += Ws[χ2hl * iy + ix] * χ2h[χ2hl * t + ix];
            }
            delta_bs[(iy) * BatchSize + _batch_idx] = _wk19;

        }
    }
#pragma omp parallel for
    for (int iy = 0; iy < Y; iy++) {
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_bo[iy] = (from t in Range(T) select delta_o[t, iy] * sigmoid_prime((from ix in Range(χ2hl) select Wo[iy, ix] * χ2h[t, ix]).Sum() + bo[iy])).Sum()
            float _wk21 = 0;
            for(int t = 0; t < T; t++){
                _wk21 += delta_o[(Y * t + iy) * BatchSize + _batch_idx] * sigmoid_prime(_wk22 + bo[iy]);
            }
            float _wk22 = 0;
            for(int ix = 0; ix < χ2hl; ix++){
                _wk22 += Wo[χ2hl * iy + ix] * χ2h[χ2hl * t + ix];
            }
            delta_bo[(iy) * BatchSize + _batch_idx] = _wk21;

        }
    }
#pragma omp parallel for
        for (int n = 0; n < N; n++) {
            for (int iw = 0; iw < W; iw++) {
                for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                    // delta_M[t, n, iw] = delta_M[t + 1, n, iw] * (1 - ww[t + 1, n] * e[t + 1, iw]) + (from ir in Range(R) select delta_r[t, ir, iw] * wr[t, ir, n]).Sum()
                    float _wk23 = 0;
                    for(int ir = 0; ir < R; ir++){
                        _wk23 += delta_r[(W * (R * t + ir) + iw) * BatchSize + _batch_idx] * wr[(N * (R * t + ir) + n) * BatchSize + _batch_idx];
                    }
                    delta_M[(W * (N * t + n) + iw) * BatchSize + _batch_idx] = (t + 1 < T ? delta_M[(W * (N * (t + 1) + n) + iw) * BatchSize + _batch_idx] : 0) * (1 - (t + 1 < T ? ww[(N * (t + 1) + n) * BatchSize + _batch_idx] : 0) * (t + 1 < T ? e[W * (t + 1) + iw] : 0)) + _wk23;

                }
            }
        }
#pragma omp parallel for
        for (int ir = 0; ir < R; ir++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_βr[t, ir] = (from n in Range(N) select 0).Sum()
                int _wk24 = 0;
                for(int n = 0; n < N; n++){
                    _wk24 += 0;
                }
                delta_βr[(R * t + ir) * BatchSize + _batch_idx] = _wk24;

            }
        }
#pragma omp parallel for
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_βw[t] = (from n in Range(N) select 0).Sum()
            int _wk25 = 0;
            for(int n = 0; n < N; n++){
                _wk25 += 0;
            }
            delta_βw[(t) * BatchSize + _batch_idx] = _wk25;

        }
#pragma omp parallel for
        for (int iw = 0; iw < W; iw++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_e[t, iw] = (from n in Range(N) select delta_M[t, n, iw] * M[t -1, n, iw] * ww[t, n]).Sum()
                float _wk26 = 0;
                for(int n = 0; n < N; n++){
                    _wk26 += delta_M[(W * (N * t + n) + iw) * BatchSize + _batch_idx] * (0 <= t - 1 ? M[(W * (N * (t -1) + n) + iw) * BatchSize + _batch_idx] : 0) * ww[(N * t + n) * BatchSize + _batch_idx];
                }
                delta_e[(W * t + iw) * BatchSize + _batch_idx] = _wk26;

            }
        }
#pragma omp parallel for
        for (int iw = 0; iw < W; iw++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_ν[t, iw] = (from n in Range(N) select delta_M[t, n, iw] * ww[t, n]).Sum()
                float _wk27 = 0;
                for(int n = 0; n < N; n++){
                    _wk27 += delta_M[(W * (N * t + n) + iw) * BatchSize + _batch_idx] * ww[(N * t + n) * BatchSize + _batch_idx];
                }
                delta_ν[(W * t + iw) * BatchSize + _batch_idx] = _wk27;

            }
        }
#pragma omp parallel for
        for (int n = 0; n < N; n++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_u[t, n] = 2 * delta_u[t + 1, n] * (1 + ww[t, n]) * ψ[t + 1, n]
                delta_u[(N * t + n) * BatchSize + _batch_idx] = 2 * (t + 1 < T ? delta_u[(N * (t + 1) + n) * BatchSize + _batch_idx] : 0) * (1 + ww[(N * t + n) * BatchSize + _batch_idx]) * (t + 1 < T ? ψ[(N * (t + 1) + n) * BatchSize + _batch_idx] : 0);

            }
        }
#pragma omp parallel for
        for (int n = 0; n < N; n++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_ψ[t, n] = delta_u[t, n] * (u[t -1, n] + ww[t -1, n] - u[t -1, n] * ww[t -1, n])
                delta_ψ[(N * t + n) * BatchSize + _batch_idx] = delta_u[(N * t + n) * BatchSize + _batch_idx] * ((0 <= t - 1 ? u[(N * (t -1) + n) * BatchSize + _batch_idx] : 0) + (0 <= t - 1 ? ww[(N * (t -1) + n) * BatchSize + _batch_idx] : 0) - (0 <= t - 1 ? u[(N * (t -1) + n) * BatchSize + _batch_idx] : 0) * (0 <= t - 1 ? ww[(N * (t -1) + n) * BatchSize + _batch_idx] : 0));

            }
        }
#pragma omp parallel for
        for (int i_ir = 0; i_ir < R; i_ir++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_gf[t, i_ir] = (from n in Range(N) select delta_ψ[t, n] * wr[t -1, i_ir, n]).Sum()
                float _wk28 = 0;
                for(int n = 0; n < N; n++){
                    _wk28 += delta_ψ[(N * t + n) * BatchSize + _batch_idx] * (0 <= t - 1 ? wr[(N * (R * (t -1) + i_ir) + n) * BatchSize + _batch_idx] : 0);
                }
                delta_gf[(R * t + i_ir) * BatchSize + _batch_idx] = _wk28;

            }
        }
#pragma omp parallel for
        for (int n = 0; n < N; n++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_p[t, n] = delta_p[t + 1, n] * (1 - (from i in Range(N) select ww[t + 1, i]).Sum()) + (from n in Range(N) select delta_L[t + 1, n, j] * ww[t + 1, n]).Sum()
                float _wk29 = 0;
                for(int i = 0; i < N; i++){
                    _wk29 += (t + 1 < T ? ww[(N * (t + 1) + i) * BatchSize + _batch_idx] : 0);
                }
                float _wk30 = 0;
                for(int n = 0; n < N; n++){
                    _wk30 += (t + 1 < T ? delta_L[(N * (N * (t + 1) + n) + j) * BatchSize + _batch_idx] : 0) * (t + 1 < T ? ww[(N * (t + 1) + n) * BatchSize + _batch_idx] : 0);
                }
                delta_p[(N * t + n) * BatchSize + _batch_idx] = (t + 1 < T ? delta_p[(N * (t + 1) + n) * BatchSize + _batch_idx] : 0) * (1 - _wk29) + _wk30;

            }
        }
#pragma omp parallel for
        for (int ir = 0; ir < R; ir++) {
            for (int n = 0; n < N; n++) {
                for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                    // delta_wr[t, ir, n] = delta_ψ[t + 1, n] * gf[t + 1, ir] + (from n in Range(N) select delta_f[t + 1, ir, n] * L[t + 1, n, n]).Sum() + (from n in Range(N) select delta_b[t + 1, ir, n] * L[t + 1, n, n]).Sum() + (from iw in Range(W) select delta_r[t, ir, iw] * M[t, n, iw]).Sum()
                    float _wk31 = 0;
                    for(int n = 0; n < N; n++){
                        _wk31 += (t + 1 < T ? delta_f[(N * (R * (t + 1) + ir) + n) * BatchSize + _batch_idx] : 0) * (t + 1 < T ? L[(N * (N * (t + 1) + n) + n) * BatchSize + _batch_idx] : 0);
                    }
                    float _wk32 = 0;
                    for(int n = 0; n < N; n++){
                        _wk32 += (t + 1 < T ? delta_b[(N * (R * (t + 1) + ir) + n) * BatchSize + _batch_idx] : 0) * (t + 1 < T ? L[(N * (N * (t + 1) + n) + n) * BatchSize + _batch_idx] : 0);
                    }
                    float _wk33 = 0;
                    for(int iw = 0; iw < W; iw++){
                        _wk33 += delta_r[(W * (R * t + ir) + iw) * BatchSize + _batch_idx] * M[(W * (N * t + n) + iw) * BatchSize + _batch_idx];
                    }
                    delta_wr[(N * (R * t + ir) + n) * BatchSize + _batch_idx] = (t + 1 < T ? delta_ψ[(N * (t + 1) + n) * BatchSize + _batch_idx] : 0) * (t + 1 < T ? gf[R * (t + 1) + ir] : 0) + _wk31 + _wk32 + _wk33;

                }
            }
        }
#pragma omp parallel for
        for (int ir = 0; ir < R; ir++) {
            for (int n = 0; n < N; n++) {
                for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                    // delta_f[t, ir, n] = delta_wr[t, ir, n] * π3[t, ir]
                    delta_f[(N * (R * t + ir) + n) * BatchSize + _batch_idx] = delta_wr[(N * (R * t + ir) + n) * BatchSize + _batch_idx] * π3[R * t + ir];

                }
            }
        }
#pragma omp parallel for
        for (int ir = 0; ir < R; ir++) {
            for (int n = 0; n < N; n++) {
                for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                    // delta_b[t, ir, n] = delta_wr[t, ir, n] * π1[t, ir]
                    delta_b[(N * (R * t + ir) + n) * BatchSize + _batch_idx] = delta_wr[(N * (R * t + ir) + n) * BatchSize + _batch_idx] * π1[R * t + ir];

                }
            }
        }
#pragma omp parallel for
        for (int n = 0; n < N; n++) {
            for (int j = 0; j < N; j++) {
                for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                    // delta_L[t, n, j] = delta_L[t + 1, n, j] * (1 - ww[t + 1, n] - ww[t + 1, j]) + (from ir in Range(R) select delta_f[t, ir, n] * wr[t -1, ir, j]).Sum() + (from ir in Range(R) select delta_b[t, ir, n] * wr[t -1, ir, n]).Sum()
                    float _wk34 = 0;
                    for(int ir = 0; ir < R; ir++){
                        _wk34 += delta_f[(N * (R * t + ir) + n) * BatchSize + _batch_idx] * (0 <= t - 1 ? wr[(N * (R * (t -1) + ir) + j) * BatchSize + _batch_idx] : 0);
                    }
                    float _wk35 = 0;
                    for(int ir = 0; ir < R; ir++){
                        _wk35 += delta_b[(N * (R * t + ir) + n) * BatchSize + _batch_idx] * (0 <= t - 1 ? wr[(N * (R * (t -1) + ir) + n) * BatchSize + _batch_idx] : 0);
                    }
                    delta_L[(N * (N * t + n) + j) * BatchSize + _batch_idx] = (t + 1 < T ? delta_L[(N * (N * (t + 1) + n) + j) * BatchSize + _batch_idx] : 0) * (1 - (t + 1 < T ? ww[(N * (t + 1) + n) * BatchSize + _batch_idx] : 0) - (t + 1 < T ? ww[(N * (t + 1) + j) * BatchSize + _batch_idx] : 0)) + _wk34 + _wk35;

                }
            }
        }
#pragma omp parallel for
        for (int n = 0; n < N; n++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_ww[t, n] = 2 * delta_u[t + 1, n] * (1 + u[t, n]) * ψ[t + 1, n] + (from iw in Range(W) select delta_M[t, n, iw] * (M[t -1, n, iw] * e[t, iw] + ν[t, iw])).Sum() + (from iw in Range(W) select delta_M[t, n, iw] * (M[t -1, n, iw] * e[t, iw] + ν[t, iw])).Sum() + (from n in Range(N) select delta_p[t, n] * (p[t -1, n] + 1)).Sum() + delta_p[t, n] + (from j in Range(N) select delta_L[t, n, j] * p[t -1, j]).Sum() + (from n in Range(N) select delta_L[t, n, j] * p[t -1, n]).Sum() + (from j in Range(N) select delta_L[t, n, j] * p[t -1, j]).Sum()
                float _wk36 = 0;
                for(int iw = 0; iw < W; iw++){
                    _wk36 += delta_M[(W * (N * t + n) + iw) * BatchSize + _batch_idx] * ((0 <= t - 1 ? M[(W * (N * (t -1) + n) + iw) * BatchSize + _batch_idx] : 0) * e[W * t + iw] + ν[W * t + iw]);
                }
                float _wk37 = 0;
                for(int iw = 0; iw < W; iw++){
                    _wk37 += delta_M[(W * (N * t + n) + iw) * BatchSize + _batch_idx] * ((0 <= t - 1 ? M[(W * (N * (t -1) + n) + iw) * BatchSize + _batch_idx] : 0) * e[W * t + iw] + ν[W * t + iw]);
                }
                float _wk38 = 0;
                for(int n = 0; n < N; n++){
                    _wk38 += delta_p[(N * t + n) * BatchSize + _batch_idx] * ((0 <= t - 1 ? p[(N * (t -1) + n) * BatchSize + _batch_idx] : 0) + 1);
                }
                float _wk39 = 0;
                for(int j = 0; j < N; j++){
                    _wk39 += delta_L[(N * (N * t + n) + j) * BatchSize + _batch_idx] * (0 <= t - 1 ? p[(N * (t -1) + j) * BatchSize + _batch_idx] : 0);
                }
                float _wk40 = 0;
                for(int n = 0; n < N; n++){
                    _wk40 += delta_L[(N * (N * t + n) + j) * BatchSize + _batch_idx] * (0 <= t - 1 ? p[(N * (t -1) + n) * BatchSize + _batch_idx] : 0);
                }
                float _wk41 = 0;
                for(int j = 0; j < N; j++){
                    _wk41 += delta_L[(N * (N * t + n) + j) * BatchSize + _batch_idx] * (0 <= t - 1 ? p[(N * (t -1) + j) * BatchSize + _batch_idx] : 0);
                }
                delta_ww[(N * t + n) * BatchSize + _batch_idx] = 2 * (t + 1 < T ? delta_u[(N * (t + 1) + n) * BatchSize + _batch_idx] : 0) * (1 + u[(N * t + n) * BatchSize + _batch_idx]) * (t + 1 < T ? ψ[(N * (t + 1) + n) * BatchSize + _batch_idx] : 0) + _wk36 + _wk37 + _wk38 + delta_p[(N * t + n) * BatchSize + _batch_idx] + _wk39 + _wk40 + _wk41;

            }
        }
#pragma omp parallel for
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_ga[t] = (from n in Range(N) select delta_ww[t, n] * gw[t] * a[t, n]).Sum() + (from n in Range(N) select delta_ww[t, n] * gw[t] * a[t, n]).Sum()
            float _wk42 = 0;
            for(int n = 0; n < N; n++){
                _wk42 += delta_ww[(N * t + n) * BatchSize + _batch_idx] * gw[t] * a[N * t + n];
            }
            float _wk43 = 0;
            for(int n = 0; n < N; n++){
                _wk43 += delta_ww[(N * t + n) * BatchSize + _batch_idx] * gw[t] * a[N * t + n];
            }
            delta_ga[(t) * BatchSize + _batch_idx] = _wk42 + _wk43;

        }
#pragma omp parallel for
        for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

            // delta_gw[t] = (from n in Range(N) select delta_ww[t, n] * (ga[t] * a[t, n] + (1 - ga[t]) * cw[t, n])).Sum()
            float _wk44 = 0;
            for(int n = 0; n < N; n++){
                _wk44 += delta_ww[(N * t + n) * BatchSize + _batch_idx] * (ga[t] * a[N * t + n] + (1 - ga[t]) * cw[(N * t + n) * BatchSize + _batch_idx]);
            }
            delta_gw[(t) * BatchSize + _batch_idx] = _wk44;

        }
#pragma omp parallel for
        for (int n = 0; n < N; n++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_a[t, n] = delta_ww[t, n] * gw[t] * ga[t]
                delta_a[(N * t + n) * BatchSize + _batch_idx] = delta_ww[(N * t + n) * BatchSize + _batch_idx] * gw[t] * ga[t];

            }
        }
#pragma omp parallel for
        for (int n = 0; n < N; n++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_cw[t, n] = delta_ww[t, n] * gw[t] * (1 - ga[t])
                delta_cw[(N * t + n) * BatchSize + _batch_idx] = delta_ww[(N * t + n) * BatchSize + _batch_idx] * gw[t] * (1 - ga[t]);

            }
        }
#pragma omp parallel for
        for (int ir = 0; ir < R; ir++) {
            for (int n = 0; n < N; n++) {
                for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                    // delta_cr[t, ir, n] = delta_wr[t, ir, n] * π2[t, ir]
                    delta_cr[(N * (R * t + ir) + n) * BatchSize + _batch_idx] = delta_wr[(N * (R * t + ir) + n) * BatchSize + _batch_idx] * π2[R * t + ir];

                }
            }
        }
#pragma omp parallel for
        for (int ir = 0; ir < R; ir++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_π1[t, ir] = (from n in Range(N) select delta_wr[t, ir, n] * b[t, ir, n]).Sum()
                float _wk45 = 0;
                for(int n = 0; n < N; n++){
                    _wk45 += delta_wr[(N * (R * t + ir) + n) * BatchSize + _batch_idx] * b[(N * (R * t + ir) + n) * BatchSize + _batch_idx];
                }
                delta_π1[(R * t + ir) * BatchSize + _batch_idx] = _wk45;

            }
        }
#pragma omp parallel for
        for (int ir = 0; ir < R; ir++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_π2[t, ir] = (from n in Range(N) select delta_wr[t, ir, n] * cr[t, ir, n]).Sum()
                float _wk46 = 0;
                for(int n = 0; n < N; n++){
                    _wk46 += delta_wr[(N * (R * t + ir) + n) * BatchSize + _batch_idx] * cr[(N * (R * t + ir) + n) * BatchSize + _batch_idx];
                }
                delta_π2[(R * t + ir) * BatchSize + _batch_idx] = _wk46;

            }
        }
#pragma omp parallel for
        for (int ir = 0; ir < R; ir++) {
            for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {

                // delta_π3[t, ir] = (from n in Range(N) select delta_wr[t, ir, n] * f[t, ir, n]).Sum()
                float _wk47 = 0;
                for(int n = 0; n < N; n++){
                    _wk47 += delta_wr[(N * (R * t + ir) + n) * BatchSize + _batch_idx] * f[(N * (R * t + ir) + n) * BatchSize + _batch_idx];
                }
                delta_π3[(R * t + ir) * BatchSize + _batch_idx] = _wk47;

            }
        }
}

void DNCF::UpdateParameter_0(){
	int _count = T * χl;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_χ[offset + i];
			}
			χ[_idx] -= LearningRate * sum;
		}
	}
}

void DNCF::UpdateParameter_1(){
	int _count = T * χ2hl;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_χ2h[offset + i];
			}
			χ2h[_idx] -= LearningRate * sum;
		}
	}
}

void DNCF::UpdateParameter_2(){
	int _count = Y * χ2hl;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_Wi[offset + i];
			}
			Wi[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_Wf[offset + i];
			}
			Wf[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_Ws[offset + i];
			}
			Ws[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_Wo[offset + i];
			}
			Wo[_idx] -= LearningRate * sum;
		}
	}
}

void DNCF::UpdateParameter_3(){
	int _count = Y;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_bi[offset + i];
			}
			bi[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_bf[offset + i];
			}
			bf[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_bs[offset + i];
			}
			bs[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_bo[offset + i];
			}
			bo[_idx] -= LearningRate * sum;
		}
	}
}

void DNCF::UpdateParameter_4(){
	int _count = T * Y;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_v[offset + i];
			}
			v[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_z[offset + i];
			}
			z[_idx] -= LearningRate * sum;
		}
	}
}

void DNCF::UpdateParameter_5(){
	int _count = T * R * W;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_kr[offset + i];
			}
			kr[_idx] -= LearningRate * sum;
		}
	}
}

void DNCF::UpdateParameter_6(){
	int _count = T * R;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_βr[offset + i];
			}
			βr[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_gf[offset + i];
			}
			gf[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_π1[offset + i];
			}
			π1[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_π2[offset + i];
			}
			π2[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_π3[offset + i];
			}
			π3[_idx] -= LearningRate * sum;
		}
	}
}

void DNCF::UpdateParameter_7(){
	int _count = T * W;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_kw[offset + i];
			}
			kw[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_e[offset + i];
			}
			e[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_ν[offset + i];
			}
			ν[_idx] -= LearningRate * sum;
		}
	}
}

void DNCF::UpdateParameter_8(){
	int _count = T;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_βw[offset + i];
			}
			βw[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_ga[offset + i];
			}
			ga[_idx] -= LearningRate * sum;
		}
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_gw[offset + i];
			}
			gw[_idx] -= LearningRate * sum;
		}
	}
}

void DNCF::UpdateParameter_9(){
	int _count = T * N;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_a[offset + i];
			}
			a[_idx] -= LearningRate * sum;
		}
	}
}

void DNCF::UpdateParameter_10(){
	int _count = R * W * Y;
#pragma omp parallel for
	for(int _idx = 0; _idx < _count; _idx++) {
		int offset = _idx * BatchSize;
		{
			float sum = 0;
			for (int i = 0; i < BatchSize; i++) {
				sum += delta_Wr[offset + i];
			}
			Wr[_idx] -= LearningRate * sum;
		}
	}
}
void DNCF::UpdateParameter(){
	UpdateParameter_0();
	UpdateParameter_1();
	UpdateParameter_2();
	UpdateParameter_3();
	UpdateParameter_4();
	UpdateParameter_5();
	UpdateParameter_6();
	UpdateParameter_7();
	UpdateParameter_8();
	UpdateParameter_9();
	UpdateParameter_10();
}

extern "C" DllExport Layer* MakeDNCF(){
	return new DNCF();
}
