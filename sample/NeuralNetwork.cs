using System;
using System.Collections.Generic;
using System.Linq;

public abstract class Layer {
    public static float e_;
    public float E;    // 損失関数
    
    public abstract void Forward();

    public int[] Range(int n) {
        return null;
    }

    public float σ(float z) {
        return (float)(1.0 / (1.0 + exp(-z)));
    }

    public float σ_prime(float z) {
        return σ(z) * (1 - σ(z));
    }

    public int[] Indexes(object a) {
        return null;
    }

    public int Dim(object a, int i) {
        return 0;
    }

    public float tanh(float x) {
        return 0;
    }

    public float tanh_prime(float x) {
        return 0;
    }

    public float log(float x) {
        return 0;
    }

    public float pow(float x, float y) {
        return 0;
    }

    public float exp(float x) {
        return 0;
    }

    public float sqrt(float x) {
        return 0;
    }

    public float[] Row(float[,] m, int i) {
        return null;
    }

    public float[] Row(float[,,] a, int t, int i) {
        return null;
    }

    public float[] Col(float[,] m, int j) {
        return null;
    }

    public float[,] Mat(float[,,] a, int i) {
        return null;
    }

    public float oneplus(float x) {
        return (float)(1 + log(1.0f + pow(e_, x)));
    }

    public float softmax(float[] x, int i) {
        return pow(e_, x[i]) / (from xj in x select pow(e_, xj)).Sum();
    }

    public float Length(float[] u) {
        return sqrt((from i in Indexes(u) select u[i] * u[i]).Sum());
    }

    public float Dot(float[] u, float[] v) {
        return (from i in Indexes(u) select u[i] * v[i]).Sum();
    }

    public float C(float[,] M, float[] k, float β, int i) {
        return exp(D(k, Row(M, i)) * β) / (from j in Indexes(M) select exp(D(k, Row(M, j)) * β)).Sum();
    }

    public float D(float[] u, float[] v) {
        return Dot(u, v) / (Length(u) * Length(v));
    }

    public float Prod(object a) {
        return 0;
    }
}

public class FullyConnectedLayer : Layer {
    public int X;
    public int Y;

    public float[] x;
    public float[] y;

    public float[,] w;
    public float[] b;
    public float[] u;

    public FullyConnectedLayer(int x_size, int y_size) {
        X = x_size;
        Y = y_size;

        x = new float[X];
        y = new float[Y];

        w = new float[Y, X];
        b = new float[Y];
        u = new float[Y];
    }

public override void Forward() {
        foreach (int i in Range(Y)) {
            u[i] = (from j in Range(X) select x[j] * w[i, j]).Sum() + b[i];
            y[i] = σ(u[i]);
        }
    }
}

public class ConvolutionalLayer : Layer {
    public int M;   // 行数
    public int N;   // 列数
    public int K;   // フィルター数
    public int H;

    public float[,] x;
    public float[,,] y;

    public float[,,] u;
    public float[,,] h;
    public float[] b;

    public ConvolutionalLayer(int m_size, int n_size, int k_size, int h_size) {
        M = m_size;
        N = n_size;
        K = k_size;
        H = h_size;

        x = new float[M, N];
        u = new float[M - H + 1, N - H + 1, K];
        y = new float[M - H + 1, N - H + 1, K];

        h = new float[H, H, K];
        b = new float[K];
    }

public override void Forward() {
        foreach (int i in Range(M - H + 1)) {
            foreach (int j in Range(N - H + 1)) {
                foreach (int k in Range(K)) {
                    u[i, j, k] = (from p in Range(H) from q in Range(H) select x[i + p, j + q] * h[p, q, k]).Sum() + b[k];
                    y[i, j, k] = σ(u[i, j, k]);
                }
            }
        }
    }
}

public class MaxPoolingLayer : Layer {
    public int M;   // 入力行数
    public int N;   // 入力列数
    public int K;   // フィルター数
    public int H;   // フィルターの幅
    public int MH;  // 出力行数
    public int NH;  // 出力列数

    public float[,,] x;
    public float[,,] y;

    public MaxPoolingLayer(int m_size, int n_size, int k_size, int h_size) {
        M = m_size;
        N = n_size;
        K = k_size;
        H = h_size;
        MH = M / H;
        NH = N / H;

        x = new float[M, N, K];
        y = new float[MH, NH, K];
    }

    public override void Forward() {
        foreach (int i in Range(MH)) {
            foreach (int j in Range(NH)) {
                foreach (int k in Range(K)) {
                    y[i, j, k] = (from p in Range(H) from q in Range(H) select x[i * H + p, j * H + q, k]).Max();
                }
            }
        }
    }
}


public class RecurrentLayer : Layer {
    public int T;
    public int X;
    public int Y;

    public float[,] x;
    public float[,] y;

    public float[,] win;
    public float[,] w;

    public float[] b;

    public float[,] u;

    public RecurrentLayer(int t_size, int x_size, int y_size) {
        T = t_size;
        X = x_size;
        Y = y_size;

        x = new float[T, X];
        y = new float[T, Y];
        win = new float[Y, X];
        w = new float[Y, Y];
        b = new float[Y];
        u = new float[T, Y];
    }

    public override void Forward() {
        foreach (int t in Range(T)) {
            foreach (int j in Range(Y)) {
                u[t, j] = (from i in Range(X) select x[t, i] * win[j, i]).Sum()
                    + (from i in Range(Y) select w[j, i] * y[t - 1, i]).Sum() + b[j];
                y[t, j] = σ(u[t, j]);
            }
        }
    }
}

public class LSTMLayer : Layer {
    public int T;
    public int X;
    public int Y;

    public float[,] x;
    public float[,] y;

    public float[,] wIin;
    public float[,] wFin;
    public float[,] wOin;
    public float[,] win;

    public float[,] wIr;
    public float[,] wFr;
    public float[,] wOr;
    public float[,] wr;

    public float[] wI;
    public float[] wF;
    public float[] wO;

    public float[] bO;
    public float[] bF;
    public float[] bI;
    public float[] b;

    public float[,] u;
    public float[,] s;

    public float[,] uI;
    public float[,] uF;
    public float[,] uO;

    public LSTMLayer(int t_size, int x_size, int y_size) {
        T = t_size;
        X = x_size;
        Y = y_size;
        x = new float[T, X];
        y = new float[T, Y];

        wIin = new float[Y, X];
        wFin = new float[Y, X];
        wOin = new float[Y, X];
        win = new float[Y, X];

        wIr = new float[Y, Y];
        wFr = new float[Y, Y];
        wOr = new float[Y, Y];
        wr = new float[Y, Y];

        wI = new float[Y];
        wF = new float[Y];
        wO = new float[Y];
        bO = new float[Y];
        bF = new float[Y];
        bI = new float[Y];
        b = new float[Y];

        u = new float[T, Y];
        s = new float[T, Y];
        uI = new float[T, Y];
        uF = new float[T, Y];
        uO = new float[T, Y];
    }

    public override void Forward() {
        foreach (int t in Range(T)) {
            foreach (int j in Range(Y)) {
                y[t, j] = σ(uO[t, j]) * σ(s[t, j]);
                s[t, j] = σ(uF[t, j]) * s[t - 1, j] + σ(uI[t, j]) * σ(u[t, j]);
                uO[t, j] = (from i in Range(X) select wOin[j, i] * x[t, i]).Sum() + (from i in Range(Y) select wOr[j, i] * y[t - 1, i]).Sum() + wO[j] * s[t, j] + bO[j];
                uF[t, j] = (from i in Range(X) select wFin[j, i] * x[t, i]).Sum() + (from i in Range(Y) select wFr[j, i] * y[t - 1, i]).Sum() + wF[j] * s[t - 1, j] + bF[j];
                uI[t, j] = (from i in Range(X) select wIin[j, i] * x[t, i]).Sum() + (from i in Range(Y) select wIr[j, i] * y[t - 1, i]).Sum() + wI[j] * s[t - 1, j] + bI[j];
                u[t, j] = (from i in Range(X) select win[j, i] * x[t, i]).Sum() + (from i in Range(Y) select wr[j, i] * y[t - 1, i]).Sum() + b[j];
            }
        }
    }
}

public class DNC : Layer {
     public static int T;
     public static int N;           // number of memory locations
     public static int W;           // memory word size
     public static int R;           // number of read heads
     public static int X;           // xの長さ
     public static int Y;           // yの長さ
     public static int χl;          // χの長さ
     public static int χ2hl;        // χ+h+hの長さ

    // LSTM
    public float[,] χ;     // input vector
    public float[,] χ2h;     // input vector + h + h
    public float[,] gin;   // input gate
    public float[,] gfo;   // forget gate
    public float[,] s;     // state
    public float[,] o;     // output gate
    public float[,] h;     // hidden

    public float[,] Wi;    // weight : input gate
    public float[,] Wf;    // weight : forget gate
    public float[,] Ws;    // weight : state
    public float[,] Wo;    // weight : output gate

    public float[] bi;     // bias : input gate
    public float[] bf;     // bias : forget gate
    public float[] bs;     // bias : state
    public float[] bo;     // bias : output gate


    public float[,] x = new float[T, X];     // input vector RX
    public float[,] y = new float[T, Y];     // output vector RX
    public float[,] v;     // output vector RX
    public float[,] z;     // target vector
    public float[,,] M;    // memory matrix

    public float[,,] kr;   // read key
    public float[,,] r;  // read vector
    public float[,] βr; // read strength

    public float[,] kw;   // write key
    public float[] βw; // write strength

    public float[,] e;   // erase vector
    public float[,] ν;   // write vector

    public float[,] gf; // free gate
    public float[] ga; // allocation gate
    public float[] gw; // write gate

    public float[,] ψ; // memory retention vector
    public float[,] u; // memory usage vector
    public int[,] φ;    // indices of slots sorted by usage
    public float[,] a; // allocation weighting
    public float[,] cw;    // write content weighting
    public float[,] ww;    // write weighting
    public float[,] p;     // precedence weighting
    //public float[,] E;     // matrix of ones
    public float[,,] L;     // temporal link matrix
    public float[,,] f;     // forward weighting
    public float[,,] b;     // backward weighting
    public float[,,] cr;    // read content weighting
    public float[,,] wr;    // read weighting
    public float[,] π1;     // read mode
    public float[,] π2;     // read mode
    public float[,] π3;     // read mode
    public float[,,] Wr;   // read key weights

    public DNC() {
        // LSTM
        χl = X + R * W;
        χ = new float[T, χl];
        gin = new float[T, Y];
        gfo = new float[T, Y];
        s = new float[T, Y];
        o = new float[T, Y];
        h = new float[T, Y];

        χ2hl = χl + Y + Y;
        χ2h = new float[T, χ2hl];
        Wi = new float[Y, χ2hl];
        Wf = new float[Y, χ2hl];
        Ws = new float[Y, χ2hl];
        Wo = new float[Y, χ2hl];

        bi = new float[Y];
        bf = new float[Y];
        bs = new float[Y];
        bo = new float[Y];

        v = new float[T, Y];
        z = new float[T, Y];
        M = new float[T, N, W];

        kr = new float[T, R, W];
        r = new float[T, R, W];
        βr = new float[T, R];

        kw = new float[T, W];
        βw = new float[T];

        e = new float[T, W];
        ν = new float[T, W];
        gf = new float[T, R];
        ga = new float[T];
        gw = new float[T];
        ψ = new float[T, N];
        u = new float[T, N];
        φ = new int[T, N];
        a = new float[T, N];
        cw = new float[T, N];
        ww = new float[T, N];
        p = new float[T, N];
        //E = new float[N, W];
        L = new float[T, N, N];
        f = new float[T, R, N];
        b = new float[T, R, N];
        cr = new float[T, R, N];
        wr = new float[T, R, N];
        π1 = new float[T, R];
        π2 = new float[T, R];
        π3 = new float[T, R];
        Wr = new float[R, W, Y];
    }

    public override void Forward() {
        foreach (int t in Range(T)) {
            foreach (int iy in Range(Y)) {
                gin[t, iy] = σ((from ix in Range(χ2hl) select Wi[iy, ix] * χ2h[t, ix]).Sum() + bi[iy]);
                gfo[t, iy] = σ((from ix in Range(χ2hl) select Wf[iy, ix] * χ2h[t, ix]).Sum() + bf[iy]);
                s[t, iy] = gfo[t, iy] * s[t - 1, iy] + gin[t, iy] * tanh((from ix in Range(χ2hl) select Ws[iy, ix] * χ2h[t, ix]).Sum() + bs[iy]);
                o[t, iy] = σ((from ix in Range(χ2hl) select Wo[iy, ix] * χ2h[t, ix]).Sum() + bo[iy]);
                h[t, iy] = o[t, iy] * tanh(s[t, iy]);
            }
            foreach (int n in Range(N)) {

                ψ[t, n] = Prod(from ir in Range(R) select 1.0 - gf[t, ir] * wr[t - 1, ir, n]);
                u[t, n] = (u[t - 1, n] + ww[t - 1, n] - (u[t - 1, n] * ww[t - 1, n])) * ψ[t, n];
                φ[t, n] = 0;// SortIndicesAscending(ut)
//                a[t, φ[t, n]] = (1 - u[t, φ[t, n]]) * Prod(from i in Range(N) select u[t, φ[t, i]]);
                cw[t, n] = C(Mat(M, t - 1), Row(kw, t), βw[t], n);
                ww[t, n] = gw[t] * (ga[t] * a[t, n] + (1 - ga[t]) * cw[t, n]);
                foreach (int iw in Range(W)) {
                    M[t, n, iw] = M[t - 1, n, iw] * (1 - ww[t, n] * e[t, iw]) + ww[t, n] * ν[t, iw];
                }
                p[t, n] = (1 - (from i in Range(N) select ww[t, i]).Sum()) * p[t - 1, n] + ww[t, n];
                foreach (int j in Range(N)) {
                    L[t, n, j] = (1 - ww[t, n] - ww[t, j]) * L[t - 1, n, j] + ww[t, n] * p[t - 1, j];
                }
            }

            foreach (int ir in Range(R)) {
                foreach (int n in Range(N)) {
                    f[t, ir, n] = (from j in Range(N) select L[t, n, j] * wr[t - 1, ir, j]).Sum();
                    b[t, ir, n] = (from j in Range(N) select L[t, j, n] * wr[t - 1, ir, j]).Sum();
                    cr[t, ir, n] = C(Mat(M, t), Row(kr, t, ir), βr[t, ir], n);
                    wr[t, ir, n] = π1[t, ir] * b[t, ir, n] + π2[t, ir] * cr[t, ir, n] + π3[t, ir] * f[t, ir, n];
                }
                foreach (int iw in Range(W)) {
                    r[t, ir, iw] = (from n in Range(N) select M[t, n, iw] * wr[t, ir, n]).Sum();
                }
            }
            foreach (int iy in Range(Y)) {
                y[t, iy] = 0;// Wr[r1t; ... ; rRt ] +υt
            }
        }
    }
}
