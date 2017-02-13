using System;
using System.Collections.Generic;
using System.Linq;

//namespace Miyu {
public abstract class Layer {
    public static double e_;
    public double E;    // 損失関数

    public abstract void Forward();

    public int[] Range(int n) {
        return null;
    }

    public double σ(double z) {
        return 1.0 / (1.0 + exp(-z));
    }

    public double σ_prime(double z) {
        return σ(z) * (1 - σ(z));
    }

    public int[] Indexes(object a) {
        return null;
    }

    public int Dim(object a, int i) {
        return 0;
    }

    public double tanh(double x) {
        return 0;
    }

    public double tanh_prime(double x) {
        return 0;
    }

    public double log(double x) {
        return 0;
    }

    public double pow(double x, double y) {
        return 0;
    }

    public double exp(double x) {
        return 0;
    }

    public double sqrt(double x) {
        return 0;
    }

    public double[] Row(double[,] m, int i) {
        return null;
    }

    public double[] Row(double[,,] a, int t, int i) {
        return null;
    }

    public double[] Col(double[,] m, int j) {
        return null;
    }

    public double[,] Mat(double[,,] a, int i) {
        return null;
    }

    public double oneplus(double x) {
        return 1 + log(1.0 + pow(e_, x));
    }

    public double softmax(double[] x, int i) {
        return pow(e_, x[i]) / (from xj in x select pow(e_, xj)).Sum();
    }

    public double Length(double[] u) {
        return sqrt((from i in Indexes(u) select u[i] * u[i]).Sum());
    }

    public double Dot(double[] u, double[] v) {
        return (from i in Indexes(u) select u[i] * v[i]).Sum();
    }

    public double C(double[,] M, double[] k, double β, int i) {
        return exp(D(k, Row(M, i)) * β) / (from j in Indexes(M) select exp(D(k, Row(M, j)) * β)).Sum();
    }

    public double D(double[] u, double[] v) {
        return Dot(u, v) / (Length(u) * Length(v));
    }

    public double Prod(object a) {
        return 0;
    }
}

public class FullyConnectedLayer : Layer {
    public int X;
    public int Y;

    public double[] x;
    public double[] y;

    public double[,] w;
    public double[] b;
    public double[] u;

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

    public double[,] x;
    public double[,,] y;

    public double[,,] u;
    public double[,,] h;
    public double[] b;

    public override void Forward() {
        foreach (int i in Range(M)) {
            foreach (int j in Range(N)) {
                foreach (int k in Range(K)) {
                    u[i, j, k] = (from p in Range(H) from q in Range(H) select x[i + p, j + q] * h[p, q, k]).Sum() + b[k];
                    y[i, j, k] = σ(u[i, j, k]);
                }
            }
        }
    }
}

public class MaxPoolingLayer : Layer {
    public double[,,] x;
    public double[,,] y;

    public int M;   // 行数
    public int N;   // 列数
    public int K;   // フィルター数
    public int H;

    public override void Forward() {
        foreach (int i in Range(M)) {
            foreach (int j in Range(N)) {
                foreach (int k in Range(K)) {
                    y[i, j, k] = (from p in Range(H) from q in Range(H) select x[i + p, j + q, k]).Max();
                }
            }
        }
    }
}

public class RecurrentLayer : Layer {
    public int T;
    public int Y;
    public int X;

    public double[,] x;
    public double[,] y;

    public double[,] win;
    public double[,] w;

    public double[] b;

    public double[,] u;

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

    //public double[,] z;
    public double[,] wZ;

    public double[,] x;
    public double[,] y;

    public double[,] wIin;
    public double[,] wFin;
    public double[,] wOin;
    public double[,] win;

    public double[,] wIr;
    public double[,] wFr;
    public double[,] wOr;
    public double[,] wr;

    public double[] wI;
    public double[] wF;
    public double[] wO;
    public double[] w;

    public double[] bO;
    public double[] bF;
    public double[] bI;
    public double[] b;

    public double[,] u;
    public double[,] s;

    public double[,] uI;
    public double[,] uF;
    public double[,] uO;

    public override void Forward() {
        foreach (int t in Range(T)) {
            //foreach (int k in Range(Dim(z, 1))) {
            //    z[t, k] = (from j in Range(Dim(y, 1)) select wZ[k, j] * y[t, j]).Sum();
            //}
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
    public int T;
    public int N;           // number of memory locations
    public int W;           // memory word size
    public int R;           // number of read heads
    public int X;           // xの長さ
    public int Y;           // yの長さ
    public int χl;          // χの長さ
    public int χ2hl;        // χ+h+hの長さ

    // LSTM
    public double[,] χ;     // input vector
    public double[,] χ2h;     // input vector + h + h
    public double[,] gin;   // input gate
    public double[,] gfo;   // forget gate
    public double[,] s;     // state
    public double[,] o;     // output gate
    public double[,] h;     // hidden

    public double[,] Wi;    // weight : input gate
    public double[,] Wf;    // weight : forget gate
    public double[,] Ws;    // weight : state
    public double[,] Wo;    // weight : output gate

    public double[] bi;     // bias : input gate
    public double[] bf;     // bias : forget gate
    public double[] bs;     // bias : state
    public double[] bo;     // bias : output gate


    public double[,] x;     // input vector RX
    public double[,] y;     // output vector RX
    public double[,] v;     // output vector RX
    public double[,] z;     // target vector
    public double[,,] M;    // memory matrix

    public double[,,] kr;   // read key
    public double[,,] r;  // read vector
    public double[,] βr; // read strength

    public double[,] kw;   // write key
    public double[] βw; // write strength

    public double[,] e;   // erase vector
    public double[,] ν;   // write vector

    public double[,] gf; // free gate
    public double[] ga; // allocation gate
    public double[] gw; // write gate

    public double[,] ψ; // memory retention vector
    public double[,] u; // memory usage vector
    public int[,] φ;    // indices of slots sorted by usage
    public double[,] a; // allocation weighting
    public double[,] cw;    // write content weighting
    public double[,] ww;    // write weighting
    public double[,] p;     // precedence weighting
    //public double[,] E;     // matrix of ones
    public double[,,] L;     // temporal link matrix
    public double[,,] f;     // forward weighting
    public double[,,] b;     // backward weighting
    public double[,,] cr;    // read content weighting
    public double[,,] wr;    // read weighting
    public double[,] π1;     // read mode
    public double[,] π2;     // read mode
    public double[,] π3;     // read mode
    public double[,,] Wr;   // read key weights

    public DNC() {
        // LSTM
        χl = X + R * W;
        χ = new double[T, χl];
        gin = new double[T, Y];
        gfo = new double[T, Y];
        s = new double[T, Y];
        o = new double[T, Y];
        h = new double[T, Y];

        χ2hl = χl + Y + Y;
        χ2h = new double[T, χ2hl];
        Wi = new double[Y, χ2hl];
        Wf = new double[Y, χ2hl];
        Ws = new double[Y, χ2hl];
        Wo = new double[Y, χ2hl];

        bi = new double[Y];
        bf = new double[Y];
        bs = new double[Y];
        bo = new double[Y];

        x = new double[T, X];
        y = new double[T, Y];
        v = new double[T, Y];
        z = new double[T, Y];
        M = new double[T, N, W];

        kr = new double[T, R, W];
        r = new double[T, R, W];
        βr = new double[T, R];

        kw = new double[T, W];
        βw = new double[T];

        e = new double[T, W];
        ν = new double[T, W];
        gf = new double[T, R];
        ga = new double[T];
        gw = new double[T];
        ψ = new double[T, N];
        u = new double[T, N];
        φ = new int[T, N];
        a = new double[T, N];
        cw = new double[T, N];
        ww = new double[T, N];
        p = new double[T, N];
        //E = new double[N, W];
        L = new double[T, N, N];
        f = new double[T, R, N];
        b = new double[T, R, N];
        cr = new double[T, R, N];
        wr = new double[T, R, N];
        π1 = new double[T, R];
        π2 = new double[T, R];
        π3 = new double[T, R];
        Wr = new double[R, W, Y];
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
//}