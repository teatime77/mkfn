using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using System.Diagnostics;
using System.Windows.Forms;
using System.Reflection;

//[assembly: AssemblyKeyFile("ComInterop.snk")]

namespace LayerNET {
    public abstract class LayerInf {
        public abstract Layer ToLayer();

        public int[] InputShape;
        public int[] OutputShape;
    }

    public class Dense : LayerInf {
        public int        OutputDim;
        public Activation Act;
        public int        InputDim;

        public Dense(int output_dim, Activation act, int input_dim = 0) {
            OutputDim = output_dim;
            Act = act;
            InputDim = input_dim;
        }

        public override Layer ToLayer() {
            return new Layer(DLL.MakeFullyConnectedLayerF(OutputDim, InputDim)); ;
        }
    }

    public class Convolution2D : LayerInf {
        public int nbFilter;
        public int nbRow;
        public int nbCol;
        public Activation Act;

        public Convolution2D(int nb_filter, int nb_row, int nb_col, Activation act, int[] input_shape = null) {
            nbFilter = nb_filter;
            nbRow = nb_row;
            nbCol = nb_col;
            Act = act;
        }

        public override Layer ToLayer() {
            return new Layer(DLL.MakeConvolution2DLayerF(InputShape[1], InputShape[2], nbFilter, nbRow, nbCol)); ;
        }
    }

    public class MaxPooling2D : LayerInf {
        public int nbFilter;
        public int nbRow;
        public int nbCol;
        public Activation Act;

        public MaxPooling2D(int nb_row, int nb_col, Activation act, int[] input_shape = null) {
            nbRow = nb_row;
            nbCol = nb_col;
            Act = act;
        }

        public override Layer ToLayer() {
            return new Layer(DLL.MakeMaxPooling2DLayerF(InputShape[1], InputShape[2], nbFilter, nbRow, nbCol));
        }
    }

    [ComVisible(true)]
    public enum Activation {
        Sigmoid,
        Softmax
    }

    [ComVisible(true)]
    public enum Optimizer {
        SGD
    }

    [ComVisible(true)]
    public enum Loss {
        MeanSquaredError
    }

    [ComVisible(true)]
    public interface IDeepLearning {
        void Array1(ref float[] a, ref float[] b);
        void Array2(ref float[,] a, ref float[,] b);
        void Array3(ref float[,,] a, ref float[] b);
    }

    [ClassInterface(ClassInterfaceType.None)]
    public class DeepLearning : IDeepLearning {
        public void Array1(ref float[] a, ref float[] b) {
            for(int i = 0; i < a.Length; i++) {
                b[i] = a[i] * 2;
            }
        }

        public void Array2(ref float[,] a, ref float[,] b) {
            for (int i = 0; i < a.GetLength(0); i++) {
                for(int j = 0; j < a.GetLength(1); j++) {
                    b[i,j] = a[i,j] * 3;
                }
            }
        }

        public void Array3(ref float[,,] a, ref float[] b) {
            //int idx = 0;
            //for (int i = 0; i < a.GetLength(0); i++) {
            //    for (int j = 0; j < a.GetLength(1); j++) {
            //        for(int k = 0; k < a.GetLength(2); k++) {

            //            b[idx] = a[i, j, k] * 4;
            //            idx++;
            //        }
            //    }
            //}

            Buffer.BlockCopy(a, 0, b, 0, b.Length * sizeof(float));
//            a.CopyTo(b, 0);
            //Array.Copy(a, b, b.Length);
            for (int i = 0; i < b.Length; i++) {
                b[i] *= 5;
            }
        }
    }

    [ComVisible(true)]
    public interface IModel {
        void AddDense(int output_dim, Activation ac, int input_dim = 0);
        void Compile(Optimizer optimizer, Loss loss);
        //void Fit(ref object x, ref object y, int batch_size= 32, int nb_epoch= 10);
        //void FitAsync(ref object x, ref object y, int batch_size = 32, int nb_epoch = 10);
        //void Predict(ref object x, int batch_size= 32);
    }


    [ComVisible(true)]
    public interface IModel11 : IModel {
        void Fit(ref float[] x, ref float[] y, int batch_size = 32, int nb_epoch = 10);
        void FitAsync(ref float[] x, ref float[] y, int batch_size = 32, int nb_epoch = 10);
        void Predict(ref float[] x, ref float[] y, int batch_size = 32);
    }

    [ComVisible(true)]
    public interface IModel21 : IModel {
        void Fit(ref float[,] x, ref float[] y, int batch_size = 32, int nb_epoch = 10);
        void FitAsync(ref float[,] x, ref float[] y, int batch_size = 32, int nb_epoch = 10);
        void Predict(ref float[,] x, ref float[] y, int batch_size = 32);
    }

    [ComVisible(true)]
    public interface IModel22 : IModel {
        void Fit(ref float[,] x, ref float[,] y, int batch_size = 32, int nb_epoch = 10);
        void FitAsync(ref float[,] x, ref float[,] y, int batch_size = 32, int nb_epoch = 10);
        void Predict(ref float[,] x, ref float[,] y, int batch_size = 32);
    }

    [ComVisible(true)]
    public interface IModel32 : IModel {
        void Fit(ref float[,,] x, ref float[,] y, int batch_size = 32, int nb_epoch = 10);
        void FitAsync(ref float[,,] x, ref float[,] y, int batch_size = 32, int nb_epoch = 10);
        void Predict(ref float[,,] x, ref float[,] y, int batch_size = 32);
    }

    [ClassInterface(ClassInterfaceType.None)]
    public class Model11 : IModel11 {

        public void AddDense(int output_dim, Activation act, int input_dim = 0) {
        }

        public void AddConvolution2D(int nb_filter, int nb_row, int nb_col, Activation act) {
        }

        public void AddMaxPooling2D(int nb_row, int nb_col, Activation act) {
        }

        public void Compile(Optimizer optimizer, Loss loss) {
        }

        public void Fit(ref float[] x, ref float[] y, int batch_size = 32, int nb_epoch = 10) {
        }

        public void FitAsync(ref float[] x, ref float[] y, int batch_size = 32, int nb_epoch = 10) {
        }

        public void Predict(ref float[] x, ref float[] y, int batch_size = 32) {
        }
    }

    [ClassInterface(ClassInterfaceType.None)]
    public class Model : IModel {
        public LayerUtil Util;
        public Device Dev;
        public string DataDir;

        public int TrainCnt;
        public int TestCnt;
        public int DomainLen;
        public int RangeLen;
        public int EpochIdx;
        public int MiniBatchIdx;
        public int CostCount;
        public int UpdateMiniBatchCount;

        public List<LayerInf> LayerInfs = new List<LayerInf>();
        public List<Layer> Layers = new List<Layer>();
        public Layer FirstLayer;
        public Layer LastLayer;

        public float CostSum;

        public void AddDense(int output_dim, Activation act, int input_dim = 0) {
            LayerInfs.Add(new Dense(output_dim, act, input_dim));
        }

        public void AddConvolution2D(int nb_filter, int nb_row, int nb_col, Activation act) {
            LayerInfs.Add(new Convolution2D(nb_filter, nb_row, nb_col, act));
        }

        public void AddMaxPooling2D(int nb_row, int nb_col, Activation act) {
            LayerInfs.Add(new MaxPooling2D(nb_row, nb_col, act));
        }

        public void Compile(Optimizer optimizer, Loss loss) {
        }

        public void FitInit() {
            foreach (Layer layer in Layers) {
                layer.Destroy();
            }
            Layers.Clear();

            foreach (LayerInf inf in LayerInfs) {
                Layers.Add(inf.ToLayer());
            }

            FirstLayer = Layers[0];
            LastLayer = Layers[Layers.Count - 1];

            DomainLen = FirstLayer.GetInputCount();
            RangeLen = LastLayer.GetOutputCount();
        }

        public void Fit(ref object x, ref object y, int batch_size = 32, int nb_epoch = 10) {
            FitInit();
            FirstLayer.SetInputData(ref x);
//            SGD();
        }

        public void FitAsync(ref object x, ref object y, int batch_size = 32, int nb_epoch = 10) {
            Task.Run(() => {
                FitInit();
//                SGD();
            });
        }

        public void Predict(ref object x, int batch_size = 32) {
//            TestBatchSize = batch_size;
        }


        /*
          [0, all_count-1]の範囲から、sample_count個の整数を抜き出して、その配列を返します。
        */
        public int[] RandomSampling(int all_count, int sample_count) {
            int[] ret = new int[sample_count];

            int[] numbers = new int[all_count];
            for (int i = 0; i < all_count; i++) {
                numbers[i] = i;
            }

            Random r = new Random(0);
            for (int i = 0; i < sample_count; i++) {
                int n = r.Next(1024 * 1024 * 1024) % (all_count - i);

                ret[i] = numbers[n];
                numbers[n] = numbers[all_count - i - 1];
            }

            //for (int i = 0; i < sample_count; i++) {
            //    for (int j = i + 1; j < sample_count; j++) {
            //        Debug.Assert(ret[i] != ret[j]);
            //    }
            //}

            return ret;
        }

        /*
        すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
        */
        public void AllocateConnectLayers(int batch_size) {
            IntPtr p = Dev.DeviceMalloc(batch_size * DomainLen * sizeof(float));
            FirstLayer.SetInput(p);

            p = Dev.DeviceMalloc(batch_size * RangeLen * sizeof(float));
            LastLayer.SetOutputDelta(p);

            for (int i = 0; i < Layers.Count; i++) {
                Layers[i].BatchSize = batch_size;
                Layers[i].Allocate();
            }

            // レイヤーの入出力を結合します。
            for (int i = 0; i + 1 < Layers.Count; i++) {

                if (Layers[i].GetTimeCount() == 0) {
                    Debug.Assert(Layers[i].GetOutputCount() == Layers[i + 1].GetInputCount());

                    // 次のレイヤーの入力は、現在のレイヤーの出力にします。(順伝播)
                    Layers[i + 1].SetInput(Layers[i].GetOutput());

                    // 現在のレイヤーの出力のデルタは、次のレイヤーの入力のデルタにします。(逆伝播)
                    Layers[i].SetOutputDelta(Layers[i + 1].GetInputDelta());
                }

                Layers[i].ConnectLayer(Layers[i + 1]);
            }
        }

        public void SetBatchData(float[] X, float[] batch_X, float[] Y, float[] batch_Y, int batch_size, int[] idxes) {
            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                int idx = MiniBatchIdx * batch_size + batch_idx;

                idx = idxes[idx];

                for (int ix = 0; ix < DomainLen; ix++) {
                    batch_X[ix * batch_size + batch_idx] = X[idx * DomainLen + ix];
                }

                for (int iy = 0; iy < RangeLen; iy++) {
                    batch_Y[iy * batch_size + batch_idx] = Y[idx * RangeLen + iy];
                }
            }
        }

        /*
            損失関数の微分	
        */
        public void CostDerivative(float[] cost_derivative, float[] last_y, float[] batch_Y, int size) {
            for (int i = 0; i < size; i++) {
                cost_derivative[i] = last_y[i] - batch_Y[i];
            }
        }


        /*
        損失関数
        */
        public float Cost(float[] cost_derivative, int size) {
            double sum = 0;
            for (int i = 0; i < size; i++) {
                float cd = cost_derivative[i];
                sum += cd * cd;
            }

            return (float)(sum / size);
        }


        /*
        損失関数の微分
        */
        public float SoftMax(float[] cost_derivative, float[] last_y, float[] batch_Y, float[] exp_work, int range_len, int batch_size, int batch_idx) {
            float max_val = -10000;
            for (int i = 0; i < range_len; i++) {
                int k = i * batch_size + batch_idx;

                if (max_val < last_y[k]) {
                    max_val = last_y[k];
                }
            }

            float sum = 0;
            for (int i = 0; i < range_len; i++) {
                int k = i * batch_size + batch_idx;

                float d = (float)Math.Exp(last_y[k] - max_val);
                sum += d;
                exp_work[k] = d;
            }

            float cost_sum = 0;
            for (int i = 0; i < range_len; i++) {
                int k = i * batch_size + batch_idx;

                float y = exp_work[k] / sum;
                cost_derivative[k] = y - batch_Y[k];

                cost_sum += (float)(batch_Y[k] * Math.Log(y));
            }

            return -cost_sum;
        }

        /*
            ミニバッチごとにパラメータを更新します。
        */
        public unsafe void UpdateMiniBatch(float[] batch_X, float[] batch_Y, float[] last_y, float[] cost_derivative, int batch_size) {
            //-------------------------------------------------- 入力をセットします。

            FirstLayer.SetInputData(batch_X);

            for (int i = 0; i < Layers.Count; i++) {
                Layers[i].Forward();
            }

            //-------------------------------------------------- 出力を得ます。
            int last_y_len = batch_size * RangeLen;

            LastLayer.GetOutputData(last_y);

            //-------------------------------------------------- 損失関数を計算します。
            CostDerivative(cost_derivative, last_y, batch_Y, last_y_len);

            float cost = Cost(cost_derivative, last_y_len);

            //-------------------------------------------------- δyをセットします。
            LastLayer.SetOutputDeltaData(cost_derivative);

            for (int i = (int)Layers.Count - 1; 0 <= i; i--) {
                Layers[i].Backward();
            }
            Dev.DeviceSynchronize();

            for (int i = (int)Layers.Count - 1; 0 <= i; i--) {
                Layers[i].UpdateParameter();
            }

            if (MiniBatchIdx % 100 == 0) {

                int iw = FirstLayer.GetFieldIndexByName("w");
                int[] sz = FirstLayer.GetFieldSize(iw);
                Debug.Assert(sz[1] == 28 * 28);
                int cnt = FirstLayer.GetFieldElementCount(iw);
                float[,] w = null;
                FirstLayer.GetFieldValue(iw, ref w);
                float[,] buf = new float[28, 28];
                for (int i = 0; i < sz[0]; i++) {
                    for (int y = 0; y < 28; y++) {
                        for (int x = 0; x < 28; x++) {
                            buf[y, x] = w[i, y * 28 + x];
                        }
                    }
                    string path = string.Format("{0}\\img\\a{1}.png", DataDir, i);
                    Util.SaveImage(path, buf);
                }
            }

            /*
                    Dmp("y0", (float[])Layers[0].GetOutput(), TrainBatchSize * Layers[0].GetOutputCount());
                    ConvolutionalLayer* cl = (ConvolutionalLayer*)FirstLayer;
                    Dmp("h", cl.h, cl.H * cl.H * cl.K);
                    FullyConnectedLayerF* fc = (FullyConnectedLayerF*)Layers[3];
                    Dmp("fc3-w", fc.w, fc.Y * fc.X);
            */

            Dev.DeviceSynchronize();
        }

        public int ArgMax(float[] result_Y, int batch_size, byte[] arg_max, byte[] label) {
            int eq_cnt = 0;

            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {

                float max_val = -10000;
                int max_idx = 0;
                for (int i = 0; i < RangeLen; i++) {
                    float val = result_Y[i * batch_size + batch_idx];
                    if (max_val < val) {

                        max_val = val;
                        max_idx = i;
                    }
                }

                arg_max[batch_idx] = (byte)max_idx;

                if (max_idx == label[MiniBatchIdx * batch_size + batch_idx]) {
                    eq_cnt++;
                }
            }

            return eq_cnt;
        }

        /*
        すべてのレイヤーのメモリを解放します。
        */
        public void FreeLayers() {
            for (int i = 0; i < Layers.Count; i++) {
                Layers[i].Free();
            }

            Dev.DeviceFree(FirstLayer.GetInput());
            Dev.DeviceFree(LastLayer.GetOutputDelta());
        }

        /*
        確率的勾配降下法 (stochastic gradient descent, SGD)
        */
        public void SGD(float[] train_X, float[] train_Y, int batch_size, int nb_epoch) {

            int train_batch_cnt = TrainCnt / batch_size;

            float[] train_batch_X = new float[batch_size * DomainLen];
            float[] train_batch_Y = new float[batch_size * RangeLen];
            float[] train_last_Y = new float[batch_size * RangeLen];

            float[] cost_derivative = new float[batch_size * RangeLen];

            // すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
            AllocateConnectLayers(batch_size);

            for (EpochIdx = 0; EpochIdx < nb_epoch; EpochIdx++) {

                int[] idxes = RandomSampling(TrainCnt, TrainCnt);

                for (MiniBatchIdx = 0; MiniBatchIdx < train_batch_cnt; MiniBatchIdx++) {

                    SetBatchData(train_X, train_batch_X, train_Y, train_batch_Y, batch_size, idxes);

                    UpdateMiniBatch(train_batch_X, train_batch_Y, train_last_Y, cost_derivative, batch_size);
                }
            }

            FreeLayers();
        }

        /*
        テスト
        */
        public void TestEvaluate(float[] test_X, float[] test_Y, int batch_size) {
            float[] test_batch_X = new float[batch_size * DomainLen];
            float[] test_batch_Y = new float[batch_size * RangeLen];

            // すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
            AllocateConnectLayers(batch_size);

            int eq_cnt_sum = 0;
            for (MiniBatchIdx = 0; MiniBatchIdx < batch_size; MiniBatchIdx++) {

                for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                    int idx = MiniBatchIdx * batch_size + batch_idx;

                    for (int ix = 0; ix < DomainLen; ix++) {
                        test_batch_X[ix * batch_size + batch_idx] = test_X[idx * DomainLen + ix];
                    }
                }

                FirstLayer.SetInputData(test_batch_X);

                for (int i = 0; i < Layers.Count; i++) {
                    Layers[i].Forward();
                }

                LastLayer.GetOutputData(test_batch_Y);

                for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                    int idx = MiniBatchIdx * batch_size + batch_idx;

                    for (int iy = 0; iy < RangeLen; iy++) {
                        test_Y[idx * RangeLen + iy] = test_batch_Y[iy * batch_size + batch_idx];
                    }
                }
            }
            Debug.WriteLine("epoch {0} : {1} / {2}", EpochIdx, eq_cnt_sum, TestCnt);

            FreeLayers();
        }
    }

    [ClassInterface(ClassInterfaceType.None)]
    public class Sequential : Model {
    }
}