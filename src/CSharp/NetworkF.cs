﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LayerNET;
using System.IO;
using System.Diagnostics;

namespace CSTest {
    public class NetworkF : Network {
        float[] TrainX;
        float[] TrainY;
        float[] TestX;
        float[] TestY;
        float CostSum;


        public override void ReadMNIST() {
            string mnist_dir = DataDir + "\\MNIST\\";

            byte[] buf;

            buf = File.ReadAllBytes(mnist_dir + "train-images.idx3-ubyte");

            TrainCnt = BytesToInt(buf, 4);
            int img_h = BytesToInt(buf, 8);
            int img_w = BytesToInt(buf, 12);

            DomainLen = img_h * img_w;
            RangeLen = 10;

            int sz = TrainCnt * DomainLen;
            TrainX = new float[sz];
            for (int i = 0; i < sz; i++) {
                TrainX[i] = buf[16 + i] / 256.0f;
            }


            buf = File.ReadAllBytes(mnist_dir + "train-labels.idx1-ubyte");
            TrainLabel = new byte[TrainCnt];
            Array.Copy(buf, 8, TrainLabel, 0, TrainCnt);


            buf = File.ReadAllBytes(mnist_dir + "t10k-images.idx3-ubyte");
            TestCnt = BytesToInt(buf, 4);
            int test_sz = TestCnt * img_h * img_w;
            TestX = new float[test_sz];
            for (int i = 0; i < test_sz; i++) {
                TestX[i] = buf[16 + i] / 256.0f;
            }


            buf = File.ReadAllBytes(mnist_dir + "t10k-labels.idx1-ubyte");
            TestLabel = new byte[TestCnt];
            Array.Copy(buf, 8, TestLabel, 0, TestCnt);
        }


        /*
        すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
        */
        void AllocateConnectLayers(int batch_size) {
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


        void SetBatchData(float[] X, float[] batch_X, float[] batch_Y, byte[] label, int batch_size, int[] idxes) {
            for (int i = 0; i < batch_Y.Length; i++) {
                batch_Y[i] = 0;
            }

            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                int idx = MiniBatchIdx * batch_size + batch_idx;

                if (idxes != null) {

                    idx = idxes[idx];
                }

                for (int ix = 0; ix < DomainLen; ix++) {
                    batch_X[ix * batch_size + batch_idx] = X[idx * DomainLen + ix];
                }

                batch_Y[label[idx] * batch_size + batch_idx] = 1;
            }
        }


        /*
            損失関数の微分	
        */
        void CostDerivative(float[] cost_derivative, float[] last_y, float[] batch_Y, int size) {
            for (int i = 0; i < size; i++) {
                cost_derivative[i] = last_y[i] - batch_Y[i];
            }
        }


        /*
        損失関数
        */
        float Cost(float[] cost_derivative, int size) {
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
        float SoftMax(float[] cost_derivative, float[] last_y, float[] batch_Y, float[] exp_work, int range_len, int batch_size, int batch_idx) {
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
        unsafe void UpdateMiniBatch(float[] batch_X, float[] batch_Y, float[] last_y, float[] cost_derivative) {
            //-------------------------------------------------- 入力をセットします。

            FirstLayer.SetInputData(ref batch_X, DomainLen * TrainBatchSize * sizeof(float));

            for (int i = 0; i < Layers.Count; i++) {
                Layers[i].Forward();
            }

            //-------------------------------------------------- 出力を得ます。
            int last_y_len = TrainBatchSize * RangeLen;

            LastLayer.GetOutputData(ref last_y, last_y_len * sizeof(float));

            //-------------------------------------------------- 損失関数を計算します。
            CostDerivative(cost_derivative, last_y, batch_Y, last_y_len);

            float cost = Cost(cost_derivative, last_y_len);

            //-------------------------------------------------- δyをセットします。
            LastLayer.SetOutputDeltaData(ref cost_derivative, last_y_len * sizeof(float));

            for (int i = (int)Layers.Count - 1; 0 <= i; i--) {
                Layers[i].Backward();
            }
            Dev.DeviceSynchronize();

            for (int i = (int)Layers.Count - 1; 0 <= i; i--) {
                Layers[i].UpdateParameter();
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


        int ArgMax(float[] result_Y, int batch_size, byte[] arg_max, byte[] label) {
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

        int Evaluate(float[] batch_X, float[] batch_Y, float[] last_y, int batch_size, byte[] arg_max, byte[] label) {
            FirstLayer.SetInputData(ref batch_X, batch_size * DomainLen * sizeof(float));

            for (int i = 0; i < Layers.Count; i++) {
                Layers[i].Forward();
            }

            LastLayer.GetOutputData(ref last_y, batch_size * RangeLen * sizeof(float));

            int eq_cnt = ArgMax(last_y, batch_size, arg_max, label);

            return eq_cnt;
        }


        /*
        すべてのレイヤーのメモリを解放します。
        */
        void FreeLayers() {
            for (int i = 0; i < Layers.Count; i++) {
                Layers[i].Free();
            }

            Dev.DeviceFree(FirstLayer.GetInput());
            Dev.DeviceFree(LastLayer.GetOutputDelta());
        }

        /*
        確率的勾配降下法 (stochastic gradient descent, SGD)
        */
        public override void SGD() {

            int train_batch_cnt = TrainCnt / TrainBatchSize;
            int test_batch_cnt = TestCnt / TestBatchSize;

            float[] train_batch_X = new float[TrainBatchSize * DomainLen];
            float[] train_batch_Y = new float[TrainBatchSize * RangeLen];
            float[] train_last_Y = new float[TrainBatchSize * RangeLen];

            float[] cost_derivative = new float[TrainBatchSize * RangeLen];

            float[] test_batch_X = new float[TestBatchSize * DomainLen];
            float[] test_batch_Y = new float[TestBatchSize * RangeLen];
            float[] test_last_Y = new float[TestBatchSize * RangeLen];

            byte[] test_arg_max = new byte[TestBatchSize];

            for (EpochIdx = 0; EpochIdx < EpochSize; EpochIdx++) {

                int[] idxes = RandomSampling(TrainCnt, TrainCnt);

                // すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
                AllocateConnectLayers(TrainBatchSize);

                for (MiniBatchIdx = 0; MiniBatchIdx < train_batch_cnt; MiniBatchIdx++) {

                    SetBatchData(TrainX, train_batch_X, train_batch_Y, TrainLabel, TrainBatchSize, idxes);

                    UpdateMiniBatch(train_batch_X, train_batch_Y, train_last_Y, cost_derivative);
                }

                FreeLayers();

                // すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
                AllocateConnectLayers(TestBatchSize);

                int eq_cnt_sum = 0;
                for (MiniBatchIdx = 0; MiniBatchIdx < test_batch_cnt; MiniBatchIdx++) {

                    SetBatchData(TestX, test_batch_X, test_batch_Y, TestLabel, TestBatchSize, null);

                    int eq_cnt = Evaluate(test_batch_X, test_batch_Y, test_last_Y, TestBatchSize, test_arg_max, TestLabel);
                    eq_cnt_sum += eq_cnt;
                }
                Debug.WriteLine("epoch {0} : {1} / {2}", EpochIdx, eq_cnt_sum, TestCnt);

                FreeLayers();

            }
        }



        const int CHAR_COUNT = 0xFFFF;
        const char INVALID_CHAR = (char)0xFFFF;

        void ReadCharTable(char[] char_tbl, char[] char_tbl_inv) {
            string char_path;

            char_path = DataDir + "\\aozorabunko\\char.txt";

            string buf = File.ReadAllText(char_path, Encoding.Unicode);

            // 文字テーブルを-1でクリアします。
            for (int i = 0; i < CHAR_COUNT; i++) {
                char_tbl[i] = INVALID_CHAR;
                char_tbl_inv[i] = (char)0;
            }


            for (int i = 0; i < buf.Length; i++) {
                char_tbl[buf[i]] = (char)i;
                char_tbl_inv[i] = buf[i];
            }
        }


        static int LineIdx;
        static char[] CharBuf;

        void InitText(int batch_size, int line_len, out int train_cnt, char[] char_tbl, char[] char_tbl_inv) {
            string text_path;
            int line_NL_len = line_len + 1;


            text_path = string.Format("{0}\\aozorabunko\\Line-{1}.txt", DataDir, line_len);
            if (!File.Exists(text_path)) {
                // ファイルが存在しない場合

                train_cnt = 0;
                return;
            }

            char[] buf = File.ReadAllText(text_path, Encoding.Unicode).ToArray();

            int line_count = buf.Length / line_NL_len;

            train_cnt = (line_count / batch_size) * batch_size;

            CharBuf = new char[buf.Length];

            int[] idxes = RandomSampling(train_cnt, train_cnt);
            for (int i = 0; i < train_cnt; i++) {

                Array.Copy(buf, idxes[i] * line_NL_len, CharBuf, i * line_NL_len, line_NL_len);
            }

            if (char_tbl == null) {
                return;
            }

            for (int i = 0; i < CHAR_COUNT; i++) {
                char_tbl[i] = (char)0;
                char_tbl_inv[i] = (char)0;
            }

            int char_idx = 1;
            for (int i = 0; i < CharBuf.Length; i++) {
                char ch = CharBuf[i];
                int j = 0;
                for (; j < char_idx; j++) {
                    if (char_tbl_inv[j] == ch) {
                        break;
                    }
                }

                if (char_idx <= j) {
                    char_tbl[ch] = (char)char_idx;
                    char_tbl_inv[char_idx] = ch;

                    char_idx++;

                }
            }

            Debug.WriteLine("文字数:{0}", char_idx);

            // 文字テーブルを-1でクリアします。
            for (int i = 0; i < CHAR_COUNT; i++) {
                if (char_tbl[i] == 0) {
                    char_tbl[i] = INVALID_CHAR;
                }
            }
        }


        void ReadText(int batch_size, int line_len, int mini_batch_idx, char[] text) {
            Array.Copy(CharBuf, mini_batch_idx * batch_size * (line_len + 1), text, 0, batch_size * (line_len + 1));
        }

        void ClearText() {
            CharBuf = null;
        }


        void CharToOneHotX(char[] char_tbl, float[] batch_X, int Time, int one_hot_size, int batch_size, char[] text) {
            for (int t = 0; t < Time; t++) {
                for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                    char ch1 = text[batch_idx * (Time + 1 + 1) + t];
                    int idx1 = char_tbl[ch1];
                    if (idx1 < one_hot_size) {
                        batch_X[(t * one_hot_size + idx1) * batch_size + batch_idx] = 1;
                    }
                }
            }
        }


        void CharToOneHotY(char[] char_tbl, float[] batch_Y, int Time, int one_hot_size, int batch_size, char[] text, int t) {
            for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
                char ch2 = text[batch_idx * (Time + 1 + 1) + t + 1];
                int idx2 = char_tbl[ch2];
                if (idx2 < one_hot_size) {
                    batch_Y[idx2 * batch_size + batch_idx] = 1;
                }
            }
        }
        void SetZero(float[] v) {
            for (int i = 0; i < v.Length; i++) {
                v[i] = 0;
            }
        }

        /*
        ミニバッチごとにパラメータを更新します。
        */
        void RNNUpdateMiniBatch(float[] batch_X, float[] batch_Y, float[] last_y, float[] cost_derivative, float[] exp_work, char[] char_tbl, char[] char_tbl_inv, char[] text) {
            char[] input = new char[100];
            //		wchar_t	input2[100];
            char[] output = new char[100];

            int Time = FirstLayer.GetTimeCount();


            //-------------------------------------------------- 入力をセットします。
            FirstLayer.SetInputData(ref batch_X, DomainLen * TrainBatchSize * sizeof(float));

            // 順方向の時刻
            for (int t = 0; t < Time; t++) {

                SetZero(batch_Y);
                CharToOneHotY(char_tbl, batch_Y, Time, RangeLen, TrainBatchSize, text, t);

                FirstLayer.t = t;

                //-------------------------------------------------- レイヤーの入出力を結合します。
                for (int i = 0; i + 1 < Layers.Count; i++) {
                    Debug.Assert(Layers[i].GetTimeOutputCount() == Layers[i + 1].GetInputCount());

                    // 次のレイヤーの入力は、現在のレイヤーの出力にします。(順伝播)
                    Layers[i + 1].SetInput(Layers[i].GetOutput(t));
                }

                //-------------------------------------------------- 順伝播
                for (int i = 0; i < Layers.Count; i++) {
                    Layers[i].Forward();
                }

                //float[] p = (float[])fc.GetOutput(t);
                //for (int i = 0; i < fc.Y * TrainBatchSize; i++) {
                //	Log("%f", p[i]);
                //}

                //-------------------------------------------------- 出力を得ます。
                int last_y_len = TrainBatchSize * RangeLen;
                LastLayer.GetOutputData(ref last_y, last_y_len * sizeof(float));

                float cost = 0;
                for (int batch_idx = 0; batch_idx < TrainBatchSize; batch_idx++) {
                    float cost2 = SoftMax(cost_derivative, last_y, batch_Y, exp_work, RangeLen, TrainBatchSize, batch_idx);
                    /*
                                    if (MiniBatchIdx % 100 == 0 && batch_idx == TrainBatchSize - 1) {
                                        for (int i = 0; i < RangeLen; i++) {
                                            int k = i * TrainBatchSize + batch_idx;
                                            float sv = last_y[k];

                                            float dy = 0.0001;
                                            last_y[k] += dy;

                                            float cost3 = SoftMax(cost_derivative, last_y, batch_Y, exp_work, RangeLen, TrainBatchSize, batch_idx);
                                            Log("diff: %.16e  δ:%.16e", cost2, cost3, cost3 - cost2, dy * cost_derivative[k]);

                                            last_y[k] = sv;
                                        }
                                    }
                    */


                    cost += cost2;
                }
                cost /= TrainBatchSize;

                //-------------------------------------------------- 損失関数を計算します。
                CostSum += cost;
                CostCount++;


                if (UpdateMiniBatchCount % 141 == 0) {//MiniBatchIdx
                    int batch_idx = UpdateMiniBatchCount % TrainBatchSize;
                    float max_val = -10000;
                    int max_idx = 0;
                    for (int i = 0; i < RangeLen; i++) {
                        float val = last_y[i * TrainBatchSize + batch_idx];
                        if (max_val < val) {

                            max_val = val;
                            max_idx = i;
                        }
                    }
                    output[t] = char_tbl_inv[max_idx];

                    max_val = -10000;
                    max_idx = 0;
                    for (int i = 0; i < RangeLen; i++) {
                        float val = batch_Y[i * TrainBatchSize + batch_idx];
                        if (max_val < val) {

                            max_val = val;
                            max_idx = i;
                        }
                    }
                    //				input2[t] = char_tbl_inv[max_idx];


                    input[t] = text[batch_idx * (Time + 1 + 1) + t];

                    if (t == Time - 1) {
                        input[t + 1] = (char)0;
                        //					input2[t + 1] = 0;
                        output[t + 1] = (char)0;

                        //Log("IN2: %ls", input2);
                        Debug.WriteLine("IN : {0}", new string(input, 0, t), "");
                        Debug.WriteLine("OUT: {0}", new string(output, 0, t), "");

                        Debug.WriteLine("epock : {0}  cost : {1}", EpochIdx, CostSum / CostCount);
                    }
                }

                //-------------------------------------------------- δyをセットします。
                LastLayer.SetOutputDeltaData(ref cost_derivative, last_y_len * sizeof(float));

                //-------------------------------------------------- 逆伝播
                // RNN以外のレイヤーの逆伝播をします。
                for (int i = (int)Layers.Count - 1; 1 <= i; i--) {
                    IntPtr input_delta_save = new IntPtr(0);

                    // 現在のレイヤーの出力のデルタは、次のレイヤーの入力のデルタにします。(逆伝播)
                    if (Layers[i - 1].GetTimeCount() != 0) {

                        input_delta_save = Layers[i].GetInputDelta(t);
                        Layers[i].SetIputDelta(Layers[i - 1].GetOutputDelta(t));
                    }
                    else {

                        Layers[i - 1].SetOutputDelta(Layers[i].GetInputDelta(t));
                    }

                    Layers[i].Backward();

                    if (input_delta_save.ToInt64() != 0) {

                        Layers[i].SetIputDelta(input_delta_save);
                    }
                }
            }


            // 時刻の逆方向に、RNNのレイヤーの逆伝播をします。
            for (int t = Time - 1; 0 <= t; t--) {
                FirstLayer.t = t;
                FirstLayer.Backward();
            }

            // パラメータを更新します。
            for (int i = (int)Layers.Count - 1; 0 <= i; i--) {
                Layers[i].UpdateParameter();
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


        /*
            RNN用SGD
        */
        public override void RNNSGD() {
            char[] char_tbl = new char[CHAR_COUNT];
            char[] char_tbl_inv = new char[CHAR_COUNT];

            ReadCharTable(char_tbl, char_tbl_inv);

            CostSum = 0;
            CostCount = 0;
            UpdateMiniBatchCount = 0;
            while (true) {

                for (EpochIdx = 0; EpochIdx < EpochSize; EpochIdx++) {
                    int time_len = EpochIdx + 5;
                    int line_len = time_len + 1;

                    InitText(TrainBatchSize, line_len, out TrainCnt, char_tbl, char_tbl_inv);
                    if (TrainCnt == 0) {
                        break;
                    }

                    FirstLayer.SetTimeCount(time_len);
                    int X = FirstLayer.GetTimeInputCount();
                    int Y = FirstLayer.GetTimeOutputCount();
                    int Time = FirstLayer.GetTimeCount();

                    DomainLen = FirstLayer.GetInputCount();
                    RangeLen = LastLayer.GetOutputCount();

                    int train_batch_cnt = TrainCnt / TrainBatchSize;

                    float[] train_batch_X = new float[DomainLen * TrainBatchSize];
                    float[] train_batch_Y = new float[RangeLen * TrainBatchSize];
                    float[] train_last_Y = new float[RangeLen * TrainBatchSize];

                    float[] cost_derivative = new float[RangeLen * TrainBatchSize];
                    float[] exp_work = new float[RangeLen * TrainBatchSize];

                    // すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
                    AllocateConnectLayers(TrainBatchSize);

                    int delta_y_sz = TrainBatchSize * Time * Y * sizeof(double);
                    IntPtr out_delta_y = Dev.DeviceMalloc(delta_y_sz);

                    FirstLayer.SetOutputDelta(out_delta_y);

                    char[] text = new char[TrainBatchSize * (line_len + 1)];
                    for (MiniBatchIdx = 0; MiniBatchIdx < train_batch_cnt;) {//MiniBatchIdx++

                        ReadText(TrainBatchSize, line_len, MiniBatchIdx, text);

                        SetZero(train_batch_X);
                        CharToOneHotX(char_tbl, train_batch_X, Time, X, TrainBatchSize, text);

                        RNNUpdateMiniBatch(train_batch_X, train_batch_Y, train_last_Y, cost_derivative, exp_work, char_tbl, char_tbl_inv, text);
                        UpdateMiniBatchCount++;

                        if (MiniBatchIdx % 100 == 0) {
                            //Log("epock : %d   mini batch: %d  cost : %f", EpochIdx, MiniBatchIdx, CostSum / CostCount);
                        }
                    }

                    FreeLayers();
                    Dev.DeviceFree(out_delta_y);

                    //Log("epock : %d  cost : %f", EpochIdx, CostSum / CostCount);
                }
            }
        }
    }
}