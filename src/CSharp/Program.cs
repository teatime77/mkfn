using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LayerNET;
using System.IO;
using System.Reflection;
using System.Diagnostics;

namespace CSTest {

    public enum NetworkType {
        Simple,
        CNN,
        RNN,
        LSTM,
    }


    public abstract class Network {
        public string DataDir;

        public NetworkType Type;
        public int EpochSize;
        public int TrainBatchSize;
        public int TestBatchSize;
        public int TrainCnt;
        public int TestCnt;
        public int DomainLen;
        public int RangeLen;
        public byte[] TrainLabel;
        public byte[] TestLabel;
        public int EpochIdx;
        public int MiniBatchIdx;
        public int CostCount;
        public int UpdateMiniBatchCount;

        public List<LayerF> Layers;
        public LayerF FirstLayer;
        public LayerF LastLayer;

        public Network(){
            string dir = Assembly.GetExecutingAssembly().Location;

            while (true) {
                DataDir = dir + "\\data";
                if (Directory.Exists(DataDir)) {
                    break;
                }

                dir = Path.GetDirectoryName(dir);
            }

        }

        public int BytesToInt(byte[] v, int offset) {
            return v[offset] * 0x1000000 + v[offset + 1] * 0x10000 + v[offset + 2] * 0x100 + v[offset + 3];
        }



        public abstract void ReadMNIST();
        public abstract void SGD();
        public abstract void RNNSGD();

        public void DeepLearning() {
            FirstLayer = Layers[0];
            LastLayer = Layers[Layers.Count - 1];

            switch (Type) {
            case NetworkType.Simple:
            case NetworkType.CNN:
                SGD();
                break;

            case NetworkType.RNN:
            case NetworkType.LSTM:
                RNNSGD();
                break;
            }

            for (int i = 0; i < Layers.Count; i++) {
                Layers[i].Destroy();
            }
            Layers.Clear();
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
    }

    public class Device {
        static bool IsCuda = true;

        public static void Synchronize() {
            if (IsCuda) {
                Cuda.DeviceSynchronize();
            }
            else {
                DLL.DeviceSynchronize();
            }
        }

        public static void Init() {
            if (IsCuda) {
                Cuda.DeviceInit();
            }
            else {
                DLL.DeviceInit();
            }
        }

        public static IntPtr Malloc(long size) {
            if (IsCuda) {
                return Cuda.DeviceMalloc(size);
            }
            else {
                return DLL.DeviceMalloc(size);
            }
        }

        public static void Free(IntPtr p) {
            if (IsCuda) {
                Cuda.DeviceFree(p);
            }
            else {
                DLL.DeviceFree(p);
            }
        }
    }

    class Program {
        static void Init() {

        }

        static void NetworkTest() {
            // 初期処理をします。
            Init();

            Device.Init();

            NetworkF net = new NetworkF();
            net.EpochSize = 100;
            net.TestBatchSize = 20;

            float learning_rate = 1.0f;
            for (int run_idx = 0; ; run_idx++) {
                net.Type = NetworkType.Simple;
                net.Type = NetworkType.RNN;
                net.Type = NetworkType.CNN;
                net.Type = NetworkType.LSTM;
                switch (net.Type) {
                case NetworkType.Simple:
                    net.TrainBatchSize = 10;
                    net.ReadMNIST();
                    net.Layers = new List<LayerF>(){
                        LayerNET.LayerF.MakeFullyConnectedLayerF(28 * 28, 30),
                        LayerNET.LayerF.MakeFullyConnectedLayerF(30, 10)
                    };
                    break;

                case NetworkType.CNN:
                    net.TrainBatchSize = 10;
                    net.ReadMNIST();
                    net.Layers = new List<LayerF>(){
                        //new ConvolutionalLayer(28, 28, 20, 5),
                        //new MaxPoolingLayer(24, 24, 20, 2),
                        //new FullyConnectedLayerF(12 * 12 * 20, 100),
                        LayerNET.LayerF.MakeConvolutionalLayerCudaF(28, 28, 5, 5),
                        LayerNET.LayerF.MakeMaxPoolingLayerCudaF(24, 24, 5, 2),
                        LayerNET.LayerF.MakeFullyConnectedLayerCudaF(12 * 12 * 5, 100),
                        LayerNET.LayerF.MakeFullyConnectedLayerCudaF(100, 10)
                    };
                    break;

                case NetworkType.RNN:
                    learning_rate = 0.1f;
                    net.TrainBatchSize = 7;
                    net.Layers = new List<LayerF>(){
                        //new RecurrentLayer(5, 2, 10),
                        //new FullyConnectedLayerF(10, 2)
                        LayerNET.LayerF.MakeRecurrentLayerF(20, 28, 100),
                        LayerNET.LayerF.MakeFullyConnectedLayerF(10, 28)
                    };
                    break;

                case NetworkType.LSTM:
                    learning_rate = 0.1f;
                    net.TrainBatchSize = 7;
                    net.Layers = new List<LayerF>(){
                        //new LSTMLayer(50, 2000, 100),
                        //new LSTMLayer(20, 1000, 100),
                        LayerNET.LayerF.MakeLSTMLayerCudaF(20, 28, 100),
                        LayerNET.LayerF.MakeFullyConnectedLayerCudaF(100, 28)
                    };
                    break;
                }

                for (int i = 0; i < net.Layers.Count; i++) {
                    net.Layers[i].LearningRate = learning_rate / net.TrainBatchSize;//3.0f
                }

                net.DeepLearning();
            }
        }

        static void Main(string[] args) {
            NetworkTest();
        }
    }
}
