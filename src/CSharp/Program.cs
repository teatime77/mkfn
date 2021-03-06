﻿using System;
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


    public partial class Network {
        public LayerUtil Util;
        public DeviceCuda Dev;
        public string DataDir;

        public NetworkType Type;
        public int EpochSize;
        public int TrainBatchSize;
        public int TestBatchSize;
        public int RangeLen;
        public int MiniBatchIdx;
        public int CostCount;
        public int UpdateMiniBatchCount;

        public List<Layer> Layers;
        public Layer FirstLayer;
        public Layer LastLayer;

        public Network(){
            Util = new LayerUtil();

            string dir = Assembly.GetExecutingAssembly().Location;

            while (true) {
                DataDir = dir + "\\data";
                if (Directory.Exists(DataDir)) {
                    break;
                }

                dir = Path.GetDirectoryName(dir);
            }

        }

        public static int BytesToInt(byte[] v, int offset) {
            return v[offset] * 0x1000000 + v[offset + 1] * 0x10000 + v[offset + 2] * 0x100 + v[offset + 3];
        }

        public void DeepLearning() {
            FirstLayer = Layers[0];
            LastLayer = Layers[Layers.Count - 1];

            switch (Type) {
            case NetworkType.Simple:
            case NetworkType.CNN:
                float[,,] train_X;
                byte[] train_label;
                float[,,] test_X;
                byte[] test_label;
                ReadMNIST(DataDir, out train_X, out train_label, out test_X, out test_label);
                SGD(train_X, train_label, test_X, test_label);
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

    class Program {
        static Network net;
        static void Init() {

        }

        static void NetworkTest() {
            // 初期処理をします。
            Init();

            DeviceCuda dev = new DeviceCuda();
            dev.DeviceInit();

            net = new Network();
            net.Dev = dev;
            net.EpochSize = 100;
            net.TestBatchSize = 20;

            LayerFactoryCudaF factory = new LayerFactoryCudaF();

            float learning_rate = 1.0f;
            for (int run_idx = 0; ; run_idx++) {
                net.Type = NetworkType.RNN;
                net.Type = NetworkType.Simple;
                net.Type = NetworkType.CNN;
                net.Type = NetworkType.LSTM;
                switch (net.Type) {
                case NetworkType.Simple:
                    net.TrainBatchSize = 10;
                    net.Layers = new List<Layer>(){
                        factory.MakeFullyConnectedLayer(28 * 28, 30),
                        factory.MakeFullyConnectedLayer(30, 10)
                    };
                    break;

                case NetworkType.CNN:
                    net.TrainBatchSize = 10;
                    net.Layers = new List<Layer>(){
                        //new ConvolutionalLayer(28, 28, 20, 5),
                        //new MaxPoolingLayer(24, 24, 20, 2),
                        //new FullyConnectedLayerF(12 * 12 * 20, 100),
                        factory.MakeConvolution2DLayer(28, 28, 5, 5, 5),
                        factory.MakeMaxPooling2DLayer(24, 24, 5, 2, 2),
                        factory.MakeFullyConnectedLayer(12 * 12 * 5, 100),
                        factory.MakeFullyConnectedLayer(100, 10)
                    };
                    break;

                case NetworkType.RNN:
                    learning_rate = 0.1f;
                    net.TrainBatchSize = 7;
                    net.Layers = new List<Layer>(){
                        //new RecurrentLayer(5, 2, 10),
                        //new FullyConnectedLayerF(10, 2)
                        factory.MakeRecurrentLayer(20, 28, 100),
                        factory.MakeFullyConnectedLayer(10, 28)
                    };
                    break;

                case NetworkType.LSTM:
                    learning_rate = 0.1f;
                    net.TrainBatchSize = 7;
                    net.Layers = new List<Layer>(){
                        //new LSTMLayer(50, 2000, 100),
                        //new LSTMLayer(20, 1000, 100),
                        factory.MakeLSTMLayer(20, 28, 100),
                        factory.MakeFullyConnectedLayer(100, 28)
                    };
                    break;
                }

                for (int i = 0; i < net.Layers.Count; i++) {
                    net.Layers[i].LearningRate = learning_rate / net.TrainBatchSize;//3.0f
                }

                net.DeepLearning();
            }
        }

        void ModelTest() {
            Model model = new Sequential();
            model.AddDense(30, Activation.Sigmoid);
            model.AddDense(10, Activation.Sigmoid);
        }

        static void Main(string[] args) {
            Program prg = new Program();
            prg.ModelTest();

            Task t = Task.Run(() => {
                NetworkTest();
            });


            while (true) {
                Task.Delay(1000);
            }
        }
    }
}
