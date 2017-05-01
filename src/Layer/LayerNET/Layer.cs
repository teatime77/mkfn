using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace LayerNET{
    //public class Startup {
    //    public void Invoke() {

    //    }
    //    public static int Add(int a, int b) {
    //        return a + b;
    //    }
    //}


    public class Startup {
        //public Task<object> Invoke(dynamic input) {

        //    int a = (int)input.a;
        //    int b = (int)input.b;

        //    return Task.Run<object>(() => {

        //        return a + b;
        //    });
        //}

        public object Invoke(dynamic input) {

            int a = (int)input.a;
            int b = (int)input.b;

            return a + b;
        }
    }

    public class MathHelper {
        public static int Add(int a, int b) {
            return a - b;
        }

        public static int AA(float[] v, int size) {
            return 2 * size;
        }
    }


    public class DLL {
        [DllImport("Layer.Dll")]
        public static extern void DeviceSynchronize();

        [DllImport("Layer.Dll")]
        public static extern void DeviceInit();

        [DllImport("Layer.Dll")]
        public static extern IntPtr DeviceMalloc(long size);

        [DllImport("Layer.Dll")]
        public static extern void DeviceFree(IntPtr p); 

        //----------------------------------------------------------------------------------------------------

        [DllImport("Layer.Dll")]
        public static extern IntPtr MakeFullyConnectedLayerF(int x_size, int y_size);

        [DllImport("Layer.Dll")]
        public static extern IntPtr MakeConvolutionalLayerF(int m_size, int n_size, int k_size, int h_size);

        [DllImport("Layer.Dll")]
        public static extern IntPtr MakeMaxPoolingLayerF(int m_size, int n_size, int k_size, int h_size);

        [DllImport("Layer.Dll")]
        public static extern IntPtr MakeRecurrentLayerF(int t_size, int x_size, int y_size);

        [DllImport("Layer.Dll")]
        public static extern IntPtr MakeLSTMLayerF(int t_size, int x_size, int y_size);

        //----------------------------------------------------------------------------------------------------

        [DllImport("Layer.Dll")]
        public static extern IntPtr MakeFullyConnectedLayerD(int x_size, int y_size);

        [DllImport("Layer.Dll")]
        public static extern IntPtr MakeConvolutionalLayerD(int m_size, int n_size, int k_size, int h_size);

        [DllImport("Layer.Dll")]
        public static extern IntPtr MakeMaxPoolingLayerD(int m_size, int n_size, int k_size, int h_size);

        [DllImport("Layer.Dll")]
        public static extern IntPtr MakeRecurrentLayerD(int t_size, int x_size, int y_size);

        [DllImport("Layer.Dll")]
        public static extern IntPtr MakeLSTMLayerD(int t_size, int x_size, int y_size);

        //----------------------------------------------------------------------------------------------------

        [DllImport("LayerCUDA.Dll")]
        public static extern IntPtr MakeFullyConnectedLayerCudaF(int x_size, int y_size);

        [DllImport("LayerCUDA.Dll")]
        public static extern IntPtr MakeConvolutionalLayerCudaF(int m_size, int n_size, int k_size, int h_size);

        [DllImport("LayerCUDA.Dll")]
        public static extern IntPtr MakeMaxPoolingLayerCudaF(int m_size, int n_size, int k_size, int h_size);

        [DllImport("LayerCUDA.Dll")]
        public static extern IntPtr MakeRecurrentLayerCudaF(int t_size, int x_size, int y_size);

        [DllImport("LayerCUDA.Dll")]
        public static extern IntPtr MakeLSTMLayerCudaF(int t_size, int x_size, int y_size);

        //----------------------------------------------------------------------------------------------------

        [DllImport("LayerCUDA.Dll")]
        public static extern IntPtr MakeFullyConnectedLayerCudaD(int x_size, int y_size);

        [DllImport("LayerCUDA.Dll")]
        public static extern IntPtr MakeConvolutionalLayerCudaD(int m_size, int n_size, int k_size, int h_size);

        [DllImport("LayerCUDA.Dll")]
        public static extern IntPtr MakeMaxPoolingLayerCudaD(int m_size, int n_size, int k_size, int h_size);

        [DllImport("LayerCUDA.Dll")]
        public static extern IntPtr MakeRecurrentLayerCudaD(int t_size, int x_size, int y_size);

        [DllImport("LayerCUDA.Dll")]
        public static extern IntPtr MakeLSTMLayerCudaD(int t_size, int x_size, int y_size);

        //----------------------------------------------------------------------------------------------------
        [DllImport("Layer.Dll")]
        public static extern int GetBatchSize(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern void SetBatchSize(IntPtr layer, int batch_size);

        [DllImport("Layer.Dll")]
        public static extern float GetLearningRate(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern void SetLearningRate(IntPtr layer, float learning_rate);

        [DllImport("Layer.Dll")]
        public static extern int Get_t(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern void Set_t(IntPtr layer, int t);


        [DllImport("Layer.Dll")]
        public static extern void DestroyLayer(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern void Forward(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern void Backward(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern void Allocate(IntPtr layer) ;

        [DllImport("Layer.Dll")]
        public static extern void Free(IntPtr layer) ;

        [DllImport("Layer.Dll")]
        public static extern void UpdateParameter(IntPtr layer) ;

        [DllImport("Layer.Dll")]
        public static extern void SetInput(IntPtr layer, IntPtr src) ;

        [DllImport("Layer.Dll")]
        public static extern IntPtr GetInput(IntPtr layer, int t = 0) ;

        [DllImport("Layer.Dll")]
        public static extern IntPtr GetOutput(IntPtr layer, int t = 0) ;

        [DllImport("Layer.Dll")]
        public static extern void SetIputDelta(IntPtr layer, IntPtr src) ;

        [DllImport("Layer.Dll")]
        public static extern void SetOutputDelta(IntPtr layer, IntPtr src) ;

        [DllImport("Layer.Dll")]
        public static extern IntPtr GetOutputDelta(IntPtr layer, int t = 0) ;

        [DllImport("Layer.Dll")]
        public static extern IntPtr GetInputDelta(IntPtr layer, int t = 0) ;

        [DllImport("Layer.Dll")]
        public static extern void SetInputData(IntPtr layer, IntPtr src, int size);

        [DllImport("Layer.Dll")]
        public static extern void SetOutputDeltaData(IntPtr layer, IntPtr src, int size);

        [DllImport("Layer.Dll")]
        public static extern void GetOutputData(IntPtr layer, IntPtr dst, int size);

        [DllImport("Layer.Dll")]
        public static extern void ConnectLayer(IntPtr layer, IntPtr next_layer) ;

        [DllImport("Layer.Dll")]
        public static extern int GetInputCount(IntPtr layer) ;

        [DllImport("Layer.Dll")]
        public static extern int GetOutputCount(IntPtr layer) ;

        [DllImport("Layer.Dll")]
        public static extern int GetTimeCount(IntPtr layer) ;

        [DllImport("Layer.Dll")]
        public static extern int GetTimeInputCount(IntPtr layer) ;

        [DllImport("Layer.Dll")]
        public static extern int GetTimeOutputCount(IntPtr layer) ;

        [DllImport("Layer.Dll")]
        public static extern void SetTimeCount(IntPtr layer, int time_count) ;

        [DllImport("Layer.Dll")]
        public static extern bool IsGPU(IntPtr layer);
    }

    public class Cuda {
        [DllImport("LayerCUDA.Dll")]
        public static extern void DeviceSynchronize();

        [DllImport("LayerCUDA.Dll")]
        public static extern void DeviceInit();

        [DllImport("LayerCUDA.Dll")]
        public static extern IntPtr DeviceMalloc(long size);

        [DllImport("LayerCUDA.Dll")]
        public static extern void DeviceFree(IntPtr p);

        [DllImport("LayerCUDA.Dll")]
        public static extern void MemcpyDeviceToHost(IntPtr dst, IntPtr src, long size);

        [DllImport("LayerCUDA.Dll")]
        public static extern void MemcpyHostToDevice(IntPtr dst, IntPtr src, long size);
    }

    public class Layer {
        public static void DeviceSynchronize() {
            DLL.DeviceSynchronize();
        }

        public static void DeviceInit() {
            DLL.DeviceInit();
        }

        public static IntPtr DeviceMalloc(long size) {
            return DLL.DeviceMalloc(size);
        }

        public static void DeviceFree(IntPtr p) {
            DLL.DeviceFree(p);
        }

        //----------------------------------------------------------------------------------------------------

        public static Layer MakeFullyConnectedLayerD(int x_size, int y_size) {
            return new Layer(DLL.MakeFullyConnectedLayerD(x_size, y_size));
        }


        public static Layer MakeConvolutionalLayerD(int m_size, int n_size, int k_size, int h_size) {
            return new Layer(DLL.MakeConvolutionalLayerD(m_size, n_size, k_size, h_size));
        }


        public static Layer MakeMaxPoolingLayerD(int m_size, int n_size, int k_size, int h_size) {
            return new Layer(DLL.MakeMaxPoolingLayerD(m_size, n_size, k_size, h_size));
        }


        public static Layer MakeRecurrentLayerD(int t_size, int x_size, int y_size) {
            return new Layer(DLL.MakeRecurrentLayerD(t_size, x_size, y_size));
        }


        public static Layer MakeLSTMLayerD(int t_size, int x_size, int y_size) {
            return new Layer(DLL.MakeLSTMLayerD(t_size, x_size, y_size));
        }

        //----------------------------------------------------------------------------------------------------

        public static Layer MakeFullyConnectedLayerCudaD(int x_size, int y_size) {
            return new Layer(DLL.MakeFullyConnectedLayerCudaD(x_size, y_size));
        }


        public static Layer MakeConvolutionalLayerCudaD(int m_size, int n_size, int k_size, int h_size) {
            return new Layer(DLL.MakeConvolutionalLayerCudaD(m_size, n_size, k_size, h_size));
        }


        public static Layer MakeMaxPoolingLayerCudaD(int m_size, int n_size, int k_size, int h_size) {
            return new Layer(DLL.MakeMaxPoolingLayerCudaD(m_size, n_size, k_size, h_size));
        }


        public static Layer MakeRecurrentLayerCudaD(int t_size, int x_size, int y_size) {
            return new Layer(DLL.MakeRecurrentLayerCudaD(t_size, x_size, y_size));
        }


        public static Layer MakeLSTMLayerCudaD(int t_size, int x_size, int y_size) {
            return new Layer(DLL.MakeLSTMLayerCudaD(t_size, x_size, y_size));
        }

        //----------------------------------------------------------------------------------------------------

        protected IntPtr Handle;

        public Layer() {
        }

        public Layer(IntPtr h) {
            Handle = h;
        }

        public int BatchSize {
            get { return DLL.GetBatchSize(Handle); }
            set { DLL.SetBatchSize(Handle, value); }
        }

        public float LearningRate {
            get { return DLL.GetLearningRate(Handle); }
            set { DLL.SetLearningRate(Handle, value); }
        }

        public int t {
            get { return DLL.Get_t(Handle); }
            set { DLL.Set_t(Handle, value); }
        }

        public void Destroy() {
            DLL.DestroyLayer(Handle);
        }

        public void Forward() {
            DLL.Forward(Handle);
        }

        public void Backward() {
            DLL.Backward(Handle);
        }

        public void Allocate() {
            DLL.Allocate(Handle);
        }

        public void Free() {
            DLL.Free(Handle);
        }

        public void UpdateParameter() {
            DLL.UpdateParameter(Handle);
        }

        public void ConnectLayer(Layer next_layer) {
            DLL.ConnectLayer(Handle, next_layer.Handle);
        }

        public int GetInputCount() {
            return DLL.GetInputCount(Handle);
        }

        public int GetOutputCount() {
            return DLL.GetOutputCount(Handle);
        }

        public int GetTimeCount() {
            return DLL.GetTimeCount(Handle);
        }

        public int GetTimeInputCount() {
            return DLL.GetTimeInputCount(Handle);
        }

        public int GetTimeOutputCount() {
            return DLL.GetTimeOutputCount(Handle);
        }

        public void SetTimeCount(int time_count) {
            DLL.SetTimeCount(Handle, time_count);
        }

        public bool IsGPU() {
            //return DLL.IsGPU(Handle);
            bool b = DLL.IsGPU(Handle);
            return b;
        }
    }

    public class LayerF : Layer {
        public LayerF(){
        }

        public LayerF(IntPtr h) : base(h) {
        }

        public static LayerF MakeFullyConnectedLayerF(int x_size, int y_size) {
            return new LayerF(DLL.MakeFullyConnectedLayerF(x_size, y_size));
        }

        public static LayerF MakeConvolutionalLayerF(int m_size, int n_size, int k_size, int h_size) {
            return new LayerF(DLL.MakeConvolutionalLayerF(m_size, n_size, k_size, h_size));
        }


        public static LayerF MakeMaxPoolingLayerF(int m_size, int n_size, int k_size, int h_size) {
            return new LayerF(DLL.MakeMaxPoolingLayerF(m_size, n_size, k_size, h_size));
        }


        public static LayerF MakeRecurrentLayerF(int t_size, int x_size, int y_size) {
            return new LayerF(DLL.MakeRecurrentLayerF(t_size, x_size, y_size));
        }


        public static LayerF MakeLSTMLayerF(int t_size, int x_size, int y_size) {
            return new LayerF(DLL.MakeLSTMLayerF(t_size, x_size, y_size));
        }

        //----------------------------------------------------------------------------------------------------

        public static LayerF MakeFullyConnectedLayerCudaF(int x_size, int y_size) {
            return new LayerF(DLL.MakeFullyConnectedLayerCudaF(x_size, y_size));
        }

        public static LayerF MakeConvolutionalLayerCudaF(int m_size, int n_size, int k_size, int h_size) {
            return new LayerF(DLL.MakeConvolutionalLayerCudaF(m_size, n_size, k_size, h_size));
        }


        public static LayerF MakeMaxPoolingLayerCudaF(int m_size, int n_size, int k_size, int h_size) {
            return new LayerF(DLL.MakeMaxPoolingLayerCudaF(m_size, n_size, k_size, h_size));
        }


        public static LayerF MakeRecurrentLayerCudaF(int t_size, int x_size, int y_size) {
            return new LayerF(DLL.MakeRecurrentLayerCudaF(t_size, x_size, y_size));
        }


        public static LayerF MakeLSTMLayerCudaF(int t_size, int x_size, int y_size) {
            return new LayerF(DLL.MakeLSTMLayerCudaF(t_size, x_size, y_size));
        }

        //----------------------------------------------------------------------------------------------------

        public void SetInput(IntPtr src) {
            DLL.SetInput(Handle, src);
        }

        public IntPtr GetInput(int t = 0) {
            return DLL.GetInput(Handle, t);
        }

        public IntPtr GetOutput(int t = 0) {
            return DLL.GetOutput(Handle, t);
        }

        public void SetIputDelta(IntPtr src) {
            DLL.SetIputDelta(Handle, src);
        }

        public void SetOutputDelta(IntPtr src) {
            DLL.SetOutputDelta(Handle, src);
        }

        public IntPtr GetOutputDelta(int t = 0) {
            return DLL.GetOutputDelta(Handle, t);
        }

        public IntPtr GetInputDelta(int t = 0) {
            return DLL.GetInputDelta(Handle, t);
        }

        unsafe public void SetInputData(float[] src, int size) {
            bool b = IsGPU();
            if (IsGPU()) {

                fixed(float* p = src) {
                    Cuda.MemcpyHostToDevice(DLL.GetInput(Handle), new IntPtr(p), size);
                }
            }
            else {

                Marshal.Copy(src, 0, DLL.GetInput(Handle), size / sizeof(float));
            }
        }

        unsafe public void SetOutputDeltaData(float[] src, int size) {
            if (IsGPU()) {

                fixed (float* p = src) {
                    Cuda.MemcpyHostToDevice(DLL.GetOutputDelta(Handle), new IntPtr(p), size);
                }
            }
            else {

                Marshal.Copy(src, 0, DLL.GetOutputDelta(Handle), size / sizeof(float));
            }
        }

        unsafe public void GetOutputData(float[] dst, int size) {
            if (IsGPU()) {

                fixed (float* p = dst) {
                    Cuda.MemcpyDeviceToHost(new IntPtr(p), DLL.GetOutput(Handle), size);
                }
            }
            else {

                Marshal.Copy(DLL.GetOutput(Handle), dst, 0, size / sizeof(float));
            }
        }
    }
}
