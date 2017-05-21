using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using System.Diagnostics;

namespace LayerNET {
    enum CopyDir {
        Set,
        Get
    }
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
        public static extern void DeviceEnd();

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
        public static extern int GetFieldCount(IntPtr layer);

        [DllImport("Layer.Dll", CharSet = CharSet.Unicode)]
        public static extern int GetFieldIndexByName(IntPtr layer, string name);

        [DllImport("Layer.Dll", CharSet = CharSet.Unicode)]
        public static extern void GetFieldName(IntPtr layer, int field_idx, StringBuilder name);

        [DllImport("Layer.Dll")]
        public static extern int GetFieldDimension(IntPtr layer, int field_idx);

        [DllImport("Layer.Dll")]
        public static extern IntPtr GetFieldSize(IntPtr layer, int field_idx);

        [DllImport("Layer.Dll")]
        public static extern int GetFieldElementCount(IntPtr layer, int field_idx);

        [DllImport("Layer.Dll")]
        public static extern void GetFieldValue(IntPtr layer, int field_idx, IntPtr dst);

        [DllImport("Layer.Dll")]
        public static extern void SetFieldValue(IntPtr layer, int field_idx, IntPtr src);


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
        public static extern void Destroy(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern void Forward(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern void Backward(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern void Allocate(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern void Free(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern void UpdateParameter(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern void SetInput(IntPtr layer, IntPtr src);

        [DllImport("Layer.Dll")]
        public static extern IntPtr GetInput(IntPtr layer, int t = 0);

        [DllImport("Layer.Dll")]
        public static extern IntPtr GetOutput(IntPtr layer, int t = 0);

        [DllImport("Layer.Dll")]
        public static extern void SetIputDelta(IntPtr layer, IntPtr src);

        [DllImport("Layer.Dll")]
        public static extern void SetOutputDelta(IntPtr layer, IntPtr src);

        [DllImport("Layer.Dll")]
        public static extern IntPtr GetOutputDelta(IntPtr layer, int t = 0);

        [DllImport("Layer.Dll")]
        public static extern IntPtr GetInputDelta(IntPtr layer, int t = 0);

        [DllImport("Layer.Dll")]
        public static extern void SetInputData(IntPtr layer, IntPtr src, int size);

        [DllImport("Layer.Dll")]
        public static extern void SetOutputDeltaData(IntPtr layer, IntPtr src, int size);

        [DllImport("Layer.Dll")]
        public static extern void GetOutputData(IntPtr layer, IntPtr dst, int size);

        [DllImport("Layer.Dll")]
        public static extern void ConnectLayer(IntPtr layer, IntPtr next_layer);

        [DllImport("Layer.Dll")]
        public static extern int GetInputCount(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern int GetOutputCount(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern int GetTimeCount(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern int GetTimeInputCount(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern int GetTimeOutputCount(IntPtr layer);

        [DllImport("Layer.Dll")]
        public static extern void SetTimeCount(IntPtr layer, int time_count);

        [DllImport("Layer.Dll")]
        public static extern bool IsGPU(IntPtr layer);
    }

    public class Cuda {
        [DllImport("LayerCUDA.Dll")]
        public static extern void DeviceSynchronize();

        [DllImport("LayerCUDA.Dll")]
        public static extern void DeviceInit();

        [DllImport("LayerCUDA.Dll")]
        public static extern void DeviceEnd();

        [DllImport("LayerCUDA.Dll")]
        public static extern IntPtr DeviceMalloc(long size);

        [DllImport("LayerCUDA.Dll")]
        public static extern void DeviceFree(IntPtr p);

        [DllImport("LayerCUDA.Dll")]
        public static extern void MemcpyDeviceToHost(IntPtr dst, IntPtr src, long size);

        [DllImport("LayerCUDA.Dll")]
        public static extern void MemcpyHostToDevice(IntPtr dst, IntPtr src, long size);
    }


    [ComVisible(true)]
    public interface ILayerUtil {
        void SaveImage(string path, int with, int height,ref float[] buf);
        void SaveImage2(string path, ref float[,] buf);
    }

    [ClassInterface(ClassInterfaceType.None)]
    public class LayerUtil : ILayerUtil {
        unsafe public void SaveImage(string path, int with, int height, float min1, float max1, float* buf) {
            Bitmap bmp = new Bitmap(with, height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);

            float span = max1 - min1;
            if (span != 0) {

                float scale = 256 / span;

                BitmapData bmpData = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadWrite, bmp.PixelFormat);

                byte[] vb = new byte[bmpData.Stride * height];
                int src = 0, dst = 0;

                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < with; w++) {
                        int idx = dst + w * 3;

                        float l = Math.Max(0, Math.Min(255, scale * (buf[src] - min1)));
                        Debug.Assert(0 <= l && l <= 255);
                        float r, g, b;
                        if (l < 64) {
                            r = 0;
                            g = 255 * l / 64.0f;
                            b = 255;

                        }
                        else if(l < 128) {
                            r = 0;
                            g = 255;
                            b = 255 * (128 - l) / 64.0f;

                        }
                        else if (l < 192) {
                            r = 255 * (l - 128) / 64.0f;
                            g = 255;
                            b = 0;

                        }
                        else {
                            r = 255;
                            g = 255 * (256 - l) / 64.0f;
                            b = 0;
                        }
                        Debug.Assert(0 <= r && r <= 255);
                        Debug.Assert(0 <= g && g <= 255);
                        Debug.Assert(0 <= b && b <= 255);

                        vb[idx] = (byte)b;
                        vb[idx + 1] = (byte)g;
                        vb[idx + 2] = (byte)r;

                        src++;
                    }

                    dst += bmpData.Stride;
                }

                Marshal.Copy(vb, 0, bmpData.Scan0, vb.Length);

                bmp.UnlockBits(bmpData);
            }

            bmp.Save(path);

        }

        unsafe public void SaveImage(string path, int with, int height,ref float[] buf) {
            fixed (float* pf = buf) {
                SaveImage(path, with, height, buf.Min(), buf.Max(), pf);
            }
        }

        unsafe public void SaveImage2(string path, ref float[,] buf) {
            int h = buf.GetLength(0);
            int w = buf.GetLength(1);
            float min1 = float.MaxValue;
            float max1 = -float.MaxValue;
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    min1 = Math.Min(min1, buf[y, x]);
                    max1 = Math.Max(max1, buf[y, x]);
                }
            }

            fixed (float* pf = buf) {
                SaveImage(path, w, h, min1, max1, pf);
            }
        }
    }

    [ComVisible(true)]
    public interface IDevice {
        void DeviceSynchronize();
        void DeviceInit();
        void DeviceEnd();
        IntPtr DeviceMalloc(long size);
        void DeviceFree(IntPtr p);
    }

    [ClassInterface(ClassInterfaceType.None)]
    public class Device : IDevice {
        public void DeviceSynchronize() {
            DLL.DeviceSynchronize();
        }

        public void DeviceInit() {
            DLL.DeviceInit();
        }

        public void DeviceEnd() {
            DLL.DeviceEnd();
        }

        public IntPtr DeviceMalloc(long size) {
            return DLL.DeviceMalloc(size);
        }

        public void DeviceFree(IntPtr p) {
            DLL.DeviceFree(p);
        }
    }

    [ClassInterface(ClassInterfaceType.None)]
    public class DeviceCuda : IDevice {
        public void DeviceSynchronize() {
            Cuda.DeviceSynchronize();
        }

        public void DeviceInit() {
            Cuda.DeviceInit();
        }

        public void DeviceEnd() {
            Cuda.DeviceEnd();
        }

        public IntPtr DeviceMalloc(long size) {
            return Cuda.DeviceMalloc(size);
        }

        public void DeviceFree(IntPtr p) {
            Cuda.DeviceFree(p);
        }
    }

    [ComVisible(true)]
    public interface ILayerFactory {
        Layer MakeFullyConnectedLayer(int x_size, int y_size);
        Layer MakeConvolutionalLayer(int m_size, int n_size, int k_size, int h_size);
        Layer MakeMaxPoolingLayer(int m_size, int n_size, int k_size, int h_size);
        Layer MakeRecurrentLayer(int t_size, int x_size, int y_size);
        Layer MakeLSTMLayer(int t_size, int x_size, int y_size);
    }

    [ClassInterface(ClassInterfaceType.None)]
    public class LayerFactoryF : ILayerFactory {
        public Layer MakeFullyConnectedLayer(int x_size, int y_size) {
            return new Layer(DLL.MakeFullyConnectedLayerF(x_size, y_size));
        }

        public Layer MakeConvolutionalLayer(int m_size, int n_size, int k_size, int h_size) {
            return new Layer(DLL.MakeConvolutionalLayerF(m_size, n_size, k_size, h_size));
        }

        public Layer MakeMaxPoolingLayer(int m_size, int n_size, int k_size, int h_size) {
            return new Layer(DLL.MakeMaxPoolingLayerF(m_size, n_size, k_size, h_size));
        }

        public Layer MakeRecurrentLayer(int t_size, int x_size, int y_size) {
            return new Layer(DLL.MakeRecurrentLayerF(t_size, x_size, y_size));
        }

        public Layer MakeLSTMLayer(int t_size, int x_size, int y_size) {
            return new Layer(DLL.MakeLSTMLayerF(t_size, x_size, y_size));
        }
    }

    [ClassInterface(ClassInterfaceType.None)]
    public class LayerFactoryD : ILayerFactory {
        public Layer MakeFullyConnectedLayer(int x_size, int y_size) {
            return new Layer(DLL.MakeFullyConnectedLayerD(x_size, y_size));
        }

        public Layer MakeConvolutionalLayer(int m_size, int n_size, int k_size, int h_size) {
            return new Layer(DLL.MakeConvolutionalLayerD(m_size, n_size, k_size, h_size));
        }

        public Layer MakeMaxPoolingLayer(int m_size, int n_size, int k_size, int h_size) {
            return new Layer(DLL.MakeMaxPoolingLayerD(m_size, n_size, k_size, h_size));
        }

        public Layer MakeRecurrentLayer(int t_size, int x_size, int y_size) {
            return new Layer(DLL.MakeRecurrentLayerD(t_size, x_size, y_size));
        }

        public Layer MakeLSTMLayer(int t_size, int x_size, int y_size) {
            return new Layer(DLL.MakeLSTMLayerD(t_size, x_size, y_size));
        }
    }

    [ClassInterface(ClassInterfaceType.None)]
    public class LayerFactoryCudaF : ILayerFactory {
        public Layer MakeFullyConnectedLayer(int x_size, int y_size) {
            return new Layer(DLL.MakeFullyConnectedLayerCudaF(x_size, y_size));
        }

        public Layer MakeConvolutionalLayer(int m_size, int n_size, int k_size, int h_size) {
            return new Layer(DLL.MakeConvolutionalLayerCudaF(m_size, n_size, k_size, h_size));
        }

        public Layer MakeMaxPoolingLayer(int m_size, int n_size, int k_size, int h_size) {
            return new Layer(DLL.MakeMaxPoolingLayerCudaF(m_size, n_size, k_size, h_size));
        }

        public Layer MakeRecurrentLayer(int t_size, int x_size, int y_size) {
            return new Layer(DLL.MakeRecurrentLayerCudaF(t_size, x_size, y_size));
        }

        public Layer MakeLSTMLayer(int t_size, int x_size, int y_size) {
            return new Layer(DLL.MakeLSTMLayerCudaF(t_size, x_size, y_size));
        }
    }

    [ClassInterface(ClassInterfaceType.None)]
    public class LayerFactoryCudaD : ILayerFactory {
        public Layer MakeFullyConnectedLayer(int x_size, int y_size) {
            return new Layer(DLL.MakeFullyConnectedLayerCudaD(x_size, y_size));
        }

        public Layer MakeConvolutionalLayer(int m_size, int n_size, int k_size, int h_size) {
            return new Layer(DLL.MakeConvolutionalLayerCudaD(m_size, n_size, k_size, h_size));
        }

        public Layer MakeMaxPoolingLayer(int m_size, int n_size, int k_size, int h_size) {
            return new Layer(DLL.MakeMaxPoolingLayerCudaD(m_size, n_size, k_size, h_size));
        }

        public Layer MakeRecurrentLayer(int t_size, int x_size, int y_size) {
            return new Layer(DLL.MakeRecurrentLayerCudaD(t_size, x_size, y_size));
        }

        public Layer MakeLSTMLayer(int t_size, int x_size, int y_size) {
            return new Layer(DLL.MakeLSTMLayerCudaD(t_size, x_size, y_size));
        }
    }

    [ComVisible(true)]
    public interface ILayer {
        int BatchSize {
            get;
            set;
        }

        float LearningRate {
            get;
            set;
        }

        int t {
            get;
            set;
        }

        int GetFieldCount();
        int GetFieldIndexByName(string name);
        string GetFieldName(int field_idx);
        int GetFieldDimension(int field_idx);
        int[] GetFieldSize(int field_idx);
        int GetFieldElementCount(int field_idx);
        void GetFieldValue(int field_idx, IntPtr dst);
        void SetFieldValue(int field_idx, IntPtr src);
        void GetFieldValue(int field_idx, ref float[] dst);
        void SetFieldValue(int field_idx, ref float[] src);
        void GetFieldValue(int field_idx, ref float[,] dst);
        void SetFieldValue(int field_idx, ref float[,] src);

        void Destroy();
        void Forward();
        void Backward();
        void Allocate();
        void Free();
        void UpdateParameter();
        void ConnectLayer(Layer next_layer);
        int GetInputCount();
        int GetOutputCount();
        int GetTimeCount();
        int GetTimeInputCount();
        int GetTimeOutputCount();
        void SetTimeCount(int time_count);
        bool IsGPU();
        void SetInput(IntPtr src);
        IntPtr GetInput(int t = 0);
        IntPtr GetOutput(int t = 0);
        void SetIputDelta(IntPtr src);
        void SetOutputDelta(IntPtr src);
        IntPtr GetOutputDelta(int t = 0);
        IntPtr GetInputDelta(int t = 0);
        void GetInputData(ref object app);
        void SetInputData(ref object app);
        void SetOutputDeltaData(ref object app);
        void GetOutputData(ref object app);
    }

    [ClassInterface(ClassInterfaceType.None)]
    public class Layer : ILayer {
        [DllImport("kernel32.dll")]
        static extern void CopyMemory(IntPtr dst, IntPtr src, uint length);

        protected IntPtr Handle;

        public Layer() {
        }

        public Layer(IntPtr h) {
            Handle = h;
        }


        public int GetFieldCount() {
            return DLL.GetFieldCount(Handle);
        }

        public int GetFieldIndexByName(string name) {
            return DLL.GetFieldIndexByName(Handle, name);
        }

        public string GetFieldName(int field_idx) {
            StringBuilder buf = new StringBuilder(256);
            DLL.GetFieldName(Handle, field_idx, buf);

            return buf.ToString();
        }

        public int GetFieldDimension(int field_idx) {
            return DLL.GetFieldDimension(Handle, field_idx);
        }

        unsafe public int[] GetFieldSize(int field_idx) {
            IntPtr p = DLL.GetFieldSize(Handle, field_idx);
            if (p.ToInt64() == 0) {
                return new int[0];
            }
            int* size_ptr = (int*)p.ToPointer();
            int dim_cnt = GetFieldDimension(field_idx);
            int[] size = new int[dim_cnt];
            for (int i = 0; i < dim_cnt; i++) {
                size[i] = size_ptr[i];
            }

            return size;
        }

        public int GetFieldElementCount(int field_idx) {
            return DLL.GetFieldElementCount(Handle, field_idx);
        }

        public void GetFieldValue(int field_idx, IntPtr dst) {
            DLL.GetFieldValue(Handle, field_idx, dst);
        }

        public void SetFieldValue(int field_idx, IntPtr src) {
            DLL.GetFieldValue(Handle, field_idx, src);
        }

        void AllocFieldValue(int field_idx, out float[] data) {
            int[] sz = GetFieldSize(field_idx);

            data = new float[sz[0]];
        }

        void AllocFieldValue(int field_idx, out float[,] data) {
            int[] sz = GetFieldSize(field_idx);

            data = new float[sz[0], sz[1]];
        }

        unsafe public void GetFieldValue(int field_idx, ref float[] dst) {
            if (dst == null) {
                AllocFieldValue(field_idx, out dst);
            }

            fixed (float* pf = dst) {

                IntPtr int_p = new IntPtr(pf);
                DLL.GetFieldValue(Handle, field_idx, int_p);
            }
        }

        public void SetFieldValue(int field_idx, ref float[] src) {

        }

        unsafe public void GetFieldValue(int field_idx, ref float[,] dst) {
            if (dst == null) {
                AllocFieldValue(field_idx, out dst);
            }

            fixed (float* pf = dst) {

                IntPtr int_p = new IntPtr(pf);
                DLL.GetFieldValue(Handle, field_idx, int_p);
            }
        }

        public void SetFieldValue(int field_idx, ref float[,] src) {
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

        //----------------------------------------------------------------------------------------------------

        public void Destroy() {
            DLL.Destroy(Handle);
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

        void CopyData(IntPtr dev, IntPtr app, uint size, CopyDir dir) {
            if (IsGPU()) {

                if(dir == CopyDir.Set) {

                    Cuda.MemcpyHostToDevice(dev, app, size);
                }
                else {

                    Cuda.MemcpyDeviceToHost(app, dev, size);
                }
            }
            else {

                if (dir == CopyDir.Set) {

                    CopyMemory(dev, app, size);
                }
                else {

                    CopyMemory(app, dev, size);
                }
            }
        }

        unsafe void CopyArrayData(IntPtr dev, object app, CopyDir dir) {
            if(!(app is Array)) {

                throw new Exception("CopyArrayData : 引数が配列でありません。");
            }
            Array arr = app as Array;

            if(arr.GetType().GetElementType() == typeof(float)) {
                switch (arr.Rank) {
                case 1:
                    fixed (float* p = (float[])arr) {
                        CopyData(dev, new IntPtr(p), (uint)arr.Length * sizeof(float), dir);
                    }
                    break;

                case 2:
                    fixed (float* p = (float[,])arr) {
                        CopyData(dev, new IntPtr(p), (uint)arr.Length * sizeof(float), dir);
                    }
                    break;

                case 3:
                    fixed (float* p = (float[,,])arr) {
                        CopyData(dev, new IntPtr(p), (uint)arr.Length * sizeof(float), dir);
                    }
                    break;

                case 4:
                    fixed (float* p = (float[,,,])arr) {
                        CopyData(dev, new IntPtr(p), (uint)arr.Length * sizeof(float), dir);
                    }
                    break;

                case 5:
                    fixed (float* p = (float[,,,,])arr) {
                        CopyData(dev, new IntPtr(p), (uint)arr.Length * sizeof(float), dir);
                    }
                    break;

                default:
                    throw new Exception("CopyArrayData : 引数が配列の次元が5を超えています。");
                }
            }
            else if (arr.GetType().GetElementType() == typeof(double)) {
                switch (arr.Rank) {
                case 1:
                    fixed (double* p = (double[])arr) {
                        CopyData(dev, new IntPtr(p), (uint)arr.Length * sizeof(double), dir);
                    }
                    break;

                case 2:
                    fixed (double* p = (double[,])arr) {
                        CopyData(dev, new IntPtr(p), (uint)arr.Length * sizeof(double), dir);
                    }
                    break;

                case 3:
                    fixed (double* p = (double[,,])arr) {
                        CopyData(dev, new IntPtr(p), (uint)arr.Length * sizeof(double), dir);
                    }
                    break;

                case 4:
                    fixed (double* p = (double[,,,])arr) {
                        CopyData(dev, new IntPtr(p), (uint)arr.Length * sizeof(double), dir);
                    }
                    break;

                case 5:
                    fixed (double* p = (double[,,,,])arr) {
                        CopyData(dev, new IntPtr(p), (uint)arr.Length * sizeof(double), dir);
                    }
                    break;

                default:
                    throw new Exception("CopyArrayData : 引数が配列の次元が5を超えています。");
                }
            }
            else {

                throw new Exception("CopyArrayData : 引数の配列の要素型が不正です。");
            }
        }

        unsafe public void SetInputData(ref object app) {
            CopyArrayData(DLL.GetInput(Handle), app, CopyDir.Set);
        }

        unsafe public void GetInputData(ref object app) {
            CopyArrayData(DLL.GetInput(Handle), app, CopyDir.Get);
        }

        public void GetOutputData(ref object app) {
            CopyArrayData(DLL.GetOutput(Handle), app, CopyDir.Get);
        }

        public void SetOutputDeltaData(ref object app) {
            CopyArrayData(DLL.GetOutputDelta(Handle), app, CopyDir.Set);
        }
    }
}
