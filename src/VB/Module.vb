
Imports System
Imports System.Collections.Generic
Imports System.Linq
Imports System.Text
Imports System.Threading.Tasks
Imports LayerNET
Imports System.IO
Imports System.Reflection
Imports System.Diagnostics
Imports System.Runtime.InteropServices

Public Enum NetworkType
    Simple
    CNN
    RNN
    LSTM
End Enum

Module Module1
    Dim Dev As Device

    Dim CharToCharIdx(CHAR_COUNT) As UShort
    Dim CharIdxToChar(CHAR_COUNT) As Char

    Sub Main()
        NetworkTest()
    End Sub

    Public DataDir As String

    Public Type As NetworkType
    Public EpochSize As Integer
    Public TrainBatchSize As Integer
    Public TestBatchSize As Integer
    Public TrainCnt As Integer
    Public TestCnt As Integer
    Public DomainLen As Integer
    Public RangeLen As Integer
    Public TrainLabel As Byte()
    Public TestLabel As Byte()
    Public EpochIdx As Integer
    Public MiniBatchIdx As Integer
    Public CostCount As Integer
    Public UpdateMiniBatchCount As Integer

    Public Layers As List(Of Layer)
    Public FirstLayer As Layer
    Public LastLayer As Layer


    Dim TrainX() As Single
    Dim TrainY() As Single
    Dim TestX() As Single
    Dim TestY() As Single
    Dim CostSum As Single


    Public Sub SetDataDir()
        Dim dir As String = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)

        While (True)
            DataDir = dir + "\data"
            If Directory.Exists(DataDir) Then
                Exit While
            End If

            dir = Path.GetDirectoryName(dir)
        End While
    End Sub


    Public Function BytesToInt(v As Byte(), offset As Integer) As Integer
        Return v(offset) * &H1000000 + v(offset + 1) * &H10000 + v(offset + 2) * &H100 + v(offset + 3)
    End Function

    Public Sub DeepLearning()
        FirstLayer = Layers(0)
        LastLayer = Layers(Layers.Count - 1)

        Select Case Type
            Case NetworkType.Simple
            Case NetworkType.CNN
                SGD()


            Case NetworkType.RNN
            Case NetworkType.LSTM
                RNNSGD()

        End Select

        For i = 0 To Layers.Count - 1
            Layers(i).Destroy()
        Next
        Layers.Clear()
    End Sub

    '  (0, all_count-1)の範囲から、sample_count個の整数を抜き出して、その配列を返します。
    Public Function RandomSampling(all_count As Integer, sample_count As Integer) As Integer()
        Dim ret(sample_count - 1) As Integer

        Dim numbers(all_count - 1) As Integer
        For i = 0 To all_count - 1
            numbers(i) = i
        Next

        Dim r As Random = New Random(0)
        For i = 0 To sample_count - 1
            Dim n As Integer = r.Next(1024 * 1024 * 1024) Mod (all_count - i)

            ret(i) = numbers(n)
            numbers(n) = numbers(all_count - i - 1)
        Next

        'For i As Integer = 0 To sample_count - 1
        '    For j As Integer = i + 1 To sample_count - 1
        '        Debug.Assert(ret(i) <> ret(j))
        '    Next
        'Next

        Return ret
    End Function

    Sub NetworkTest()
        SetDataDir()

        Dev = New Device
        Dev.DeviceInit()

        EpochSize = 100
        TestBatchSize = 20

        Dim factory As LayerFactoryF = New LayerFactoryF()

        Dim learning_rate As Single = 1.0F
        For run_idx As Integer = 0 To Integer.MaxValue
            Type = NetworkType.Simple
            Type = NetworkType.RNN
            Type = NetworkType.CNN
            Type = NetworkType.LSTM

            Select Case Type
                Case NetworkType.Simple
                    TrainBatchSize = 10
                    ReadMNIST()
                    Layers = New List(Of Layer) From {
                        factory.MakeFullyConnectedLayer(28 * 28, 30),
                        factory.MakeFullyConnectedLayer(30, 10)
                    }

                Case NetworkType.CNN
                    TrainBatchSize = 10
                    ReadMNIST()
                    Layers = New List(Of Layer) From {
                        factory.MakeConvolutionalLayer(28, 28, 5, 5),
                        factory.MakeMaxPoolingLayer(24, 24, 5, 2),
                        factory.MakeFullyConnectedLayer(12 * 12 * 5, 100),
                        factory.MakeFullyConnectedLayer(100, 10)
                    }

                Case NetworkType.RNN
                    learning_rate = 0.1F
                    TrainBatchSize = 7
                    Layers = New List(Of Layer) From {
                        factory.MakeRecurrentLayer(20, 28, 100),
                        factory.MakeFullyConnectedLayer(10, 28)
                    }

                Case NetworkType.LSTM
                    learning_rate = 0.1F
                    TrainBatchSize = 7
                    Layers = New List(Of Layer) From {
                        factory.MakeLSTMLayer(20, 28, 100),
                        factory.MakeFullyConnectedLayer(100, 28)
                    }
            End Select

            For i = 0 To Layers.Count - 1
                Layers(i).LearningRate = learning_rate / TrainBatchSize '3.0f
            Next

            DeepLearning()
        Next
    End Sub

    Public Sub ReadMNIST()
        Dim mnist_dir As String = DataDir + "\MNIST\"

        Dim buf As Byte() = File.ReadAllBytes(mnist_dir + "train-images.idx3-ubyte")

        TrainCnt = BytesToInt(buf, 4)
        Dim img_h As Integer = BytesToInt(buf, 8)
        Dim img_w As Integer = BytesToInt(buf, 12)

        DomainLen = img_h * img_w
        RangeLen = 10

        Dim sz As Integer = TrainCnt * DomainLen
        TrainX = New Single(sz) {}
        For i = 0 To sz - 1
            TrainX(i) = buf(16 + i) / 256.0F
        Next


        buf = File.ReadAllBytes(mnist_dir + "train-labels.idx1-ubyte")
        TrainLabel = New Byte(TrainCnt) {}
        Array.Copy(buf, 8, TrainLabel, 0, TrainCnt)


        buf = File.ReadAllBytes(mnist_dir + "t10k-images.idx3-ubyte")
        TestCnt = BytesToInt(buf, 4)
        Dim test_sz As Integer = TestCnt * img_h * img_w
        TestX = New Single(test_sz) {}
        For i = 0 To test_sz - 1
            TestX(i) = buf(16 + i) / 256.0F
        Next


        buf = File.ReadAllBytes(mnist_dir + "t10k-labels.idx1-ubyte")
        TestLabel = New Byte(TestCnt) {}
        Array.Copy(buf, 8, TestLabel, 0, TestCnt)
    End Sub

    ' すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
    Sub AllocateConnectLayers(batch_size As Integer)
        Dim p As IntPtr = Dev.DeviceMalloc(batch_size * DomainLen * Marshal.SizeOf(Of Single))
        FirstLayer.SetInput(p)

        p = Dev.DeviceMalloc(batch_size * RangeLen * Marshal.SizeOf(Of Single))
        LastLayer.SetOutputDelta(p)

        For i = 0 To Layers.Count - 1
            Layers(i).BatchSize = batch_size
            Layers(i).Allocate()
        Next

        ' レイヤーの入出力を結合します。
        For i = 0 To Layers.Count - 2

            If Layers(i).GetTimeCount() = 0 Then
                Debug.Assert(Layers(i).GetOutputCount() = Layers(i + 1).GetInputCount())

                ' 次のレイヤーの入力は、現在のレイヤーの出力にします。(順伝播)
                Layers(i + 1).SetInput(Layers(i).GetOutput())

                ' 現在のレイヤーの出力のデルタは、次のレイヤーの入力のデルタにします。(逆伝播)
                Layers(i).SetOutputDelta(Layers(i + 1).GetInputDelta())
            End If

            Layers(i).ConnectLayer(Layers(i + 1))
        Next
    End Sub


    Sub SetBatchData(X() As Single, batch_X() As Single, batch_Y() As Single, label() As Byte, batch_size As Integer, idxes() As Integer)
        For i = 0 To batch_Y.Length - 1
            batch_Y(i) = 0
        Next

        For batch_idx = 0 To batch_size - 1
            Dim idx As Integer = MiniBatchIdx * batch_size + batch_idx

            If idxes IsNot Nothing Then

                idx = idxes(idx)
            End If

            For ix = 0 To DomainLen - 1
                batch_X(ix * batch_size + batch_idx) = X(idx * DomainLen + ix)
            Next

            batch_Y(label(idx) * batch_size + batch_idx) = 1
        Next
    End Sub

    ' 損失関数の微分	        
    Sub CostDerivative(cost_derivative() As Single, last_y() As Single, batch_Y As Single(), size As Integer)
        For i = 0 To size - 1
            cost_derivative(i) = last_y(i) - batch_Y(i)
        Next
    End Sub

    ' 損失関数
    Function Cost(cost_derivative As Single(), size As Integer) As Single
        Dim sum As Double = 0
        For i = 0 To size - 1
            Dim cd As Single = cost_derivative(i)
            sum += cd * cd
        Next

        Dim f1 As Single = CType(sum / size, Single)
        Dim f2 As Single = cost_derivative.Select(Function(x) x * x).Sum() / size
        Debug.Assert(Math.Abs(f1 - f2) < 0.0001)

        Return f2
    End Function

    ' 損失関数の微分

    Function SoftMax(cost_derivative As Single(), last_y As Single(), batch_Y As Single(), exp_work As Single(), range_len As Integer, batch_size As Integer, batch_idx As Integer) As Single
        Dim max_val As Single = -10000
        For i = 0 To range_len - 1
            Dim k As Integer = i * batch_size + batch_idx

            If max_val < last_y(k) Then
                max_val = last_y(k)
            End If
        Next

        Dim sum As Single = 0
        For i = 0 To range_len - 1
            Dim k As Integer = i * batch_size + batch_idx

            Dim d As Single = CType(Math.Exp(last_y(k) - max_val), Single)
            sum += d
            exp_work(k) = d
        Next

        Dim cost_sum As Single = 0
        For i = 0 To range_len - 1
            Dim k As Integer = i * batch_size + batch_idx

            Dim y As Single = exp_work(k) / sum
            cost_derivative(k) = y - batch_Y(k)

            cost_sum += CType(batch_Y(k) * Math.Log(y), Single)
        Next

        Return -cost_sum
    End Function


    '    ミニバッチごとにパラメータを更新します。

    Sub UpdateMiniBatch(batch_X As Single(), batch_Y As Single(), last_y As Single(), cost_derivative As Single())
        '-------------------------------------------------- 入力をセットします。

        FirstLayer.SetInputData(batch_X, DomainLen * TrainBatchSize * Marshal.SizeOf(Of Single))

        For i = 0 To Layers.Count - 1
            Layers(i).Forward()
        Next

        '-------------------------------------------------- 出力を得ます。
        Dim last_y_len As Integer = TrainBatchSize * RangeLen

        LastLayer.GetOutputData(last_y, last_y_len * Marshal.SizeOf(Of Single))

        '-------------------------------------------------- 損失関数を計算します。
        CostDerivative(cost_derivative, last_y, batch_Y, last_y_len)

        Dim cost1 As Single = Cost(cost_derivative, last_y_len)

        '-------------------------------------------------- δyをセットします。
        LastLayer.SetOutputDeltaData(cost_derivative, last_y_len * Marshal.SizeOf(Of Single))

        For i = Layers.Count - 1 To 0 Step -1
            Layers(i).Backward()
        Next
        Dev.DeviceSynchronize()

        For i = Layers.Count - 1 To 0 Step -1
            Layers(i).UpdateParameter()
        Next

        Dev.DeviceSynchronize()
    End Sub


    Function ArgMax(result_Y As Single(), batch_size As Integer, arg_max As Byte(), label As Byte()) As Integer
        Dim eq_cnt As Integer = 0

        For batch_idx = 0 To batch_size - 1

            Dim max_val As Single = -10000
            Dim max_idx As Integer = 0
            For i = 0 To RangeLen - 1
                Dim val As Single = result_Y(i * batch_size + batch_idx)
                If max_val < val Then

                    max_val = val
                    max_idx = i
                End If
            Next

            arg_max(batch_idx) = max_idx

            If max_idx = label(MiniBatchIdx * batch_size + batch_idx) Then
                eq_cnt = eq_cnt + 1
            End If
        Next

        Return eq_cnt
    End Function



    Function Evaluate(batch_X As Single(), batch_Y As Single(), last_y As Single(), batch_size As Integer, arg_max As Byte(), label As Byte()) As Integer
        FirstLayer.SetInputData(batch_X, batch_size * DomainLen * Marshal.SizeOf(Of Single))

        For i = 0 To Layers.Count - 1
            Layers(i).Forward()
        Next

        LastLayer.GetOutputData(last_y, batch_size * RangeLen * Marshal.SizeOf(Of Single))

        Dim eq_cnt As Integer = ArgMax(last_y, batch_size, arg_max, label)

        Return eq_cnt
    End Function

    ' すべてのレイヤーのメモリを解放します。
    Sub FreeLayers()
        For i = 0 To Layers.Count - 1
            Layers(i).Free()
        Next

        Dev.DeviceFree(FirstLayer.GetInput())
        Dev.DeviceFree(LastLayer.GetOutputDelta())
    End Sub


    ' 確率的勾配降下法 (stochastic gradient descent, SGD)

    Sub SGD()

        Dim train_batch_cnt As Integer = TrainCnt / TrainBatchSize
        Dim test_batch_cnt As Integer = TestCnt / TestBatchSize

        Dim train_batch_X(TrainBatchSize * DomainLen - 1) As Single
        Dim train_batch_Y(TrainBatchSize * RangeLen - 1) As Single
        Dim train_last_Y(TrainBatchSize * RangeLen - 1) As Single

        Dim cost_derivative(TrainBatchSize * RangeLen - 1) As Single

        Dim test_batch_X(TestBatchSize * DomainLen - 1) As Single
        Dim test_batch_Y(TestBatchSize * RangeLen - 1) As Single
        Dim test_last_Y(TestBatchSize * RangeLen - 1) As Single

        Dim test_arg_max(TestBatchSize) As Byte

        For EpochIdx = 0 To EpochSize - 1

            Dim idxes As Integer() = RandomSampling(TrainCnt, TrainCnt)

            ' すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
            AllocateConnectLayers(TrainBatchSize)

            For MiniBatchIdx = 0 To train_batch_cnt - 1

                SetBatchData(TrainX, train_batch_X, train_batch_Y, TrainLabel, TrainBatchSize, idxes)

                UpdateMiniBatch(train_batch_X, train_batch_Y, train_last_Y, cost_derivative)
            Next

            FreeLayers()

            ' すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
            AllocateConnectLayers(TestBatchSize)

            Dim eq_cnt_sum As Integer = 0
            For MiniBatchIdx = 0 To test_batch_cnt - 1

                SetBatchData(TestX, test_batch_X, test_batch_Y, TestLabel, TestBatchSize, Nothing)

                Dim eq_cnt As Integer = Evaluate(test_batch_X, test_batch_Y, test_last_Y, TestBatchSize, test_arg_max, TestLabel)
                eq_cnt_sum += eq_cnt
            Next
            Debug.WriteLine("epoch {0} : {1} / {2}", EpochIdx, eq_cnt_sum, TestCnt)

            FreeLayers()

        Next
    End Sub

    Const CHAR_COUNT As Integer = &HFFFF
    Const INVALID_CHAR_IDX As UShort = &HFFFF
    Const INVALID_CHAR As Char = ChrW(INVALID_CHAR_IDX)

    Sub ReadCharTable()
        Dim char_path As String = DataDir + "\aozorabunko\char.txt"

        Dim buf As String = File.ReadAllText(char_path, Encoding.Unicode)

        ' 文字テーブルを-1でクリアします。
        For i = 0 To CHAR_COUNT - 1
            CharToCharIdx(i) = INVALID_CHAR_IDX
            CharIdxToChar(i) = ChrW(0)
        Next

        For i = 0 To buf.Length - 1
            CharToCharIdx(AscW(buf(i))) = i
            CharIdxToChar(i) = buf(i)
        Next
    End Sub

    Dim LineIdx As Integer
    Dim CharBuf As Char()


    Function InitText(batch_size As Integer, line_len As Integer) As Integer
        Dim text_path As String
        Dim line_NL_len As Integer = line_len + 1


        text_path = String.Format("{0}\aozorabunko\Line-{1}.txt", DataDir, line_len)
        If Not File.Exists(text_path) Then
            ' ファイルが存在しない場合

            Return 0
        End If

        Dim buf As Char() = File.ReadAllText(text_path, Encoding.Unicode).ToArray()

        Dim line_count As Integer = buf.Length / line_NL_len

        Dim train_cnt As Integer = (line_count / batch_size) * batch_size

        CharBuf = New Char(buf.Length) {}

        Dim idxes As Integer() = RandomSampling(train_cnt, train_cnt)
        For i = 0 To train_cnt - 1

            Array.Copy(buf, idxes(i) * line_NL_len, CharBuf, i * line_NL_len, line_NL_len)
        Next

        If CharToCharIdx Is Nothing Then
            Return train_cnt
        End If

        For i = 0 To CHAR_COUNT - 1
            CharToCharIdx(i) = 0
            CharIdxToChar(i) = ChrW(0)
        Next

        Dim char_idx As Integer = 1
        For i = 0 To CharBuf.Length - 1
            Dim ch As Char = CharBuf(i)
            Dim j As Integer = 0
            For j = 0 To char_idx - 1
                If CharIdxToChar(j) = ch Then
                    Exit For
                End If
            Next

            If char_idx <= j Then
                CharToCharIdx(AscW(ch)) = char_idx
                CharIdxToChar(char_idx) = ch

                char_idx += 1
            End If
        Next

        Debug.WriteLine("文字数:{0}", char_idx)

        ' 文字テーブルを-1でクリアします。
        For i = 0 To CHAR_COUNT - 1
            If CharToCharIdx(i) = 0 Then
                CharToCharIdx(i) = INVALID_CHAR_IDX
            End If
        Next

        Return train_cnt
    End Function


    Sub ReadText(batch_size As Integer, line_len As Integer, mini_batch_idx As Integer, text As Char())
        Array.Copy(CharBuf, mini_batch_idx * batch_size * (line_len + 1), text, 0, batch_size * (line_len + 1))
    End Sub


    Sub ClearText()
        CharBuf = Nothing
    End Sub


    Sub CharToOneHotX(batch_X As Single(), Time As Integer, one_hot_size As Integer, batch_size As Integer, text As Char())
        For t = 0 To Time - 1
            For batch_idx = 0 To batch_size - 1
                Dim ch1 As Char = text(batch_idx * (Time + 1 + 1) + t)
                Dim idx1 As Integer = CharToCharIdx(AscW(ch1))
                If idx1 < one_hot_size Then
                    batch_X((t * one_hot_size + idx1) * batch_size + batch_idx) = 1
                End If
            Next
        Next
    End Sub

    Sub CharToOneHotY(batch_Y As Single(), Time As Integer, one_hot_size As Integer, batch_size As Integer, text As Char(), t As Integer)
        For batch_idx = 0 To batch_size - 1
            Dim ch2 As Char = text(batch_idx * (Time + 1 + 1) + t + 1)
            Dim idx2 As Integer = CharToCharIdx(AscW(ch2))
            If idx2 < one_hot_size Then
                batch_Y(idx2 * batch_size + batch_idx) = 1
            End If
        Next
    End Sub

    Sub SetZero(v As Single())
        For i = 0 To v.Length - 1
            v(i) = 0
        Next
    End Sub

    ' ミニバッチごとにパラメータを更新します。

    Sub RNNUpdateMiniBatch(batch_X As Single(), batch_Y As Single(), last_y As Single(), cost_derivative As Single(), exp_work As Single(), text As Char())
        Dim input(100) As Char
        '		wchar_t	input2(100)
        Dim output(100) As Char

        Dim Time As Integer = FirstLayer.GetTimeCount()


        '-------------------------------------------------- 入力をセットします。
        FirstLayer.SetInputData(batch_X, DomainLen * TrainBatchSize * Marshal.SizeOf(Of Single))

        ' 順方向の時刻
        For t = 0 To Time - 1

            SetZero(batch_Y)
            CharToOneHotY(batch_Y, Time, RangeLen, TrainBatchSize, text, t)

            FirstLayer.t = t

            '-------------------------------------------------- レイヤーの入出力を結合します。
            For i = 0 To Layers.Count - 2
                Debug.Assert(Layers(i).GetTimeOutputCount() = Layers(i + 1).GetInputCount())

                ' 次のレイヤーの入力は、現在のレイヤーの出力にします。(順伝播)
                Layers(i + 1).SetInput(Layers(i).GetOutput(t))
            Next

            '-------------------------------------------------- 順伝播
            For i = 0 To Layers.Count - 1
                Layers(i).Forward()
            Next

            'Single() p = (Single())fc.GetOutput(t)
            'for (Integer i = 0; i < fc.Y * TrainBatchSize; i++) {
            '	Log("%f", p(i))
            '}

            '-------------------------------------------------- 出力を得ます。
            Dim last_y_len As Integer = TrainBatchSize * RangeLen
            LastLayer.GetOutputData(last_y, last_y_len * Marshal.SizeOf(Of Single))

            Dim cost1 As Single = 0
            For batch_idx = 0 To TrainBatchSize
                Dim cost2 As Single = SoftMax(cost_derivative, last_y, batch_Y, exp_work, RangeLen, TrainBatchSize, batch_idx)

                cost1 += cost2
            Next
            cost1 /= TrainBatchSize

            '-------------------------------------------------- 損失関数を計算します。
            CostSum += cost1
            CostCount += 1


            If UpdateMiniBatchCount Mod 141 = 0 Then 'MiniBatchIdx
                Dim batch_idx As Integer = UpdateMiniBatchCount Mod TrainBatchSize
                Dim max_val As Single = -10000
                Dim max_idx As Integer = 0
                For i = 0 To RangeLen - 1
                    Dim val As Single = last_y(i * TrainBatchSize + batch_idx)
                    If max_val < val Then

                        max_val = val
                        max_idx = i
                    End If
                Next
                output(t) = CharIdxToChar(max_idx)

                max_val = -10000
                max_idx = 0
                For i = 0 To RangeLen - 1
                    Dim val As Single = batch_Y(i * TrainBatchSize + batch_idx)
                    If max_val < val Then

                        max_val = val
                        max_idx = i
                    End If
                Next
                '				input2(t) = CharIdxToChar(max_idx)


                input(t) = text(batch_idx * (Time + 1 + 1) + t)

                If t = Time - 1 Then

                    Debug.WriteLine("IN : {0}", New String(input, 0, t), "")
                    Debug.WriteLine("OUT: {0}", New String(output, 0, t), "")
                    Debug.WriteLine("epock : {0}  cost : {1}", EpochIdx, CostSum / CostCount)
                End If
            End If

            '-------------------------------------------------- δyをセットします。
            LastLayer.SetOutputDeltaData(cost_derivative, last_y_len * Marshal.SizeOf(Of Single))

            '-------------------------------------------------- 逆伝播
            ' RNN以外のレイヤーの逆伝播をします。
            For i = Layers.Count - 1 To 1 Step -1
                Dim input_delta_save As IntPtr = New IntPtr(0)

                ' 現在のレイヤーの出力のデルタは、次のレイヤーの入力のデルタにします。(逆伝播)
                If Layers(i - 1).GetTimeCount() <> 0 Then

                    input_delta_save = Layers(i).GetInputDelta(t)
                    Layers(i).SetIputDelta(Layers(i - 1).GetOutputDelta(t))

                Else

                    Layers(i - 1).SetOutputDelta(Layers(i).GetInputDelta(t))
                End If

                Layers(i).Backward()

                If input_delta_save.ToInt64() <> 0 Then

                    Layers(i).SetIputDelta(input_delta_save)
                End If
            Next
        Next


        ' 時刻の逆方向に、RNNのレイヤーの逆伝播をします。
        For t = Time - 1 To 0 Step -1
            FirstLayer.t = t
            FirstLayer.Backward()
        Next

        ' パラメータを更新します。
        For i = Layers.Count - 1 To 0 Step -1
            Layers(i).UpdateParameter()
        Next


        'Dmp("y0", (Single())Layers(0).GetOutput(), TrainBatchSize * Layers(0).GetOutputCount())
        'ConvolutionalLayer* cl = (ConvolutionalLayer*)FirstLayer
        'Dmp("h", cl.h, cl.H * cl.H * cl.K)
        'FullyConnectedLayerF* fc = (FullyConnectedLayerF*)Layers(3)
        'Dmp("fc3-w", fc.w, fc.Y * fc.X)


        Dev.DeviceSynchronize()
    End Sub



    '    RNN用SGD

    Sub RNNSGD()
        ReadCharTable()

        CostSum = 0
        CostCount = 0
        UpdateMiniBatchCount = 0
        While (True)

            For EpochIdx = 0 To EpochSize - 1
                Dim time_len As Integer = EpochIdx + 5
                Dim line_len As Integer = time_len + 1

                TrainCnt = InitText(TrainBatchSize, line_len)
                If TrainCnt = 0 Then
                    Exit For
                End If

                FirstLayer.SetTimeCount(time_len)
                Dim X As Integer = FirstLayer.GetTimeInputCount()
                Dim Y As Integer = FirstLayer.GetTimeOutputCount()
                Dim Time As Integer = FirstLayer.GetTimeCount()

                DomainLen = FirstLayer.GetInputCount()
                RangeLen = LastLayer.GetOutputCount()

                Dim train_batch_cnt As Integer = TrainCnt / TrainBatchSize

                Dim train_batch_X(DomainLen * TrainBatchSize) As Single
                Dim train_batch_Y(RangeLen * TrainBatchSize) As Single
                Dim train_last_Y(RangeLen * TrainBatchSize) As Single

                Dim cost_derivative(RangeLen * TrainBatchSize) As Single
                Dim exp_work(RangeLen * TrainBatchSize) As Single

                ' すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
                AllocateConnectLayers(TrainBatchSize)

                Dim delta_y_sz As Integer = TrainBatchSize * Time * Y * Marshal.SizeOf(Of Double)
                Dim out_delta_y As IntPtr = Dev.DeviceMalloc(delta_y_sz)

                FirstLayer.SetOutputDelta(out_delta_y)

                Dim text(TrainBatchSize * (line_len + 1) - 1) As Char

                ' For MiniBatchIdx = 0 To train_batch_cnt - 1
                MiniBatchIdx = 0
                While MiniBatchIdx = 0

                    ReadText(TrainBatchSize, line_len, MiniBatchIdx, text)

                    SetZero(train_batch_X)
                    CharToOneHotX(train_batch_X, Time, X, TrainBatchSize, text)

                    RNNUpdateMiniBatch(train_batch_X, train_batch_Y, train_last_Y, cost_derivative, exp_work, text)
                    UpdateMiniBatchCount += 1

                    If MiniBatchIdx Mod 100 = 0 Then
                        'Log("epock : %d   mini batch: %d  cost : %f", EpochIdx, MiniBatchIdx, CostSum / CostCount)
                    End If
                End While

                FreeLayers()
                Dev.DeviceFree(out_delta_y)

                'Log("epock : %d  cost : %f", EpochIdx, CostSum / CostCount)
            Next

        End While
    End Sub



End Module
