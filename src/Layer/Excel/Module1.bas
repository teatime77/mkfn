Attribute VB_Name = "Module1"
Option Explicit

Declare PtrSafe Function GetTickCount64 Lib "kernel32" () As LongLong

Public Const IsCuda As Boolean = True


Public CorrectAns() As Long

Public MySeries As Series


Enum NetworkType
    Simple = 1
    CNN = 2
    RNN = 3
    LSTM = 4
End Enum

Const SizeOfSingle As Long = 4
Const SizeOfDouble As Long = 8

Const LONG_MAX As Long = 2147483647
    Const CHAR_COUNT As Long = 256& * 256&
    Const INVALID_CHAR_IDX As Long = &HFFFF
'    Const INVALID_CHAR As Char = ChrW(INVALID_CHAR_IDX)

    Dim FSO As FileSystemObject
    Dim Dev As Device

    Dim CharToCharIdx(CHAR_COUNT - 1) As Long
    Dim CharIdxToChar(CHAR_COUNT - 1) As Long

    Public DataDir As String

    Public NetType As NetworkType
    Public EpochSize As Long
    Public TrainBatchSize As Long
    Public TestBatchSize As Long
    Public TrainCnt As Long
    Public TestCnt As Long
    Public DomainLen As Long
    Public RangeLen As Long
    Public TrainLabel() As Byte
    Public TestLabel() As Byte
    Public EpochIdx As Long
    Public MiniBatchIdx As Long
    Public CostCount As Long
    Public UpdateMiniBatchCount As Long

    Public Layers() As Layer
    Public FirstLayer As Layer
    Public LastLayer As Layer


    Dim TrainX() As Single
    Dim TrainY() As Single
    Dim TestX() As Single
    Dim TestY() As Single
    Dim CostSum As Single

    Dim LineIdx As Long
    Dim CharBuf As String
    Dim LastTick As LongLong
    Public StopNetworkTest As Boolean

    Public Sub SetDataDir()
        Set FSO = CreateObject("Scripting.FileSystemObject")
        Dim path As String: path = FSO.GetParentFolderName(ThisWorkbook.FullName)

        Do While True
            DataDir = path + "\data"
            If dir(DataDir, vbDirectory) <> "" Then
                Exit Do
            End If

        path = FSO.GetParentFolderName(path)
        Loop
        
'        Set FSO = Nothing
    End Sub


    Public Function BytesToInt(v() As Byte, offset As Long) As Long
        BytesToInt = v(offset) * &H1000000 + v(offset + 1) * &H10000 + v(offset + 2) * &H100& + v(offset + 3)
    End Function

    Public Sub DeepLearning()
        Dim i As Long
        
        Set FirstLayer = Layers(0)
        Set LastLayer = Layers(UBound(Layers))

        
        Dim cnt As Long: cnt = FirstLayer.GetFieldCount()
        For i = 0 To cnt - 1
            Dim name As String: name = FirstLayer.GetFieldName(i)
            Debug.Assert FirstLayer.GetFieldIndexByName(name) = i
            Dim v As Variant: v = FirstLayer.GetFieldSize(i)
            Dim s As String: s = IIf(IsArray2(v), "S", "A")
            Debug.Print name, FirstLayer.GetFieldDimension(i), FirstLayer.GetFieldElementCount(i), s
        Next

        Select Case NetType
            Case NetworkType.Simple
            Case NetworkType.CNN
                ReadMNIST
                SGD

            Case NetworkType.RNN
            Case NetworkType.LSTM
                RNNSGD

        End Select

        For i = 0 To UBound(Layers)
            Layers(i).Destroy
        Next
        Erase Layers
    End Sub

    '  (0, all_count-1)の範囲から、sample_count個の整数を抜き出して、その配列を返します。
    Public Function RandomSampling(all_count As Long, sample_count As Long) As Long()
        Dim i As Long
        
        ReDim ret(sample_count - 1) As Long

        ReDim numbers(all_count - 1) As Long
        For i = 0 To all_count - 1
            numbers(i) = i
        Next

        For i = 0 To sample_count - 1
            Dim n As Long: n = CLng(Rnd * 1024 * 1024 * 1024) Mod (all_count - i)

            ret(i) = numbers(n)
            numbers(n) = numbers(all_count - i - 1)
        Next

        'For i As Long = 0 To sample_count - 1
        '    For j As Long = i + 1 To sample_count - 1
        '        Debug.Assert(ret(i) <> ret(j))
        '    Next
        'Next

        RandomSampling = ret
    End Function
    
    Public Function IsArray2(x As Variant) As Boolean
        Dim n As Long: n = -1
        On Error Resume Next
        n = UBound(x)
        Err.Clear
        IsArray2 = (n <> -1)
    End Function

Sub Msg(s As String)
        ThisWorkbook.ActiveSheet.Cells(2, 1) = s
End Sub


Public Function ReadAllBytes(path As String) As Byte()
    Dim buf() As Byte
    
    Open path For Binary As #1
    ReDim buf(LOF(1))
    Get #1, , buf
    Close #1
    
    ReadAllBytes = buf
End Function


Public Sub WriteAllBytes(path As String, ByRef buf() As Byte)
    Open path For Binary As #1
    Put #1, , buf
    Close #1
End Sub

    Public Sub ReadMNIST()
        Dim mnist_dir As String: mnist_dir = DataDir + "\MNIST\"

        Dim buf() As Byte, i As Long, j As Long, idx As Long
        
        buf = ReadAllBytes(mnist_dir + "train-images.idx3-ubyte")

        TrainCnt = BytesToInt(buf, 4)
        Dim img_h As Long: img_h = BytesToInt(buf, 8)
        Dim img_w As Long: img_w = BytesToInt(buf, 12)

        DomainLen = img_h * img_w
        RangeLen = 10

        ReDim TrainX(TrainCnt - 1, DomainLen - 1)
        idx = 0
        For i = 0 To TrainCnt - 1
            For j = 0 To DomainLen - 1
                TrainX(i, j) = buf(16 + idx) / 256!
                idx = idx + 1
            Next
        Next


        buf = ReadAllBytes(mnist_dir + "train-labels.idx1-ubyte")
        ReDim TrainLabel(TrainCnt - 1)
        'Array.Copy(buf, 8, TrainLabel, 0, TrainCnt)
        For i = 0 To TrainCnt - 1
            TrainLabel(i) = buf(8 + i)
        Next


        buf = ReadAllBytes(mnist_dir + "t10k-images.idx3-ubyte")
        TestCnt = BytesToInt(buf, 4)
        ReDim TestX(TestCnt - 1, DomainLen - 1)
        
        idx = 0
        For i = 0 To TestCnt - 1
            For j = 0 To DomainLen - 1
                TestX(i, j) = buf(16 + idx) / 256!
                idx = idx + 1
            Next
        Next


        buf = ReadAllBytes(mnist_dir + "t10k-labels.idx1-ubyte")
        ReDim TestLabel(TestCnt - 1)
        'Array.Copy(buf, 8, TestLabel, 0, TestCnt)
        For i = 0 To TestCnt - 1
            TestLabel(i) = buf(8 + i)
        Next
    End Sub

    ' すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
    Sub AllocateConnectLayers(batch_size As Long)
        Dim i As Long
        Dim p As LongPtr: p = Dev.DeviceMalloc(batch_size * DomainLen * SizeOfSingle)
        FirstLayer.SetInput (p)

        p = Dev.DeviceMalloc(batch_size * RangeLen * SizeOfSingle)
        LastLayer.SetOutputDelta (p)

        For i = 0 To UBound(Layers)
            Layers(i).BatchSize = batch_size
            Layers(i).Allocate
        Next

        ' レイヤーの入出力を結合します。
        For i = 0 To UBound(Layers) - 1

            If Layers(i).GetTimeCount() = 0 Then
                Debug.Assert (Layers(i).GetOutputCount() = Layers(i + 1).GetInputCount())

                ' 次のレイヤーの入力は、現在のレイヤーの出力にします。(順伝播)
                Layers(i + 1).SetInput (Layers(i).GetOutput())

                ' 現在のレイヤーの出力のデルタは、次のレイヤーの入力のデルタにします。(逆伝播)
                Layers(i).SetOutputDelta (Layers(i + 1).GetInputDelta())
            End If

            Layers(i).ConnectLayer Layers(i + 1)
        Next
    End Sub


    Sub SetBatchData(x() As Single, batch_X() As Single, batch_Y() As Single, label() As Byte, batch_size As Long, idxes() As Long)
        Dim i As Long, batch_idx As Long
        Dim is_array As Boolean: is_array = IsArray2(idxes)
        
        For i = 0 To UBound(batch_Y, 1)
            For batch_idx = 0 To UBound(batch_Y, 2)
                batch_Y(i, batch_idx) = 0
            Next
        Next

        For batch_idx = 0 To batch_size - 1
            Dim idx As Long: idx = MiniBatchIdx * batch_size + batch_idx

            If is_array Then

                idx = idxes(idx)
            End If

            For i = 0 To DomainLen - 1
                batch_X(i, batch_idx) = x(idx, i)
            Next

            batch_Y(label(idx), batch_idx) = 1
        Next
    End Sub

    ' 損失関数の微分
    Sub CostDerivative(cost_derivative() As Single, last_y() As Single, batch_Y() As Single)
        Dim i As Long, batch_idx As Long
        
        For i = 0 To UBound(last_y, 1)
            For batch_idx = 0 To UBound(last_y, 2)
                cost_derivative(i, batch_idx) = last_y(i, batch_idx) - batch_Y(i, batch_idx)
            Next
        Next
    End Sub


    ' 損失関数
    Function Cost(cost_derivative() As Single) As Single
        Dim i As Long, batch_idx As Long, cd As Single
        Dim sum As Double: sum = 0
        
        For i = 0 To UBound(cost_derivative, 1)
            For batch_idx = 0 To UBound(cost_derivative, 2)
            
                cd = cost_derivative(i, batch_idx)
                sum = sum + cd * cd
            Next
        Next

        Dim size As Long: size = (UBound(cost_derivative, 1) + 1) * (UBound(cost_derivative, 2) + 1)
        Cost = CSng(sum / size)
    End Function

    ' 損失関数の微分

    Function SoftMax(cost_derivative() As Single, last_y() As Single, batch_Y() As Single, exp_work() As Single, range_len As Long, batch_size As Long, ByVal batch_idx As Integer) As Single
        Dim max_val As Single: max_val = -10000
        Dim k As Long, i As Long
        
        For i = 0 To range_len - 1
            k = i * batch_size + batch_idx

            If max_val < last_y(k) Then
                max_val = last_y(k)
            End If
        Next

        Dim sum As Single: sum = 0
        For i = 0 To range_len - 1
            k = i * batch_size + batch_idx

            Dim d As Single: d = CSng(Math.Exp(last_y(k) - max_val))
            sum = sum + d
            exp_work(k) = d
        Next

        Dim cost_sum As Single: cost_sum = 0
        For i = 0 To range_len - 1
            k = i * batch_size + batch_idx

            Dim y As Single: y = exp_work(k) / sum
            cost_derivative(k) = y - batch_Y(k)

            cost_sum = cost_sum + CSng(batch_Y(k) * Math.Log(y))
        Next

        SoftMax = -cost_sum
    End Function


    '    ミニバッチごとにパラメータを更新します。

    Sub UpdateMiniBatch(batch_X() As Single, batch_Y() As Single, last_y() As Single, cost_derivative() As Single)
        Dim i As Long
        Dim batch_size As Long: batch_size = TrainBatchSize
        
        ' 入力をセットします。
        FirstLayer.SetInputData batch_X

        ' 順伝播の計算をします。
        For i = 0 To UBound(Layers)
            Layers(i).Forward
        Next
        
        Dev.DeviceSynchronize

        ' 出力を得ます。
        LastLayer.GetOutputData last_y
                
                
        ' 損失関数の微分を計算します。
        CostDerivative cost_derivative, last_y, batch_Y


        ' 損失関数を計算します。
        Dim cost1 As Single: cost1 = Cost(cost_derivative)


        ' δyをセットします。
        LastLayer.SetOutputDeltaData cost_derivative

        ' 誤差逆伝播の計算をします。
        For i = UBound(Layers) To 0 Step -1
            Layers(i).Backward
        Next
        
        Dev.DeviceSynchronize

        ' パラメータを更新します。
        For i = UBound(Layers) To 0 Step -1
            Layers(i).UpdateParameter
        Next

        Dev.DeviceSynchronize
    End Sub


    Function ArgMax(result_Y() As Single, batch_size As Long, arg_max() As Byte, label() As Byte) As Long
        Dim batch_idx As Long, i As Long
        Dim eq_cnt As Long: eq_cnt = 0

        For batch_idx = 0 To batch_size - 1

            Dim max_val As Single: max_val = -10000
            Dim max_idx As Long: max_idx = 0
            For i = 0 To RangeLen - 1
                Dim val As Single: val = result_Y(i, batch_idx)
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

        ArgMax = eq_cnt
    End Function



    Function Evaluate(batch_X() As Single, batch_Y() As Single, last_y() As Single, batch_size As Long, arg_max() As Byte, label() As Byte) As Long
        Dim i As Long
        
        FirstLayer.SetInputData batch_X

        For i = 0 To UBound(Layers)
            Layers(i).Forward
        Next

        LastLayer.GetOutputData last_y

        Dim eq_cnt As Long: eq_cnt = ArgMax(last_y, batch_size, arg_max, label)

        Evaluate = eq_cnt
    End Function

    ' すべてのレイヤーのメモリを解放します。
    Sub FreeLayers()
        Dim i As Long
        
        For i = 0 To UBound(Layers)
            Layers(i).Free
        Next

        Dev.DeviceFree (FirstLayer.GetInput())
        Dev.DeviceFree (LastLayer.GetOutputDelta())
    End Sub

    Sub HandleDoEvents()
        If LastTick = 0 Then
            LastTick = GetTickCount64
        Else
            Dim t As LongLong: t = GetTickCount64() - LastTick
            If 1000 < t Then
            
              DoEvents
              LastTick = GetTickCount64
            End If
        End If
    End Sub

    ' 確率的勾配降下法 (stochastic gradient descent, SGD)

    Sub SGD()

        Dim train_batch_cnt As Long: train_batch_cnt = TrainCnt / TrainBatchSize
        Dim test_batch_cnt As Long: test_batch_cnt = TestCnt / TestBatchSize

        ReDim train_batch_X(DomainLen - 1, TrainBatchSize - 1) As Single
        ReDim train_batch_Y(RangeLen - 1, TrainBatchSize - 1) As Single
        ReDim train_last_Y(RangeLen - 1, TrainBatchSize - 1) As Single

        ReDim cost_derivative(RangeLen - 1, TrainBatchSize - 1) As Single

        ReDim test_batch_X(DomainLen - 1, TestBatchSize - 1) As Single
        ReDim test_batch_Y(RangeLen - 1, TestBatchSize - 1) As Single
        ReDim test_last_Y(RangeLen - 1, TestBatchSize - 1) As Single

        ReDim test_arg_max(TestBatchSize) As Byte

        Dim idxes() As Long
    
        For EpochIdx = 0 To EpochSize - 1

            idxes = RandomSampling(TrainCnt, TrainCnt)

            ' すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
            AllocateConnectLayers (TrainBatchSize)

            For MiniBatchIdx = 0 To train_batch_cnt - 1

                SetBatchData TrainX, train_batch_X, train_batch_Y, TrainLabel, TrainBatchSize, idxes

                UpdateMiniBatch train_batch_X, train_batch_Y, train_last_Y, cost_derivative
                
                
                If StopNetworkTest Then
                    Exit For
                End If
                HandleDoEvents
            Next

            FreeLayers

            Erase idxes
            
            If StopNetworkTest Then
                Exit For
            End If

            ' すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
            AllocateConnectLayers (TestBatchSize)

            Dim eq_cnt_sum As Long: eq_cnt_sum = 0
            For MiniBatchIdx = 0 To test_batch_cnt - 1

                SetBatchData TestX, test_batch_X, test_batch_Y, TestLabel, TestBatchSize, idxes

                Dim eq_cnt As Long: eq_cnt = Evaluate(test_batch_X, test_batch_Y, test_last_Y, TestBatchSize, test_arg_max, TestLabel)
                eq_cnt_sum = eq_cnt_sum + eq_cnt
            Next
            Msg "epoch " + Format(EpochIdx) + " : " + Format(eq_cnt_sum) + " / " + Format(TestCnt)
            AddCorrectAns eq_cnt_sum

            FreeLayers
        Next
    End Sub


'    Sub Main()
'        NetworkTest
'    End Sub

    Function ReadAllText(path As String) As String
        Dim buf As String
        Dim ts As TextStream: Set ts = FSO.GetFile(path).OpenAsTextStream(ForReading, TristateTrue)
        
        buf = ts.ReadAll
        ts.Close
        
        
        ReadAllText = buf
    End Function

    Function AscW2(s As String) As Long
        Dim n As Long: n = AscW(s)
        If n < 0 Then
            n = n + &H10000
        End If
        AscW2 = n
    End Function
    
    Sub ReadCharTable()
        Dim i As Long
        Dim char_path As String: char_path = DataDir + "\aozorabunko\char.txt"

        Dim buf As String: buf = ReadAllText(char_path) ' Encoding.Unicode

        ' 文字テーブルを-1でクリアします。
        For i = 0 To CHAR_COUNT - 1
            CharToCharIdx(i) = INVALID_CHAR_IDX
            CharIdxToChar(i) = 0
        Next

        For i = 0 To Len(buf) - 1
            Dim ach As Long: ach = AscW2(Mid(buf, 1 + i, 1))
            CharToCharIdx(ach) = i
            CharIdxToChar(i) = ach
        Next
    End Sub

    Function InitText(batch_size As Long, line_len As Long) As Long
        Dim text_path As String
        Dim line_NL_len As Long: line_NL_len = line_len + 1
        Dim i As Long

        text_path = DataDir + "\aozorabunko\Line-" + Format(line_len) + ".txt"
        If dir(text_path, vbNormal) = "" Then
            ' ファイルが存在しない場合

            InitText = 0
            Exit Function
        End If

        Dim buf As String: buf = ReadAllText(text_path) ', Encoding.Unicode

        Dim line_count As Long: line_count = Len(buf) / line_NL_len

        Dim train_cnt As Long: train_cnt = (line_count / batch_size) * batch_size

        CharBuf = String(Len(buf), ChrW(0))

        Dim idxes() As Long: idxes = RandomSampling(train_cnt, train_cnt)
        For i = 0 To train_cnt - 1
            Dim src As Long: src = idxes(i) * line_NL_len
            Dim dst As Long: dst = i * line_NL_len
'            For k = 0 To line_NL_len - 1
'                CharBuf(dst + k) = buf(src + k)
'            Next
            Mid(CharBuf, 1 + dst, line_NL_len) = Mid(buf, 1 + src, line_NL_len)
        Next

'        If CharToCharIdx Is Nothing Then
'            InitText = train_cnt
'            Exit Function
'        End If

        For i = 0 To CHAR_COUNT - 1
            CharToCharIdx(i) = 0
            CharIdxToChar(i) = 0
        Next

        Dim char_idx As Long: char_idx = 1
        For i = 0 To Len(CharBuf) - 1
            Dim ch As String: ch = Mid(CharBuf, 1 + i, 1)
            Dim ach As Long: ach = AscW2(ch)
            Dim j As Long: j = 0
            For j = 0 To char_idx - 1
                If CharIdxToChar(j) = ach Then
                    Exit For
                End If
            Next

            If char_idx <= j Then
                CharToCharIdx(ach) = char_idx
                CharIdxToChar(char_idx) = ach

                char_idx = char_idx + 1
            End If
        Next

        Msg "文字数:" + Format(char_idx)

        ' 文字テーブルを-1でクリアします。
        For i = 0 To CHAR_COUNT - 1
            If CharToCharIdx(i) = 0 Then
                CharToCharIdx(i) = INVALID_CHAR_IDX
            End If
        Next

        InitText = train_cnt
    End Function

    Sub ReadText(batch_size As Long, line_len As Long, mini_batch_idx As Long, text As String)
        Dim src As Long: src = mini_batch_idx * batch_size * (line_len + 1)
        Dim len1 As Long: len1 = batch_size * (line_len + 1)
'        For k = 0 To len1 - 1
'            text(k) = CharBuf(src + k)
'        Next
        Mid(text, 1, len1) = Mid(CharBuf, 1 + src, len1)
    End Sub


    Sub ClearText()
        CharBuf = ""
    End Sub

    Sub CharToOneHotX(batch_X() As Single, Time As Long, one_hot_size As Long, batch_size As Long, text As String)
        Dim t As Long, batch_idx As Long
        
        For t = 0 To Time - 1
            For batch_idx = 0 To batch_size - 1
                Dim ch1 As String: ch1 = Mid(text, 1 + batch_idx * (Time + 1 + 1) + t, 1)
                Dim idx1 As Long: idx1 = CharToCharIdx(AscW2(ch1))
                If idx1 < one_hot_size Then
                    batch_X((t * one_hot_size + idx1) * batch_size + batch_idx) = 1
                End If
            Next
        Next
    End Sub

    Sub CharToOneHotY(batch_Y() As Single, Time As Long, one_hot_size As Long, batch_size As Long, text As String, ByVal t As Integer)
        Dim batch_idx As Long
        
        For batch_idx = 0 To batch_size - 1
            Dim ch2 As String: ch2 = Mid(text, 1 + batch_idx * (Time + 1 + 1) + t + 1, 1)
            Dim idx2 As Long: idx2 = CharToCharIdx(AscW2(ch2))
            If idx2 < one_hot_size Then
                batch_Y(idx2 * batch_size + batch_idx) = 1
            End If
        Next
    End Sub

    Sub SetZero(v() As Single)
        Dim i As Long
        
        For i = 0 To UBound(v)
            v(i) = 0
        Next
    End Sub

    ' ミニバッチごとにパラメータを更新します。

    Sub RNNUpdateMiniBatch(batch_X() As Single, batch_Y() As Single, last_y() As Single, cost_derivative() As Single, exp_work() As Single, text As String)
        Dim input_text As String * 100
        '       wchar_t input2(100)
        Dim output_text As String * 100
        Dim batch_idx As Long
        Dim val As Single

        Dim Time As Long: Time = FirstLayer.GetTimeCount()
        Dim t As Long, i As Long

        '-------------------------------------------------- 入力をセットします。
        FirstLayer.SetInputData batch_X

        ' 順方向の時刻
        For t = 0 To Time - 1

            SetZero batch_Y
            CharToOneHotY batch_Y, Time, RangeLen, TrainBatchSize, text, t

            FirstLayer.t = t

            '-------------------------------------------------- レイヤーの入出力を結合します。
            For i = 0 To UBound(Layers) - 1
                Debug.Assert (Layers(i).GetTimeOutputCount() = Layers(i + 1).GetInputCount())

                ' 次のレイヤーの入力は、現在のレイヤーの出力にします。(順伝播)
                Layers(i + 1).SetInput (Layers(i).GetOutput(t))
            Next

            '-------------------------------------------------- 順伝播
            For i = 0 To UBound(Layers)
                Layers(i).Forward
            Next

            'Single() p = (Single())fc.GetOutput(t)
            'for (Long i = 0; i < fc.Y * TrainBatchSize; i++) {
            '   Log("%f", p(i))
            '}

            '-------------------------------------------------- 出力を得ます。
            Dim last_y_len As Long: last_y_len = TrainBatchSize * RangeLen
            LastLayer.GetOutputData last_y

            Dim cost1 As Single: cost1 = 0
            For batch_idx = 0 To TrainBatchSize
                Dim cost2 As Single: cost2 = SoftMax(cost_derivative, last_y, batch_Y, exp_work, RangeLen, TrainBatchSize, batch_idx)

                cost1 = cost1 + cost2
            Next
            cost1 = cost1 / TrainBatchSize

            '-------------------------------------------------- 損失関数を計算します。
            CostSum = CostSum + cost1
            CostCount = CostCount + 1

            If StopNetworkTest Then
                Exit Sub
            End If
            HandleDoEvents

            If UpdateMiniBatchCount Mod 141 = 0 Then 'MiniBatchIdx
                batch_idx = UpdateMiniBatchCount Mod TrainBatchSize
                Dim max_val As Single: max_val = -10000
                Dim max_idx As Long: max_idx = 0
                For i = 0 To RangeLen - 1
                    val = last_y(i * TrainBatchSize + batch_idx)
                    If max_val < val Then

                        max_val = val
                        max_idx = i
                    End If
                Next
                Mid(output_text, 1 + t, 1) = ChrW(CharIdxToChar(max_idx))

                max_val = -10000
                max_idx = 0
                For i = 0 To RangeLen - 1
                    val = batch_Y(i * TrainBatchSize + batch_idx)
                    If max_val < val Then

                        max_val = val
                        max_idx = i
                    End If
                Next
                '               input2(t) = CharIdxToChar(max_idx)


                Mid(input_text, 1 + t, 1) = Mid(text, 1 + batch_idx * (Time + 1 + 1) + t, 1)

                If t = Time - 1 Then

                    Debug.Print "IN : " + Left(input_text, t)
                    Debug.Print "OUT: " + Left(output_text, t)
                    Debug.Print "epock : " + Format(EpochIdx) + "  cost : " + Format(CostSum / CostCount)
                End If
            End If

            '-------------------------------------------------- δyをセットします。
            LastLayer.SetOutputDeltaData cost_derivative

            '-------------------------------------------------- 逆伝播
            ' RNN以外のレイヤーの逆伝播をします。
            For i = UBound(Layers) To 1 Step -1
                Dim input_delta_save As LongPtr: input_delta_save = 0

                ' 現在のレイヤーの出力のデルタは、次のレイヤーの入力のデルタにします。(逆伝播)
                If Layers(i - 1).GetTimeCount() <> 0 Then

                    input_delta_save = Layers(i).GetInputDelta(t)
                    Layers(i).SetIputDelta (Layers(i - 1).GetOutputDelta(t))

                Else

                    Layers(i - 1).SetOutputDelta (Layers(i).GetInputDelta(t))
                End If

                Layers(i).Backward

                If input_delta_save <> 0 Then

                    Layers(i).SetIputDelta (input_delta_save)
                End If
            Next
        Next


        ' 時刻の逆方向に、RNNのレイヤーの逆伝播をします。
        For t = Time - 1 To 0 Step -1
            FirstLayer.t = t
            FirstLayer.Backward
        Next

        ' パラメータを更新します。
        For i = UBound(Layers) To 0 Step -1
            Layers(i).UpdateParameter
        Next


        'Dmp("y0", (Single())Layers(0).GetOutput(), TrainBatchSize * Layers(0).GetOutputCount())
        'ConvolutionalLayer* cl = (ConvolutionalLayer*)FirstLayer
        'Dmp("h", cl.h, cl.H * cl.H * cl.K)
        'FullyConnectedLayerF* fc = (FullyConnectedLayerF*)Layers(3)
        'Dmp("fc3-w", fc.w, fc.Y * fc.X)


        Dev.DeviceSynchronize
    End Sub



    '    RNN用SGD

    Sub RNNSGD()
        ReadCharTable

        CostSum = 0
        CostCount = 0
        UpdateMiniBatchCount = 0
        Do While True

            For EpochIdx = 0 To EpochSize - 1
                Dim time_len As Long: time_len = EpochIdx + 5
                Dim line_len As Long: line_len = time_len + 1

                TrainCnt = InitText(TrainBatchSize, line_len)
                If TrainCnt = 0 Then
                    Exit For
                End If

                FirstLayer.SetTimeCount (time_len)
                Dim x As Long: x = FirstLayer.GetTimeInputCount()
                Dim y As Long: y = FirstLayer.GetTimeOutputCount()
                Dim Time As Long: Time = FirstLayer.GetTimeCount()

                DomainLen = FirstLayer.GetInputCount()
                RangeLen = LastLayer.GetOutputCount()

                Dim train_batch_cnt As Long: train_batch_cnt = TrainCnt / TrainBatchSize

                ReDim train_batch_X(DomainLen * TrainBatchSize) As Single
                ReDim train_batch_Y(RangeLen * TrainBatchSize) As Single
                ReDim train_last_Y(RangeLen * TrainBatchSize) As Single

                ReDim cost_derivative(RangeLen * TrainBatchSize) As Single
                ReDim exp_work(RangeLen * TrainBatchSize) As Single

                ' すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
                AllocateConnectLayers (TrainBatchSize)

                Dim delta_y_sz As Long: delta_y_sz = TrainBatchSize * Time * y * SizeOfDouble
                Dim out_delta_y As LongPtr: out_delta_y = Dev.DeviceMalloc(delta_y_sz)

                FirstLayer.SetOutputDelta (out_delta_y)

                Dim text_len As Long: text_len = TrainBatchSize * (line_len + 1) - 1
                Dim text As String: text = String(text_len, ChrW(0))

                ' For MiniBatchIdx = 0 To train_batch_cnt - 1
                MiniBatchIdx = 0
                
                Do While MiniBatchIdx = 0

                    ReadText TrainBatchSize, line_len, MiniBatchIdx, text

                    SetZero train_batch_X
                    CharToOneHotX train_batch_X, Time, x, TrainBatchSize, text

                    RNNUpdateMiniBatch train_batch_X, train_batch_Y, train_last_Y, cost_derivative, exp_work, text
                    UpdateMiniBatchCount = UpdateMiniBatchCount + 1

                    If StopNetworkTest Then
                    
                        Exit Do
                    End If
                Loop

                FreeLayers
                Dev.DeviceFree (out_delta_y)

                If StopNetworkTest Then
                
                    Exit Sub
                End If
            Next
        Loop
    End Sub

Public Sub NetworkTest()
        Dim run_idx As Long, i As Long
        
        Msg "開始しました。"
        
        StopNetworkTest = False
        SetDataDir

        Set Dev = New DeviceCuda
        Dev.DeviceInit

        EpochSize = 100
        TestBatchSize = 20

        Dim factory As LayerFactoryF: Set factory = New LayerFactoryCudaF
        
        Dim learning_rate As Single: learning_rate = 1#
        For run_idx = 0 To LONG_MAX
            NetType = NetworkType.Simple
            NetType = NetworkType.RNN
            NetType = NetworkType.LSTM
            NetType = NetworkType.CNN

            Select Case NetType
                Case NetworkType.Simple
                    TrainBatchSize = 10
                    ReDim Layers(1)
                    
                    Set Layers(0) = factory.MakeFullyConnectedLayer(28 * 28, 30)
                    Set Layers(1) = factory.MakeFullyConnectedLayer(30, 10)

                Case NetworkType.CNN
                    TrainBatchSize = 10
                    
                    ReDim Layers(3)
                    Set Layers(0) = factory.MakeConvolutionalLayer(28, 28, 5, 5)
                    Set Layers(1) = factory.MakeMaxPoolingLayer(24, 24, 5, 2)
                    Set Layers(2) = factory.MakeFullyConnectedLayer(12 * 12 * 5, 100)
                    Set Layers(3) = factory.MakeFullyConnectedLayer(100, 10)

                Case NetworkType.RNN
                    learning_rate = 0.1
                    TrainBatchSize = 7
                    
                    ReDim Layers(1)
                    Set Layers(0) = factory.MakeRecurrentLayer(20, 28, 100)
                    Set Layers(1) = factory.MakeFullyConnectedLayer(10, 28)

                Case NetworkType.LSTM
                    learning_rate = 0.1
                    TrainBatchSize = 7
                    
                    ReDim Layers(1)
                    Set Layers(0) = factory.MakeLSTMLayer(20, 28, 100)
                    Set Layers(1) = factory.MakeFullyConnectedLayer(100, 28)
            End Select

            For i = 0 To UBound(Layers)
                Layers(i).LearningRate = learning_rate / TrainBatchSize '3.0f
            Next

            DeepLearning
            If StopNetworkTest Then
                Exit For
            End If
        Next
        
        Dev.DeviceEnd
    
        Msg "終了しました。"
    End Sub

'型宣言文字  >変数の型
'％  Integer(整数型)
'＆  Long(長整数型)
'＄  String(文字列型)
'！  Single(単精度浮動小数点型)
'＃  Double(倍精度浮動小数点型)
'＠  Currency(通貨型)

' Data Type Summary
' https://msdn.microsoft.com/ja-jp/library/office/gg251528.aspx


Public Sub AddArray(arr As Variant, x As Variant)
    If Not IsArray2(arr) Then
        ReDim arr(0)
    Else
        ReDim Preserve arr(UBound(arr) + 1)
    End If
    
    arr(UBound(arr)) = x
End Sub

Public Sub AddCorrectAns(n As Long)
    AddArray CorrectAns, n

    With MySeries
        .Values = CorrectAns
'        .XValues = Array(1, 2, 3, 4, 5, 6, 7, 8)
    End With

End Sub


