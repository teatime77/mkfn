Attribute VB_Name = "RNN"
Option Explicit

Const CHAR_COUNT As Long = 256& * 256&
Const INVALID_CHAR_IDX As Long = &HFFFF
'    Const INVALID_CHAR As Char = ChrW(INVALID_CHAR_IDX)

Public FSO As FileSystemObject

Dim CharToCharIdx(CHAR_COUNT - 1) As Long
Dim CharIdxToChar(CHAR_COUNT - 1) As Long
Dim CharBuf As String
Dim CostSum As Single
Dim CostCount As Long
Dim DataDir As String

Dim UpdateMiniBatchCount As Long

Sub SetDataDir()
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

Function ReadAllText(path As String) As String
    Dim buf As String
    Dim ts As TextStream: Set ts = FSO.GetFile(path).OpenAsTextStream(ForReading, TristateTrue)
    
    buf = ts.ReadAll
    ts.Close
    
    
    ReadAllText = buf
End Function

Public Function AscW2(s As String) As Long
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
                batch_X(t * one_hot_size + idx1, batch_idx) = 1
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
            batch_Y(idx2, batch_idx) = 1
        End If
    Next
End Sub


Public Sub SetZero(v() As Single)
    Dim i As Long, j As Long
    
    For i = 0 To UBound(v, 1)
        For j = 0 To UBound(v, 2)
            v(i, j) = 0
        Next
    Next
End Sub


' 損失関数の微分
Function SoftMax(cost_derivative() As Single, last_y() As Single, batch_Y() As Single, exp_work() As Single, range_len As Long, batch_size As Long, ByVal batch_idx As Integer) As Single
    Dim max_val As Single: max_val = -10000
    Dim i As Long
    
    For i = 0 To range_len - 1
        If max_val < last_y(i, batch_idx) Then
            max_val = last_y(i, batch_idx)
        End If
    Next

    Dim sum As Single: sum = 0
    For i = 0 To range_len - 1

        Dim d As Single: d = CSng(Math.Exp(last_y(i, batch_idx) - max_val))
        sum = sum + d
        exp_work(i, batch_idx) = d
    Next

    Dim cost_sum As Single: cost_sum = 0
    For i = 0 To range_len - 1

        Dim y As Single: y = exp_work(i, batch_idx) / sum
        cost_derivative(i, batch_idx) = y - batch_Y(i, batch_idx)

        cost_sum = cost_sum + CSng(batch_Y(i, batch_idx) * Math.Log(y))
    Next

    SoftMax = -cost_sum
End Function


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
        For batch_idx = 0 To TrainBatchSize - 1
            Dim cost2 As Single: cost2 = SoftMax(cost_derivative, last_y, batch_Y, exp_work, RangeLen, TrainBatchSize, batch_idx)

            cost1 = cost1 + cost2
        Next
        cost1 = cost1 / TrainBatchSize

        '-------------------------------------------------- 損失関数を計算します。
        CostSum = CostSum + cost1
        CostCount = CostCount + 1

        If StopFlag Then
            ' 停止ボタンが押された場合
            
            Exit Sub
        End If
        HandleDoEvents

        If UpdateMiniBatchCount Mod 141 = 0 Then 'MiniBatchIdx
            batch_idx = UpdateMiniBatchCount Mod TrainBatchSize
            Dim max_val As Single: max_val = -10000
            Dim max_idx As Long: max_idx = 0
            For i = 0 To RangeLen - 1
                val = last_y(i, batch_idx)
                If max_val < val Then

                    max_val = val
                    max_idx = i
                End If
            Next
            Mid(output_text, 1 + t, 1) = ChrW(CharIdxToChar(max_idx))

            max_val = -10000
            max_idx = 0
            For i = 0 To RangeLen - 1
                val = batch_Y(i, batch_idx)
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
Public Sub RNNSGD()
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

            ReDim train_batch_X(DomainLen - 1, TrainBatchSize - 1) As Single
            ReDim train_batch_Y(RangeLen - 1, TrainBatchSize - 1) As Single
            ReDim train_last_Y(RangeLen - 1, TrainBatchSize - 1) As Single

            ReDim cost_derivative(RangeLen - 1, TrainBatchSize - 1) As Single
            ReDim exp_work(RangeLen - 1, TrainBatchSize - 1) As Single

            ' すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
            AllocateConnectLayers (TrainBatchSize)

            Dim delta_y_sz As Long: delta_y_sz = TrainBatchSize * Time * y * SizeOfDouble
            Dim out_delta_y As LongPtr: out_delta_y = Dev.DeviceMalloc32(delta_y_sz)

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

                If StopFlag Then
                    ' 停止ボタンが押された場合
                
                    Exit Do
                End If
            Loop

            FreeLayers
            Dev.DeviceFree (out_delta_y)

            If StopFlag Then
                ' 停止ボタンが押された場合
            
                Exit Sub
            End If
        Next
    Loop
End Sub

Public Sub TestRNN()
    Msg "開始しました。"
    
    ' 実行停止フラグをクリアします。
    StopFlag = False
    
    SetDataDir

    Set Dev = New Device ' DeviceCuda
    Dev.DeviceInit

    EpochSize = 100
    TestBatchSize = 20

    Dim layer_factory As LayerFactoryF: Set layer_factory = New LayerFactoryF ' LayerFactoryCudaF
    
    NetType = NetworkType.RNN
    NetType = NetworkType.LSTM

    Dim learning_rate As Single: learning_rate = 1#
    Select Case NetType
        Case NetworkType.RNN
            learning_rate = 0.1
            TrainBatchSize = 7
            
            ReDim Layers(1)
            Set Layers(0) = layer_factory.MakeRecurrentLayer(20, 28, 100)
            Set Layers(1) = layer_factory.MakeFullyConnectedLayer(10, 28)

        Case NetworkType.LSTM
            learning_rate = 0.1
            TrainBatchSize = 7
            
            ReDim Layers(1)
            Set Layers(0) = layer_factory.MakeLSTMLayer(20, 28, 100)
            Set Layers(1) = layer_factory.MakeFullyConnectedLayer(100, 28)
    End Select

    ' 学習率をセットします。
    Dim i As Long
    For i = 0 To UBound(Layers)
        Layers(i).LearningRate = learning_rate / TrainBatchSize '3.0f
    Next

    
    Set FirstLayer = Layers(0)
    Set LastLayer = Layers(UBound(Layers))

    
    Dim cnt As Long: cnt = FirstLayer.GetFieldCount()
    For i = 0 To cnt - 1
        Dim name As String: name = FirstLayer.GetFieldName(i)
        Debug.Assert FirstLayer.GetFieldIndexByName(name) = i
        Dim v As Variant: v = FirstLayer.GetFieldSize(i)
        Dim s As String: s = IIf(ValidArray(v), "S", "A")
        Debug.Print name, FirstLayer.GetFieldDimension(i), FirstLayer.GetFieldElementCount(i), s
    Next

    RNNSGD

    For i = 0 To UBound(Layers)
        Layers(i).Destroy
    Next
    Erase Layers
    
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

