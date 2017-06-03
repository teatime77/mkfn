Attribute VB_Name = "MNIST"
Option Explicit

Declare PtrSafe Function GetTickCount Lib "kernel32" () As Long

Public CorrectAns() As Long

Public MySeries As Series

Enum NetworkType
    Simple = 1
    CNN = 2
    RNN = 3
    LSTM = 4
End Enum

Public Const SizeOfSingle As Long = 4
Public Const SizeOfDouble As Long = 8

Public Dev As Device

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
Public UpdateMiniBatchCount As Long

Public Layers() As Layer
Public FirstLayer As Layer
Public LastLayer As Layer


Public TrainX() As Single
Public TrainY() As Single
Public TestX() As Single
Public TestY() As Single

Public LastTick As Long
Public StopDeepLearning As Boolean

' 配列に要素を追加します。
Public Sub AddArray(arr As Variant, x As Variant)
    If Not IsArray2(arr) Then
    
        ReDim arr(0)
    Else
        ReDim Preserve arr(UBound(arr) + 1)
    End If
    
    arr(UBound(arr)) = x
End Sub

Public Function BytesToInt(v() As Byte, offset As Long) As Long
    BytesToInt = v(offset) * &H1000000 + v(offset + 1) * &H10000 + v(offset + 2) * &H100& + v(offset + 3)
End Function

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

' ファイルからバイト配列を読み込みます。
Public Function ReadAllBytes(path As String) As Byte()
    Dim buf() As Byte
    
    Open path For Binary As #1
    ReDim buf(LOF(1))
    Get #1, , buf
    Close #1
    
    ReadAllBytes = buf
End Function


' MNISTのデータファイルを読み込みます。
Public Sub ReadMNIST()
    Dim k As Long: k = InStrRev(ThisWorkbook.FullName, "\")
    Dim mnist_dir As String: mnist_dir = Left(ThisWorkbook.FullName, k) + "MNIST\"

    Dim buf() As Byte, i As Long, j As Long, idx As Long
    
    ' トレーニング データの入力値のファイルからバイト配列を読み込みます。
    buf = ReadAllBytes(mnist_dir + "train-images.idx3-ubyte")

    ' トレーニング データの数
    TrainCnt = BytesToInt(buf, 4)
    
    ' 画像の高さ
    Dim img_h As Long: img_h = BytesToInt(buf, 8)
    
    ' 画像の幅
    Dim img_w As Long: img_w = BytesToInt(buf, 12)

    ' 入力ベクトルの次元
    DomainLen = img_h * img_w
    
    ' 出力ベクトルの次元
    RangeLen = 10

    ReDim TrainX(TrainCnt - 1, DomainLen - 1)
    idx = 0
    ' すべてのトレーニング データに対して
    For i = 0 To TrainCnt - 1
    
        '
        For j = 0 To DomainLen - 1
            TrainX(i, j) = buf(16 + idx) / 256!
            idx = idx + 1
        Next
    Next


    ' トレーニング データの出力値のファイルからバイト配列を読み込みます。
    buf = ReadAllBytes(mnist_dir + "train-labels.idx1-ubyte")
    ReDim TrainLabel(TrainCnt - 1)
    'Array.Copy(buf, 8, TrainLabel, 0, TrainCnt)
    
    ' すべてのトレーニング データに対して
    For i = 0 To TrainCnt - 1
        TrainLabel(i) = buf(8 + i)
    Next


    ' テスト データの入力値のファイルからバイト配列を読み込みます。
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


    ' テスト データの出力値のファイルからバイト配列を読み込みます。
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
    
    ' 最初のレイヤーの入力のメモリを割り当てます。
    Dim p As LongPtr: p = Dev.DeviceMalloc32(batch_size * DomainLen * SizeOfSingle)
    FirstLayer.SetInput (p)

    ' 最後のレイヤーの出力の微分のメモリを割り当てます。
    p = Dev.DeviceMalloc32(batch_size * RangeLen * SizeOfSingle)
    LastLayer.SetOutputDelta (p)

    ' すべてのレイヤーに対して
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

' ミニバッチのデータをセットします。
Sub SetBatchData(x() As Single, batch_X() As Single, batch_Y() As Single, label() As Byte, batch_size As Long, idxes() As Long)
    Dim i As Long, batch_idx As Long
    Dim is_array As Boolean: is_array = IsArray2(idxes)
    
    ' 出力ベクトルの各次元に対して
    For i = 0 To UBound(batch_Y, 1)
    
        ' バッチ内の各データに対して
        For batch_idx = 0 To UBound(batch_Y, 2)
        
            ' one-hotベクトルの値をクリアします。
            batch_Y(i, batch_idx) = 0
        Next
    Next

    ' バッチ内の各データに対して
    For batch_idx = 0 To batch_size - 1
        Dim idx As Long: idx = MiniBatchIdx * batch_size + batch_idx

        If is_array Then
            ' データのインデックスが指定された場合

            idx = idxes(idx)
        End If

        ' 入力ベクトルの各次元に対して
        For i = 0 To DomainLen - 1
            
            ' 入力ベクトルの値をセットします。
            batch_X(i, batch_idx) = x(idx, i)
        Next

        ' 出力の one-hotベクトルの値をセットします。
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
    
    ' 出力ベクトルの各次元に対して
    For i = 0 To UBound(cost_derivative, 1)
    
        ' バッチ内のデータに対して
        For batch_idx = 0 To UBound(cost_derivative, 2)
        
            ' 損失関数の微分
            cd = cost_derivative(i, batch_idx)
            
            ' 損失関数の微分の二乗を加算します。
            sum = sum + cd * cd
        Next
    Next

    Dim size As Long: size = (UBound(cost_derivative, 1) + 1) * (UBound(cost_derivative, 2) + 1)
    Cost = CSng(sum / size)
End Function

' ミニバッチごとにパラメータを更新します。
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

' 最大値のインデックスが正解のインデックスと等しい個数を得ます。
Function ArgMax(result_Y() As Single, batch_size As Long, arg_max() As Byte, label() As Byte) As Long
    Dim batch_idx As Long, i As Long
    Dim eq_cnt As Long: eq_cnt = 0

    ' バッチ内のすべてのデータに対して
    For batch_idx = 0 To batch_size - 1

        Dim max_val As Single: max_val = -10000
        Dim max_idx As Long: max_idx = 0
        
        ' 出力の各次元に対して
        For i = 0 To RangeLen - 1
            Dim val As Single: val = result_Y(i, batch_idx)
            If max_val < val Then
                ' 現在までの最大値より大きい場合

                ' 最大値と最大値のインデックスを更新します。
                max_val = val
                max_idx = i
            End If
        Next

        ' 最大値のインデックスをセットします。
        arg_max(batch_idx) = max_idx

        If max_idx = label(MiniBatchIdx * batch_size + batch_idx) Then
            ' 最大値のインデックスが正解のインデックスと等しい場合
            
            ' 正解数を加算します。
            eq_cnt = eq_cnt + 1
        End If
    Next

    ' 正解数を返します。
    ArgMax = eq_cnt
End Function


' テスト データの評価をします。
Function Evaluate(batch_X() As Single, batch_Y() As Single, last_y() As Single, batch_size As Long, arg_max() As Byte, label() As Byte) As Long
    Dim i As Long
    
    ' 入力をセットします。
    FirstLayer.SetInputData batch_X

    ' 順伝播の計算をします。
    For i = 0 To UBound(Layers)
        Layers(i).Forward
    Next

    ' 出力を得ます。
    LastLayer.GetOutputData last_y

    ' 最大値のインデックスが正解のインデックスと等しい個数を得ます。
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

' フリーズ状態にならないようにDoEventsを実行します。
Sub HandleDoEvents()
    If LastTick = 0 Then
        ' 最初の場合
        
        LastTick = GetTickCount
    Else
        ' 2回目以降の場合
        
        Dim t As Long: t = GetTickCount() - LastTick
        If 500 < t Then
            ' 前回から500ミリ秒経過した場合
        
            DoEvents
            LastTick = GetTickCount
        End If
    End If
End Sub

' 確率的勾配降下法 (stochastic gradient descent, SGD)
Sub SGD()

    ' トレーニング データのバッチサイズ
    Dim train_batch_cnt As Long: train_batch_cnt = TrainCnt / TrainBatchSize
    
    ' テスト データのバッチサイズ
    Dim test_batch_cnt As Long: test_batch_cnt = TestCnt / TestBatchSize

    ' トレーニング データの入力値
    ReDim train_batch_X(DomainLen - 1, TrainBatchSize - 1) As Single
    
    ' トレーニング データの正解の出力値
    ReDim train_batch_Y(RangeLen - 1, TrainBatchSize - 1) As Single
    
    ' トレーニング データの計算した出力値
    ReDim train_last_Y(RangeLen - 1, TrainBatchSize - 1) As Single

    ' 損失関数の微分
    ReDim cost_derivative(RangeLen - 1, TrainBatchSize - 1) As Single

    ' テスト データの入力値
    ReDim test_batch_X(DomainLen - 1, TestBatchSize - 1) As Single
    
    ' テスト データの正解の出力値
    ReDim test_batch_Y(RangeLen - 1, TestBatchSize - 1) As Single
    
    ' テスト データの計算した出力値
    ReDim test_last_Y(RangeLen - 1, TestBatchSize - 1) As Single

    ReDim test_arg_max(TestBatchSize) As Byte

    ' 入力データのインデックスの配列
    Dim idxes() As Long

    ' エポックのループ
    For EpochIdx = 0 To EpochSize - 1

        ' [0, TrainCnt-1]の範囲の整数をランダムに並べ変えます。
        idxes = RandomSampling(TrainCnt, TrainCnt)

        ' すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
        AllocateConnectLayers (TrainBatchSize)

        ' トレーニング データのミニバッチのループ
        For MiniBatchIdx = 0 To train_batch_cnt - 1

            ' ミニバッチのデータをセットします。
            SetBatchData TrainX, train_batch_X, train_batch_Y, TrainLabel, TrainBatchSize, idxes

            ' ミニバッチごとにパラメータを更新します。
            UpdateMiniBatch train_batch_X, train_batch_Y, train_last_Y, cost_derivative
                        
            If StopDeepLearning Then
                ' 停止ボタンが押された場合
                
                Exit For
            End If
            
            ' フリーズ状態にならないようにDoEventsを実行します。
            HandleDoEvents
        Next

        ' すべてのレイヤーのメモリを解放します。
        FreeLayers

        ' 入力データのインデックスの配列を削除します。
        Erase idxes
        
        If StopDeepLearning Then
            ' 停止ボタンが押された場合
            
            Exit For
        End If

        ' すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
        AllocateConnectLayers (TestBatchSize)

        Dim eq_cnt_sum As Long: eq_cnt_sum = 0
                
        ' テスト データのミニバッチのループ
        For MiniBatchIdx = 0 To test_batch_cnt - 1

            ' ミニバッチのデータをセットします。
            SetBatchData TestX, test_batch_X, test_batch_Y, TestLabel, TestBatchSize, idxes

            ' テスト データの評価をします。
            Dim eq_cnt As Long: eq_cnt = Evaluate(test_batch_X, test_batch_Y, test_last_Y, TestBatchSize, test_arg_max, TestLabel)
            eq_cnt_sum = eq_cnt_sum + eq_cnt
        Next
        Msg "エポック = " + Format(EpochIdx) + " : 正解数 = " + Format(eq_cnt_sum) + " / " + Format(TestCnt)
        
        ' 正解数の配列に追加します。
        AddArray CorrectAns, eq_cnt_sum
        
        ' チャートの表示を更新します。
        MySeries.Values = CorrectAns
        
        FreeLayers
    Next
End Sub


Public Sub DeepLearning()
    Dim use_cuda As Boolean: use_cuda = True
    
    Msg "開始しました。"
    
    ' 処理の中断フラグをクリア
    StopDeepLearning = False

    If use_cuda Then
        ' CUDAを使う場合
        
        Set Dev = New DeviceCuda
    Else
        ' CUDAを使わない場合
        
        Set Dev = New Device
    End If
    
    ' デバイスを開始します。(CUDAでのみ必要)
    Dev.DeviceInit

    EpochSize = 100
    TrainBatchSize = 10
    TestBatchSize = 20

    Dim factory As LayerFactoryF:
    
    If use_cuda Then
        ' CUDAを使う場合
        
        Set factory = New LayerFactoryCudaF
    Else
        ' CUDAを使わない場合
        
        Set factory = New LayerFactoryF
    End If
    
    NetType = NetworkType.Simple
    NetType = NetworkType.CNN

    Select Case NetType
    Case NetworkType.Simple
        ' 2層のレイヤーを作ります。
        ReDim Layers(1)
        
        Set Layers(0) = factory.MakeFullyConnectedLayer(28 * 28, 30)
        Set Layers(1) = factory.MakeFullyConnectedLayer(30, 10)

    Case NetworkType.CNN
        
        ' 4層のレイヤーを作ります。
        ReDim Layers(3)
        Set Layers(0) = factory.MakeConvolutionalLayer(28, 28, 5, 5)
        Set Layers(1) = factory.MakeMaxPoolingLayer(24, 24, 5, 2)
        Set Layers(2) = factory.MakeFullyConnectedLayer(12 * 12 * 5, 100)
        Set Layers(3) = factory.MakeFullyConnectedLayer(100, 10)
    End Select

    ' 学習率をセットします。
    Dim i As Long
    Dim learning_rate As Single: learning_rate = 1#
    For i = 0 To UBound(Layers)
        Layers(i).LearningRate = learning_rate / TrainBatchSize '3.0f
    Next

    ' 最初のレイヤー
    Set FirstLayer = Layers(0)
    
    ' 最後のレイヤー
    Set LastLayer = Layers(UBound(Layers))

    ' MNISTのデータファイルを読み込みます。
    ReadMNIST
    
    ' 確率的勾配降下法 (stochastic gradient descent, SGD)
    SGD

    ' レイヤーを削除します。
    For i = 0 To UBound(Layers)
        Layers(i).Destroy
    Next
    Erase Layers
    
    ' デバイスを終了します。(CUDAでのみ必要)
    Dev.DeviceEnd

    Msg "終了しました。"
End Sub


' 停止ボタンのクリック処理
Sub StopButton_OnAction()
    StopDeepLearning = True
End Sub

' 停止ボタンを作ります。
Sub MakeStopButton()
    If ActiveSheet.Buttons.Count() <> 0 Then
        ' ボタンを作成済みの場合
        
        ' 削除します。
        ActiveSheet.Buttons.Delete
    End If
    
    ' 停止ボタンを作ります。
    With ActiveSheet.Buttons.Add(Range("B5").Left, Range("B5").Top, Range("B5:C6").Width, Range("B5:C6").Height)
        .OnAction = "StopButton_OnAction"
        .Characters.text = "停止"
    End With
End Sub

' 正解数のチャートを作ります。
Sub MakeChart()
    If ActiveSheet.ChartObjects.Count() <> 0 Then
        ' チャートを作成済みの場合
        
        ' 削除します。
        ActiveSheet.ChartObjects.Delete
    End If
    
    
    ' 正解数のチャートを作ります。
    With ActiveSheet.ChartObjects.Add(250, 20, 300, 200).Chart
        .ChartType = xlColumnClustered
        Set MySeries = .SeriesCollection.NewSeries
        With MySeries
'            .Values = Array(1, 3, 5, 7, 11, 13, 17, 19)
'            .Values = Array(1, 3, 5, 7, 11, 13, 17, 19)
            .name = "正解数"
        End With
    End With
    
    ' 正解数の配列をクリアします。
    If IsArray2(CorrectAns) Then
        Erase CorrectAns
    End If
End Sub

' MNISTのテスト
Public Sub TestMNIST()
    MakeStopButton
    MakeChart
    DeepLearning
End Sub
