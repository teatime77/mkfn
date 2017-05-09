Attribute VB_Name = "PureVBA"
Option Explicit

Dim TrainCnt As Long
Dim TrainBatchSize As Long
Dim DomainLen As Long
Dim RangeLen As Long
Dim EpochIdx As Long
Dim EpochSize As Long
Dim MiniBatchIdx As Long
Public StopNetworkTest As Boolean

Dim TrainX() As Single
Dim TrainY() As Single
Dim TrainLabel() As Byte

Dim Layers() As ILayerVBA
Dim FirstLayer As ILayerVBA
Dim LastLayer As ILayerVBA

Dim Costs As Collection


Public Function Sigmoid(z As Single) As Single
    Sigmoid = 1 / (1 + Exp(-z))
End Function

Public Function SigmoidPrime(z As Single) As Single
    SigmoidPrime = Sigmoid(z) * (1 - Sigmoid(z))
End Function


' 損失関数の微分
Sub CostDerivative(ByRef cost_derivative() As Single, ByRef last_y() As Single, ByRef batch_Y() As Single)
    Dim i As Long, j As Long
    For i = 0 To UBound(cost_derivative, 1)
        For j = 0 To UBound(cost_derivative, 2)
            cost_derivative(i, j) = last_y(i, j) - batch_Y(i, j)
        Next
    Next
End Sub

' 損失関数
Function Cost(ByRef cost_derivative() As Single) As Single
    Dim sum As Double: sum = 0
    
    Dim i As Long, j As Long
    For i = 0 To UBound(cost_derivative, 1)
        For j = 0 To UBound(cost_derivative, 2)
            Dim d As Single: d = cost_derivative(i, j)
            sum = sum + d * d
        Next
    Next

    Dim cnt As Long: cnt = (UBound(cost_derivative, 1) + 1) * (UBound(cost_derivative, 2) + 1)
    Cost = CSng(sum / cnt)
End Function

Function ArgMax(ByRef last_y() As Single, ByRef train_label() As Byte, ByRef arg_max As String) As Long
    Dim batch_idx As Long, i As Long
    Dim eq_cnt As Long: eq_cnt = 0

    For batch_idx = 0 To UBound(last_y, 2)

        Dim max_val As Single: max_val = -10000
        Dim max_idx As Long: max_idx = 0
        For i = 0 To RangeLen - 1
            Dim val As Single: val = last_y(i, batch_idx)
            If max_val < val Then

                max_val = val
                max_idx = i
            End If
        Next

        Mid(arg_max, 1 + batch_idx, 1) = Hex(max_idx)

        If max_idx = train_label(batch_idx) Then
            eq_cnt = eq_cnt + 1
        End If
    Next

    ArgMax = eq_cnt
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


'    ミニバッチごとにパラメータを更新します。
Sub UpdateMiniBatch(ByRef batch_X() As Single, ByRef batch_Y() As Single, ByRef last_y() As Single, cost_derivative() As Single, ByRef train_label() As Byte, idxes() As Long)
    Dim i As Long, j As Long
    Dim wk() As Single
    
    '-------------------------------------------------- 入力をセットします。
    FirstLayer.SetInputData batch_X

    For i = 0 To UBound(Layers)
        Layers(i).Forward
        If i <> UBound(Layers) Then
        
            Layers(i).GetOutputData wk
            Layers(i + 1).SetInputData wk
        End If
    Next

    '-------------------------------------------------- 出力を得ます。
    LastLayer.GetOutputData last_y

    '-------------------------------------------------- 損失関数を計算します。
    CostDerivative cost_derivative, last_y, batch_Y

    MiniBatchIdx = MiniBatchIdx + 1
    If MiniBatchIdx Mod 100 = 0 Then
        Dim cost1 As Single: cost1 = Cost(cost_derivative)
        
        Costs.Add cost1

        Dim arg_max As String * 10
        Dim eq_cnt As Long: eq_cnt = ArgMax(last_y, train_label, arg_max)
        
        
        ThisWorkbook.ActiveSheet.Cells(3, 1) = "ミニバッチ"
        ThisWorkbook.ActiveSheet.Cells(3, 2) = MiniBatchIdx
        
        ThisWorkbook.ActiveSheet.Cells(4, 1) = "正解数"
        ThisWorkbook.ActiveSheet.Cells(4, 2) = eq_cnt
        
        Dim ch As ChartObject
        Dim ys() As Single
        
        If MiniBatchIdx Mod 100 = 0 Then
'            Debug.Print MiniBatchIdx; " cost:" & Str(cost1); " EQ:"; eq_cnt; " "; arg_max
            
'            Application.ScreenUpdating = True
            
            ReDim ys(UBound(last_y, 1))
            For i = 0 To UBound(last_y, 2)
                For j = 0 To UBound(last_y, 1)
                    ys(j) = last_y(j, i)
'                    ys(j) = cost_derivative(i, j)
                Next
                
                
                Set ch = GetChartByIdx(idxes(i))
                With ch.Chart.SeriesCollection(1)
                    .Values = ys
                End With
                With ch.Chart
                    .HasTitle = False
                End With
                
                ch.Chart.Refresh
            Next
        End If
        
        ReDim ys(Costs.Count - 1)
        For i = 0 To UBound(ys)
            ys(i) = Costs.Item(1 + i)
        Next
        Set ch = GetChartByName("Cost", 580, 10, 400, 400)
        With ch.Chart
            .ChartType = xlLine
            .SeriesCollection(1).name = "損失関数"
            .HasTitle = False
        End With
        
        With ch.Chart.SeriesCollection(1)
            .Values = ys
        End With
        ch.Chart.Refresh
        
        DoEvents
        DoEvents
    End If

    '-------------------------------------------------- δyをセットします。
    LastLayer.SetOutputDeltaData cost_derivative

    For i = UBound(Layers) To 0 Step -1
        If i <> UBound(Layers) Then
        
            Layers(i + 1).GeInputDeltaData wk
            Layers(i).SetOutputDeltaData wk
        End If
        
        Layers(i).Backward
    Next

    For i = UBound(Layers) To 0 Step -1
        Layers(i).UpdateParameter
    Next
End Sub

Sub SetBatchData(batch_X() As Single, batch_Y() As Single, train_label() As Byte, idxes() As Long)
    Dim batch_idx As Long, j As Long, idx As Long
    
    For batch_idx = 0 To TrainBatchSize - 1
        idx = idxes(batch_idx)
        
        For j = 0 To UBound(batch_X, 1)
            batch_X(j, batch_idx) = TrainX(j, idx)
        Next
        
        For j = 0 To UBound(batch_Y, 1)
            batch_Y(j, batch_idx) = TrainY(j, idx)
            Debug.Assert (TrainY(j, idx) = 1) = (j = TrainLabel(idx))
        Next
        
        train_label(batch_idx) = TrainLabel(idx)
    Next
End Sub

Sub SGD()
    Dim i As Long
    
    ReDim TrainX(DomainLen - 1, TrainCnt - 1)
    ReDim TrainY(RangeLen - 1, TrainCnt - 1)
    ReDim TrainLabel(TrainCnt - 1)
    
    Dim train_batch_X() As Single
    Dim train_batch_Y() As Single
    Dim train_last_Y() As Single
    Dim cost_derivative() As Single
    Dim train_label() As Byte
    
    ReDim train_batch_X(DomainLen - 1, TrainBatchSize - 1)
    ReDim train_batch_Y(RangeLen - 1, TrainBatchSize - 1)
    ReDim train_last_Y(RangeLen - 1, TrainBatchSize - 1)
    ReDim cost_derivative(RangeLen - 1, TrainBatchSize - 1)
    ReDim train_label(TrainBatchSize - 1)

    ' すべてのレイヤーのメモリを割り当て、レイヤーの入出力を結合します。
    For i = 0 To UBound(Layers)
        Layers(i).BatchSize = TrainBatchSize
        Layers(i).Allocate
    Next

    SetSevenSegmentData TrainX, TrainY, TrainLabel
    
    Dim idxes() As Long
    
    For EpochIdx = 0 To EpochSize - 1
        idxes = RandomSampling(TrainCnt, TrainBatchSize)
        
        SetBatchData train_batch_X, train_batch_Y, train_label, idxes
    
        UpdateMiniBatch train_batch_X, train_batch_Y, train_last_Y, cost_derivative, train_label, idxes
                        
        If StopNetworkTest Then
            Exit Sub
        End If
    Next
    
    ' すべてのレイヤーのメモリを解放します。
    For i = 0 To UBound(Layers)
        Layers(i).Free
    Next
End Sub


Sub Msg(s As String)
        ThisWorkbook.ActiveSheet.Cells(2, 2) = s
End Sub

Function MakeFullyConnectedLayer(x_len As Long, y_len As Long) As FullyConnectedLayerVBA
    Dim layer As FullyConnectedLayerVBA
    
    Set layer = New FullyConnectedLayerVBA
    layer.SetSize x_len, y_len
    
    Set MakeFullyConnectedLayer = layer
End Function


Function MakeSegment() As Object
    Dim o As Object
    
    Set o = ActiveSheet.OLEObjects.Add(ClassType:="Forms.Label.1", Link:=False, DisplayAsIcon:=False, Left:=303, Top:=82.5, Width:=45, Height:=17.25)
    
    Set MakeSegment = o
End Function

Public Sub VBAMain()
    Dim i As Long
    
'    Dim o As Object
'    Set o = MakeSegment
     
     
    Msg "開始しました。"
    
    TrainCnt = 10
    TrainBatchSize = 7
    DomainLen = 7
    RangeLen = 10
    EpochSize = 200000
    StopNetworkTest = False
    MiniBatchIdx = 0
    Set Costs = New Collection

    Dim learning_rate As Single: learning_rate = 0.5

    ReDim Layers(1)
        
    Set Layers(0) = MakeFullyConnectedLayer(DomainLen, 30)
'    Set Layers(1) = MakeFullyConnectedLayer(30, 30)
    Set Layers(1) = MakeFullyConnectedLayer(30, RangeLen)

    For i = 0 To UBound(Layers)
        Layers(i).LearningRate = learning_rate / TrainBatchSize '3.0f
    Next

    Set FirstLayer = Layers(0)
    Set LastLayer = Layers(UBound(Layers))

    SGD
    
    Erase Layers

    Msg "終了しました。"
End Sub

Public Function GetChartByName(name As String, x As Long, y As Long, w As Long, h As Long) As ChartObject
    Dim ch As ChartObject: Set ch = Nothing
    Dim ch2 As ChartObject

    For Each ch2 In ActiveSheet.ChartObjects
        If ch2.name = name Then
            Set ch = ch2
            Exit For
        End If
    Next

    If ch Is Nothing Then
        Set ch = ActiveSheet.ChartObjects.Add(x, y, w, h)
        ch.name = name
        
        With ch.Chart
            .ChartType = xlColumnClustered
            .HasTitle = False
            
            Dim ser As Series
            Set ser = .SeriesCollection.NewSeries
            With ser
                .Values = Array(1, 3, 5, 7, 11, 13, 17, 19)
    '                .Name = "Sample"
                .XValues = Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
            End With
        End With
    End If

    Set GetChartByName = ch
End Function

Public Function GetChartByIdx(idx As Long) As ChartObject
    Dim ch As ChartObject: Set ch = Nothing
    
    Dim w As Long: w = 200
    Dim h As Long: h = 100
    Dim x As Long, y As Long
    
    If idx <= 4 Then
        x = 150
        y = 10 + idx * (h + 10)
    Else
        x = 150 + w + 10
        y = 10 + (idx - 5) * (h + 10)
    End If
    
    Dim name As String: name = "Chart_" + Str(idx)
    
    Set ch = GetChartByName(name, x, y, w, h)
    
    With ch.Chart
        .SeriesCollection(1).name = "ラベル " + Str(idx)
        .HasTitle = False
    End With
            
'    ch.Left = x
'    ch.Top = y
'    ch.Width = w
'    ch.Height = h

    Set GetChartByIdx = ch
End Function

