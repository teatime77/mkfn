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

' �z��ɗv�f��ǉ����܂��B
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

'  (0, all_count-1)�͈̔͂���Asample_count�̐����𔲂��o���āA���̔z���Ԃ��܂��B
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

' �t�@�C������o�C�g�z���ǂݍ��݂܂��B
Public Function ReadAllBytes(path As String) As Byte()
    Dim buf() As Byte
    
    Open path For Binary As #1
    ReDim buf(LOF(1))
    Get #1, , buf
    Close #1
    
    ReadAllBytes = buf
End Function


' MNIST�̃f�[�^�t�@�C����ǂݍ��݂܂��B
Public Sub ReadMNIST()
    Dim k As Long: k = InStrRev(ThisWorkbook.FullName, "\")
    Dim mnist_dir As String: mnist_dir = Left(ThisWorkbook.FullName, k) + "MNIST\"

    Dim buf() As Byte, i As Long, j As Long, idx As Long
    
    ' �g���[�j���O �f�[�^�̓��͒l�̃t�@�C������o�C�g�z���ǂݍ��݂܂��B
    buf = ReadAllBytes(mnist_dir + "train-images.idx3-ubyte")

    ' �g���[�j���O �f�[�^�̐�
    TrainCnt = BytesToInt(buf, 4)
    
    ' �摜�̍���
    Dim img_h As Long: img_h = BytesToInt(buf, 8)
    
    ' �摜�̕�
    Dim img_w As Long: img_w = BytesToInt(buf, 12)

    ' ���̓x�N�g���̎���
    DomainLen = img_h * img_w
    
    ' �o�̓x�N�g���̎���
    RangeLen = 10

    ReDim TrainX(TrainCnt - 1, DomainLen - 1)
    idx = 0
    ' ���ׂẴg���[�j���O �f�[�^�ɑ΂���
    For i = 0 To TrainCnt - 1
    
        '
        For j = 0 To DomainLen - 1
            TrainX(i, j) = buf(16 + idx) / 256!
            idx = idx + 1
        Next
    Next


    ' �g���[�j���O �f�[�^�̏o�͒l�̃t�@�C������o�C�g�z���ǂݍ��݂܂��B
    buf = ReadAllBytes(mnist_dir + "train-labels.idx1-ubyte")
    ReDim TrainLabel(TrainCnt - 1)
    'Array.Copy(buf, 8, TrainLabel, 0, TrainCnt)
    
    ' ���ׂẴg���[�j���O �f�[�^�ɑ΂���
    For i = 0 To TrainCnt - 1
        TrainLabel(i) = buf(8 + i)
    Next


    ' �e�X�g �f�[�^�̓��͒l�̃t�@�C������o�C�g�z���ǂݍ��݂܂��B
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


    ' �e�X�g �f�[�^�̏o�͒l�̃t�@�C������o�C�g�z���ǂݍ��݂܂��B
    buf = ReadAllBytes(mnist_dir + "t10k-labels.idx1-ubyte")
    ReDim TestLabel(TestCnt - 1)
    'Array.Copy(buf, 8, TestLabel, 0, TestCnt)
    For i = 0 To TestCnt - 1
        TestLabel(i) = buf(8 + i)
    Next
End Sub

' ���ׂẴ��C���[�̃����������蓖�āA���C���[�̓��o�͂��������܂��B
Sub AllocateConnectLayers(batch_size As Long)
    Dim i As Long
    
    ' �ŏ��̃��C���[�̓��͂̃����������蓖�Ă܂��B
    Dim p As LongPtr: p = Dev.DeviceMalloc32(batch_size * DomainLen * SizeOfSingle)
    FirstLayer.SetInput (p)

    ' �Ō�̃��C���[�̏o�͂̔����̃����������蓖�Ă܂��B
    p = Dev.DeviceMalloc32(batch_size * RangeLen * SizeOfSingle)
    LastLayer.SetOutputDelta (p)

    ' ���ׂẴ��C���[�ɑ΂���
    For i = 0 To UBound(Layers)
        Layers(i).BatchSize = batch_size
        Layers(i).Allocate
    Next

    ' ���C���[�̓��o�͂��������܂��B
    For i = 0 To UBound(Layers) - 1

        If Layers(i).GetTimeCount() = 0 Then
            Debug.Assert (Layers(i).GetOutputCount() = Layers(i + 1).GetInputCount())

            ' ���̃��C���[�̓��͂́A���݂̃��C���[�̏o�͂ɂ��܂��B(���`�d)
            Layers(i + 1).SetInput (Layers(i).GetOutput())

            ' ���݂̃��C���[�̏o�͂̃f���^�́A���̃��C���[�̓��͂̃f���^�ɂ��܂��B(�t�`�d)
            Layers(i).SetOutputDelta (Layers(i + 1).GetInputDelta())
        End If

        Layers(i).ConnectLayer Layers(i + 1)
    Next
End Sub

' �~�j�o�b�`�̃f�[�^���Z�b�g���܂��B
Sub SetBatchData(x() As Single, batch_X() As Single, batch_Y() As Single, label() As Byte, batch_size As Long, idxes() As Long)
    Dim i As Long, batch_idx As Long
    Dim is_array As Boolean: is_array = IsArray2(idxes)
    
    ' �o�̓x�N�g���̊e�����ɑ΂���
    For i = 0 To UBound(batch_Y, 1)
    
        ' �o�b�`���̊e�f�[�^�ɑ΂���
        For batch_idx = 0 To UBound(batch_Y, 2)
        
            ' one-hot�x�N�g���̒l���N���A���܂��B
            batch_Y(i, batch_idx) = 0
        Next
    Next

    ' �o�b�`���̊e�f�[�^�ɑ΂���
    For batch_idx = 0 To batch_size - 1
        Dim idx As Long: idx = MiniBatchIdx * batch_size + batch_idx

        If is_array Then
            ' �f�[�^�̃C���f�b�N�X���w�肳�ꂽ�ꍇ

            idx = idxes(idx)
        End If

        ' ���̓x�N�g���̊e�����ɑ΂���
        For i = 0 To DomainLen - 1
            
            ' ���̓x�N�g���̒l���Z�b�g���܂��B
            batch_X(i, batch_idx) = x(idx, i)
        Next

        ' �o�͂� one-hot�x�N�g���̒l���Z�b�g���܂��B
        batch_Y(label(idx), batch_idx) = 1
    Next
End Sub

' �����֐��̔���
Sub CostDerivative(cost_derivative() As Single, last_y() As Single, batch_Y() As Single)
    Dim i As Long, batch_idx As Long
    
    For i = 0 To UBound(last_y, 1)
        For batch_idx = 0 To UBound(last_y, 2)
            cost_derivative(i, batch_idx) = last_y(i, batch_idx) - batch_Y(i, batch_idx)
        Next
    Next
End Sub

' �����֐�
Function Cost(cost_derivative() As Single) As Single
    Dim i As Long, batch_idx As Long, cd As Single
    Dim sum As Double: sum = 0
    
    ' �o�̓x�N�g���̊e�����ɑ΂���
    For i = 0 To UBound(cost_derivative, 1)
    
        ' �o�b�`���̃f�[�^�ɑ΂���
        For batch_idx = 0 To UBound(cost_derivative, 2)
        
            ' �����֐��̔���
            cd = cost_derivative(i, batch_idx)
            
            ' �����֐��̔����̓������Z���܂��B
            sum = sum + cd * cd
        Next
    Next

    Dim size As Long: size = (UBound(cost_derivative, 1) + 1) * (UBound(cost_derivative, 2) + 1)
    Cost = CSng(sum / size)
End Function

' �~�j�o�b�`���ƂɃp�����[�^���X�V���܂��B
Sub UpdateMiniBatch(batch_X() As Single, batch_Y() As Single, last_y() As Single, cost_derivative() As Single)
    Dim i As Long
    Dim batch_size As Long: batch_size = TrainBatchSize
    
    ' ���͂��Z�b�g���܂��B
    FirstLayer.SetInputData batch_X

    ' ���`�d�̌v�Z�����܂��B
    For i = 0 To UBound(Layers)
        Layers(i).Forward
    Next
    
    Dev.DeviceSynchronize

    ' �o�͂𓾂܂��B
    LastLayer.GetOutputData last_y
            
    ' �����֐��̔������v�Z���܂��B
    CostDerivative cost_derivative, last_y, batch_Y

    ' �����֐����v�Z���܂��B
    Dim cost1 As Single: cost1 = Cost(cost_derivative)

    ' ��y���Z�b�g���܂��B
    LastLayer.SetOutputDeltaData cost_derivative

    ' �덷�t�`�d�̌v�Z�����܂��B
    For i = UBound(Layers) To 0 Step -1
        Layers(i).Backward
    Next
    
    Dev.DeviceSynchronize

    ' �p�����[�^���X�V���܂��B
    For i = UBound(Layers) To 0 Step -1
        Layers(i).UpdateParameter
    Next

    Dev.DeviceSynchronize
End Sub

' �ő�l�̃C���f�b�N�X�������̃C���f�b�N�X�Ɠ��������𓾂܂��B
Function ArgMax(result_Y() As Single, batch_size As Long, arg_max() As Byte, label() As Byte) As Long
    Dim batch_idx As Long, i As Long
    Dim eq_cnt As Long: eq_cnt = 0

    ' �o�b�`���̂��ׂẴf�[�^�ɑ΂���
    For batch_idx = 0 To batch_size - 1

        Dim max_val As Single: max_val = -10000
        Dim max_idx As Long: max_idx = 0
        
        ' �o�͂̊e�����ɑ΂���
        For i = 0 To RangeLen - 1
            Dim val As Single: val = result_Y(i, batch_idx)
            If max_val < val Then
                ' ���݂܂ł̍ő�l���傫���ꍇ

                ' �ő�l�ƍő�l�̃C���f�b�N�X���X�V���܂��B
                max_val = val
                max_idx = i
            End If
        Next

        ' �ő�l�̃C���f�b�N�X���Z�b�g���܂��B
        arg_max(batch_idx) = max_idx

        If max_idx = label(MiniBatchIdx * batch_size + batch_idx) Then
            ' �ő�l�̃C���f�b�N�X�������̃C���f�b�N�X�Ɠ������ꍇ
            
            ' ���𐔂����Z���܂��B
            eq_cnt = eq_cnt + 1
        End If
    Next

    ' ���𐔂�Ԃ��܂��B
    ArgMax = eq_cnt
End Function


' �e�X�g �f�[�^�̕]�������܂��B
Function Evaluate(batch_X() As Single, batch_Y() As Single, last_y() As Single, batch_size As Long, arg_max() As Byte, label() As Byte) As Long
    Dim i As Long
    
    ' ���͂��Z�b�g���܂��B
    FirstLayer.SetInputData batch_X

    ' ���`�d�̌v�Z�����܂��B
    For i = 0 To UBound(Layers)
        Layers(i).Forward
    Next

    ' �o�͂𓾂܂��B
    LastLayer.GetOutputData last_y

    ' �ő�l�̃C���f�b�N�X�������̃C���f�b�N�X�Ɠ��������𓾂܂��B
    Dim eq_cnt As Long: eq_cnt = ArgMax(last_y, batch_size, arg_max, label)

    Evaluate = eq_cnt
End Function

' ���ׂẴ��C���[�̃�������������܂��B
Sub FreeLayers()
    Dim i As Long
    
    For i = 0 To UBound(Layers)
        Layers(i).Free
    Next

    Dev.DeviceFree (FirstLayer.GetInput())
    Dev.DeviceFree (LastLayer.GetOutputDelta())
End Sub

' �t���[�Y��ԂɂȂ�Ȃ��悤��DoEvents�����s���܂��B
Sub HandleDoEvents()
    If LastTick = 0 Then
        ' �ŏ��̏ꍇ
        
        LastTick = GetTickCount
    Else
        ' 2��ڈȍ~�̏ꍇ
        
        Dim t As Long: t = GetTickCount() - LastTick
        If 500 < t Then
            ' �O�񂩂�500�~���b�o�߂����ꍇ
        
            DoEvents
            LastTick = GetTickCount
        End If
    End If
End Sub

' �m���I���z�~���@ (stochastic gradient descent, SGD)
Sub SGD()

    ' �g���[�j���O �f�[�^�̃o�b�`�T�C�Y
    Dim train_batch_cnt As Long: train_batch_cnt = TrainCnt / TrainBatchSize
    
    ' �e�X�g �f�[�^�̃o�b�`�T�C�Y
    Dim test_batch_cnt As Long: test_batch_cnt = TestCnt / TestBatchSize

    ' �g���[�j���O �f�[�^�̓��͒l
    ReDim train_batch_X(DomainLen - 1, TrainBatchSize - 1) As Single
    
    ' �g���[�j���O �f�[�^�̐����̏o�͒l
    ReDim train_batch_Y(RangeLen - 1, TrainBatchSize - 1) As Single
    
    ' �g���[�j���O �f�[�^�̌v�Z�����o�͒l
    ReDim train_last_Y(RangeLen - 1, TrainBatchSize - 1) As Single

    ' �����֐��̔���
    ReDim cost_derivative(RangeLen - 1, TrainBatchSize - 1) As Single

    ' �e�X�g �f�[�^�̓��͒l
    ReDim test_batch_X(DomainLen - 1, TestBatchSize - 1) As Single
    
    ' �e�X�g �f�[�^�̐����̏o�͒l
    ReDim test_batch_Y(RangeLen - 1, TestBatchSize - 1) As Single
    
    ' �e�X�g �f�[�^�̌v�Z�����o�͒l
    ReDim test_last_Y(RangeLen - 1, TestBatchSize - 1) As Single

    ReDim test_arg_max(TestBatchSize) As Byte

    ' ���̓f�[�^�̃C���f�b�N�X�̔z��
    Dim idxes() As Long

    ' �G�|�b�N�̃��[�v
    For EpochIdx = 0 To EpochSize - 1

        ' [0, TrainCnt-1]�͈̔͂̐����������_���ɕ��וς��܂��B
        idxes = RandomSampling(TrainCnt, TrainCnt)

        ' ���ׂẴ��C���[�̃����������蓖�āA���C���[�̓��o�͂��������܂��B
        AllocateConnectLayers (TrainBatchSize)

        ' �g���[�j���O �f�[�^�̃~�j�o�b�`�̃��[�v
        For MiniBatchIdx = 0 To train_batch_cnt - 1

            ' �~�j�o�b�`�̃f�[�^���Z�b�g���܂��B
            SetBatchData TrainX, train_batch_X, train_batch_Y, TrainLabel, TrainBatchSize, idxes

            ' �~�j�o�b�`���ƂɃp�����[�^���X�V���܂��B
            UpdateMiniBatch train_batch_X, train_batch_Y, train_last_Y, cost_derivative
                        
            If StopDeepLearning Then
                ' ��~�{�^���������ꂽ�ꍇ
                
                Exit For
            End If
            
            ' �t���[�Y��ԂɂȂ�Ȃ��悤��DoEvents�����s���܂��B
            HandleDoEvents
        Next

        ' ���ׂẴ��C���[�̃�������������܂��B
        FreeLayers

        ' ���̓f�[�^�̃C���f�b�N�X�̔z����폜���܂��B
        Erase idxes
        
        If StopDeepLearning Then
            ' ��~�{�^���������ꂽ�ꍇ
            
            Exit For
        End If

        ' ���ׂẴ��C���[�̃����������蓖�āA���C���[�̓��o�͂��������܂��B
        AllocateConnectLayers (TestBatchSize)

        Dim eq_cnt_sum As Long: eq_cnt_sum = 0
                
        ' �e�X�g �f�[�^�̃~�j�o�b�`�̃��[�v
        For MiniBatchIdx = 0 To test_batch_cnt - 1

            ' �~�j�o�b�`�̃f�[�^���Z�b�g���܂��B
            SetBatchData TestX, test_batch_X, test_batch_Y, TestLabel, TestBatchSize, idxes

            ' �e�X�g �f�[�^�̕]�������܂��B
            Dim eq_cnt As Long: eq_cnt = Evaluate(test_batch_X, test_batch_Y, test_last_Y, TestBatchSize, test_arg_max, TestLabel)
            eq_cnt_sum = eq_cnt_sum + eq_cnt
        Next
        Msg "�G�|�b�N = " + Format(EpochIdx) + " : ���� = " + Format(eq_cnt_sum) + " / " + Format(TestCnt)
        
        ' ���𐔂̔z��ɒǉ����܂��B
        AddArray CorrectAns, eq_cnt_sum
        
        ' �`���[�g�̕\�����X�V���܂��B
        MySeries.Values = CorrectAns
        
        FreeLayers
    Next
End Sub


Public Sub DeepLearning()
    Dim use_cuda As Boolean: use_cuda = True
    
    Msg "�J�n���܂����B"
    
    ' �����̒��f�t���O���N���A
    StopDeepLearning = False

    If use_cuda Then
        ' CUDA���g���ꍇ
        
        Set Dev = New DeviceCuda
    Else
        ' CUDA���g��Ȃ��ꍇ
        
        Set Dev = New Device
    End If
    
    ' �f�o�C�X���J�n���܂��B(CUDA�ł̂ݕK�v)
    Dev.DeviceInit

    EpochSize = 100
    TrainBatchSize = 10
    TestBatchSize = 20

    Dim factory As LayerFactoryF:
    
    If use_cuda Then
        ' CUDA���g���ꍇ
        
        Set factory = New LayerFactoryCudaF
    Else
        ' CUDA���g��Ȃ��ꍇ
        
        Set factory = New LayerFactoryF
    End If
    
    NetType = NetworkType.Simple
    NetType = NetworkType.CNN

    Select Case NetType
    Case NetworkType.Simple
        ' 2�w�̃��C���[�����܂��B
        ReDim Layers(1)
        
        Set Layers(0) = factory.MakeFullyConnectedLayer(28 * 28, 30)
        Set Layers(1) = factory.MakeFullyConnectedLayer(30, 10)

    Case NetworkType.CNN
        
        ' 4�w�̃��C���[�����܂��B
        ReDim Layers(3)
        Set Layers(0) = factory.MakeConvolutionalLayer(28, 28, 5, 5)
        Set Layers(1) = factory.MakeMaxPoolingLayer(24, 24, 5, 2)
        Set Layers(2) = factory.MakeFullyConnectedLayer(12 * 12 * 5, 100)
        Set Layers(3) = factory.MakeFullyConnectedLayer(100, 10)
    End Select

    ' �w�K�����Z�b�g���܂��B
    Dim i As Long
    Dim learning_rate As Single: learning_rate = 1#
    For i = 0 To UBound(Layers)
        Layers(i).LearningRate = learning_rate / TrainBatchSize '3.0f
    Next

    ' �ŏ��̃��C���[
    Set FirstLayer = Layers(0)
    
    ' �Ō�̃��C���[
    Set LastLayer = Layers(UBound(Layers))

    ' MNIST�̃f�[�^�t�@�C����ǂݍ��݂܂��B
    ReadMNIST
    
    ' �m���I���z�~���@ (stochastic gradient descent, SGD)
    SGD

    ' ���C���[���폜���܂��B
    For i = 0 To UBound(Layers)
        Layers(i).Destroy
    Next
    Erase Layers
    
    ' �f�o�C�X���I�����܂��B(CUDA�ł̂ݕK�v)
    Dev.DeviceEnd

    Msg "�I�����܂����B"
End Sub


' ��~�{�^���̃N���b�N����
Sub StopButton_OnAction()
    StopDeepLearning = True
End Sub

' ��~�{�^�������܂��B
Sub MakeStopButton()
    If ActiveSheet.Buttons.Count() <> 0 Then
        ' �{�^�����쐬�ς݂̏ꍇ
        
        ' �폜���܂��B
        ActiveSheet.Buttons.Delete
    End If
    
    ' ��~�{�^�������܂��B
    With ActiveSheet.Buttons.Add(Range("B5").Left, Range("B5").Top, Range("B5:C6").Width, Range("B5:C6").Height)
        .OnAction = "StopButton_OnAction"
        .Characters.text = "��~"
    End With
End Sub

' ���𐔂̃`���[�g�����܂��B
Sub MakeChart()
    If ActiveSheet.ChartObjects.Count() <> 0 Then
        ' �`���[�g���쐬�ς݂̏ꍇ
        
        ' �폜���܂��B
        ActiveSheet.ChartObjects.Delete
    End If
    
    
    ' ���𐔂̃`���[�g�����܂��B
    With ActiveSheet.ChartObjects.Add(250, 20, 300, 200).Chart
        .ChartType = xlColumnClustered
        Set MySeries = .SeriesCollection.NewSeries
        With MySeries
'            .Values = Array(1, 3, 5, 7, 11, 13, 17, 19)
'            .Values = Array(1, 3, 5, 7, 11, 13, 17, 19)
            .name = "����"
        End With
    End With
    
    ' ���𐔂̔z����N���A���܂��B
    If IsArray2(CorrectAns) Then
        Erase CorrectAns
    End If
End Sub

' MNIST�̃e�X�g
Public Sub TestMNIST()
    MakeStopButton
    MakeChart
    DeepLearning
End Sub
