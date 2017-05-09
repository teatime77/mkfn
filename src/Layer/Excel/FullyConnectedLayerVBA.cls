VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "FullyConnectedLayerVBA"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Option Explicit

Implements ILayerVBA

Private m_BatchSize As Long
Private m_LearningRate As Single

Public XLen As Long
Public YLen As Long

Dim x() As Single
Dim u() As Single
Dim y() As Single

Dim w() As Single
Dim b() As Single
    
Dim delta_x() As Single
Dim delta_u() As Single
Dim delta_y() As Single

Dim delta_w() As Single
Dim delta_b() As Single

Public Sub SetSize(x_len As Long, y_len As Long)
    XLen = x_len
    YLen = y_len
End Sub
   

Private Property Get ILayerVBA_BatchSize() As Long
    ILayerVBA_BatchSize = m_BatchSize
End Property

Private Property Let ILayerVBA_BatchSize(ByVal batch_size As Long)
    m_BatchSize = batch_size
End Property
    
    
Private Property Get ILayerVBA_LearningRate() As Single
    ILayerVBA_LearningRate = m_LearningRate
End Property

Private Property Let ILayerVBA_LearningRate(ByVal learning_rate As Single)
    m_LearningRate = learning_rate
End Property
        
Public Sub ILayerVBA_Allocate()
    Dim i As Long, j As Long
    
    ReDim x(XLen - 1, m_BatchSize - 1)
    ReDim u(YLen - 1, m_BatchSize - 1)
    ReDim y(YLen - 1, m_BatchSize - 1)
    
    ReDim w(YLen - 1, XLen - 1)
    ReDim b(YLen - 1)
    
    For i = 0 To YLen - 1
        For j = 0 To XLen - 1
            w(i, j) = Rnd
        Next
        b(i) = Rnd
    Next

    ReDim delta_x(XLen - 1, m_BatchSize - 1)
    ReDim delta_u(YLen - 1, m_BatchSize - 1)
    ReDim delta_y(YLen - 1, m_BatchSize - 1)
    
    ReDim delta_w(YLen - 1, XLen - 1, m_BatchSize - 1)
    ReDim delta_b(YLen - 1, m_BatchSize - 1)
End Sub

Public Sub ILayerVBA_Free()
    Erase x
    Erase u
    Erase y
    
    Erase w
    Erase b

    Erase delta_x
    Erase delta_u
    Erase delta_y
    
    Erase delta_w
    Erase delta_b
End Sub

Public Sub ILayerVBA_Forward()
    Dim i As Long, j As Long, batch_idx As Long
    
    For batch_idx = 0 To m_BatchSize - 1
        For i = 0 To YLen - 1
            Dim sum As Single: sum = 0
            For j = 0 To XLen - 1
                sum = sum + x(j, batch_idx) * w(i, j)
            Next
            u(i, batch_idx) = sum + b(i)
            
            y(i, batch_idx) = Sigmoid(u(i, batch_idx))
        Next
    Next
End Sub

Public Sub ILayerVBA_Backward()
    Dim i As Long, j As Long, batch_idx As Long
    
    For batch_idx = 0 To m_BatchSize - 1
        For i = 0 To YLen - 1
            delta_u(i, batch_idx) = delta_y(i, batch_idx) * SigmoidPrime(u(i, batch_idx))
        Next
        
        For j = 0 To XLen - 1
            Dim wk As Single: wk = 0
            For i = 0 To YLen - 1
                wk = wk + delta_u(i, batch_idx) * w(i, j)
            Next
            delta_x(j, batch_idx) = wk
        Next
        
        For i = 0 To YLen - 1
            For j = 0 To XLen - 1
                delta_w(i, j, batch_idx) = delta_u(i, batch_idx) * x(j, batch_idx)
            Next
        Next
        
         For i = 0 To YLen - 1
            delta_b(i, batch_idx) = delta_u(i, batch_idx)
         Next
    Next
End Sub

Public Sub ILayerVBA_UpdateParameter()
    Dim i As Long, j As Long, batch_idx As Long
    
    For i = 0 To YLen - 1
        For j = 0 To XLen - 1
            Dim delta_w_sum As Single: delta_w_sum = 0
            For batch_idx = 0 To m_BatchSize - 1
                delta_w_sum = delta_w_sum + delta_w(i, j, batch_idx)
            Next
            w(i, j) = w(i, j) - m_LearningRate * delta_w_sum
        Next
    Next
    
    For i = 0 To YLen - 1
        Dim delta_b_sum As Single: delta_b_sum = 0
        For batch_idx = 0 To m_BatchSize - 1
            delta_b_sum = delta_b_sum + delta_b(i, batch_idx)
        Next
        b(i) = b(i) - m_LearningRate * delta_b_sum
    Next
End Sub

Public Sub ILayerVBA_SetInputData(ByRef src() As Single)
    x = src
End Sub

Public Sub ILayerVBA_GetOutputData(ByRef dst() As Single)
    dst = y
End Sub

Public Sub ILayerVBA_SetOutputDeltaData(ByRef src() As Single)
    delta_y = src
End Sub

Public Sub ILayerVBA_GeInputDeltaData(ByRef dst() As Single)
    dst = delta_x
End Sub

