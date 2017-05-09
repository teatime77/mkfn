VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "ILayerVBA"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Option Explicit

Public BatchSize As Long
Public LearningRate As Single

Public Sub Allocate()
End Sub

Public Sub Free()
End Sub

Public Sub Forward()
End Sub

Public Sub Backward()
End Sub

Public Sub UpdateParameter()
End Sub

Public Sub SetInputData(ByRef src() As Single)
End Sub

Public Sub GetOutputData(ByRef dst() As Single)
End Sub

Public Sub SetOutputDeltaData(ByRef src() As Single)
End Sub

Public Sub GeInputDeltaData(ByRef dst() As Single)
End Sub

