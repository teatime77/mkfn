VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "Sheet1"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = True
Option Explicit

Private Sub StartButton_Click()
    NetworkTest
End Sub

Private Sub StopButton_Click()
    StopNetworkTest = True
End Sub

Private Sub GraphButton_Click()
    With ActiveSheet.ChartObjects.Add(30, 50, 300, 200).Chart
        .ChartType = xlColumnClustered
        Set MySeries = .SeriesCollection.NewSeries
        With MySeries
            .Values = Array(1, 3, 5, 7, 11, 13, 17, 19)
            .name = "Sample"
        End With
    End With

End Sub

Private Sub ImageButton_Click()
    With ActiveSheet.Shapes.AddShape(msoShapeRectangle, 144, 144, 72, 72)
        .name = "Red Square"
        .Fill.ForeColor.RGB = RGB(255, 0, 0)
        .Line.DashStyle = msoLineDashDot
    End With
    
    Dim myShape As Shape
    
    Dim path As String: path = ActiveWorkbook.path & "\A.png"
    
    '--(1) 選択位置に画像ファイルを挿入し、変数myShapeに格納
    Set myShape = ActiveSheet.Shapes.AddPicture(Filename:=path, LinkToFile:=True, SaveWithDocument:=False, Left:=400, Top:=100, Width:=128, Height:=128)
    
    Debug.Print ""

    Dim MyImage As Object
    Set MyImage = ActiveSheet.OLEObjects.Add(ClassType:="Forms.Image.1", Link:=False, DisplayAsIcon:=False, Left:=100, Top:=200, Width:=300, Height:=400)

End Sub


