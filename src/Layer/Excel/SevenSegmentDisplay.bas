Attribute VB_Name = "SevenSegmentDisplay"
Option Explicit

Public Sub SetSevenSegmentData(ByRef x() As Single, ByRef y() As Single, ByRef label() As Byte)
    Dim i As Long, j As Long
    
    For i = 0 To UBound(y, 1)
        For j = 0 To UBound(y, 2)
            If i = j Then
                y(i, j) = 1
            Else
                y(i, j) = 0
            End If
        Next
        
        label(i) = i
    Next

    '0  0�~3F   0�~7E   on  on  on  on  on  on  off
    x(0, 0) = 1
    x(1, 0) = 1
    x(2, 0) = 1
    x(3, 0) = 1
    x(4, 0) = 1
    x(5, 0) = 1
    x(6, 0) = 0
    
    '1  0�~06   0�~30   off on  on  off off off off
    x(0, 1) = 0
    x(1, 1) = 1
    x(2, 1) = 1
    x(3, 1) = 0
    x(4, 1) = 0
    x(5, 1) = 0
    x(6, 1) = 0
    
    '2  0�~5B   0�~6D   on  on  off on  on  off on
    x(0, 2) = 1
    x(1, 2) = 1
    x(2, 2) = 0
    x(3, 2) = 1
    x(4, 2) = 1
    x(5, 2) = 0
    x(6, 2) = 1
    
    '3  0�~4F   0�~79   on  on  on  on  off off on
    x(0, 3) = 1
    x(1, 3) = 1
    x(2, 3) = 1
    x(3, 3) = 1
    x(4, 3) = 0
    x(5, 3) = 0
    x(6, 3) = 1
    
    '4  0�~66   0�~33   off on  on  off off on  on
    x(0, 4) = 0
    x(1, 4) = 1
    x(2, 4) = 1
    x(3, 4) = 0
    x(4, 4) = 0
    x(5, 4) = 1
    x(6, 4) = 1
    
    '5  0�~6D   0�~5B   on  off on  on  off on  on
    x(0, 5) = 1
    x(1, 5) = 0
    x(2, 5) = 1
    x(3, 5) = 1
    x(4, 5) = 0
    x(5, 5) = 1
    x(6, 5) = 1
    
    '6  0�~7D   0�~5F   on  off on  on  on  on  on
    x(0, 6) = 1
    x(1, 6) = 0
    x(2, 6) = 1
    x(3, 6) = 1
    x(4, 6) = 1
    x(5, 6) = 1
    x(6, 6) = 1
    
    '7  0�~07   0�~70   on  on  on  off off off off
    x(0, 7) = 1
    x(1, 7) = 1
    x(2, 7) = 1
    x(3, 7) = 0
    x(4, 7) = 0
    x(5, 7) = 0
    x(6, 7) = 0
    
    '8  0�~7F   0�~7F   on  on  on  on  on  on  on
    x(0, 8) = 1
    x(1, 8) = 1
    x(2, 8) = 1
    x(3, 8) = 1
    x(4, 8) = 1
    x(5, 8) = 1
    x(6, 8) = 1
    
    '9  0�~6F   0�~7B   on  on  on  on  off on  on
    x(0, 9) = 1
    x(1, 9) = 1
    x(2, 9) = 1
    x(3, 9) = 1
    x(4, 9) = 0
    x(5, 9) = 1
    x(6, 9) = 1
    
End Sub
    'A  0�~77   0�~77   on  on  on  off on  on  on
    'b  0�~7C   0�~1F   off off on  on  on  on  on
    'C  0�~39   0�~4E   on  off off on  on  on  off
    'd  0�~5E   0�~3D   off on  on  on  on  off on
    'E  0�~79   0�~4F   on  off off on  on  on  on
    'F  0�~71   0�~47   on  off off off on  on  on

