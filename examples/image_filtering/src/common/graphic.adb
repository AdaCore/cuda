Package Body Graphic Is

    Procedure Normalize (Img : Image_Access) Is
    Begin
        For I In Img.All'Range(1) Loop
            For J In Img.All'Range(2) Loop
                Img (I, J).R := Img (I, J).R / 255.0;
                Img (I, J).G := Img (I, J).G / 255.0;
                Img (I, J).B := Img (I, J).B / 255.0;
            End Loop;
        End Loop;
    End;
    
End Graphic;
