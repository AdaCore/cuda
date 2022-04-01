With Ada.Strings;       Use Ada.Strings;
With Ada.Strings.Fixed; Use Ada.Strings.Fixed;
With Ada.Text_IO;       Use Ada.Text_IO;

Package Body Exporter Is

    Procedure Write_Image (File_Path : String; Img : G.Image) Is
        File   : File_Type;
        Width  : Natural := Img'Length (1);
        Height : Natural := Img'Length (2);
    Begin
        Create (File, Out_File, File_Path);
        Put_Line (File, "P3");
        Put_Line (File, "#median filtered image");
        Put_Line (File, Trim (Width'Image, Left) & " " & Trim (Height'Image, Left));
        Put_Line (File, "255");
        For J In Img'Range (2) Loop
            For I In Img'Range (1) Loop
                Put_Line (File, Trim (Integer(Img (I, J).R)'Image, Left) & " " & 
                                Trim (Integer(Img (I, J).G)'Image, Left) & " " & 
                                Trim (Integer(Img (I, J).B)'Image, Left));
            End Loop;
        End Loop;
    End;

End Exporter;
