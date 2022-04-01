With Ada.Text_Io;
With Ada.Directories; Use Ada.Directories;

With GNAT.Spitbol.Patterns;  

With Ada.Strings.Unbounded; Use Ada.Strings.Unbounded;

Package Body Importer Is

    Procedure Get_Image_Infos (File_Path : String; Width: Out Natural; Height: Out Natural) Is
    Use GNAT.Spitbol.Patterns;
    Use Ada.Text_Io;
    Input_File : File_Type;

    Begin
        Width  := 0;
        Height := 0;
        Open (Input_File, In_File, File_Path);
        Declare
            Magic_Number   : String := Get_Line(Input_File);
            Note           : String := Get_Line(Input_File);
            Natural_P      : Constant Pattern := Span("0123456789");
            W, H           : Vstring_Var;
            Width_Height_P : Constant Pattern := Pos(0) & Natural_P * W & Span(' ') & Natural_P * H;
            Width_Height   : Vstring_Var := To_Unbounded_String(Get_Line(Input_File));
        Begin
            If Match (Width_Height, Width_Height_P, "") Then
                Width  := Natural'Value (To_String(W));
                Height := Natural'Value (To_String(H));
            End If;
        End;
        Close (Input_File);
    End;

    Procedure Fill_Image (File_Path : String; Width: Natural; Height: Natural; Img : In Out G.Image) Is
        Use GNAT.Spitbol.Patterns;
        Use Ada.Text_Io;
        Input_File : File_Type;
    Begin
        Open (Input_File, In_File, File_Path);
        Declare
            Color_Value   : Vstring_Var;
            Color_Value_P : Constant Pattern := Span("0123456789") * Color_Value;

            Magic_Number  : String := Get_Line(Input_File);
            Note          : String := Get_Line(Input_File);
            Width_Height  : String := Get_Line(Input_File);
            Max_Value     : String := Get_Line(Input_File);

            Component_Counter : Natural := 0;
            Done              : Boolean := False;

            Col, Row          : Natural;
        Begin
            While Not Done Loop
                Component_Counter := Component_Counter + 1;
                Col               := ((Component_Counter-1) Mod Width) + 1;
                Row               := (Component_Counter + (Width - 1)) / Width;
                For I In 1 .. 3 Loop
                    Declare
                        Vline : Vstring_Var := To_Unbounded_String (Get_Line (Input_File));
                    Begin
                        If Match (Vline, Color_Value_P, "") Then
                        Null;
                            Case I Is
                                When 1 =>
                                    Img (Col, Row).R := Float'Value (To_String (Color_Value));
                                When 2 =>
                                    Img (Col, Row).G := Float'Value (To_String (Color_Value));
                                When 3 =>
                                    Img (Col, Row).B := Float'Value (To_String (Color_Value));
                            End Case;
                        End If;
                    End;
                End Loop;
                If End_Of_File (Input_File) Then
                    Done := True;
                End If;
            End Loop;
        End;
        Close (Input_File);
    End;

End Importer;
