Package Graphic Is
    Type Rgb Is Record
        R, G, B : Float;
    End Record;

    Type Image Is Array (Natural Range <>, Natural Range <>) Of Rgb;
    Type Image_Access Is Access All Image;

    Procedure Normalize (Img : Image_Access);
End Graphic;
