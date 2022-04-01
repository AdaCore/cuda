With Graphic;

Package Importer Is

    Package G Renames Graphic;

    Procedure Get_Image_Infos (File_Path : String; 
                               Width     : Out Natural; 
                               Height    : Out Natural);

    Procedure Fill_Image (File_Path : String; 
                          Width     : Natural; 
                          Height    : Natural; 
                          Img       : In Out G.Image);

End Importer;
