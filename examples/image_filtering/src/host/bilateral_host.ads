With Graphic;

Package Bilateral_Host Is

    Package G Renames Graphic;

    Procedure Bilateral_Cpu (Host_Img          : G.Image_Access; 
                             Host_Filtered_Img : G.Image_Access;
                             Width             : Integer;
                             Height            : Integer;
                             Spatial_Stdev     : Float;
                             Color_Dist_Stdev  : Float);

    Procedure Bilateral_Cuda (Host_Img          : G.Image_Access; 
                              Host_Filtered_Img : G.Image_Access;
                              Width             : Integer;
                              Height            : Integer;
                              Spatial_Stdev     : Float;
                              Color_Dist_Stdev  : Float);
End Bilateral_Host;
