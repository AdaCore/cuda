With System;

Package Bilateral_Kernel Is

    Procedure Bilateral (Img_Addr          : System.Address; 
                         Filtered_Img_Addr : System.Address;
                         Width             : Integer;
                         Height            : Integer;
                         Spatial_Stdev     : Float;
                         Color_Dist_Stdev  : Float;
                         I                 : Integer;
                         J                 : Integer);


    Procedure Bilateral_Cuda (Device_Img          : System.Address; 
                              Device_Filtered_Img : System.Address;
                              Width               : Integer;
                              Height              : Integer;
                              Spatial_Stdev       : Float;
                              Color_Dist_Stdev    : Float) With Cuda_Global;

End Bilateral_Kernel;
