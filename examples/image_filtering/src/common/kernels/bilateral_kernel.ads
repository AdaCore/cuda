with system;

package bilateral_kernel is

    procedure bilateral (img_addr          : system.address; 
                         filtered_img_addr : system.address;
                         width             : integer;
                         height            : integer;
                         spatial_stdev     : float;
                         color_dist_stdev  : float;
                         i                 : integer;
                         j                 : integer);


    procedure bilateral_cuda (device_img          : system.address; 
                              device_filtered_img : system.address;
                              width               : integer;
                              height              : integer;
                              spatial_stdev       : float;
                              color_dist_stdev    : float) with cuda_global;

end bilateral_kernel;