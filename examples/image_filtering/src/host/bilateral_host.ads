with graphic;

package bilateral_host is

    package G renames graphic;

    procedure bilateral_cpu (host_img          : G.image_access; 
                             host_filtered_img : G.image_access;
                             width             : integer;
                             height            : integer;
                             spatial_stdev     : float;
                             color_dist_stdev  : float);

    procedure bilateral_cuda (host_img          : G.image_access; 
                              host_filtered_img : G.image_access;
                              width             : integer;
                              height            : integer;
                              spatial_stdev     : float;
                              color_dist_stdev  : float);
end bilateral_host;