with system;

with cuda.runtime_api;
use cuda.runtime_api; -- block_dim, block_idx, thread_idx
 
with interfaces.c;
use interfaces.c; -- operators for block_dim, block_idx, thread_idx

with graphic;

package body bilateral_kernel is

    package G renames graphic; 

    procedure bilateral (img_addr          : system.address; 
                         filtered_img_addr : system.address;
                         width             : integer;
                         height            : integer;
                         spatial_stdev     : float;
                         color_dist_stdev  : float;
                         i                 : integer;
                         j                 : integer) is

        kernel_size         : integer := integer (2.0 * spatial_stdev * 3.0);
        half_size           : natural := (kernel_size - 1) / 2;

        spatial_gaussian    : float := 0.0;
        color_dist_gaussian : float := 0.0;
        sg_cdg              : float := 0.0;
        sum_sg_cdg          : float := 0.0;
        rgb_dist            : float := 0.0;
        filtered_rgb        : g.rgb := (0.0, 0.0, 0.0);

        img          : g.image (1 .. width, 1 .. height) with address => img_addr;
        filtered_img : g.image (1 .. width, 1 .. height) with address => filtered_img_addr;

        function exponential (n : integer; x : float) return float is
            sum : float := 1.0;
        begin
            for i in reverse 1 .. n loop
                sum := 1.0 + x * sum / float(i);
            end loop;
            return sum;
        end;

        function sqrt(x : float; t : float) return float is
            y : float := 1.0;
        begin
            while abs (x/y - y) > t loop
                y := (y + x / y) / 2.0;
            end loop;
            return y;
        end;

        function compute_spatial_gaussian (m : float; n : float) return float is
            spatial_variance : float := spatial_stdev * spatial_stdev;
            two_pi_variance  : float := 2.0*3.1416*spatial_variance;
            exp              : float := exponential (10, -0.5 * ((m*m + n*n)/spatial_variance));
        begin
            return (1.0 / (two_pi_variance)) * exp;
        end;

        function compute_color_dist_gaussian (k : float) return float is
            color_dist_variance : float := color_dist_stdev * color_dist_stdev;
        begin
            return (1.0 / (sqrt(2.0*3.1416, 0.001)*color_dist_stdev)) * exponential (10, -0.5 * ((k*k)/color_dist_variance));
        end;

        -- compute kernel bounds
        xb : integer := i - half_size;
        xe : integer := i + half_size;
        yb : integer := j - half_size;
        ye : integer := j + half_size;

        test : float := 0.0;

    begin
        for x in xb .. xe loop
            for y in yb .. ye loop
                if x >= 1 and x <= width and y >= 1 and y <= height then
                    -- compute color distance
                    rgb_dist := sqrt ((img (i, j).r - img (x, y).r) * (img (i, j).r - img (x, y).r) +
                                      (img (i, j).g - img (x, y).g) * (img (i, j).g - img (x, y).g) +
                                      (img (i,j ).b - img (x, y).b) * (img (i, j).b - img (x, y).b), 0.001);

                    -- compute gaussians
                    spatial_gaussian := compute_spatial_gaussian (float(i - x), float(j - y));
                    color_dist_gaussian := compute_color_dist_gaussian (rgb_dist);

                    -- multiply gaussians
                    sg_cdg := spatial_gaussian*color_dist_gaussian;

                    -- accumulate
                    filtered_rgb.r := filtered_rgb.r + sg_cdg * img (x,y).r;
                    filtered_rgb.g := filtered_rgb.g + sg_cdg * img (x,y).g;
                    filtered_rgb.b := filtered_rgb.b + sg_cdg * img (x,y).b;

                    -- track to normalize intensity
                    sum_sg_cdg := sum_sg_cdg + sg_cdg;
                end if;
            end loop;
        end loop;

        -- normalize intensity
        filtered_img (i,j).r := filtered_rgb.r / sum_sg_cdg;
        filtered_img (i,j).g := filtered_rgb.g / sum_sg_cdg;
        filtered_img (i,j).b := filtered_rgb.b / sum_sg_cdg;
    end;

    procedure bilateral_cuda (device_img          : system.address; 
                              device_filtered_img : system.address;
                              width               : integer;
                              height              : integer;
                              spatial_stdev       : float;
                              color_dist_stdev    : float) is
        i : integer := integer (block_idx.x);
        j : integer := integer (block_idx.y);
    begin
        bilateral (device_img, device_filtered_img, width, height, spatial_stdev, color_dist_stdev, i, j);
    end;

end bilateral_kernel;