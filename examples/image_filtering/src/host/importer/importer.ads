with Graphic;

package importer is

    package G renames Graphic;

    procedure get_image_infos (file_path : string; 
                               width     : out natural; 
                               height    : out natural);

    procedure fill_image (file_path : string; 
                          width     : natural; 
                          height    : natural; 
                          img       : in out G.Image);

end importer;