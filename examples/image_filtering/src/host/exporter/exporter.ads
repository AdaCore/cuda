with graphic;

package exporter is

    package G renames graphic;
    procedure write_image (file_path : string; img : G.Image);

end exporter;