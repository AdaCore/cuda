with Ada.Strings;       use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Text_IO;       use Ada.Text_IO;

package body exporter is

    procedure write_image (file_path : string; img : G.Image) is
        file   : File_Type;
        width  : natural := img'length (1);
        height : natural := img'length (2);
    begin
        create (file, out_file, file_path);
        put_line (file, "P3");
        put_line (file, "#median filtered image");
        put_line (file, trim (width'image, left) & " " & trim (height'image, left));
        put_line (file, "255");
        for j in img'range (2) loop
            for i in img'range (1) loop
                put_line (file, trim (integer(img (i, j).r)'image, left) & " " & 
                                trim (integer(img (i, j).g)'image, left) & " " & 
                                trim (integer(img (i, j).b)'image, left));
            end loop;
        end loop;
    end;

end exporter;