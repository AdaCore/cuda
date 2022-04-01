with ada.text_io;
with Ada.Directories; use Ada.Directories;

with GNAT.Spitbol.Patterns;  

with ada.strings.unbounded; use ada.strings.Unbounded;

package body importer is

    procedure get_image_infos (file_path : string; width: out natural; height: out natural) is
    use GNAT.Spitbol.Patterns;
    use ada.text_io;
    input_file : file_type;

    begin
        width  := 0;
        height := 0;
        open (input_file, in_file, file_path);
        declare
            magic_number   : string := get_line(input_file);
            note           : string := get_line(input_file);
            natural_p      : constant Pattern := Span("0123456789");
            w, h           : vstring_var;
            width_height_p : constant Pattern := Pos(0) & natural_p * w & Span(' ') & natural_p * h;
            width_height   : vstring_var := to_unbounded_string(get_line(input_file));
        begin
            if match (width_height, width_height_p, "") then
                width  := Natural'Value (to_string(w));
                height := Natural'Value (to_string(h));
            end if;
        end;
        close (input_file);
    end;

    procedure fill_image (file_path : string; width: natural; height: natural; img : in out G.Image) is
        use GNAT.Spitbol.Patterns;
        use ada.text_io;
        input_file : file_type;
    begin
        open (input_file, in_file, file_path);
        declare
            color_value   : vstring_var;
            color_value_p : constant Pattern := Span("0123456789") * color_value;

            magic_number  : string := get_line(input_file);
            note          : string := get_line(input_file);
            width_height  : string := get_line(input_file);
            max_value     : string := get_line(input_file);

            component_counter : natural := 0;
            done              : boolean := False;

            col, row          : natural;
        begin
            while not done loop
                component_counter := component_counter + 1;
                col               := ((component_counter-1) mod width) + 1;
                row               := (component_counter + (width - 1)) / width;
                for i in 1 .. 3 loop
                    declare
                        vline : vstring_var := to_unbounded_string (get_line (input_file));
                    begin
                        if match (vline, color_value_p, "") then
                        null;
                            case i is
                                when 1 =>
                                    img (col, row).r := float'value (to_string (color_value));
                                when 2 =>
                                    img (col, row).g := float'value (to_string (color_value));
                                when 3 =>
                                    img (col, row).b := float'value (to_string (color_value));
                            end case;
                        end if;
                    end;
                end loop;
                if end_of_file (input_file) then
                    done := true;
                end if;
            end loop;
        end;
        close (input_file);
    end;

end importer;