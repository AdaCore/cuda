package body graphic is

    procedure normalize (img : image_access) is
    begin
        for i in img.all'range(1) loop
            for j in img.all'range(2) loop
                img (i, j).r := img (i, j).r / 255.0;
                img (i, j).g := img (i, j).g / 255.0;
                img (i, j).b := img (i, j).b / 255.0;
            end loop;
        end loop;
    end;
    
end graphic;