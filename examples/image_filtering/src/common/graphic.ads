package graphic is
    type rgb is record
        r, g, b : Float;
    end record;

    type image is array (Natural range <>, Natural range <>) of rgb;
    type image_access is access all image;

    procedure normalize (img : image_access);
end graphic;
