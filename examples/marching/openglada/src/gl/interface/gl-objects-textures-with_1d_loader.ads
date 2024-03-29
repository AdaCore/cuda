--------------------------------------------------------------------------------
-- Copyright (c) 2013, Felix Krause <contact@flyx.org>
--
-- Permission to use, copy, modify, and/or distribute this software for any
-- purpose with or without fee is hereby granted, provided that the above
-- copyright notice and this permission notice appear in all copies.
--
-- THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
-- WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
-- MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
-- ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
-- WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
-- ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
-- OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
--------------------------------------------------------------------------------

generic
   type Base (<>) is new Texture_Proxy with private;
package GL.Objects.Textures.With_1D_Loader is
   pragma Preelaborate;

   type Target is new Base with null record;

   procedure Load_Empty_Texture
     (Object: Target; Level : Mipmap_Level;
      Internal_Format : Pixels.Internal_Format;
      Width : Types.Size);

   procedure Storage (Object : Target; Levels : Types.Size;
                      Internal_Format : Pixels.Internal_Format;
                      Width : Types.Size);

   type Fillable_Target is new With_1D_Loader.Target with null record;

   procedure Load_From_Data
     (Object : Fillable_Target; Level : Mipmap_Level;
      Internal_Format : Pixels.Internal_Format;
      Width           : Types.Size;
      Source_Format   : Pixels.Data_Format;
      Source_Type     : Pixels.Data_Type;
      Source          : Image_Source);

   procedure Load_Sub_Image_From_Data
     (Object : Fillable_Target; Level : Mipmap_Level;
       X_Offset, Y_Offset :Int;
       Width, Height : Size;
       Format        : Pixels.Data_Format;
       Data_Type     : Pixels.Data_Type;
      Source         : Image_Source);

   procedure Load_Compressed
     (Object            : Fillable_Target;
      Level             : Mipmap_Level;
      Internal_Format   : Pixels.Internal_Format;
      Width, Image_Size : Types.Size;
      Source            : Image_Source);
end GL.Objects.Textures.With_1D_Loader;
