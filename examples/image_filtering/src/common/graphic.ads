------------------------------------------------------------------------------
--                       Copyright (C) 2017, AdaCore                        --
-- This is free software;  you can redistribute it  and/or modify it  under --
-- terms of the  GNU General Public License as published  by the Free Soft- --
-- ware  Foundation;  either version 3,  or (at your option) any later ver- --
-- sion.  This software is distributed in the hope  that it will be useful, --
-- but WITHOUT ANY WARRANTY;  without even the implied warranty of MERCHAN- --
-- TABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public --
-- License for  more details.  You should have  received  a copy of the GNU --
-- General  Public  License  distributed  with  this  software;   see  file --
-- COPYING3.  If not, go to http://www.gnu.org/licenses for a complete copy --
-- of the license.                                                          --
------------------------------------------------------------------------------

package Graphic is

   type Rgb is record
      R, G, B : Float;
   end record;

   function "/" (Left : Rgb; Right : Float) return Rgb is
     (Left.R / Right, Left.G / Right, Left.B / Right);

   function "*" (Left : Rgb; Right : Float) return Rgb is
     (Left.R * Right, Left.G * Right, Left.B * Right);

   function "+" (Left : Rgb; Right : Rgb) return Rgb is
     (Left.R + Right.R, Left.G + Right.G, Left.B + Right.B);

   function Distance_Square (Left : Rgb; Right : Rgb) return Float is
     ((Left.R - Right.R) * (Left.R - Right.R) +
      (Left.G - Right.G) * (Left.G - Right.G) +
      (Left.B - Right.B) * (Left.B - Right.B));

   type Image is array (Natural range <>, Natural range <>) of Rgb;
   type Image_Access is access all Image;

   --type Array_Device_Access is access Float_Array
   --  with Designated_Storage_Model => CUDA.Storage_Models.Model;

end Graphic;
