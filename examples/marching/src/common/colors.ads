------------------------------------------------------------------------------
--                       Copyright (C) 2021, AdaCore                        --
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

package Colors is

   type RGB_T is record
      R, G, B : Float;
   end record;

   function "+" (Left, Right : RGB_T) return RGB_T is
     (Left.R + Right.R, Left.G + Right.G, Left.B + Right.B);

   function "-" (Left, Right : RGB_T) return RGB_T is
     (Left.R - Right.R, Left.G - Right.G, Left.B - Right.B);

   function "*" (Left : RGB_T; Right : Float) return RGB_T is
     (Left.R * Right, Left.G * Right, Left.B * Right);

   function "/" (Left : RGB_T; Right : Float) return RGB_T is
     (Left.R / Right, Left.G / Right, Left.B / Right);

   type HSL_T is record
      H, S, L: Float;
   end record;

   function HSL_To_RGB (Src : HSL_T) return RGB_T;

   function RGB_To_HSL (src : RGB_T) return HSL_T;

end Colors;
