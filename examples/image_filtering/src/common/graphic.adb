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

package body Graphic is

   procedure Normalize (Img : Image_Access) is
   begin
      for I in Img.all'Range (1) loop
         for J in Img.all'Range (2) loop
            Img (I, J).R := Img (I, J).R / 255.0;
            Img (I, J).G := Img (I, J).G / 255.0;
            Img (I, J).B := Img (I, J).B / 255.0;
         end loop;
      end loop;
   end;

end Graphic;
