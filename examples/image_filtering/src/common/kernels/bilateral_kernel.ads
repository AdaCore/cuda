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

with System;

with Graphic;

package Bilateral_Kernel is

   package G renames Graphic;

   procedure Bilateral
     (Img               : G.Image; 
      Filtered_Img      : in out G.Image;
      Width             : Integer; Height : Integer; Spatial_Stdev : Float;
      Color_Dist_Stdev  : Float; I : Integer; J : Integer);

   procedure Bilateral_Cuda
     (Device_Img                      : G.Image_Device_Access; 
      Device_Filtered_Img             : G.Image_Device_Access;
      Width, Height                   : Integer; 
      Spatial_Stdev, Color_Dist_Stdev : Float) 
        with Cuda_Global;

end Bilateral_Kernel;
