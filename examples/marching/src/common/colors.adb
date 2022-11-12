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

package body Colors is

   -- see https://stackoverflow.com/questions/2353211/hsl-to-rgb-color-conversion

   function Hue_To_RGB (P, Q, T : Float) return Float is
      T_Clamped : Float := T;
   begin
      if T_Clamped < 0.0 then
         T_Clamped := @ + 1.0;
      end if;

      if T_Clamped > 1.0 then
         T_Clamped := @ - 1.0;
      end if;

      if T_Clamped < 1.0 / 6.0 then
         return P + (Q - P) * 6.0 * T_Clamped;
      end if;

      if T_Clamped < 1.0 / 2.0 then
         return Q;
      end if;

      if T_Clamped < 2.0 / 3.0 then
         return P + (Q - P) * (2.0 / 3.0 - T_Clamped) * 6.0;
      end if;

      return P;
   end Hue_To_RGB;

   function HSL_To_RGB (Src : HSL_T) return RGB_T is
      Res : RGB_T;
   begin
      if Src.S = 0.0 then
         Res.R := Src.L;
         Res.G := Src.L;
         Res.B := Src.L;
      else
         declare
            Q : Float :=
              (if Src.L < 0.5 then Src.L * (1.0 + Src.S)
               else Src.L + Src.S - Src.L * Src.S);
            P : Float := 2.0 * Src.L - Q;
         begin
            Res.R := Hue_To_RGB (P, Q, Src.H + 1.0 / 3.0);
            Res.G := Hue_To_RGB (P, Q, Src.H);
            Res.B := Hue_To_RGB (P, Q, Src.H - 1.0 / 3.0);
         end;
      end if;

      return Res;
   end HSL_To_RGB;

   function RGB_To_HSL (src : RGB_T) return HSL_T is
      Max : Float := Float'Max (src.R, Float'Max (src.G, src.B));
      Min : Float := Float'Min (src.R, Float'Min (src.G, src.B));
      Res : HSL_T;
   begin
      Res.L := (Max + Min) / 2.0;

      if Max = Min then
         Res.H := 0.0;
         Res.S := 0.0;
      else
         declare
            D : Float := Max - Min;
         begin
            Res.S :=
              (if Res.L > 0.5 then D / (2.0 - Max - Min) else D / (Max + Min));

            if src.R >= src.G and src.R >= src.B then
               Res.H :=
                 (src.G - src.B) / D + (if src.G < src.B then 6.0 else 0.0);
            elsif src.G >= src.B then
               Res.H := (src.B - src.R) / D + 2.0;
            else
               Res.H := (src.R - src.G) / D + 4.0;
            end if;

            Res.H := @ / 6.0;
         end;
      end if;

      return Res;
   end RGB_To_HSL;

end Colors;
