--------------------------------------------------------------------------------
-- Copyright (c) 2012, Felix Krause <contact@flyx.org>
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

package body GL.Helpers is

   function Float_Array (Value : Colors.Color) return Low_Level.Single_Array is
      use GL.Types.Colors;
   begin
      return Low_Level.Single_Array' (1 => Value (R),
                                     2 => Value (G),
                                     3 => Value (B),
                                     4 => Value (A));
   end Float_Array;

   function Color (Value : Low_Level.Single_Array) return Colors.Color is
      use GL.Types.Colors;
   begin
      return Colors.Color' (R => Value (1), G => Value (2), B => Value (3),
                           A => Value (4));
   end Color;

end GL.Helpers;
