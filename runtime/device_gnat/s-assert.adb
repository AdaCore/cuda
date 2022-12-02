------------------------------------------------------------------------------
--                                                                          --
--                          GNAT RUN-TIME COMPONENTS                        --
--                                                                          --
--                     S Y S T E M . A S S E R T I O N S                    --
--                                                                          --
--                                 B o d y                                  --
--                                                                          --
--          Copyright (C) 1992-2022, Free Software Foundation, Inc.         --
--                                                                          --
-- GNAT is free software;  you can  redistribute it  and/or modify it under --
-- terms of the  GNU General Public License as published  by the Free Soft- --
-- ware  Foundation;  either version 3,  or (at your option) any later ver- --
-- sion.  GNAT is distributed in the hope that it will be useful, but WITH- --
-- OUT ANY WARRANTY;  without even the  implied warranty of MERCHANTABILITY --
-- or FITNESS FOR A PARTICULAR PURPOSE.                                     --
--                                                                          --
-- As a special exception under Section 7 of GPL version 3, you are granted --
-- additional permissions described in the GCC Runtime Library Exception,   --
-- version 3.1, as published by the Free Software Foundation.               --
--                                                                          --
-- You should have received a copy of the GNU General Public License and    --
-- a copy of the GCC Runtime Library Exception along with this program;     --
-- see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see    --
-- <http://www.gnu.org/licenses/>.                                          --
--                                                                          --
-- GNAT was originally developed  by the GNAT team at  New York University. --
-- Extensive contributions were provided by Ada Core Technologies Inc.      --
--                                                                          --
------------------------------------------------------------------------------

with Interfaces.C; use Interfaces.C;

package body System.Assertions is

   --  See CUDA API:
   --  extern __host__ __device__ void
   --  __assertfail(const char * __assertion,
   --  const char *__file,
   --  unsigned int __line,
   --  const char *__function,
   --  size_t charsize);

   procedure Assert_Fail
     (Assertion : char_array;
     File : char_array;
     Line : unsigned;
     Func : char_array;
     CharSize : size_t)
      with Import,
      Convention => C,
      Link_Name => "__assertfail",
      No_Return;

   --------------------------
   -- Raise_Assert_Failure --
   --------------------------

   procedure Raise_Assert_Failure (Msg : String) is
      --  TODO: This procedure could be simpler, GNAT LLVM currently
      --  has issues with constant arrays with ASCII.NUL symbols
      --  as well as dynamic frames.

      Empty_Str : char_array (1 .. 1);
      Msg_With_Null : char_array (1 .. 400);
      Last_Msg_Index : constant Integer :=
         (if Msg'Length < Msg_With_Null'Length
         then Msg'Length
         else Msg_With_Null'Length - 1);
   begin
      Empty_Str (1) := char (ASCII.NUL);

      for I in 1 .. Last_Msg_Index loop
         Msg_With_Null (size_t (I)) := char (Msg (Msg'First + I - 1));
      end loop;

      Msg_With_Null (size_t (Msg'Length) + 1) := char (ASCII.NUL);

      Assert_Fail (Msg_With_Null, Empty_Str, 0, Empty_Str, 1);
   end Raise_Assert_Failure;

end System.Assertions;
