------------------------------------------------------------------------------
--                                                                          --
--                         GNAT RUN-TIME COMPONENTS                         --
--                                                                          --
--               S Y S T E M . A T O M I C _ P R I M I T I V E S            --
--                                                                          --
--                                 S p e c                                  --
--                                                                          --
--              Copyright (C) 2012-2020, Free Software Foundation, Inc.     --
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

--  This package contains both atomic primitives defined from gcc built-in
--  functions and operations used by the compiler to generate the lock-free
--  implementation of protected objects.

with Interfaces.C;

package System.Atomic_Primitives is
   pragma Pure;

   Relaxed : constant := 0;
   Consume : constant := 1;
   Acquire : constant := 2;
   Release : constant := 3;
   Acq_Rel : constant := 4;
   Seq_Cst : constant := 5;
   Last    : constant := 6;

   subtype Mem_Model is Integer range Relaxed .. Last;

   function Atomic_Always_Lock_Free
     (Size : Interfaces.C.size_t;
      Ptr  : System.Address := System.Null_Address) return Boolean;
   pragma Import
     (Intrinsic, Atomic_Always_Lock_Free, "__atomic_always_lock_free");

end System.Atomic_Primitives;
