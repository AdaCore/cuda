------------------------------------------------------------------------------
--                                                                          --
--                         GNAT RUN-TIME COMPONENTS                         --
--                                                                          --
--                         S Y S T E M . M E M O R Y                        --
--                                                                          --
--                                 S p e c                                  --
--                                                                          --
--          Copyright (C) 2001-2021, Free Software Foundation, Inc.         --
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

--  This is the CUDA-specific version of this package. This package differs
--  from other versions of System.Memory as it does attempt to align memory
--  requests nor raise exceptions if memory cannot be allocated. This is due
--  to the nature of CUDA target.

--  This package provides the low level memory allocation/deallocation
--  mechanisms used by GNAT.

package System.Memory is
   pragma Elaborate_Body;

   type size_t is mod 2 ** Standard'Address_Size;
   --  Note: the reason we redefine this here instead of using the
   --  definition in Interfaces.C is that we do not want to drag in
   --  all of Interfaces.C just because System.Memory is used.

   function Alloc (Size : size_t) return System.Address;
   --  This is the low level allocation routine. It is a wrapper around the
   --  standard C malloc call.

   procedure Free (Ptr : System.Address);
   --  This is the low level free routine. It frees a block previously
   --  allocated with a call to Alloc. It is a wrapper around the standard C
   --  free call.

private
   --  The following names are used from the generated compiler code

   pragma Export (C, Alloc,   "__gnat_malloc");
   pragma Export (C, Free, "__gnat_free");

end System.Memory;
