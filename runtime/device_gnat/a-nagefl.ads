------------------------------------------------------------------------------
--                                                                          --
--                         GNAT RUN-TIME COMPONENTS                         --
--                                                                          --
--       A D A . N U M E R I C S . A U X _ G E N E R I C _ F L O A T        --
--                                                                          --
--                                 S p e c                                  --
--                            (Generic Wrapper)                             --
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

--  This package provides the basic computational interface for the generic
--  elementary functions. The C library version interfaces with the routines
--  in the C mathematical library.

--  This version here is for use with normal Unix math functions.

generic
   type T is digits <>;
package Ada.Numerics.Aux_Generic_Float is
   pragma Pure;

   function Sin (X : T) return T
      with Import, External_Name => "__nv_fast_sinf";

   function Cos (X : T) return T
      with Import, External_Name => "__nv_fast_cosf";

   function Tan (X : T) return T
      with Import, External_Name => "__nv_fast_tanf";

   function Exp (X : T) return T
      with Import, External_Name => "__nv_fast_expf";

   function Sqrt (X : T) return T
      with Import, External_Name => "__nv_rsqrtf";

   function Log (X : T) return T with Import, External_Name => "__nv_logf";

   function Acos (X : T) return T with Import, External_Name => "__nv_acosf";

   function Asin (X : T) return T with Import, External_Name => "__nv_asinf";

   function Atan (X : T) return T with Import, External_Name => "__nv_atanf";

   function Sinh (X : T) return T with Import, External_Name => "__nv_sinhf";

   function Cosh (X : T) return T with Import, External_Name => "__nv_coshf";

   function Tanh (X : T) return T with Import, External_Name => "__nv_tanhf";

   function Pow (X, Y : T) return T with Import, External_Name => "__nv_powf";

end Ada.Numerics.Aux_Generic_Float;
