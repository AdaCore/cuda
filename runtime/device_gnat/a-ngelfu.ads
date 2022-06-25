------------------------------------------------------------------------------
--                                                                          --
--                         GNAT RUN-TIME COMPONENTS                         --
--                                                                          --
--                ADA.NUMERICS.GENERIC_ELEMENTARY_FUNCTIONS                 --
--                                                                          --
--                                 S p e c                                  --
--                                                                          --
--          Copyright (C) 2012-2022, Free Software Foundation, Inc.         --
--                                                                          --
-- This specification is derived from the Ada Reference Manual for use with --
-- GNAT. The copyright notice above, and the license provisions that follow --
-- apply solely to the Post aspects that have been added to the spec.       --
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

generic
   type Float_Type is digits <>;

package Ada.Numerics.Generic_Elementary_Functions with
  SPARK_Mode => On
is
   pragma Pure;

   function Sin (X : Float_Type'Base) return Float_Type'Base
      with Import, External_Name => "__nv_fast_sinf";

   function Cos (X : Float_Type'Base) return Float_Type'Base
      with Import, External_Name => "__nv_fast_cosf";

   function Tan (X : Float_Type'Base) return Float_Type'Base
      with Import, External_Name => "__nv_fast_tanf";

   function Exp (X : Float_Type'Base) return Float_Type'Base
      with Import, External_Name => "__nv_fast_expf";

   function Sqrt (X : Float_Type'Base) return Float_Type'Base
      with Import, External_Name => "__nv_sqrtf";

   function Log (X : Float_Type'Base) return Float_Type'Base
      with Import, External_Name => "__nv_logf";

   function Acos (X : Float_Type'Base) return Float_Type'Base
      with Import, External_Name => "__nv_acosf";

   function Asin (X : Float_Type'Base) return Float_Type'Base
      with Import, External_Name => "__nv_asinf";

   function Atan (X : Float_Type'Base) return Float_Type'Base
      with Import, External_Name => "__nv_atanf";

   function Sinh (X : Float_Type'Base) return Float_Type'Base
      with Import, External_Name => "__nv_sinhf";

   function Cosh (X : Float_Type'Base) return Float_Type'Base
      with Import, External_Name => "__nv_coshf";

   function Tanh (X : Float_Type'Base) return Float_Type'Base
      with Import, External_Name => "__nv_tanhf";

   function Pow (X, Y : Float_Type'Base) return Float_Type'Base
      with Import, External_Name => "__nv_powf";

end Ada.Numerics.Generic_Elementary_Functions;
