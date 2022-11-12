------------------------------------------------------------------------------
--                    Copyright (C) 2017-2020, AdaCore                      --
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

with Geometry; use Geometry;

with Interfaces;   use Interfaces;
with Interfaces.C; use Interfaces.C;
with System;       use System;
with Data;         use Data;

with CUDA.Storage_Models; use CUDA.Storage_Models;

package Marching_Cubes is
   type Atomic_Integer is new Integer with
     Atomic;

   type Unsigned32_Array is array (Integer range <>) of aliased Unsigned_32;

   type Int_Access is access all Integer;

   procedure Clear_Lattice (XI : Integer);

   type Device_Ball_Array_Access is access Ball_Array with
     Designated_Storage_Model => CUDA.Storage_Models.Model;

   type Device_Triangle_Array_Access is access Triangle_Array with
     Designated_Storage_Model => CUDA.Storage_Models.Model;

   type Device_Vertex_Array_Access is access Vertex_Array with
     Designated_Storage_Model => CUDA.Storage_Models.Model;

   type Device_Point_Real_Array_Access is access Point_Real_Array with
     Designated_Storage_Model => CUDA.Storage_Models.Model;

   type Device_Int_Access is access Integer with
     Designated_Storage_Model => CUDA.Storage_Models.Model;

   procedure Mesh
     (Balls               :    Ball_Array; Triangles : in out Triangle_Array;
      Vertices : in out Vertex_Array; Start : Point_Real; Stop : Point_Real;
      Lattice_Size        : Point_Int; Last_Triangle : not null access Integer;
      Last_Vertex         :        not null access Integer;
      Interpolation_Steps :        Positive := 4; XI, YI, ZI : Integer);

   procedure Clear_Lattice_CUDA with
     Cuda_Global;

   procedure Mesh_CUDA
     (D_Balls             : Device_Ball_Array_Access;
      D_Triangles         : Device_Triangle_Array_Access;
      D_Vertices          : Device_Vertex_Array_Access; Start : Point_Real;
      Stop                : Point_Real; Lattice_Size : Point_Int;
      Last_Triangle       : Device_Int_Access; Last_Vertex : Device_Int_Access;
      Interpolation_Steps : Positive := 4;
      Debug_Value         : Device_Int_Access) with
     Cuda_Global;

   procedure Last_Chance_Handler is null;
   pragma Export (C, Last_Chance_Handler, "__gnat_last_chance_handler");

end Marching_Cubes;
