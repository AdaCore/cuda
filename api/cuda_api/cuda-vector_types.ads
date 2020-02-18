with Interfaces.C;
with Interfaces.C.Extensions; use Interfaces.C;

package CUDA.Vector_Types is
   type Char1 is record
      X : aliased signed_char;
   end record with
      Convention => C_Pass_By_Copy;

   type Uchar1 is record
      X : aliased unsigned_char;
   end record with
      Convention => C_Pass_By_Copy;

   type Char2 is record
      X : aliased signed_char;
      Y : aliased signed_char;
   end record with
      Convention => C_Pass_By_Copy;

   type Uchar2 is record
      X : aliased unsigned_char;
      Y : aliased unsigned_char;
   end record with
      Convention => C_Pass_By_Copy;

   type Char3 is record
      X : aliased signed_char;
      Y : aliased signed_char;
      Z : aliased signed_char;
   end record with
      Convention => C_Pass_By_Copy;

   type Uchar3 is record
      X : aliased unsigned_char;
      Y : aliased unsigned_char;
      Z : aliased unsigned_char;
   end record with
      Convention => C_Pass_By_Copy;

   type Char4 is record
      X : aliased signed_char;
      Y : aliased signed_char;
      Z : aliased signed_char;
      W : aliased signed_char;
   end record with
      Convention => C_Pass_By_Copy;

   type Uchar4 is record
      X : aliased unsigned_char;
      Y : aliased unsigned_char;
      Z : aliased unsigned_char;
      W : aliased unsigned_char;
   end record with
      Convention => C_Pass_By_Copy;

   type Short1 is record
      X : aliased short;
   end record with
      Convention => C_Pass_By_Copy;

   type Ushort1 is record
      X : aliased unsigned_short;
   end record with
      Convention => C_Pass_By_Copy;

   type Short2 is record
      X : aliased short;
      Y : aliased short;
   end record with
      Convention => C_Pass_By_Copy;

   type Ushort2 is record
      X : aliased unsigned_short;
      Y : aliased unsigned_short;
   end record with
      Convention => C_Pass_By_Copy;

   type Short3 is record
      X : aliased short;
      Y : aliased short;
      Z : aliased short;
   end record with
      Convention => C_Pass_By_Copy;

   type Ushort3 is record
      X : aliased unsigned_short;
      Y : aliased unsigned_short;
      Z : aliased unsigned_short;
   end record with
      Convention => C_Pass_By_Copy;

   type Short4 is record
      X : aliased short;
      Y : aliased short;
      Z : aliased short;
      W : aliased short;
   end record with
      Convention => C_Pass_By_Copy;

   type Ushort4 is record
      X : aliased unsigned_short;
      Y : aliased unsigned_short;
      Z : aliased unsigned_short;
      W : aliased unsigned_short;
   end record with
      Convention => C_Pass_By_Copy;

   type Int1 is record
      X : aliased int;
   end record with
      Convention => C_Pass_By_Copy;

   type Uint1 is record
      X : aliased unsigned;
   end record with
      Convention => C_Pass_By_Copy;

   type Int2 is record
      X : aliased int;
      Y : aliased int;
   end record with
      Convention => C_Pass_By_Copy;

   type Uint2 is record
      X : aliased unsigned;
      Y : aliased unsigned;
   end record with
      Convention => C_Pass_By_Copy;

   type Int3 is record
      X : aliased int;
      Y : aliased int;
      Z : aliased int;
   end record with
      Convention => C_Pass_By_Copy;

   type Uint3 is record
      X : aliased unsigned;
      Y : aliased unsigned;
      Z : aliased unsigned;
   end record with
      Convention => C_Pass_By_Copy;

   type Int4 is record
      X : aliased int;
      Y : aliased int;
      Z : aliased int;
      W : aliased int;
   end record with
      Convention => C_Pass_By_Copy;

   type Uint4 is record
      X : aliased unsigned;
      Y : aliased unsigned;
      Z : aliased unsigned;
      W : aliased unsigned;
   end record with
      Convention => C_Pass_By_Copy;

   type Long1 is record
      X : aliased long;
   end record with
      Convention => C_Pass_By_Copy;

   type Ulong1 is record
      X : aliased unsigned_long;
   end record with
      Convention => C_Pass_By_Copy;

   type Long2 is record
      X : aliased long;
      Y : aliased long;
   end record with
      Convention => C_Pass_By_Copy;

   type Ulong2 is record
      X : aliased unsigned_long;
      Y : aliased unsigned_long;
   end record with
      Convention => C_Pass_By_Copy;

   type Long3 is record
      X : aliased long;
      Y : aliased long;
      Z : aliased long;
   end record with
      Convention => C_Pass_By_Copy;

   type Ulong3 is record
      X : aliased unsigned_long;
      Y : aliased unsigned_long;
      Z : aliased unsigned_long;
   end record with
      Convention => C_Pass_By_Copy;

   type Long4 is record
      X : aliased long;
      Y : aliased long;
      Z : aliased long;
      W : aliased long;
   end record with
      Convention => C_Pass_By_Copy;

   type Ulong4 is record
      X : aliased unsigned_long;
      Y : aliased unsigned_long;
      Z : aliased unsigned_long;
      W : aliased unsigned_long;
   end record with
      Convention => C_Pass_By_Copy;

   type Float1 is record
      X : aliased Float;
   end record with
      Convention => C_Pass_By_Copy;

   type Float2 is record
      X : aliased Float;
      Y : aliased Float;
   end record with
      Convention => C_Pass_By_Copy;

   type Float3 is record
      X : aliased Float;
      Y : aliased Float;
      Z : aliased Float;
   end record with
      Convention => C_Pass_By_Copy;

   type Float4 is record
      X : aliased Float;
      Y : aliased Float;
      Z : aliased Float;
      W : aliased Float;
   end record with
      Convention => C_Pass_By_Copy;

   type Longlong1 is record
      X : aliased Long_Long_Integer;
   end record with
      Convention => C_Pass_By_Copy;

   type Ulonglong1 is record
      X : aliased Extensions.unsigned_long_long;
   end record with
      Convention => C_Pass_By_Copy;

   type Longlong2 is record
      X : aliased Long_Long_Integer;
      Y : aliased Long_Long_Integer;
   end record with
      Convention => C_Pass_By_Copy;

   type Ulonglong2 is record
      X : aliased Extensions.unsigned_long_long;
      Y : aliased Extensions.unsigned_long_long;
   end record with
      Convention => C_Pass_By_Copy;

   type Longlong3 is record
      X : aliased Long_Long_Integer;
      Y : aliased Long_Long_Integer;
      Z : aliased Long_Long_Integer;
   end record with
      Convention => C_Pass_By_Copy;

   type Ulonglong3 is record
      X : aliased Extensions.unsigned_long_long;
      Y : aliased Extensions.unsigned_long_long;
      Z : aliased Extensions.unsigned_long_long;
   end record with
      Convention => C_Pass_By_Copy;

   type Longlong4 is record
      X : aliased Long_Long_Integer;
      Y : aliased Long_Long_Integer;
      Z : aliased Long_Long_Integer;
      W : aliased Long_Long_Integer;
   end record with
      Convention => C_Pass_By_Copy;

   type Ulonglong4 is record
      X : aliased Extensions.unsigned_long_long;
      Y : aliased Extensions.unsigned_long_long;
      Z : aliased Extensions.unsigned_long_long;
      W : aliased Extensions.unsigned_long_long;
   end record with
      Convention => C_Pass_By_Copy;

   type Double1 is record
      X : aliased double;
   end record with
      Convention => C_Pass_By_Copy;

   type Double2 is record
      X : aliased double;
      Y : aliased double;
   end record with
      Convention => C_Pass_By_Copy;

   type Double3 is record
      X : aliased double;
      Y : aliased double;
      Z : aliased double;
   end record with
      Convention => C_Pass_By_Copy;

   type Double4 is record
      X : aliased double;
      Y : aliased double;
      Z : aliased double;
      W : aliased double;
   end record with
      Convention => C_Pass_By_Copy;

   type Dim3 is record
      X : aliased unsigned;
      Y : aliased unsigned;
      Z : aliased unsigned;
   end record with
      Convention => C_Pass_By_Copy;
end CUDA.Vector_Types;
