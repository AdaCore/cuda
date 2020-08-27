with Interfaces.C; use Interfaces.C;
with Interfaces.C.Extensions;

package CUDA.Vector_Types is

   type Char1 is record
      X : signed_char;

   end record with
      Convention => C;

   type Uchar1 is record
      X : unsigned_char;

   end record with
      Convention => C;

   type Char2 is record
      X : signed_char;
      Y : signed_char;

   end record with
      Convention => C;

   type Uchar2 is record
      X : unsigned_char;
      Y : unsigned_char;

   end record with
      Convention => C;

   type Char3 is record
      X : signed_char;
      Y : signed_char;
      Z : signed_char;

   end record with
      Convention => C;

   type Uchar3 is record
      X : unsigned_char;
      Y : unsigned_char;
      Z : unsigned_char;

   end record with
      Convention => C;

   type Char4 is record
      X : signed_char;
      Y : signed_char;
      Z : signed_char;
      W : signed_char;

   end record with
      Convention => C;

   type Uchar4 is record
      X : unsigned_char;
      Y : unsigned_char;
      Z : unsigned_char;
      W : unsigned_char;

   end record with
      Convention => C;

   type Short1 is record
      X : short;

   end record with
      Convention => C;

   type Ushort1 is record
      X : unsigned_short;

   end record with
      Convention => C;

   type Short2 is record
      X : short;
      Y : short;

   end record with
      Convention => C;

   type Ushort2 is record
      X : unsigned_short;
      Y : unsigned_short;

   end record with
      Convention => C;

   type Short3 is record
      X : short;
      Y : short;
      Z : short;

   end record with
      Convention => C;

   type Ushort3 is record
      X : unsigned_short;
      Y : unsigned_short;
      Z : unsigned_short;

   end record with
      Convention => C;

   type Short4 is record
      X : short;
      Y : short;
      Z : short;
      W : short;

   end record with
      Convention => C;

   type Ushort4 is record
      X : unsigned_short;
      Y : unsigned_short;
      Z : unsigned_short;
      W : unsigned_short;

   end record with
      Convention => C;

   type Int1 is record
      X : int;

   end record with
      Convention => C;

   type Uint1 is record
      X : unsigned;

   end record with
      Convention => C;

   type Int2 is record
      X : int;
      Y : int;

   end record with
      Convention => C;

   type Uint2 is record
      X : unsigned;
      Y : unsigned;

   end record with
      Convention => C;

   type Int3 is record
      X : int;
      Y : int;
      Z : int;

   end record with
      Convention => C;

   type Uint3 is record
      X : unsigned;
      Y : unsigned;
      Z : unsigned;

   end record with
      Convention => C;

   type Int4 is record
      X : int;
      Y : int;
      Z : int;
      W : int;

   end record with
      Convention => C;

   type Uint4 is record
      X : unsigned;
      Y : unsigned;
      Z : unsigned;
      W : unsigned;

   end record with
      Convention => C;

   type Long1 is record
      X : long;

   end record with
      Convention => C;

   type Ulong1 is record
      X : unsigned_long;

   end record with
      Convention => C;

   type Long2 is record
      X : long;
      Y : long;

   end record with
      Convention => C;

   type Ulong2 is record
      X : unsigned_long;
      Y : unsigned_long;

   end record with
      Convention => C;

   type Long3 is record
      X : long;
      Y : long;
      Z : long;

   end record with
      Convention => C;

   type Ulong3 is record
      X : unsigned_long;
      Y : unsigned_long;
      Z : unsigned_long;

   end record with
      Convention => C;

   type Long4 is record
      X : long;
      Y : long;
      Z : long;
      W : long;

   end record with
      Convention => C;

   type Ulong4 is record
      X : unsigned_long;
      Y : unsigned_long;
      Z : unsigned_long;
      W : unsigned_long;

   end record with
      Convention => C;

   type Float1 is record
      X : Float;

   end record with
      Convention => C;

   type Float2 is record
      X : Float;
      Y : Float;

   end record with
      Convention => C;

   type Float3 is record
      X : Float;
      Y : Float;
      Z : Float;

   end record with
      Convention => C;

   type Float4 is record
      X : Float;
      Y : Float;
      Z : Float;
      W : Float;

   end record with
      Convention => C;

   type Longlong1 is record
      X : Long_Long_Integer;

   end record with
      Convention => C;

   type Ulonglong1 is record
      X : Extensions.unsigned_long_long;

   end record with
      Convention => C;

   type Longlong2 is record
      X : Long_Long_Integer;
      Y : Long_Long_Integer;

   end record with
      Convention => C;

   type Ulonglong2 is record
      X : Extensions.unsigned_long_long;
      Y : Extensions.unsigned_long_long;

   end record with
      Convention => C;

   type Longlong3 is record
      X : Long_Long_Integer;
      Y : Long_Long_Integer;
      Z : Long_Long_Integer;

   end record with
      Convention => C;

   type Ulonglong3 is record
      X : Extensions.unsigned_long_long;
      Y : Extensions.unsigned_long_long;
      Z : Extensions.unsigned_long_long;

   end record with
      Convention => C;

   type Longlong4 is record
      X : Long_Long_Integer;
      Y : Long_Long_Integer;
      Z : Long_Long_Integer;
      W : Long_Long_Integer;

   end record with
      Convention => C;

   type Ulonglong4 is record
      X : Extensions.unsigned_long_long;
      Y : Extensions.unsigned_long_long;
      Z : Extensions.unsigned_long_long;
      W : Extensions.unsigned_long_long;

   end record with
      Convention => C;

   type Double1 is record
      X : double;

   end record with
      Convention => C;

   type Double2 is record
      X : double;
      Y : double;

   end record with
      Convention => C;

   type Double3 is record
      X : double;
      Y : double;
      Z : double;

   end record with
      Convention => C;

   type Double4 is record
      X : double;
      Y : double;
      Z : double;
      W : double;

   end record with
      Convention => C;

   type Dim3 is record
      X : unsigned;
      Y : unsigned;
      Z : unsigned;

   end record with
      Convention => C;

end CUDA.Vector_Types;
