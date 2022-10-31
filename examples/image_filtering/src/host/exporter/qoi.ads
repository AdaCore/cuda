with Interfaces; use Interfaces;

with System.Storage_Elements; use System.Storage_Elements;

package Qoi with
  SPARK_Mode
is

   type Colorspace_Kind is (Srgb, Srgb_Linear_Alpha);

   type Qoi_Desc is record
      Width, Height : Storage_Count;
      Channels      : Storage_Count;
      Colorspace    : Colorspace_Kind;
   end record;

   Qoi_Header_Size : constant                        := 14;
   Qoi_Padding : constant Storage_Array (1 .. 8) := (0, 0, 0, 0, 0, 0, 0, 1);

   function Valid_Size (Desc : Qoi_Desc) return Boolean;
   --  Return True if the the Desc describes an image that can be encoded.
   --  Image can be invalid because of the wrong nuber of channels (only 3 or
   --  4 supported by QOI) or dimensions that would lead to integer overflows.

   function Encode_Worst_Case (Desc : Qoi_Desc) return Storage_Count is
     (Desc.Width * Desc.Height * (Desc.Channels + 1) + Qoi_Header_Size +
      Qoi_Padding'Length) with
     Pre  => Valid_Size (Desc),
     Post => Encode_Worst_Case'Result >= Qoi_Header_Size + Qoi_Padding'Length;
   --  Return the worst case output size of QOI encoding. Use this function to
   --  allocate an output buffer for the Encode procedure.

   procedure Encode
     (Pix         : Storage_Array; Desc : Qoi_Desc; Output : out Storage_Array;
      Output_Size : out Storage_Count) with
     Relaxed_Initialization => Output,
     Pre                    =>
      Valid_Size (Desc) and then Output'First >= 0
      and then Output'Last < Storage_Count'Last
      and then Output'Length >= Encode_Worst_Case (Desc)
      and then Pix'First >= 1
      and then Pix'Length = (Desc.Width * Desc.Height * Desc.Channels),
     Post =>
      Output (Output'First .. Output'First - 1 + Output_Size)'Initialized;
   --  Encode an RGB or RGBA image to QOI.
   --
   --  If the provided Output buffer is not large enough (see
   --  Encode_Worst_Case), the procedure will return with Output_Size = 0.
   --  Otherwise, Output_Size will contain the size of encoded data in the
   --  Output buffer.

   procedure Get_Desc (Data : Storage_Array; Desc : out Qoi_Desc) with
     Pre => Data'First >= 0 and then Data'Last < Storage_Count'Last;
   --  Read a QOI descriptor from an encoded QOI buffer.  Use this function to
   --  allocate an output buffer for the Decode procedure.
   --
   --  The procedure will return with Desc set to an empty descriptor if the
   --  provided data is not vaild QOI.

   procedure Decode
     (Data : Storage_Array; Desc : out Qoi_Desc; Output : out Storage_Array;
      Output_Size : out Storage_Count) with
     Relaxed_Initialization => Output,
     Pre                    =>
      Output'First >= 0 and then Output'Last < Storage_Count'Last
      and then Data'First >= 0 and then Data'Last < Storage_Count'Last
      and then Data'Length >= Qoi_Header_Size + Qoi_Padding'Length,
     Post =>
      (if Output_Size /= 0 then
         Desc.Height <= Storage_Count'Last / Desc.Width
         and then Desc.Channels <=
           Storage_Count'Last / (Desc.Width * Desc.Height)
         and then Output_Size = Desc.Width * Desc.Height * Desc.Channels
         and then Output (Output'First .. Output'First - 1 + Output_Size)'
           Initialized);
   --  Decode an QOI buffer into RGB or RGBA image.
   --
   --  If the provided data is not valid QOI or the Output buffer is not large
   --  enough (see Get_Desc), the procedure will return with Output_Size =
   --  0. Otherwise, Output_Size will contain the size of decoded data in
   --  the Output buffer.

private

   for Colorspace_Kind'Size use 8;
   for Colorspace_Kind use (Srgb => 16#00#, Srgb_Linear_Alpha => 16#01#);

   ----------------
   -- Valid_Size --
   ----------------

   function Valid_Size (Desc : Qoi_Desc) return Boolean is
     (Desc.Width in 1 .. Storage_Count (Integer_32'Last)
      and then Desc.Height in 1 .. Storage_Count (Integer_32'Last)
      and then Desc.Channels in 3 .. 4
      and then Desc.Width <= Storage_Count'Last / Desc.Height
      and then Desc.Channels + 1 <=
        Storage_Count'Last / (Desc.Width * Desc.Height)
      and then Qoi_Header_Size + Qoi_Padding'Length <=
        Storage_Count'Last - (Desc.Width * Desc.Height * (Desc.Channels + 1)));

   -- Representations of QOI tags --

   type Tag_Op is mod 2**2 with
     Size => 2;

   Qoi_Op_Index : constant Tag_Op          := 2#00#;
   Qoi_Op_Diff  : constant Tag_Op          := 2#01#;
   Qoi_Op_Luma  : constant Tag_Op          := 2#10#;
   Qoi_Op_Run   : constant Tag_Op          := 2#11#;
   Qoi_Op_Rgb   : constant Storage_Element := 2#1111_1110#;
   Qoi_Op_Rgba  : constant Storage_Element := 2#1111_1111#;

   Qoi_Magic : constant Unsigned_32 :=
     (113 * 2**24) + (111 * 2**16) + (105 * 2**8) + 102;

   type Uint2 is mod 2**2 with
     Size => 2;
   type Uint4 is mod 2**4 with
     Size => 4;
   type Uint6 is mod 2**6 with
     Size => 6;

   type Index_Tag is record
      Index : Uint6;
      Op    : Tag_Op;
   end record with
     Size => 8, Object_Size => 8;
   for Index_Tag use record
      Index at 0 range 0 .. 5;
      Op    at 0 range 6 .. 7;
   end record;

   type Diff_Tag is record
      Db : Uint2;
      Dg : Uint2;
      Dr : Uint2;
      Op : Tag_Op;
   end record with
     Size => 8, Object_Size => 8;
   for Diff_Tag use record
      Db at 0 range 0 .. 1;
      Dg at 0 range 2 .. 3;
      Dr at 0 range 4 .. 5;
      Op at 0 range 6 .. 7;
   end record;

   type Luma_Tag_A is record
      Dg : Uint6;
      Op : Tag_Op;
   end record with
     Size => 8, Object_Size => 8;
   for Luma_Tag_A use record
      Dg at 0 range 0 .. 5;
      Op at 0 range 6 .. 7;
   end record;

   type Luma_Tag_B is record
      Dg_B : Uint4;
      Dg_R : Uint4;
   end record with
     Size => 8, Object_Size => 8;
   for Luma_Tag_B use record
      Dg_B at 0 range 0 .. 3;
      Dg_R at 0 range 4 .. 7;
   end record;

   type Run_Tag is record
      Run : Uint6;
      Op  : Tag_Op;
   end record with
     Size => 8, Object_Size => 8;
   for Run_Tag use record
      Run at 0 range 0 .. 5;
      Op  at 0 range 6 .. 7;
   end record;

end Qoi;
