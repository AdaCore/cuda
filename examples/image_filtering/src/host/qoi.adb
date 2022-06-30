with Ada.Unchecked_Conversion;

package body QOI
with SPARK_Mode
is

   pragma Compile_Time_Error
     (Storage_Element'Size /= 8, "Invalid element size");

   pragma Warnings (Off, "lower bound test optimized away");

   subtype SE is Storage_Element;

   function As_Index  is new Ada.Unchecked_Conversion (SE, Index_Tag);
   function As_Diff   is new Ada.Unchecked_Conversion (SE, Diff_Tag);
   function As_LUMA_A is new Ada.Unchecked_Conversion (SE, LUMA_Tag_A);
   function As_LUMA_B is new Ada.Unchecked_Conversion (SE, LUMA_Tag_B);
   function As_Run    is new Ada.Unchecked_Conversion (SE, Run_Tag);

   type Color is record
      R, G, B, A : SE;
   end record;

   type Index_Range is range 0 .. 63;
   subtype Run_Range is Unsigned_32 range 0 .. 62;

   function Hash (C : Color) return SE;

   ----------
   -- Hash --
   ----------

   function Hash (C : Color) return SE is
   begin
      return C.R * 3 + C.G * 5 + C.B * 7 + C.A * 11;
   end Hash;

   ------------
   -- Encode --
   ------------

   procedure Encode (Pix         :     Storage_Array;
                     Desc        :     QOI_Desc;
                     Output      : out Storage_Array;
                     Output_Size : out Storage_Count)
   is
      P   : Storage_Count := Output'First;
      Run : Run_Range     := 0;

      function Valid_Parameters return Boolean is
        (Valid_Size (Desc)
         and then Output'First >= 0
         and then Output'Last < Storage_Count'Last
         and then Output'Length >= Encode_Worst_Case (Desc))
      with Ghost;

      procedure Push (D : Unsigned_32)
      with
        Pre  =>
          Valid_Parameters
            and then P in Output'First .. Output'Last - 3
            and then Output (Output'First .. P - 1)'Initialized,
        Post =>
          P = P'Old + 4 and then Output (Output'First .. P - 1)'Initialized;

      generic
         type T is private;
      procedure Gen_Push_8 (D : T)
      with
        Pre  =>
            Valid_Parameters
            and then T'Size = 8
            and then P in Output'Range
            and then Output (Output'First .. P - 1)'Initialized,
        Post =>
          P = P'Old + 1 and then Output (Output'First .. P - 1)'Initialized;

      ----------
      -- Push --
      ----------

      procedure Push (D : Unsigned_32) is
      begin
         Output (P)     := SE (Shift_Right (D and 16#FF_00_00_00#, 24));
         Output (P + 1) := SE (Shift_Right (D and 16#00_FF_00_00#, 16));
         Output (P + 2) := SE (Shift_Right (D and 16#00_00_FF_00#, 8));
         Output (P + 3) := SE (Shift_Right (D and 16#00_00_00_FF#, 0));

         P := P + 4;
      end Push;

      ----------------
      -- Gen_Push_8 --
      ----------------

      procedure Gen_Push_8  (D : T) is
         function To_Byte is new Ada.Unchecked_Conversion (T, SE);
      begin
         Output (P) := To_Byte (D);

         P := P + 1;
      end Gen_Push_8;

      procedure Push        is new Gen_Push_8 (SE);
      procedure Push_Run    is new Gen_Push_8 (Run_Tag);
      procedure Push_Index  is new Gen_Push_8 (Index_Tag);
      procedure Push_Diff   is new Gen_Push_8 (Diff_Tag);
      procedure Push_Luma_A is new Gen_Push_8 (LUMA_Tag_A);
      procedure Push_Luma_B is new Gen_Push_8 (LUMA_Tag_B);

      Number_Of_Pixels : constant Storage_Count := Desc.Width * Desc.Height;

      subtype Pixel_Index is Storage_Count range 0 .. Number_Of_Pixels - 1;

      function Read (Index : Pixel_Index) return Color;

      ----------
      -- Read --
      ----------

      function Read (Index : Pixel_Index) return Color is
         Result : Color;
         Offset : constant Storage_Count := Index * Desc.Channels;
         Buffer_Index : constant Storage_Count := Pix'First + Offset;
      begin
         Result.R := Pix (Buffer_Index);
         Result.G := Pix (Buffer_Index + 1);
         Result.B := Pix (Buffer_Index + 2);

         if Desc.Channels = 4 then
            Result.A := Pix (Buffer_Index + 3);
         else
            Result.A := 255;
         end if;
         return Result;
      end Read;

      Index   : array (Index_Range) of Color := (others => ((0, 0, 0, 0)));
      Px_Prev : Color := (R => 0, G => 0, B => 0, A => 255);
      Px      : Color;
   begin

      if Output'Length < Encode_Worst_Case (Desc) then
         Output_Size := 0;
         return;
      end if;

      Push (QOI_MAGIC);
      Push (Unsigned_32 (Desc.Width));
      Push (Unsigned_32 (Desc.Height));
      Push (SE (Desc.Channels));
      Push (SE (Desc.Colorspace'Enum_Rep));

      pragma Assert (P = Output'First + QOI_HEADER_SIZE);
      pragma Assert (Run = 0);
      pragma Assert (Output (Output'First .. P - 1)'Initialized);
      for Px_Index in Pixel_Index loop

         pragma Loop_Invariant
           (Run in 0 .. Run_Range'Last - 1);
         pragma Loop_Invariant
           (P - Output'First in
              0
                ..
              QOI_HEADER_SIZE
            + (Desc.Channels + 1) *
            (Storage_Count (Px_Index) - Storage_Count (Run)));
         pragma Loop_Invariant (Output (Output'First .. P - 1)'Initialized);
         pragma Loop_Invariant (if Desc.Channels /= 4 then Px_Prev.A = 255);

         Px := Read (Px_Index);

         if Px = Px_Prev then
            Run := Run + 1;

            if Run = Run_Range'Last or else Px_Index = Pixel_Index'Last
            then
               Push_Run ((Op => QOI_OP_RUN, Run => Uint6 (Run - 1)));
               Run := 0;
            end if;

         else

            if Run > 0 then
               Push_Run ((Op => QOI_OP_RUN, Run => Uint6 (Run - 1)));
               Run := 0;
            end if;

            pragma Assert (Run = 0);
            pragma Assert (P - Output'First in
                             0 .. QOI_HEADER_SIZE + (Desc.Channels + 1) *
                             Storage_Count (Px_Index));

            declare
               Index_Pos : constant Index_Range :=
                 Index_Range (Hash (Px) mod Index'Length);
            begin
               if Index (Index_Pos) = Px then
                  Push_Index ((Op    => QOI_OP_INDEX,
                               Index => Uint6 (Index_Pos)));
               else
                  Index (Index_Pos) := Px;

                  if Px.A = Px_Prev.A then
                     declare
                        VR : constant Integer :=
                          Integer (Px.R) - Integer (Px_Prev.R);
                        VG : constant Integer :=
                          Integer (Px.G) - Integer (Px_Prev.G);
                        VB : constant Integer :=
                          Integer (Px.B) - Integer (Px_Prev.B);

                        VG_R : constant Integer := VR - VG;
                        VG_B : constant Integer := VB - VG;
                     begin
                        if         VR in -2 .. 1
                          and then VG in -2 .. 1
                          and then VB in -2 .. 1
                        then
                           Push_Diff ((Op  => QOI_OP_DIFF,
                                       DR  => Uint2 (VR + 2),
                                       DG  => Uint2 (VG + 2),
                                       DB  => Uint2 (VB + 2)));

                        elsif      VG_R in -8 .. 7
                          and then VG   in -32 .. 31
                          and then VG_B in -8 .. 7
                        then
                           Push_Luma_A ((Op  => QOI_OP_LUMA,
                                         DG  => Uint6 (VG + 32)));
                           Push_Luma_B ((DG_R => Uint4 (VG_R + 8),
                                         DG_B => Uint4 (VG_B + 8)));

                        else
                           Push (QOI_OP_RGB);
                           Push (Px.R);
                           Push (Px.G);
                           Push (Px.B);
                        end if;
                     end;
                  else
                     Push (QOI_OP_RGBA);
                     Push (Px.R);
                     Push (Px.G);
                     Push (Px.B);
                     Push (Px.A);
                  end if;
               end if;
            end;
         end if;

         pragma Assert (Output (Output'First .. P - 1)'Initialized);
         Px_Prev := Px;
      end loop;

      pragma Assert (Output (Output'First .. P - 1)'Initialized);
      pragma Assert (P - Output'First in
        0 .. QOI_HEADER_SIZE + (Desc.Channels + 1) * Number_Of_Pixels);

      for Index in QOI_PADDING'Range loop
         pragma Loop_Invariant
           (P - Output'First in
              0 .. Encode_Worst_Case (Desc) - QOI_PADDING'Length + Index - 1);
         pragma Loop_Invariant (Output (Output'First .. P - 1)'Initialized);
         Push (QOI_PADDING (Index));
      end loop;

      pragma Assert (Output (Output'First .. P - 1)'Initialized);
      Output_Size := P - Output'First;
   end Encode;

   --------------
   -- Get_Desc --
   --------------

   procedure Get_Desc (Data :     Storage_Array;
                       Desc : out QOI_Desc)
   is
      P : Storage_Count := Data'First;

      procedure Pop8  (Result : out SE);
      procedure Pop32 (Result : out Unsigned_32);

      procedure Pop8 (Result : out SE) is
      begin
         Result := Data (P);
         P := P + 1;
      end Pop8;

      procedure Pop32 (Result : out Unsigned_32) is
         A : constant Unsigned_32 := Unsigned_32 (Data (P));
         B : constant Unsigned_32 := Unsigned_32 (Data (P + 1));
         C : constant Unsigned_32 := Unsigned_32 (Data (P + 2));
         D : constant Unsigned_32 := Unsigned_32 (Data (P + 3));
      begin
         Result :=
           Shift_Left (A, 24)
           or Shift_Left (B, 16)
           or Shift_Left (C, 8)
           or D;
         P := P + 4;
      end Pop32;

      Magic   : Unsigned_32;
      Temp_32 : Unsigned_32;
      Temp_8  : SE;
   begin
      if Data'Length < QOI_HEADER_SIZE then
         Desc := (0, 0, 0, SRGB);
         return;
      end if;

      Pop32 (Magic);

      if Magic /= QOI_MAGIC then
         Desc := (0, 0, 0, SRGB);
         return;
      end if;

      Pop32 (Temp_32);
      Desc.Width := Storage_Count (Temp_32);
      Pop32 (Temp_32);
      Desc.Height := Storage_Count (Temp_32);
      Pop8 (Temp_8);
      Desc.Channels := Storage_Count (Temp_8);
      Pop8 (Temp_8);
      pragma Assert (P = Data'First + QOI_HEADER_SIZE);
      if Temp_8 not in SE (Colorspace_Kind'Enum_Rep (SRGB))
                     | SE (Colorspace_Kind'Enum_Rep (SRGB_Linear_Alpha))
      then
         Desc := (0, 0, 0, SRGB);
         return;
      end if;
      Desc.Colorspace := Colorspace_Kind'Enum_Val (Temp_8);
   end Get_Desc;

   ------------
   -- Decode --
   ------------

   procedure Decode (Data        :     Storage_Array;
                     Desc        : out QOI_Desc;
                     Output      : out Storage_Array;
                     Output_Size : out Storage_Count)
   is
      P         : Storage_Count;
      Out_Index : Storage_Count := Output'First;

      procedure Pop (Result : out SE) with
        Pre  =>
          P in Data'Range
            and then Data'Last < Storage_Count'Last,
        Post => P = P'Old + 1;
      procedure Push (D      :     SE) with
        Pre  =>
          Out_Index in Output'Range
            and then Output'Last < Storage_Count'Last
            and then Output (Output'First .. Out_Index - 1)'Initialized,
        Post =>
          Out_Index = Out_Index'Old + 1
            and then Output (Output'First .. Out_Index - 1)'Initialized;

      procedure Pop (Result : out SE) is
      begin
         Result := Data (P);
         P := P + 1;
      end Pop;

      procedure Push (D : SE) is
      begin
         Output (Out_Index) := D;
         Out_Index := Out_Index + 1;
      end Push;

   begin

      Get_Desc (Data, Desc);

      if Desc.Width = 0
        or else
         Desc.Height = 0
        or else
         Desc.Channels not in 3 .. 4
        or else
         Desc.Height > Storage_Count'Last / Desc.Width
        or else
         Desc.Channels > Storage_Count'Last / (Desc.Width * Desc.Height)
        or else
         Output'Length < Desc.Width * Desc.Height * Desc.Channels
      then
         Output_Size := 0;
         return;
      end if;

      P := Data'First + QOI_HEADER_SIZE;

      declare
         Number_Of_Pixels : constant Storage_Count := Desc.Width * Desc.Height;

         subtype Pixel_Index is Storage_Count range 0 .. Number_Of_Pixels - 1;

         Last_Chunk : constant Storage_Count := Data'Last - QOI_PADDING'Length;

         Index   : array (Index_Range) of Color := (others => ((0, 0, 0, 0)));
         Px      : Color := (R => 0, G => 0, B => 0, A => 255);
         B1, B2  : SE;
         VG      : SE;
         Run : Run_Range := 0;
      begin
         for Px_Index in Pixel_Index loop

            pragma Loop_Invariant (P >= Data'First);
            pragma Loop_Invariant
              (Out_Index
               = Output'First + Desc.Channels * (Px_Index - Pixel_Index'First));
            pragma Loop_Invariant
              (Output (Output'First .. Out_Index - 1)'Initialized);

            if Run > 0 then
               Run := Run - 1;
            elsif P <= Last_Chunk then
               Pop (B1);

               if B1 = QOI_OP_RGB then
                  Pop (Px.R);
                  Pop (Px.G);
                  Pop (Px.B);

               elsif B1 = QOI_OP_RGBA then
                  Pop (Px.R);
                  Pop (Px.G);
                  Pop (Px.B);
                  Pop (Px.A);

               else
                  case As_Run (B1).Op is

                  when QOI_OP_INDEX =>
                     Px := Index (Index_Range (As_Index (B1).Index));

                  when QOI_OP_DIFF =>
                     Px.R := Px.R + SE (As_Diff (B1).DR) - 2;
                     Px.G := Px.G + SE (As_Diff (B1).DG) - 2;
                     Px.B := Px.B + SE (As_Diff (B1).DB) - 2;

                  when QOI_OP_LUMA =>
                     Pop (B2);
                     VG := SE (As_LUMA_A (B1).DG) - 32;

                     Px.R := Px.R + VG + SE (As_LUMA_B (B2).DG_R) - 8;
                     Px.G := Px.G + VG;
                     Px.B := Px.B + VG + SE (As_LUMA_B (B2).DG_B) - 8;

                  when QOI_OP_RUN =>
                     Run := Run_Range (As_Run (B1).Run mod 63);

                  end case;
               end if;

               Index (Index_Range (Hash (Px) mod Index'Length)) := Px;
            end if;

            Push (Px.R);
            Push (Px.G);
            Push (Px.B);
            if Desc.Channels = 4 then
               Push (Px.A);
            end if;
         end loop;
      end;

      Output_Size := Out_Index - Output'First;
   end Decode;

end QOI;
