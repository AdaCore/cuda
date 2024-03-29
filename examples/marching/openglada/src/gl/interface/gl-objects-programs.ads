--------------------------------------------------------------------------------
-- Copyright (c) 2012, Felix Krause <contact@flyx.org>
--
-- Permission to use, copy, modify, and/or distribute this software for any
-- purpose with or without fee is hereby granted, provided that the above
-- copyright notice and this permission notice appear in all copies.
--
-- THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
-- WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
-- MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
-- ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
-- WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
-- ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
-- OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
--------------------------------------------------------------------------------

with GL.Attributes;
with GL.Buffers;
with GL.Objects.Shaders.Lists;
with GL.Uniforms;

private with GL.Low_Level;

package GL.Objects.Programs is
   pragma Preelaborate;

   subtype Subroutine_Index_Type is UInt;
   subtype Uniform_Location_Type is Int range -1 .. Int'Last;

   Unknown_Variable_Name : exception;

   Invalid_Index : constant Subroutine_Index_Type;

   type Program is new GL_Object with private;

   type Tessellation_Primitive_Mode is (Triangles, Quads, Isolines);

   type Tessellation_Spacing is (Equal, Fractional_Odd, Fractional_Even);

   type Buffer_Mode is (Interleaved_Attribs, Separate_Attribs);

   procedure Attach (Subject : Program; Shader : Shaders.Shader);

   procedure Link (Subject : Program);

   function Link_Status (Subject : Program) return Boolean;

   function Delete_Status (Subject : Program) return Boolean;

   function Validate_Status (Subject : Program) return Boolean;

   function Info_Log (Subject : Program) return String;

   function Active_Attributes (Subject : Program) return Size;

   function Active_Attribute_Max_Length (Subject : Program) return Size;

   function Active_Uniform_Blocks (Subject : Program) return Size;

   function Active_Uniform_Block_Max_Name_Length (Subject : Program) return Size;

   function Active_Uniforms (Subject : Program) return Size;

   function Active_Uniform_Max_Length (Subject : Program) return Size;

   function Tess_Control_Output_Vertices (Subject : Program) return Size;

   function Tess_Gen_Mode (Subject : Program) return Tessellation_Primitive_Mode;

   function Tess_Gen_Point_Mode (Subject : Program) return Boolean;

   function Tess_Gen_Spacing (Subject : Program) return Tessellation_Spacing;

   function Tess_Gen_Vertex_Order (Subject : Program) return Orientation;

   function Transform_Feedback_Buffer_Mode (Subject : Program)
                                            return Buffer_Mode;

   function Transform_Feedback_Varyings (Subject : Program) return Size;

   function Transform_Feedback_Varying_Max_Length (Subject : Program)
                                                   return Size;

   function Active_Subroutines (Object : Program; Shader : Shaders.Shader_Type)
                                return Size;
   function Active_Subroutine_Uniforms (Object : Program;
                                        Shader : Shaders.Shader_Type)
                                        return Size;
   function Active_Subroutine_Uniform_Locations (Object : Program;
                                                 Shader : Shaders.Shader_Type)
                                                 return Size;
   function Active_Subroutine_Uniform_Max_Length (Object : Program;
                                                  Shader : Shaders.Shader_Type)
                                                  return Size;
   function Active_Subroutine_Max_Length (Object : Program;
                                          Shader : Shaders.Shader_Type)
                                          return Size;

   function Subroutine_Index (Object : Program; Shader : Shaders.Shader_Type;
                              Name   : String)
                              return Subroutine_Index_Type;
   function Subroutine_Uniform_Locations (Object : Program;
                                          Shader : Shaders.Shader_Type;
                                          Name   : String)
                                          return Uniform_Location_Type;

   procedure Validate (Subject : Program);

   procedure Use_Program (Subject : Program);

   function Uniform_Location (Subject : Program; Name : String)
     return Uniforms.Uniform;

   procedure Bind_Attrib_Location (Subject : Program;
                                   Index : Attributes.Attribute;
                                   Name : String);
   function Attrib_Location (Subject : Program; Name : String)
     return Attributes.Attribute;

   function Attached_Shaders (Object : Program) return Shaders.Lists.List;

   procedure Bind_Frag_Data_Location
     (Object : Program; Color_Number : Buffers.Draw_Buffer_Index;
      Name : String);

   -- raises Unknown_Variable_Name if Name is not an out variable
   function Frag_Data_Location (Object : Program; Name : String)
     return Buffers.Draw_Buffer_Index;

   overriding
   procedure Initialize_Id (Object : in out Program);
private
   Invalid_Index : constant Subroutine_Index_Type := 16#FFFFFFFF#;

   type Program is new GL_Object with null record;

   for Tessellation_Primitive_Mode use (Triangles => 16#0004#,
                                        Quads     => 16#0007#,
                                        Isolines  => 16#8E7A#);
   for Tessellation_Primitive_Mode'Size use Low_Level.Enum'Size;

   for Tessellation_Spacing use (Equal           => 16#0202#,
                                 Fractional_Odd  => 16#8E7B#,
                                 Fractional_Even => 16#8E7C#);
   for Tessellation_Spacing'Size use Low_Level.Enum'Size;

   for Buffer_Mode use (Interleaved_Attribs => 16#8C8C#,
                        Separate_Attribs    => 16#8C8D#);
   for Buffer_Mode'Size use Low_Level.Enum'Size;
end GL.Objects.Programs;
