----------------------------------------------------------------------------
--            Generic Command Line Parser (gclp)
--
--               Copyright (C) 2012, Riccardo Bernardini
--
--      This file is part of gclp.
--
--      gclp is free software: you can redistribute it and/or modify
--      it under the terms of the GNU General Public License as published by
--      the Free Software Foundation, either version 2 of the License, or
--      (at your option) any later version.
--
--      gclp is distributed in the hope that it will be useful,
--      but WITHOUT ANY WARRANTY; without even the implied warranty of
--      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
--      GNU General Public License for more details.
--
--      You should have received a copy of the GNU General Public License
--      along with gclp.  If not, see <http://www.gnu.org/licenses/>.
----------------------------------------------------------------------------
--
with Ada.Command_Line;
with Ada.Strings.Fixed;
with Ada.Text_IO;
with Ada.Containers.Ordered_Maps;
with Ada.Strings.Maps.Constants;
with Ada.Containers.Doubly_Linked_Lists;

use Ada;
use Ada.Strings;
use Ada.Strings.Fixed;

package body Generic_Line_Parser is

   function To_S (X : Unbounded_String) return String
                  renames To_String;

   function To_U (X : String) return Unbounded_String
                  renames To_Unbounded_String;

   -- In order to handle parameter aliases (see comments in the specs)
   -- we keep a table that maps parameter names to parameter "index"
   package Name_To_Index_Maps is
     new Ada.Containers.Ordered_Maps (Key_Type     => Unbounded_String,
                                      Element_Type => Natural);

   --------------------
   -- Case_Normalize --
   --------------------

   -- If the user required case insensitive matching, force the
   -- name to lower case
   procedure Case_Normalize (Name : in out Unbounded_String) is
   begin
      if not Case_Sensitive then
         Translate (Name, Maps.Constants.Lower_Case_Map);
      end if;
   end Case_Normalize;

   ---------------------
   -- Fill_Name_Table --
   ---------------------

   -- Fill the Parameter Name -> parameter index table with the
   -- parameter names
   procedure Fill_Name_Table (Parameters : in     Parameter_Descriptor_Array;
                              Name_Table : in out Name_To_Index_Maps.Map)
   is
      package Name_Lists is
        new Ada.Containers.Doubly_Linked_Lists (Unbounded_String);

      use Name_Lists;

      ----------------
      -- Parse_Name --
      ----------------

      function Parse_Name (Name : Unbounded_String) return Name_Lists.List
      is
         ------------------
         -- Trimmed_Name --
         ------------------

         function Trimmed_Name (Name  : String)
                                return Unbounded_String
         is
            Trimmed : Unbounded_String;
         begin
            Trimmed := To_U (Fixed.Trim (Name, Both));
            if Unbounded.Length (Trimmed) = 0 then
               raise Constraint_Error
                  with "Empty alternative in label '" & Name & "'";
            else
               return Trimmed;
            end if;
         end Trimmed_Name;

         Result    : Name_Lists.List;
         Buffer    : String := To_S (Name);
         First     : Natural;
         Comma_Pos : Natural;
      begin
         if Fixed.Index (Buffer, "=") /= 0 then
            raise Constraint_Error with "Option label '" & Buffer & "' has '='";
         end if;

         if Buffer(Buffer'Last) = ',' then
            raise Constraint_Error
               with "Option label '" & Buffer & "' ends with ','";
         end if;

         First := Buffer'First;
         loop
            pragma Assert (First <= Buffer'Last);

            Comma_Pos := Fixed.Index (Buffer (First .. Buffer'Last), ",");
            exit when Comma_Pos = 0;

            if First = Comma_Pos then
               -- First should always point to the beginning of a
               -- label, therefore it cannot be Buffer(First) = ','
               raise Constraint_Error
                  with "Wrong syntax in Option label '" & Buffer & "'";
            end if;

            pragma Assert (Comma_Pos > First);

            Result.Append (Trimmed_Name (Buffer(First .. Comma_Pos - 1)));

            First := Comma_Pos + 1;

            -- It cannot be First > Buffer'Last since Buffer(Comma_Pos) = '='
            -- and Buffer(Buffer'Last) /= ','
            pragma Assert (First <= Buffer'Last);
         end loop;

         pragma Assert (First <= Buffer'Last);

         Result.Append (Trimmed_Name (Buffer (First .. Buffer'Last)));

         return Result;
      end Parse_Name;

      Option_Names : Name_Lists.List;
      Position : Name_Lists.Cursor;

      Name : Unbounded_String;
   begin
      for Idx in Parameters'Range loop
         Option_Names := Parse_Name (Parameters (Idx).Name);

         Position := Option_Names.First;

         while Position /= No_Element loop
            Name := Name_Lists.Element (Position);
            Name_Lists.Next (Position);

            Case_Normalize(Name);

            if Name_Table.Contains (Name) then
               raise Constraint_Error
                  with "Ambiguous label '" & To_S (Name) & "'";
            end if;

            Name_Table.Insert (Name, Idx);
         end loop;
      end loop;
   end Fill_Name_Table;


   ----------------
   -- To_Natural --
   ----------------

   function To_Natural (X : Unbounded_String)
                        return Natural is
   begin
      if X = Null_Unbounded_String then
         raise Bad_Command with "Invalid integer '" & To_S(X) & "'";
      end if;

      return Natural'Value (To_S (X));
   end To_Natural;

   --------------
   -- To_Float --
   --------------

   function To_Float (X : Unbounded_String)
                      return Float is
   begin
      if X = Null_Unbounded_String then
         raise Bad_Command with "Invalid Float '" & To_S(X) & "'";
      end if;

      return Float'Value (To_S (X));
   end To_Float;


   ------------------------
   -- Parse_Command_Line --
   ------------------------

   procedure Parse_Command_Line
      (Parameters  : in     Parameter_Descriptor_Array;
       Result      :    out Config_Data;
       Help_Line   : in     String := "";
       Help_Output : in     Ada.Text_IO.File_Type := Ada.Text_IO.Standard_Error) is

      package String_Lists is
         new Ada.Containers.Doubly_Linked_Lists (Unbounded_String);

      ---------------------
      -- Split_Parameter --
      ---------------------

      procedure Split_Parameter (Param : in     String;
                                 Name  :    out Unbounded_String;
                                 Value :    out Unbounded_String)
      is
         Idx : Natural;
      begin
         Idx := Index (Source  => Param,
                       Pattern => "=");

         if (Idx = 0) then
            Name  := To_U (Param);
            Value := Null_Unbounded_String;
         else
            Name  := To_U (Param (Param'First .. Idx - 1));
            Value := To_U (Param (Idx + 1 .. Param'Last));
         end if;

         Case_Normalize (Name);
      end Split_Parameter;

      function Missing_Message (Missing  : String_Lists.List)
                                return String
      is
         function Join (Item : String_Lists.List) return String is
            Result : Unbounded_String;

            procedure Append (Pos : String_Lists.Cursor) is
            begin
               if Result /= Null_Unbounded_String then
                  Result := Result & ", ";
               end if;

               Result := Result & "'" & String_Lists.Element (Pos) & "'";
            end Append;
         begin
            Item.Iterate (Append'Access);

            return To_String(Result);
         end Join;

         use type Ada.Containers.Count_Type;
      begin
         if Missing.Length = 1 then
            return "Missing mandatory option " & Join (Missing);
         else
            return "Missing mandatory options: " & Join (Missing);
         end if;
      end Missing_Message;


      Found : array (Parameters'Range) of Boolean := (others => False);

      Name  : Unbounded_String;
      Value : Unbounded_String;

      use Name_To_Index_Maps;

      Name_Table : Name_To_Index_Maps.Map;
      Position   : Name_To_Index_Maps.Cursor;
      Param_Idx  : Natural;
   begin
      Fill_Name_Table (Parameters, Name_Table);

      for Pos  in 1 .. Command_Line.Argument_Count loop
         Split_Parameter (Command_Line.Argument (Pos), Name, Value);

         Position := Name_Table.Find (Name);

         if Position = No_Element then
            raise Bad_Command with "Option '" & To_S (Name) & "' unknown";
         end if;

         Param_Idx := Name_To_Index_Maps.Element (Position);

         if Found (Param_Idx) and then Parameters (Param_Idx).Only_Once then
            raise Bad_Command with "Option '" & To_S (Name) & "' given twice";
         end if;

         Found (Param_Idx) := True;
         Parameters (Param_Idx).Callback (Name   => Name,
                                          Value  => Value,
                                          Result => Result);
      end loop;

      declare
         use type Name_To_Index_Maps.Cursor;

         Missing   : String_Lists.List;
         Param_Idx : Natural;
         Position  : Name_To_Index_Maps.Cursor;

         Reported  : array (Parameters'Range) of Boolean := (others => False);
         --  Reported(Idx) is true if the parameter with index Idx has
         --  already processed as missing.  We need this since we loop over
         --  the option names and more than option can refer to the same
         --  parameter.
      begin
         Position := Name_Table.First;

         while Position /= Name_To_Index_Maps.No_Element loop
            Param_Idx := Name_To_Index_Maps.Element (Position);

--              Ada.Text_IO.Put ("checking" & To_S(Parameters (Param_Idx).Name) & "->");
--              Ada.Text_IO.Put (Boolean'Image (Found (Param_Idx)));
--              Ada.Text_IO.Put_Line (" "& Boolean'Image (Reported (Param_Idx)));

            if not Found (Param_Idx) and not Reported (Param_Idx) then
               Reported (Param_Idx) := True;

               case Parameters (Param_Idx).If_Missing is
                  when Die =>
                     Missing.Append (Name_To_Index_Maps.Key (Position));

                  when Use_Default =>
                     Parameters (Param_Idx).Callback
                       (Name   => Parameters (Param_Idx).Name,
                        Value  => Parameters (Param_Idx).Default,
                        Result => Result);

                  when Ignore =>
                     null;
               end case;
            end if;

            Name_To_Index_Maps.Next (Position);
         end loop;


         if not Missing.Is_Empty then
            raise Bad_Command with Missing_Message (Missing);
         end if;
      end;
   exception
      when Bad_Command =>
         if Help_Line /= "" then
            Ada.Text_IO.Put_Line (File => Help_Output,
                                  Item => Help_Line);
         end if;

         raise;
   end Parse_Command_Line;

end Generic_Line_Parser;
