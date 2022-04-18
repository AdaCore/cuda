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
-- <summary>
-- <p>This is a generic package implementing a simple-to-use command line
-- parser.  Yes, I know, everyone makes his/her own command line parser...
-- so, I wrote mine.  As they say, every open source project starts
-- with a programmer that schratches its own itch. So I did... If
-- you find this useful, you are welcome to use it.</p>
--
-- <p>The ideas behind this package are the following
--
-- <itemize>
-- <item> <p>Parameters are nominal, non positional.  The syntax is of
--   "named parameter" type, that is, each command line parameter is
--   expected to have thefollowing format</p>
--
--      <center>label ['=' value]</center>
--
--    <p>where "label" is any string without '='.</p></item>
--
-- <item><p> Parsed value are written in a "configuration variable" whose type
--   is a formal parameter of this package.  The values are written
--   in the configuration variable by using some callbacks provided
--   by caller.</p></item>
-- </itemize>
-- </p>
-- The names of the parameters are given to the parser in "parameter
-- description array" that is an array of records that specify
--
--     + The parameter name
--
--     + A default value (if needed)
--
--     + What to do if the parameter is missing
--
--     + If it can be specified more than once
--
--     + The callback function to be called when the parameter is found
--
-- In order to parse the command line it suffices to call Parse_Command_Line
-- giving as argument the array of parameter descriptors and the configuration
-- variable to be written.  For every parameter found, the corresponding
-- callback function is called.  If at the end of the parsing there are some
-- optional parameters that were missing from the command line, the
-- corresponding callbacks are called with the default parameter.
-- </summary>

with Ada.Strings.Unbounded;
with Ada.Text_IO;

generic
   type Config_Data (<>) is limited private;
   -- The parameters read from the command line will be written in
   -- a variable of this type

   Case_Sensitive : Boolean := True;
   -- Set this to False if you want case insensitive option matching.
   -- For example, if you set this to False, "input", "Input", "INPUT"
   -- and "InPuT" will be equivalent names for the option "input"
package Generic_Line_Parser is
   use Ada.Strings.Unbounded;

   type Parameter_Callback is
     access procedure (Name   : in     Unbounded_String;
                       Value  : in     Unbounded_String;
                       Result : in out Config_Data);

   type Missing_Action is (Die, Use_Default, Ignore);
   --  Possibile alternatives about what to do if a parameter is missing
   --
   --     [Die]         The parameter is mandatory.  If it is missing, an
   --                   exception with explicative message is raised
   --
   --     [Use_Default] The parameter is optional.  If it is missing, the
   --                   corresponding callback function is called with the
   --                   specified default value (see record
   --                   Parameter_Descriptor in the following)
   --
   --     [Ignore]      The parameter is optional.  If it is missing, nothing
   --                   is done

   type Parameter_Descriptor is
      record
         Name       : Unbounded_String;    -- Parameter name
         Default    : Unbounded_String;    -- Default value used if not on C.L.
         If_Missing : Missing_Action;      -- What to do if parameter missing
         Only_Once  : Boolean;             -- Parameter MUST NOT be given more than once
         Callback   : Parameter_Callback;  -- Called when parameter found
      end record;
   -- <description>Record holding the description of a parameter.  The fields
   --  should be self-explenatory (I hope).  The only field that needs some
   -- explanation is Name since it allows to specify more than one
   -- name for each parameter.  The syntax is very simple: just separate
   -- the names with commas.  For example, if Name is "f,filename,input"
   -- one can use on the command line, with the same effect  f=/tmp/a.txt or
   -- filename=/tmp/a.txt or input=/tmp/a.txt.  Spaces at both ends of
   -- the label name are trimmed, so that, for example, "f,filename,input"
   -- is equivalent to "f ,    filename  ,input "
   -- </description>


   type Parameter_Descriptor_Array is
     array (Natural range <>) of Parameter_Descriptor;


   procedure Parse_Command_Line
     (Parameters  : in     Parameter_Descriptor_Array;
      Result      :    out Config_Data;
      Help_Line   : in     String := "";
      Help_Output : in     Ada.Text_IO.File_Type := Ada.Text_IO.Standard_Error);
   -- Main exported method.  It parses the command line and it writes
   -- the result in Result.  If some error is encountered, Bad_Command
   -- is raised with an explicative exception message.  If Help_Line is
   -- not empty, it is written to Help_Output in case of error.

   Bad_Command : exception;



   function To_Float (X : Unbounded_String)
                      return Float;
   -- Convenient conversion function to Float that raise Bad_Command if
   -- the argument has not a valid syntax

   function To_Natural (X : Unbounded_String)
                        return Natural;
   -- Convenient conversion function to Float that raise Bad_Command if
   -- the argument has not a valid syntax

end Generic_Line_Parser;
