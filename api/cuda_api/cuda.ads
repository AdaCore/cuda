with Ada.Containers.Ordered_Maps;
with Ada.Exceptions;

package CUDA is
   package Exception_Registry_Map is new Ada.Containers.Ordered_Maps (Integer, Ada.Exceptions.Exception_Id, "<", Ada.Exceptions."=");
   Exception_Registry : Exception_Registry_Map.Map;
end CUDA;
