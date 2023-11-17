use strict;
use warnings;

my $bad = <<EOF;
   function Graph_Add_Node_V2
     (Graph :     CUDA.Driver_Types.Graph_T; Dependencies : Graph_Node_Array_T;
      Dependency_Data :     CUDA.Driver_Types.Graph_Edge_Data_St;
      Node_Params     : out CUDA.Driver_Types.Graph_Node_Params)
      return CUDA.Driver_Types.Graph_Node_T
   is

      Temp_ret_1   : aliased CUDA.Driver_Types.Graph_Node_T;
      Temp_call_2  : aliased System.Address := Temp_ret_1'Address;
      Temp_local_2 : aliased udriver_types_h.cudaGraph_t with
        Address => Graph'Address, Import;
      Temp_call_7  : aliased constant udriver_types_h.cudaGraphEdgeData_st with
        Address => Dependency_Data'Address, Import;
      Temp_call_8  : aliased udriver_types_h.cudaGraphNodeParams with
        Address => Node_Params'Address, Import;
      Temp_res_9   : Integer                :=
        Integer
          (ucuda_runtime_api_h.cudaGraphAddNode_v2
             (Temp_call_2, Temp_local_2, Dependencies'Address,
              Dependencies'Length, Temp_call_7'Unchecked_Access,
              Temp_call_8'Unchecked_Access));
EOF

my $good = <<EOF;
   function Graph_Add_Node_V2
     (Graph :     CUDA.Driver_Types.Graph_T; Dependencies : Graph_Node_Array_T;
      Dependency_Data :     CUDA.Driver_Types.Graph_Edge_Data_St;
      Node_Params     : out CUDA.Driver_Types.Graph_Node_Params)
      return CUDA.Driver_Types.Graph_Node_T
   is

      Temp_ret_1   : aliased CUDA.Driver_Types.Graph_Node_T;
      Temp_call_2  : aliased System.Address := Temp_ret_1'Address;
      Temp_local_2 : aliased udriver_types_h.cudaGraph_t with
        Address => Graph'Address, Import;
      Temp_call_7  : aliased constant udriver_types_h.cudaGraphEdgeData_st with
        Address => Dependency_Data'Address, Import;
      Temp_call_8  : aliased udriver_types_h.cudaGraphNodeParams with
        Address => Node_Params'Address, Import;
      Temp_res_9   : Integer                :=
        Integer
          (ucuda_runtime_api_h.cudaGraphAddNode_v2
             (Temp_call_2, Temp_local_2, Dependencies'Address,
              Temp_call_7'Unchecked_Access, Dependencies'Length,
              Temp_call_8'Unchecked_Access));
EOF

s/\Q$bad/$good/g;
