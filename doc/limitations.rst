**************************************
Current Limitations
**************************************

 - Only one device library can be linked to a host program.
 - Exceptions propagation is not supported on the device.
 - Binding to the CUDA API is incomplete both on the host and the device.
 - Elaboration is not supported on the device.
 - Tagged types cannot be passed between the host and the device.
 - Parameters that can be passed to a ``CUDA_Execute`` calls are limited to
   scalars in input mode and accesses.