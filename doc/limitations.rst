**************************************
Current Limitations
**************************************

- Only one standalone device library can be linked to a host program. It can
  however itself include multiple static libraries.
- Exception propagation is not supported on the device.
- Binding to the CUDA API is incomplete, both on the host and the device.
- Elaboration is not supported on the device.
- Tagged types cannot be passed between the host and the device.
- Parameters that can be passed to a ``CUDA_Execute`` call are limited to
  scalars in input mode and accesses.
- The toolchain is only Linux hosted, Windows ports are not available.
- Debugging is currently not supported on the device, including compilation
  with ``-g``.
