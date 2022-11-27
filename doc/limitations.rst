**************************************
Current Limitations
**************************************

- You can only link one standalone device library to a host
  program. However, that device library can itself include multiple
  static libraries.
- Exception propagation is not supported on the device.
- Ada and GNAT checks are currently not supported on the device.
- The binding to the CUDA API is incomplete, both on the host and the device.
- Elaboration is not supported on the device.
- Tagged types cannot be passed between the host and the device.
- Parameters that can be passed to a :code:`CUDA_Execute` call are
  limited input mode.
- The toolchain is only hosted on Linux; support for Windows hosts is
  not yet available.
- Debugging is currently not supported on the device. The :switch:`-g`
  switch has no effect when compiling for the device.
