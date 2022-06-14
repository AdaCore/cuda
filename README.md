# cuda

## Building the CUDA compiler tools and runtime
Prerequisite: gnat-llvm with NVPTX support, python
 
To build the CUDA compiler wrapper *llvm-cuda* and the *device-cuda* runtime you will need a copy of the *gnat* and *bb-runtimes* sources in the folder containing the *cuda* project, i.e:

	development/
      |--> bb-runtimes
      |--> cuda
      \--> gnat
   
Once set up, you can then build and install *llvm-cuda* and the *device-cuda* runtime by running `make` from the CUDA project directory.

If required, *llvm-cuda* can be built directly using `wrapper.gpr` project file in the `wrapper` directory. 

## Building the CUDA bindings
CUDA API changes a lot from version to version, and even within a single version from configuration to configuration. It's not practical to write or keep all possible versions in version control. Instead, we can generate the API with -fdump-ada-specs and Uwrap.

For that to work, you need a recent gnat, langkit and libadalang installation, e.g. using anod:

	anod install libadalang_for_customers
	anod install langkit

	eval `anod printenv libadalang_for_customers`
	eval `anod printenv langkit`
    
	git clone git@github.com:AdaCore/uwrap.git
	cd uwrap
	cd lang_test
	make
	cd ../lang_template
	make
	cd ..
	source env.sh
	gprbuild
	export PATH=`pwd`/obj:$PATH

Set CUDA_ROOT to point to your cuda installation. For example:
	
	export CUDA_ROOT=/usr/local/cuda-10.0
	
Next you will run the cuda bind.sh script. ("cuda" is from the AdaCore github project, there's a link above.) However, this must be done from the directory it appears in, so use

	cd cuda/api
	sh ./bind.sh

This should create CUDA bindings under `cuda/api`.

## Using CUDA with your Ada application

### Device Code

A minimal project file for the device target **(device.gpr)** takes the following form:
- Where you replace `[common_sources]` with the path to your Ada source files common to the **host** and **device**.
- Where you need to specify the **cubin version** matching your CUDA device. By default it is set to `sm_75`.   
  You can find the proper cubin version, `sm_XY`, for your device at https://developer.nvidia.com/cuda-gpus.    
  Once your device found, `XY` = Compute Capability * 10.
  Note: if your **cubin version** is not in the `Cubin_Version_Option` type definition defined in the `.gpr` file, you need to add it.
- Where you need to specify your **host_target** for running the application. By default it is set to `x86_64-linux`.

Eg. With Ada application source code inside `src/common` folder, to build your project running over a `x86_64-linux` **host_target**, targeting a `GeForce GTX 1080 Ti` CUDA **device**, you would:     
1. Replace `[common_sources]` by `"src/common"` inside the following `device.gpr` file content:

```
with "../../../api/cuda_device.gpr";

library project Device is

   for Create_Missing_Dirs use "True";

   for Languages use ("Ada");
   for Source_Dirs use ([common_sources]);

   for Target use "cuda";
   for Runtime ("ada") use "device-cuda";
   for Library_Name use "kernel";

   type Cubin_Version_Option is ("sm_53", "sm_61", "sm_75"); 
   Cubin_Version : Cubin_Version_Option := external ("cubin_version", "sm_75");
   
   package Compiler is
      for Switches ("ada") use ("-gnatX", "-O2", "-gnatn", "-mcpu=" & Cubin_Version);      
   end Compiler;

   type Host_Target_Option is ("x86_64-linux", "aarch64-linux");
   Host_Target : Host_Target_Option := external ("host_target", "x86_64-linux");

   for Archive_Builder use ("cuda-gcc", "-host-target=" & Host_Target, "-mcpu=" & Cubin_Version);
   
end Device;
```

2. Build: 
```
gprbuild -Xcubin_version=sm_61 -Xhost_target=x86_64-linux -P device
```

### Host Code

Note: We assume your CUDA distribution is installed at `/usr/local/cuda` **or** you have such symbolic link pointing to your concrete CUDA distribution. For *exotic* installation please adapt the following instructions or open an issue for further assistance.

A minimal project file for the host **(host.gpr)** takes the following form:
- Where you replace `[common_sources]` with the path to your Ada source files common to the **host** and **device**.
- Where you replace `[host_sources]` with the path to your Ada source files concerning the **host**.
- Where you need to specify your `host_target` for running the application. By default it is set to `x86_64-linux`.

Eg. With Ada application source code inside `src/common` and `src/host` folders, to build your project running over a `x86_64-linux` **host**, you would:     
1. Replace `[common_sources]` by `"src/common"` and `[host_sources]` by `"src/host"` inside `host.gpr`.    

```
with "../../../api/cuda_host.gpr";

project Host is

   for Create_Missing_Dirs use "True";

   for Object_Dir use "obj/host";
   for Source_Dirs use ([common_sources], [host_sources]);
   for Main use ("main.adb");

   type Host_Target_Option is ("x86_64-linux", "aarch64-linux");
   Host_Target : Host_Target_Option := external ("host_target", "x86_64-linux");

   for Target use host_target;

   package Compiler is
      for Switches ("ada") use ("-gnatX", "-gnatd_c");
   end Compiler;

   package Linker is
      for Switches ("ada") use (
            "-L/usr/local/cuda/targets/" & Host_Target & "/lib",
            "-L/usr/local/cuda/targets/" & Host_Target & "/lib/stubs",
            "-lcudadevrt", "-lcudart_static", 
            "-lrt", 
            "-lpthread", 
            "-ldl",
            "-Wl,--unresolved-symbols=ignore-all"
         );
   end Linker;

   package Binder is
      for Default_Switches ("ada") use ("-d_c");
   end Binder;

end Host;
```

2. Build: 
```
gprbuild -Xhost_target=x86_64-linux -P host -largs $(PWD)/lib/*.fatbin.o
```

## Cross compilation

Note: To illustrate concrete cross-compilation steps, the following instructions are contextualized for cross-compiling from an `x86_64-linux` desktop (**host**) to a `aarch64-linux` Jetson Nano (**target**) running an Ubuntu 18.04 derivatives as officialy published by NVIDIA.

### In a nutshell 

You will need:
1. An Ada aarch64-linux cross-compiler. AdaCore client's can have access to such compiler on the GNAT Pro portal.
2. The CUDA libraries matching the **target** distribution CUDA capabilities and drivers; from `apt-get` the NVIDIA Jetson Nano Ubuntu 18.04 derivative ships CUDA 10.2.
3. Tell `gprbuild` to build for an `aarch64-linux` host-target with a cubin version of `sm_53`; matching the CUDA compute capability of the Jetson Nano (https://developer.nvidia.com/cuda-gpus).

Note: Because installing CUDA binaries not matching your **host** distribution can be cumbersome, we recommend to install CUDA on the **target** (**Jetson**) and mount the lib binaries to the **host** system using `sshfs`. This way you are sure to cross-compile against proper binaries version and spare your **host** system, here `x86_64`, of potential **target** alien binaries proliferation, here `aarch64`.

### Cross building

For a Jetson Nano board available at IP address **192.168.1.1** on LAN running as user **alice**:

- Build  as instructed in preceding section titled [Building the CUDA compiler tools and runtime](#building-the-cuda-compiler-tools-and-runtime) on **host**.
- Install **AdaCore** `aarch64-linux` cross-compiler on **host**. Make sure it is in your path by following accompanying instructions.
- Install `sshfs`. On **host**:
```
sudo apt install sshfs
```

- Create folders     
		 `/usr/lib/aarch64-linux-gnu/`    
         `/usr/local/cuda/targets/aarch64-linux/lib`    
		 `/usr/local/cuda/targets/aarch64-linux/lib/stubs`

- On **host**:
```
sudo mkdir -p /usr/lib/aarch64-linux-gnu/
sudo mkdir -p /usr/local/cuda/targets/aarch64-linux/lib/stubs
```

- Mount **target** `aarch64` system libs to newly created **host** directory using sshfs. On **host**:
```
sudo sshfs -o nonempty,allow_other,default_permissions alice@192.168.1.1:/usr/lib/aarch64-linux-gnu/ /usr/lib/aarch64-linux-gnu/
```

- Mount **target** `aarch64` CUDA libs to newly created **host** directory using sshfs. On **host**:
```
sudo sshfs -o nonempty,allow_other,default_permissions alice@192.168.1.1:/usr/local/cuda/targets/aarch64-linux/lib /usr/local/cuda/targets/aarch64-linux/lib
```

- Build `device.gpr` project by specifying the cubin_version and host_target. On **host**:
```
gprbuild -Xcubin_version=sm_53 -Xhost_target=aarch64-linux -P device
```

- Build `host.gpr` project by specifying the host_target. On **host**:
```
gprbuild -Xhost_target=aarch64-linux -P host -largs $(PWD)/lib/*.fatbin.o
```

- Copy executable to **target**. On **host**:
```
scp main alice@192.168.1.1:~
```

- Move to **target**. On **host**:
```
ssh alice@192.168.1.1
```

- Execute. On **target**:
```
./main
```
