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

A minimal project file for the device target takes the following form:

	with "<path to>/api/cuda_device.gpr";

	project Marching_Device_Code is
	   for Languages use ("Ada");
	   for Target use "cuda";
	   for Runtime ("ada") use "device-cuda";

	   for Source_Dirs use ("src");
	   for Object_Dir use "obj";
	end Marching_Device_Code;

### Host Code
In your host project reference the `cuda_host.gpr` project file.