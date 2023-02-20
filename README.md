# GNAT for CUDA

GNAT for CUDA® is a toolsuite that allows to compile Ada and SPARK code directly for NVIDIA GPUs.

## Documentation

For a thorough discussion about `GNAT for CUDA` please consult the official documentation. A html version can be built like so:

```
cd doc
make hmtl
```

## Status
Beta

## Quickstart

### End user - archive distribution
**Note**: During beta phase this package is available on request/invitation only.

#### Prerequisites

- GNAT toolchain - AdaCore client portal
- CUDA libraries - eg. apt-get
- GNAT aarch64-linux cross compiler toolchain - AdaCore client portal (optional)

### Developper - git repository clone
**Note**: During beta phase this repo can only be built by AdaCore engineers with `anod` acces.

#### Prerequisites

- GNAT toolchain - anod
- CUDA libraries - eg. apt-get
- GNAT aarch64-linux cross compiler toolchain - anod (optional)
- GNAT sources - anod (set root Makefile $GNAT_SRC to it)
- bb-runtimes - anod (set root Makefile $BB_SRC to it)
- CUDA_env - anod (set to system environment variables)

### Setup GNAT for CUDA
```
chmod +x setup.sh
./setup.sh
```
- **End user only**:
```
source ./env.sh 
```

### Build cuda-gcc (optional)
- First [Setup GNAT for CUDA](#setup-gnat-for-cuda). Then:
```
make
```

### Compilation of vectorAdd example program
**Note**: By default we are building for `x86_64,` Turing family GPU `sm_75`
- First [Setup GNAT for CUDA](#setup-gnat-for-cuda). Then:
```
cd examples/0_Introduction/vectorAdd
make
```

### Cross-compilation of vectorAdd example program

**Note**: To illustrate concrete cross-compilation steps, the following instructions are contextualized for cross-compiling from a `x86_64-linux` desktop (**host**) to a `aarch64-linux` Jetson Nano (**cuda_host**) Maxwell family GPU `sm_53` running an `Ubuntu 18.04` derivative as officialy published by NVIDIA. The **cuda_host** is located at LAN IP address **192.168.x.y** running as user **alice**:

- First [Setup GNAT for CUDA](#setup-gnat-for-cuda). 

- Make sure your cross toolchain is properly installed.
```
$ which aarch64-linux-gnu-gcc
[somewhere_on_your_disk]/aarch64-linux-linux64/gnat/install/bin/aarch64-linux-gnu-gcc

$ echo $ENV_PREFIX
[somewhere_on_your_disk]/aarch64-linux-linux64/system-libs/src/aarch64-linux-system
```

- As we will use the CUDA libraries found on the **cuda_host**, install `sshfs`. On **host**:
```
sudo apt install sshfs
```

- Create folders. On **host**:   
		 `/usr/lib/aarch64-linux-gnu/`        
       `/usr/local/cuda/targets/aarch64-linux/lib`       
		 `/usr/local/cuda/targets/aarch64-linux/lib/stubs`    

```
sudo mkdir -p /usr/lib/aarch64-linux-gnu/
sudo mkdir -p /usr/local/cuda/targets/aarch64-linux/lib/stubs
```

- Mount **cuda_host** `aarch64` system libs to newly created **host** directory using sshfs. On **host**:
```
sudo sshfs -o nonempty,allow_other,default_permissions alice@192.168.x.y:/usr/lib/aarch64-linux-gnu/ /usr/lib/aarch64-linux-gnu/
```

- Mount **cuda_host** `aarch64` CUDA libs to newly created **host** directory using sshfs. On **host**:
```
sudo sshfs -o nonempty,allow_other,default_permissions alice@192.168.x.y:/usr/local/cuda/targets/aarch64-linux/lib /usr/local/cuda/targets/aarch64-linux/lib
```

- Edit `examples/Makefile.include` to:
```
GPU_ARCH=sm_53
CUDA_HOST=aarch64-linux
```

- Build example program. On **host**:
```
source ./env.sh
cd examples/0_Introduction/vectorAdd
make
```

- Copy executable to **cuda_host**. On **host**:
```
scp main alice@192.168.x.y:~
```

- Move to **cuda_host**. On **host**:
```
ssh alice@192.168.x.y:~
./main
```
