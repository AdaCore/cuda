from support.bsp_sources.target import DFBBTarget

import os

# cuda library
CUDA_API = os.path.join(os.path.dirname(__file__), "..", "api")


def add_source_dir(holder, subdir, path):
    for f in os.listdir(path):
        full = os.path.join(path, f)
        if os.path.isfile(full):
            holder.add_source(subdir, full)


class CUDADevice(DFBBTarget):
    @property
    def name(self):
        return "cuda"

    @property
    def target(self):
        return "cuda"

    # @property
    # def readme_file(self):
    #     return os.path.join(os.path.dirname(__file__), 'README')

    @property
    def system_ads(self):
        return {"device": "runtime/device_gnat/system.ads"}

    @property
    def compiler_switches(self):
        # The required compiler switches
        return (f"-mcpu={self.gpu_arch}",)

    def base_profile(self, profile):
        # No base profile as the CUDA runtime uses it's own sources.
        return "none"

    def amend_rts(self, rts_profile, cfg):
        super(CUDADevice, self).amend_rts("none", cfg)
        cfg.rts_vars["Cuda_Target"] = "device"
        # Add cuda sources. Probably better to put these in their own cuda
        # directory with their own make file.
        # add_source_dir(cfg, 'gnat', os.path.join(CUDA_API, 'device_static'))

    def __init__(self):
        super(CUDADevice, self).__init__()
