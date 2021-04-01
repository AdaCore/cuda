#! /usr/bin/env python3
#
# Copyright (C) 2016-2020, AdaCore
#
# Python script to gather files for the bareboard runtime.
# Don't use any fancy features.  Ideally, this script should work with any
# Python version starting from 2.6 (yes, it's very old but that's the system
# python on oldest host).

import os
import sys

# look for --bb-dir to add it to the sys path
path = None
take_next = False
index = 0
while index < len(sys.argv):
    arg = sys.argv[index]
    if arg.startswith('--bb-dir='):
        _, path = arg.split('=')
        sys.argv.remove(arg)
        break
    elif arg == '--bb-dir':
        take_next = True
        sys.argv.remove(arg)
    elif take_next:
        path = arg
        sys.argv.remove(arg)
        break
    else:
        index += 1

assert path is not None, "missing --bb-dir switch"
sys.path.append(os.path.abspath(path))

# import the main build script
import build_rts
from support import add_source_search_path

# now our runtime support
from runtime import CUDADevice


def build_configs(target):
    "Customized targets to build specific runtimes"
    if target == 'cuda-device':
        t = CUDADevice()
    else:
        assert False, "unexpected target '%s'" % target

    return t

def instrument_bb_runtimes():
    # Add this directory in the BSP sources search path
    PWD = os.path.dirname(__file__)
    add_source_search_path(PWD)
    add_source_search_path(os.path.join(PWD, '..'))

    # Patch build_rts.build_configs to return the customized targets
    build_rts.build_configs = build_configs

if __name__ == '__main__':
    # patch bb-runtimes to use our sources
    instrument_bb_runtimes()
    # and build the runtime tree
    build_rts.main()
    # build_cuda()

