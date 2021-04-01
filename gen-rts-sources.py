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

# also add ./runtime
sys.path.append(os.path.join(os.path.dirname(__file__), 'runtime'))

# import our cuda gnat rts sources
import cuda_sources

# and replace in the original module before it is used by the
# build_rts script and its dependencies
import support.rts_sources.sources
support.rts_sources.sources.all_scenarios = cuda_sources.rts_scenarios
support.rts_sources.sources.sources = cuda_sources.rts_sources

import gen_rts_sources
from support import add_source_search_path


def instrument_bb_runtimes():
    # Add the runtime directory in the BSP sources search path
    PWD = os.path.join(os.path.dirname(__file__), 'runtime')
    add_source_search_path(PWD)


def main():
    instrument_bb_runtimes()
    gen_rts_sources.main()


if __name__ == '__main__':
    main()
