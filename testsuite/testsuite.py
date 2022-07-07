#!/usr/bin/env python3

"""
Usage::

    testsuite.py [OPTIONS]

Run the CUDA testsuite.
"""
import sys
import logging
from pathlib import Path
from e3.testsuite import Testsuite
from e3.testsuite.driver.classic import TestAbortWithError, TestAbortWithFailure, ClassicTestDriver
from e3.fs import sync_tree


ROOT = Path(sys.argv[0]).resolve().parents[1]
EXAMPLES = ROOT / "examples"


class CUDAExamplesDriver(ClassicTestDriver):
    def check_file(self, path):
        assert Path(path).is_file, f"Missing file: {path}"

    def do_run(self):
        # Copying the source to the working dir, in isolation
        TESTED_SOURCE_DIR = EXAMPLES / self.test_env["input_directory"]
        sync_tree(str(TESTED_SOURCE_DIR), self.working_dir())

        self.check_file(Path(self.working_dir()) / "Makefile")
        self.shell(["make", "-I", str(TESTED_SOURCE_DIR), "-j12"], timeout=10)
        self.shell(["./main"])

    def run(self):
        try:
            self.do_run()
        except AssertionError as ae:
            raise TestAbortWithError(ae) from ae
        except TestAbortWithFailure as e:
            logging.error(f"test caused failure: {e}")
            logging.error(self.output)
            raise
        except Exception as e:
            # those are test failures
            logging.error(f"test raised an exception {e.__class__}: {e}")
            logging.error(self.output)
            raise TestAbortWithFailure(e) from e


class CUDATestsuite(Testsuite):
    tests_subdir = "tests"
    test_driver_map = {"examples": CUDAExamplesDriver}


if __name__ == "__main__":
    sys.exit(CUDATestsuite().testsuite_main())
