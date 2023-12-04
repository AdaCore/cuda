#!/usr/bin/env python3

"""
Usage::

    testsuite.py [OPTIONS]

Run the CUDA testsuite.
"""
import subprocess
import sys
import logging
from pathlib import Path
from e3.os.process import get_rlimit
from e3.testsuite import Testsuite, TestAbort
from e3.testsuite.driver.classic import (
    ClassicTestDriver,
    ProcessResult,
    TestAbortWithError,
    TestAbortWithFailure,
)
from e3.testsuite.driver.diff import DiffTestDriver, PatternSubstitute
from e3.testsuite.result import Log, TestStatus

from e3.fs import sync_tree


ROOT = Path(sys.argv[0]).resolve().parents[1]
EXAMPLES = ROOT / "examples"


class CUDABaseDriver(DiffTestDriver):
    def check_file(self, path):
        assert Path(path).is_file, f"Missing file: {path}"

    def __init_run__(self):
        """
        Driver-specific initialization routines
        """
        self.expect_failure = self.test_env.get("expect_failure", False)

    def run(self):
        try:
            self.do_run()
        except AssertionError as ae:
            raise TestAbortWithError(ae) from ae
        except TestAbortWithFailure as e:
            logging.error(f"{self.test_name} caused failure: {e}")
            logging.error(self.output)
            raise
        except Exception as e:
            # those are test failures
            logging.error(f"{self.test_name} raised an exception {e.__class__}: {e}")
            logging.error(self.output)
            raise TestAbortWithFailure(e) from e

    def do_run(self):
        self.__init_run__()

        self.check_file(Path(self.working_dir()) / "Makefile")
        self.shell(
            ["make", "-I", str(self.tested_source_dir), "-j12"],
            timeout=10,
            analyze_output=False,
        )
        p = self.run_test_program(
            [self.working_dir("main")], self.slot, timeout=self.default_process_timeout
        )

        if p.status:
            self.output.log += ">>>program returned status code {}\n".format(p.status)

        if self.expect_failure:
            assert p.status != 0, f"Expected failure {p.out}"

    @property
    def output_refiners(self):
        """
        Certain exceptions depend on the CUDA version or the execution platform
        Substitute the exception name before comparison
        """
        return [
            PatternSubstitute(
                "CUDA\\.EXCEPTIONS\\.ERRORASSERT",
                "CUDA.EXCEPTIONS.ERROR_CUDAERRORASSERT",
            )
        ]

    def run_test_program(
        self,
        cmd,
        slot,
        test_name=None,
        result=None,
        timeout=None,
        env=None,
        cwd=None,
        copy_files_on_target=None,
        **kwargs,
    ):
        if cwd is None and "working_dir" in self.test_env:
            cwd = self.test_env["working_dir"]
        if result is None:
            result = self.result
        if test_name is None:
            test_name = self.test_name

        if self.env.is_cross:
            # Import pycross only when necessary for cross targets
            from pycross.runcross.main import run_cross

            run = run_cross(
                cmd,
                cwd=cwd,
                mode=None,
                timeout=timeout,
                slot=slot,
                copy_files_on_target=copy_files_on_target,
            )

            # Here process.out holds utf-8 encoded data.
            process = ProcessResult(run.status, run.out.encode("utf-8"))

        else:
            if timeout is not None:
                cmd = [get_rlimit(), str(timeout)] + cmd

            # Use directly subprocess instead of e3.os.process.Run,
            # since the latter does not handle binary outputs.
            subp = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            stdout, _ = subp.communicate()
            # stdout here is bytes
            process = ProcessResult(subp.returncode, stdout)

        result.processes.append(
            {
                "output": Log(process.out),
                "status": process.status,
                "cmd": cmd,
                "timeout": timeout,
                "env": env,
                "cwd": cwd,
            }
        )

        # Append the status code and process output to the log to
        # ease post-mortem investigation.
        result.log += f"Cmd_Line: {' '.join(cmd)}"
        result.log += "Status code: {}\n".format(process.status)
        result.log += "Output:\n"

        try:
            out = process.out.decode("utf-8")
        except UnicodeDecodeError:
            out = str(process.out)
        result.log += out
        self.output.log = out

        if not self.expect_failure and process.status != 0:
            if isinstance(self, ClassicTestDriver):
                raise TestAbortWithFailure("command call fails")
            else:
                result.set_status(TestStatus.FAIL, "command call fails")
                self.push_result(result)
                raise TestAbort

        return process


class CUDAExamplesDriver(CUDABaseDriver):
    def __init_run__(self):
        """
        Driver-specific initialization routines
        """
        super().__init_run__()
        self.tested_source_dir = EXAMPLES / self.test_env["input_directory"]
        # Copying the source to the working dir, in isolation
        sync_tree(str(self.tested_source_dir), self.working_dir())


class CUDATextOracleDriver(CUDABaseDriver):
    def __init_run__(self):
        """
        Driver-specific initialization routines
        """
        super().__init_run__()
        self.tested_source_dir = self.test_env["test_dir"]


class CUDATestsuite(Testsuite):
    tests_subdir = "tests"
    test_driver_map = {
        "examples": CUDAExamplesDriver,
        "text_oracle": CUDATextOracleDriver,
    }


if __name__ == "__main__":
    sys.exit(CUDATestsuite().testsuite_main())
