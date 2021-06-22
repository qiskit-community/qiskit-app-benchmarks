# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Publish asv html results """

from typing import List
import sys
import os
import argparse
import subprocess
import traceback
import shutil
import errno
import tempfile
from distutils.dir_util import copy_tree


class Publisher:
    """Publishes  results"""

    _DOCS_DIR = "docs"
    _BUILD_DIR = "_build"

    def __init__(self, repo_owner: str, root_dir: str, targets: List[str]) -> None:
        self._repo_owner = repo_owner
        self._root_dir = root_dir
        self._targets = targets
        self._docs_dir = os.path.join(self._root_dir, Publisher._DOCS_DIR)
        if not os.path.isdir(self._docs_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self._docs_dir)
        self._build_dir = os.path.join(self._docs_dir, Publisher._BUILD_DIR)

    @staticmethod
    def _run_cmd(root_dir: str, cmd: List[str]) -> str:
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        with subprocess.Popen(
            cmd,
            cwd=root_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as popen:
            out, err = popen.communicate()
            popen.wait()
            out_str = out.decode("utf-8").strip()
            err_str = err.decode("utf-8").strip()
            if err_str:
                raise Exception(f"{cmd}: {err_str}")
            return out_str

    @staticmethod
    def _exception_to_string(excp: Exception) -> str:
        stack = traceback.extract_stack()[:-3] + traceback.extract_tb(excp.__traceback__)
        pretty = traceback.format_list(stack)
        return "".join(pretty) + "\n  {} {}".format(excp.__class__, excp)

    def publish(self) -> None:
        """public results"""
        # build sphinx html
        self._run_cmd(self._root_dir, ["make", "clean"])
        self._run_cmd(self._root_dir, ["make", "html"])
        # publish asv targets and copy the html to sphinx location
        html_dir = os.path.join(self._build_dir, "html")
        for target in self._targets:
            self._run_cmd(self._root_dir, ["make", f"TARGET={target}", "ASVCMD=publish"])
            paths = [target, ".asv", "html"]
            asv_dir = os.path.join(self._root_dir, *paths)
            dest_dir = os.path.join(html_dir, target)
            if os.path.isdir(dest_dir):
                shutil.rmtree(dest_dir)
            copy_tree(asv_dir, dest_dir)

        # push them to gh-pages
        with tempfile.TemporaryDirectory() as repo_dir:
            # init git on it
            self._run_cmd(repo_dir, ["git", "init", "-q"])
            # add remote
            self._run_cmd(
                repo_dir,
                [
                    "git",
                    "remote",
                    "add",
                    "deploy",
                    f"git@github.com:{self._repo_owner}/qiskit-app-benchmarks.git",
                ],
            )
            # fetch the remote
            self._run_cmd(repo_dir, ["git", "fetch", "deploy", "-q"])
            # checkout branch
            self._run_cmd(repo_dir, ["git", "checkout", "-q", "gh-pages"])
            # copy files to repo directory
            copy_tree(html_dir, repo_dir)
            # add files to it
            self._run_cmd(repo_dir, ["git", "add", "."])
            # Publish results
            self._run_cmd(repo_dir, ["git", "commit", "-m", "Publish asv results"])
            # Push results
            self._run_cmd(repo_dir, ["git", "push", "-q", "deploy", "gh-pages"])


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Publish asv html results")
    PARSER.add_argument(
        "-user",
        type=str,
        metavar="user",
        help="User owner of destination repository.",
    )
    PARSER.add_argument(
        "-targets",
        type=str,
        metavar="targets",
        help="Comma separated targets to publish, if not passed, publish all.",
    )

    ROOT_PATH = os.path.abspath(os.path.realpath(os.path.expanduser(os.getcwd())))

    TARGETS = ["finance", "machine_learning", "nature", "optimization"]
    ARGS = PARSER.parse_args()
    if not ARGS.user:
        print("Missing repo user")
        sys.exit(os.EX_SOFTWARE)

    TARGETS_ENTERED = []
    if ARGS.targets:
        TARGETS_ENTERED = ARGS.targets.split(",")
        for TARGET in TARGETS_ENTERED:
            if TARGET not in TARGETS:
                print(f"Invalid target {TARGET}")
                sys.exit(os.EX_SOFTWARE)

    if not TARGETS_ENTERED:
        TARGETS_ENTERED = TARGETS

    Publisher(ARGS.user, ROOT_PATH, TARGETS_ENTERED).publish()
    sys.exit(os.EX_OK)
