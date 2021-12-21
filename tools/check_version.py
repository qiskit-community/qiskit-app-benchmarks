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

""" Checks that benchmark classes have a version property """

from abc import ABC
from typing import List
import sys
import os
import argparse
import inspect
import pkgutil
import importlib


class VersionChecker:
    """Check existence of version property"""

    _VERSION_NAME = "version"

    def __init__(self, root_path: str, package: str) -> None:
        self._root_path = root_path
        self._package = package

    def check(self) -> int:
        """check copyright"""
        return self._check(os.path.join(self._root_path, self._package), self._package)

    def _check(self, path: str, package: str) -> int:
        ret_code = 0
        for _, name, ispackage in pkgutil.iter_modules([path]):
            if ispackage:
                continue

            # Iterate through the modules
            fullname = package + "." + name
            modspec = importlib.util.find_spec(fullname)  # type: ignore
            mod = importlib.util.module_from_spec(modspec)  # type: ignore
            modspec.loader.exec_module(mod)
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                # Iterate through the classes defined on the module.
                if cls.__module__ == modspec.name:
                    if ABC not in cls.__bases__:
                        name = ".".join([cls.__module__, cls.__name__])
                        try:
                            _ = getattr(cls, VersionChecker._VERSION_NAME)
                        except AttributeError as ex:
                            print(f"Error: Class {name}: {ex}")
                            ret_code = 1

        for item in sorted(os.listdir(path)):
            full_path = os.path.join(path, item)
            if (
                item
                not in [
                    f"qiskit-{self._package.replace('_', '-')}",
                    "__pycache__",
                    ".asv",
                ]
                and os.path.isdir(full_path)
            ):
                ret = self._check(full_path, package + "." + item)
                if ret != 0:
                    ret_code = ret

        return ret_code


def _check_version_property(modules: List[str]) -> int:
    root = os.path.abspath(".")
    sys.path.insert(0, root)
    ret_code = 0
    for module in modules:
        ret = VersionChecker(root, module).check()
        if ret != 0:
            ret_code = ret

    return ret_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check benchmark versions.")
    parser.add_argument(
        "modules",
        type=str,
        nargs="*",
        default=["finance", "machine_learning", "nature", "optimization"],
        help="Modules to scan",
    )
    args = parser.parse_args()
    RET = _check_version_property(args.modules)
    sys.exit(RET)
