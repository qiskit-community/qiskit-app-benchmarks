#!/bin/bash
# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Script parameters
START_BASENAME=${BASH_SOURCE}

set -e

echo "Start script $START_BASENAME."

echo 'Install Rust'
curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable --profile default --no-modify-path -y

echo "End of $START_BASENAME script."
