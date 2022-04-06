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

# This script prepares the current environment by installing
# dependencies then runs unit tests

# Script parameters
ML_BASENAME=${BASH_SOURCE}
GIT_PERSONAL_TOKEN=$1

echo "Start script $ML_BASENAME."

set -e

echo 'Update benchmarks repository dependencies'
pip install -U -r requirements-dev.txt

BASE_DIR=/tmp/ml
mkdir -p ${BASE_DIR}

ML_DIR=${BASE_DIR}/qiskit-machine-learning
rm -rf ${ML_DIR}

echo 'Clone Qiskit Machine Learning'
git clone https://github.com/Qiskit/qiskit-machine-learning.git ${ML_DIR}

echo 'Run unit tests with tox'
pushd ${ML_DIR}
tox -e gpu || true
popd

echo 'Final cleanup'
rm -rf ${ML_DIR}
echo "End of $ML_BASENAME script."
