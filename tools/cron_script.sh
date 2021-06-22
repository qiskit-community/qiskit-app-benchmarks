#!/bin/bash
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

set -o xtrace

# A virtual env names benchmark has been created
# and has all the qiskit-app-benchmarks requirements-dev.txt
# dependencies installed
source /opt/benchmark/bin/activate

if [[ -f /tmp/benchmark.lock ]] ; then
    exit 0
fi
touch /tmp/benchmark.lock

# qiskit-app-benchmarks was cloned in opt and is checkout to main branch
pushd /opt/qiskit-app-benchmarks
git pull
declare -a targets=("finance" "machine_learning" "nature" "optimization")
echo “Run Benchmarks for domains”
for target in "${targets[@]}"
do
  pushd $target
  date
  asv run --launch-method spawn --record-samples NEW
  date
  popd
done
echo “Publishes html to branch gh-pages”
date
python tools/publish_html.py -user Qiskit
date
popd

rm /tmp/benchmark.lock
