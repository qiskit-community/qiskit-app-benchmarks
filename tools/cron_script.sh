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

# A virtual env names benchmark has been created
# and has all the qiskit-app-benchmarks requirements-dev.txt
# dependencies installed

if [[ -f /tmp/benchmark.lock ]] ; then
    echo 'Script still running, file /tmp/benchmark.lock exists.'
    exit 0
fi
echo 'Start script ...'
touch /tmp/benchmark.lock

# find if asv is installed
ASV_CMD="asv"
if command -v $ASV_CMD > /dev/null 2>&1; then
  echo "asv command is available in known paths."
else
  ASV_CMD="/usr/local/bin/asv"
  if command -v $ASV_CMD > /dev/null 2>&1; then
    echo "asv command is available at $ASV_CMD"
  else
    echo "asv command not found in any known path."
    rm /tmp/benchmark.lock
    exit 1
  fi
fi

echo "echo $3" > /tmp/.git-askpass
chmod +x /tmp/.git-askpass
export GIT_ASKPASS=/tmp/.git-askpass

source /opt/benchmark/bin/activate

set -e

echo 'qiskit-app-benchmarks was already cloned in opt and is checkout to main branch'
echo 'qiskit-app-benchmarks has a gh-pages branch with the html contents in it'
if [ -d /tmp/qiskit-app-benchmarks ]; then
  rm -rf /tmp/qiskit-app-benchmarks
fi

git clone https://$2@github.com/$1/qiskit-app-benchmarks.git /tmp/qiskit-app-benchmarks

git pull
make clean_sphinx
make html SPHINXOPTS=-W

echo 'Copy main docs'

pushd /tmp/qiskit-app-benchmarks
git config user.name "Qiskit Application Benchmarks Autodeploy"
git config user.email "qiskit@qiskit.org"
git checkout gh-pages
GLOBIGNORE=.git:finance:machine_learning:nature:optimization
rm -rf * .*
popd
cp -r docs/_build/html/* /tmp/qiskit-app-benchmarks
unset GLOBIGNORE

pushd /tmp/qiskit-app-benchmarks
git add .
# push only if there are changes
if git diff-index --quiet HEAD --; then
  echo 'Nothing to commit for the base doc template.'
else
  git commit -m "[Benchmarks] Base documentation update"
fi
popd

declare -a targets=("finance" "machine_learning" "nature" "optimization")
echo 'Run Benchmarks for domains'
for target in "${targets[@]}"
do
  pushd $target
  if [ -n "$(find benchmarks/* -not -name '__*' | head -1)" ]; then
    date
    echo "Run Benchmark for domain $target"
    $ASV_CMD run --launch-method spawn --record-samples NEW
    # $ASV_CMD run --quick
    date
    $ASV_CMD publish
    rm -rf /tmp/qiskit-app-benchmarks/$target/*
    cp -r .asv/html/* /tmp/qiskit-app-benchmarks/$target
  else
    rm -rf /tmp/qiskit-app-benchmarks/$target/*
    cp -r ../docs/_build/html/$target/* /tmp/qiskit-app-benchmarks/$target
    echo "No Benchmark files found for domain $target, run skipped."
  fi
  popd
  pushd /tmp/qiskit-app-benchmarks
  git add .
  # push only if there are changes
  if git diff-index --quiet HEAD --; then
    echo "Nothing to push for $target."
  else
    git commit -m "[Benchmarks $target] Automated documentation update"
    git push origin gh-pages
  fi
  popd
done

echo 'Final Cleanup'
unset GIT_ASKPASS
rm /tmp/.git-askpass
rm -rf /tmp/qiskit-app-benchmarks
rm /tmp/benchmark.lock
echo 'End of script.'
