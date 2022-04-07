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

# A virtual env names benchmark has been created
# and has all the qiskit-app-benchmarks requirements-dev.txt
# dependencies installed

# Script parameters
BENCHMARK_BASENAME=${BASH_SOURCE}
GIT_OWNER=$1
GIT_USERID=$2
GIT_PERSONAL_TOKEN=$3

set -e

echo "Start script $BENCHMARK_BASENAME."

echo 'Update benchmarks repository dependencies'
pip install -U -r requirements-dev.txt

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
    echo "End of $BENCHMARK_BASENAME script."
    exit 1
  fi
fi

export GIT_ASKPASS=/tmp/.git-askpass
rm -f $GIT_ASKPASS
echo "echo $GIT_PERSONAL_TOKEN" > $GIT_ASKPASS
chmod +x $GIT_ASKPASS

echo 'qiskit-app-benchmarks has a gh-pages branch with the html benchmarks results in it.'

make clean_sphinx
make html SPHINXOPTS=-W

rm -rf /tmp/qiskit-app-benchmarks
git clone https://$GIT_USERID@github.com/$GIT_OWNER/qiskit-app-benchmarks.git /tmp/qiskit-app-benchmarks

echo 'Copy main docs'

pushd /tmp/qiskit-app-benchmarks
git config user.name "Qiskit Application Benchmarks Autodeploy"
git config user.email "qiskit@qiskit.org"
git checkout gh-pages
GLOBIGNORE=.git:finance:machine_learning:nature:optimization
rm -rf * .*
unset GLOBIGNORE
popd

declare -a TARGETS=("finance" "nature" "optimization" "machine_learning")

# copy base html to benchmarks gh-pages branch
rm -rf /tmp/qiskit-app-benchmarks-html
mkdir /tmp/qiskit-app-benchmarks-html
cp -r docs/_build/html/. /tmp/qiskit-app-benchmarks-html
for TARGET in "${TARGETS[@]}"
do
  rm -rf /tmp/qiskit-app-benchmarks-html/$TARGET
done
cp -r /tmp/qiskit-app-benchmarks-html/. /tmp/qiskit-app-benchmarks


pushd /tmp/qiskit-app-benchmarks
git add .
# push only if there are changes
if git diff-index --quiet HEAD --; then
  echo 'Nothing to commit for the base doc template.'
else
  git commit -m "[Benchmarks] Base documentation update"
fi
popd

echo 'Run Benchmarks for domains'
for TARGET in "${TARGETS[@]}"
do
  pushd $TARGET
  if [ -n "$(find benchmarks/* -not -name '__*' | head -1)" ]; then
    date
    asv_result=0
    if [ -z "$ASV_QUICK" ]; then
      echo "Run Benchmarks for domain $TARGET"
      $ASV_CMD run --show-stderr --launch-method spawn --record-samples NEW && asv_result=$? || asv_result=$?
    else
      echo "Run Quick Benchmarks for domain $TARGET"
      $ASV_CMD run --quick --show-stderr && asv_result=$? || asv_result=$?
    fi
    date
    echo "$ASV_CMD returned $asv_result for domain $TARGET"
    if [ $asv_result == 0 ]; then
      echo "Publish Benchmark for domain $TARGET"
      retval=0
      $ASV_CMD publish && retval=$? || retval=$?
      if [ $retval == 0 ]; then
        rm -rf /tmp/qiskit-app-benchmarks/$TARGET/*
        cp -r .asv/html/. /tmp/qiskit-app-benchmarks/$TARGET
      else
        echo "$ASV_CMD failed to publish. Error:  $retval"
      fi
    fi
  else
    rm -rf /tmp/qiskit-app-benchmarks/$TARGET/*
    cp -r ../docs/_build/html/$TARGET/. /tmp/qiskit-app-benchmarks/$TARGET
    echo "No Benchmark files found for domain $TARGET, run skipped."
  fi
  popd
  pushd /tmp/qiskit-app-benchmarks
  git add .
  # push only if there are changes
  if git diff-index --quiet HEAD --; then
    echo "Nothing to push for $TARGET."
  else
    echo "Push benchmark for $TARGET."
    git commit -m "[Benchmarks $TARGET] Automated documentation update"
    git push origin gh-pages
  fi
  popd
done

echo 'Final Cleanup'
rm -f $GIT_ASKPASS
unset GIT_ASKPASS
rm -rf /tmp/qiskit-app-benchmarks
rm -rf /tmp/qiskit-app-benchmarks-html
echo "End of $BENCHMARK_BASENAME script."
