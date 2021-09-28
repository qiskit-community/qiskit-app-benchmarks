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

# lock file with this file name and containing the pid
LOCKFILE=/tmp/`basename $0`.lock

if [ -f $LOCKFILE ]; then
  if ps -p `cat $LOCKFILE` > /dev/null 2>&1; then
      echo "Script $0 is still running."
      exit 0
  fi
fi
echo 'Start script ...'
echo $$ > $LOCKFILE

# Removes the file if:
# EXIT - normal termination
# SIGHUP - termination of the controlling process
# SIGKILL - immediate program termination
# SIGINT - program interrupt INTR character
# SIGQUIT - program interrupt QUIT characte
# SIGTERM - program termination by kill
trap 'rm -f "$LOCKFILE" >/dev/null 2>&1' EXIT HUP KILL INT QUIT TERM

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

rm -rf /tmp/qiskit-app-benchmarks

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
unset GLOBIGNORE
popd

declare -a targets=("finance" "nature" "optimization" "machine_learning")

# copy base html to benchmarks gh-pages branch
rm -rf /tmp/qiskit-app-benchmarks-html
mkdir /tmp/qiskit-app-benchmarks-html
cp -r docs/_build/html/. /tmp/qiskit-app-benchmarks-html
for target in "${targets[@]}"
do
  rm -rf /tmp/qiskit-app-benchmarks-html/$target
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
for target in "${targets[@]}"
do
  pushd $target
  if [ -n "$(find benchmarks/* -not -name '__*' | head -1)" ]; then
    date
    asv_result=0
    if [ -z "$ASV_QUICK" ]; then
      echo "Run Benchmarks for domain $target"
      $ASV_CMD run --show-stderr --launch-method spawn --record-samples NEW && asv_result=$? || asv_result=$?
    else
      echo "Run Quick Benchmarks for domain $target"
      $ASV_CMD run --quick --show-stderr && asv_result=$? || asv_result=$?
    fi
    date
    echo "asv command returned $asv_result for domain $target"
    if [ $asv_result == 0 ]; then
      echo "Publish Benchmark for domain $target"
      $ASV_CMD publish
      rm -rf /tmp/qiskit-app-benchmarks/$target/*
      cp -r .asv/html/. /tmp/qiskit-app-benchmarks/$target
    fi
  else
    rm -rf /tmp/qiskit-app-benchmarks/$target/*
    cp -r ../docs/_build/html/$target/. /tmp/qiskit-app-benchmarks/$target
    echo "No Benchmark files found for domain $target, run skipped."
  fi
  popd
  pushd /tmp/qiskit-app-benchmarks
  git add .
  # push only if there are changes
  if git diff-index --quiet HEAD --; then
    echo "Nothing to push for $target."
  else
    echo "Push benchmark for $target."
    git commit -m "[Benchmarks $target] Automated documentation update"
    git push origin gh-pages
  fi
  popd
done

echo 'Final Cleanup'
unset GIT_ASKPASS
rm /tmp/.git-askpass
rm -rf /tmp/qiskit-app-benchmarks
rm -rf /tmp/qiskit-app-benchmarks-html
echo 'End of script.'
