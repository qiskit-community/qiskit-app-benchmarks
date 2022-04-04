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

echo "Start script $BENCHMARK_BASENAME."

set -e

BENCHMARK_SCRIPT_PATH=$(dirname $(readlink -f "${BENCHMARK_BASENAME}"))
ENC_FILE_PATH=$(dirname $(dirname ${BENCHMARK_SCRIPT_PATH}))/benchmarks-secrets.json.asc

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

echo "echo $GIT_PERSONAL_TOKEN" > /tmp/.git-askpass
chmod +x /tmp/.git-askpass
export GIT_ASKPASS=/tmp/.git-askpass

echo 'Update benchmarks repository dependencies'
pip install -U -r requirements-dev.txt

echo 'qiskit-app-benchmarks was already cloned in opt and is checkout to main branch'
echo 'qiskit-app-benchmarks has a gh-pages branch with the html contents in it'

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
  rm -rf /tmp/qiskit-app-benchmarks-html/TARGET
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

BASE_DIR=/tmp/benchmarks-logs
mkdir -p ${BASE_DIR}
FILE_PREFIX=benchmarks_
FILE_SUFFIX=.txt

echo 'Remove benchmark output files older than 30 days'
find ${BASE_DIR} -name ${FILE_PREFIX}*${FILE_SUFFIX} -maxdepth 1 -type f -mtime +30 -delete

echo 'Run Benchmarks for domains'
for TARGET in "${TARGETS[@]}"
do
  pushd $TARGET
  if [ -n "$(find benchmarks/* -not -name '__*' | head -1)" ]; then
    date
    asv_result=0
    DATE=$(date +%Y%m%d%H%M%S)
    BENCHMARK_LOG_FILE="${BASE_DIR}/${FILE_PREFIX}${TARGET}_${DATE}${FILE_SUFFIX}"
    set +e
    if [ -z "$ASV_QUICK" ]; then
      echo "Run Benchmarks for domain $TARGET"
      $ASV_CMD run --show-stderr --launch-method spawn --record-samples NEW 2>&1 | tee ${BENCHMARK_LOG_FILE}
    else
      echo "Run Quick Benchmarks for domain $TARGET"
      $ASV_CMD run --quick --show-stderr 2>&1 | tee ${BENCHMARK_LOG_FILE}
    fi
    asv_result=$?
    date
    echo "Posting Benchmarks log to Slack"
    python $BENCHMARK_SCRIPT_PATH/send_notification.py -key $GIT_PERSONAL_TOKEN -encryptedfile $ENC_FILE_PATH -logfile $BENCHMARK_LOG_FILE
    retval=$?
    if [ $retval -ne 0 ]; then
      echo 'Benchmarks Logs post to Slack failed.'
    else
      echo 'Benchmarks Logs post to Slack succeeded.'
    fi
    set -e
    echo "asv command returned $asv_result for domain $TARGET"
    if [ $asv_result == 0 ]; then
      echo "Publish Benchmark for domain $TARGET"
      $ASV_CMD publish
      rm -rf /tmp/qiskit-app-benchmarks/$TARGET/*
      cp -r .asv/html/. /tmp/qiskit-app-benchmarks/$TARGET
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
unset GIT_ASKPASS
rm /tmp/.git-askpass
rm -rf /tmp/qiskit-app-benchmarks
rm -rf /tmp/qiskit-app-benchmarks-html
echo "End of $BENCHMARK_BASENAME script."
