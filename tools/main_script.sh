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
MAIN_BASENAME=${BASH_SOURCE}
GIT_OWNER=$1
GIT_USERID=$2
GIT_PERSONAL_TOKEN=$3

set -e

echo "Start script $MAIN_BASENAME."

echo 'Remove previous python environment if it exists'
rm -rf /tmp/benchmarks-env

echo 'Create python environment'
python3 -m venv /tmp/benchmarks-env

echo 'Activate python environment'
source /tmp/benchmarks-env/bin/activate

echo 'Upgrade pip'
pip install -U pip

echo 'Update benchmarks repository dependencies'
pip install -U -r requirements-dev.txt

BASE_DIR=/tmp/cron-logs
mkdir -p ${BASE_DIR}
FILE_PREFIX=cron_
FILE_SUFFIX=.txt

echo 'Remove cron log files older than 30 days'
find ${BASE_DIR} -name ${FILE_PREFIX}*${FILE_SUFFIX} -maxdepth 1 -type f -mtime +30 -delete

MAIN_SCRIPT_PATH=$(dirname $(readlink -f "${MAIN_BASENAME}"))
ENC_FILE_PATH=$(dirname $(dirname ${MAIN_SCRIPT_PATH}))/benchmarks-secrets.json.asc

DATE=$(date +%Y%m%d%H%M%S)

echo "Environment HOME=$HOME"

echo 'Install Rust'
export CARGO_HOME=/tmp/cargo
export RUSTUP_HOME=/tmp/rustup
rm -rf $CARGO_HOME
rm -rf $RUSTUP_HOME

MAIN_LOG_FILE="${BASE_DIR}/${FILE_PREFIX}${DATE}_RUST${FILE_SUFFIX}"

echo 'Run Install Rust script'
rust_retval=0
. $MAIN_SCRIPT_PATH/install_rust.sh 2>&1 | tee ${MAIN_LOG_FILE}  && rust_retval=$? || rust_retval=$?

export PATH="$PATH:$CARGO_HOME/bin"
echo "Environment PATH=$PATH"

echo "Posting Rust install to Slack"
retval=0
python $MAIN_SCRIPT_PATH/send_notification.py -key $GIT_PERSONAL_TOKEN -encryptedfile $ENC_FILE_PATH -logfile $MAIN_LOG_FILE && retval=$? || retval=$?
if [ $retval -ne 0 ]; then
  echo "Rust Install post to Slack failed. Error: $retval"
else
  echo 'Rust Install post to Slack succeeded.'
fi

if [ $rust_retval -ne 0 ]; then
  echo "Rust Install failed. Error: $rust_retval"
  echo "End of $MAIN_BASENAME script."
  return $rust_retval
fi

MAIN_LOG_FILE="${BASE_DIR}/${FILE_PREFIX}${DATE}_GPU${FILE_SUFFIX}"

echo 'Run GPU Unit tests script'
. $MAIN_SCRIPT_PATH/ml_unittests.sh $GIT_PERSONAL_TOKEN 2>&1 | tee ${MAIN_LOG_FILE} || true

echo "Posting GPU logs to Slack"
retval=0
python $MAIN_SCRIPT_PATH/send_notification.py -key $GIT_PERSONAL_TOKEN -encryptedfile $ENC_FILE_PATH -logfile $MAIN_LOG_FILE && retval=$? || retval=$?
if [ $retval -ne 0 ]; then
  echo "GPU Logs post to Slack failed. Error: $retval"
else
  echo 'GPU Logs post to Slack succeeded.'
fi

MAIN_LOG_FILE="${BASE_DIR}/${FILE_PREFIX}${DATE}_ASV${FILE_SUFFIX}"

echo 'Run benchmarks script'
. $MAIN_SCRIPT_PATH/benchmarks.sh $GIT_OWNER $GIT_USERID $GIT_PERSONAL_TOKEN 2>&1 | tee ${MAIN_LOG_FILE} || true

echo "Posting Benchmarks logs to Slack"
retval=0
python $MAIN_SCRIPT_PATH/send_notification.py -key $GIT_PERSONAL_TOKEN -encryptedfile $ENC_FILE_PATH -logfile $MAIN_LOG_FILE && retval=$? || retval=$?
if [ $retval -ne 0 ]; then
  echo "Benchmarks Logs post to Slack failed. Error: $retval"
else
  echo 'Benchmarks Logs post to Slack succeeded.'
fi

echo "End of $MAIN_BASENAME script."
