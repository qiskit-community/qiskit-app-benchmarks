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

BASE_DIR=/tmp/ml
mkdir -p ${BASE_DIR}
FILE_PREFIX=ml_unittests_
FILE_SUFFIX=.txt

echo 'Remove unit test output files older than 30 days'
find ${BASE_DIR} -name ${FILE_PREFIX}*${FILE_SUFFIX} -maxdepth 1 -type f -mtime +30 -delete

ML_DIR=${BASE_DIR}/qiskit-machine-learning
rm -rf ${ML_DIR}

echo 'Clone Qiskit Machine Learning'
git clone https://github.com/Qiskit/qiskit-machine-learning.git ${ML_DIR}

echo 'Install tox'
pip install -U tox

echo 'Run unit tests with tox'
DATE=$(date +%Y%m%d%H%M%S)
ML_LOG_FILE="${BASE_DIR}/${FILE_PREFIX}${DATE}${FILE_SUFFIX}"
pushd ${ML_DIR}
tox -e gpu 2>&1 | tee ${ML_LOG_FILE}
retval=$?
popd
if [ $retval -ne 0 ]; then
  echo 'ML Unit Tests failed.'
else
  echo 'ML Unit Tests passed.'
fi

ML_SCRIPT_PATH=$(dirname $(readlink -f "${ML_BASENAME}"))
ENC_FILE_PATH=$(dirname $(dirname ${ML_SCRIPT_PATH}))/benchmarks-secrets.json.asc

echo "Posting to Slack"
python $ML_SCRIPT_PATH/send_notification.py -key $GIT_PERSONAL_TOKEN -encryptedfile $ENC_FILE_PATH -name "GPU Tests" -status $retval -logfile $ML_LOG_FILE
retval=$?
if [ $retval -ne 0 ]; then
  echo 'ML Unit Tests Logs post to Slack failed.'
else
  echo 'ML Unit Tests Logs post to Slack succeeded.'
fi

echo 'Final cleanup'
rm -rf ${ML_DIR}
echo "End of $ML_BASENAME script."
