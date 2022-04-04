#!/bin/bash
# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# A virtual env named benchmark has been created already

# Script parameters
CRON_BASENAME=$0
GIT_OWNER=$1
GIT_USERID=$2
GIT_PERSONAL_TOKEN=$3

CRON_SCRIPT_PATH=$(dirname "$CRON_BASENAME")

# lock file with this file name and containing the pid
CRON_LOCKFILE=/tmp/`basename $CRON_BASENAME`.lock

if [ -f $CRON_LOCKFILE ]; then
  if ps -p `cat $CRON_LOCKFILE` > /dev/null 2>&1; then
      echo "Script $CRON_BASENAME is still running."
      echo "End of $CRON_BASENAME script."
      exit 0
  fi
fi
echo $$ > $CRON_LOCKFILE
echo "Start script $CRON_BASENAME."

# Removes the file if:
# EXIT - normal termination
# SIGHUP - termination of the controlling process
# SIGKILL - immediate program termination
# SIGINT - program interrupt INTR character
# SIGQUIT - program interrupt QUIT character
# SIGTERM - program termination by kill
trap 'rm -f "$CRON_LOCKFILE" >/dev/null 2>&1' EXIT HUP KILL INT QUIT TERM

set -e

echo 'Pull latest benchmarks repository files'
git pull origin main --no-rebase

echo 'Remove previous python environment if it exists'
rm -rf /tmp/benchmarks-env

echo 'Create python environment'
python3 -m venv /tmp/benchmarks-env

echo 'Activate python environment'
source /tmp/benchmarks-env/bin/activate

echo 'Upgrade pip'
pip install -U pip

echo "Environment HOME=$HOME"

echo 'Install Rust'
export CARGO_HOME=/tmp/cargo
export RUSTUP_HOME=/tmp/rustup
rm -rf $CARGO_HOME
rm -rf $RUSTUP_HOME
curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable --profile default --no-modify-path -y
export PATH="$PATH:$CARGO_HOME/bin"
echo "Environment PATH=$PATH"

set +e

echo 'Run ML Unit tests script'
. $CRON_SCRIPT_PATH/ml_unittests.sh $GIT_PERSONAL_TOKEN

echo 'Run benchmarks script'
. $CRON_SCRIPT_PATH/benchmarks.sh $GIT_OWNER $GIT_USERID $GIT_PERSONAL_TOKEN

set -e

echo "End of $CRON_BASENAME script."
