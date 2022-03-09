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

echo 'Activate environment'
source /opt/benchmark/bin/activate

echo 'Update benchmarks repository dependencies'
git pull
pip install -U -r requirements-dev.txt

echo 'Run ML Unit tests script'
. $CRON_SCRIPT_PATH/ml_unittests.sh $GIT_PERSONAL_TOKEN

echo 'Run benchmarks script'
. $CRON_SCRIPT_PATH/benchmarks.sh $GIT_OWNER $GIT_USERID $GIT_PERSONAL_TOKEN

echo "End of $CRON_BASENAME script."
