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

# Script parameters
CRON_BASENAME=$0
GIT_OWNER=$1
GIT_USERID=$2
GIT_PERSONAL_TOKEN=$3

set -e

echo "Start script $CRON_BASENAME."

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

# Removes the file if:
# EXIT - normal termination
# SIGHUP - termination of the controlling process
# SIGKILL - immediate program termination
# SIGINT - program interrupt INTR character
# SIGQUIT - program interrupt QUIT character
# SIGTERM - program termination by kill
trap 'rm -f "$CRON_LOCKFILE" >/dev/null 2>&1' EXIT HUP KILL INT QUIT TERM

echo 'Pull latest benchmarks repository files'
git pull origin main --no-rebase

CRON_SCRIPT_PATH=$(dirname $(readlink -f "${CRON_BASENAME}"))

echo 'Run main script'
. $CRON_SCRIPT_PATH/main_script.sh $GIT_OWNER $GIT_USERID $GIT_PERSONAL_TOKEN || true

echo "End of $CRON_BASENAME script."
