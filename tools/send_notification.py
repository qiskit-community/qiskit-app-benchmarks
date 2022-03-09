#!/usr/bin/env python3
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

"""Utility script to send notifications to Slack"""

from typing import Tuple, Union, List
import sys
import os
import json
import argparse
from pathlib import Path
import subprocess
import requests


def _cmd_execute(args: List[str]) -> Tuple[str, Union[None, str]]:
    """execute command"""
    env = {}
    for k in ["SYSTEMROOT", "PATH"]:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env["LANGUAGE"] = "C"
    env["LANG"] = "C"
    env["LC_ALL"] = "C"
    with subprocess.Popen(
        args,
        cwd=os.getcwd(),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as popen:
        out, err = popen.communicate()
        popen.wait()
        out_str = out.decode("utf-8").strip()
        err_str = err.decode("utf-8").strip()
        err_str = err_str if err_str else None
        return out_str, err_str


def _get_webhook_url(key: str, encryptedfile: str) -> str:
    """decrypts file and return web hook URL"""
    path = Path(encryptedfile).resolve()
    path_str = str(path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"GPG error: Invalid file path {path_str}")
    out_str, err_str = _cmd_execute(["gpg", "-d", "--batch", "--passphrase", key, path_str])
    if not out_str:
        if err_str:
            raise ValueError(f"GPG error: {err_str}")
        raise ValueError("GPG error: empty decrypted data")

    data = json.loads(out_str)
    return data["secrets"]["slack-app-url"]


def _send_notification(key: str, encryptedfile: str, name: str, status: int, path: str) -> None:
    """Sends notification to Slack"""
    webhook_url = _get_webhook_url(key, encryptedfile)
    msg_status = "succeeded" if status == 0 else "failed"
    with open(path, "rt", encoding="utf8") as file:
        text = file.read()

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"{name} {msg_status}."}},
        {
            "type": "rich_text",
            "elements": [
                {"type": "rich_text_preformatted", "elements": [{"type": "text", "text": text}]}
            ],
        },
    ]
    slack_data = {
        "blocks": blocks,
    }
    response = requests.post(
        webhook_url, data=json.dumps(slack_data), headers={"Content-Type": "application/json"}
    )
    if response.status_code != 200:
        raise ValueError(
            f"Request to Slack returned an error {response.status_code}, "
            f"the response is:\n{response.text}"
        )


def _check_path(path: str) -> str:
    """valid path argument"""
    if path and os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"path:{path} is not a valid file path")
    return path


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Sends notifications to Slack")
    PARSER.add_argument("-key", metavar="key", required=True, help="Encrypted file key")
    PARSER.add_argument(
        "-encryptedfile",
        type=_check_path,
        metavar="encryptedfile",
        required=True,
        help="Encrypted file path.",
    )
    PARSER.add_argument("-name", metavar="name", required=True, help="Application name")
    PARSER.add_argument(
        "-status", type=int, metavar="status", required=True, help="Status Success(0) or Failure"
    )
    PARSER.add_argument(
        "-logfile",
        type=_check_path,
        metavar="logfile",
        required=True,
        help="Log path path.",
    )

    STATUS = 0
    try:
        ARGS = PARSER.parse_args()
        _send_notification(ARGS.key, ARGS.encryptedfile, ARGS.name, ARGS.status, ARGS.logfile)
    except Exception as ex:  # pylint: disable=broad-except
        print(str(ex))
        STATUS = 1

    sys.exit(STATUS)
