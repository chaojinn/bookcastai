from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


ENV_USERNAME = "SCP_USERNAME"
ENV_PASSWORD = "SCP_PASSWORD"
DEFAULT_PORT = 22


class SCPConfigurationError(RuntimeError):
    """Raised when required credentials or tools are missing."""


def _load_credentials() -> tuple[str, str]:
    load_dotenv()

    username = os.getenv(ENV_USERNAME)
    password = os.getenv(ENV_PASSWORD)

    if not username:
        raise SCPConfigurationError(f"Environment variable {ENV_USERNAME} is required.")
    if not password:
        raise SCPConfigurationError(f"Environment variable {ENV_PASSWORD} is required.")

    return username, password


def _parse_remote(remote: str) -> tuple[str, int]:
    if remote.count(":") == 1 and "[" not in remote:
        host, port_text = remote.split(":", 1)
        try:
            return host, int(port_text)
        except ValueError:
            pass
    return remote, DEFAULT_PORT


def _resolve_sshpass() -> str:
    binary = shutil.which("sshpass")
    if not binary:
        raise FileNotFoundError(
            "sshpass executable not found. Install it (e.g. `apt install sshpass`) and retry."
        )
    return binary


def run_scp(source: Path, destination: str, remote: str) -> None:
    username, password = _load_credentials()
    sshpass = _resolve_sshpass()
    host, port = _parse_remote(remote)

    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")

    command = [
        sshpass,
        "-p",
        password,
        "scp",
        "-P",
        str(port),
        str(source),
        f"{username}@{host}:{destination}",
    ]

    print("Executing:", " ".join(shlex.quote(part) for part in command))
    completed = subprocess.run(command, capture_output=True, text=True)

    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        error_output = completed.stderr.strip() or f"scp exited with status {completed.returncode}"
        raise RuntimeError(error_output)


def run_ssh(remote: str, cmd: str) -> None:
    username, password = _load_credentials()
    sshpass = _resolve_sshpass()
    host, port = _parse_remote(remote)

    command = [
        sshpass,
        "-p",
        password,
        "ssh",
        "-p",
        str(port),
        f"{username}@{host}",
        cmd,
    ]

    print("Executing:", " ".join(shlex.quote(part) for part in command))
    completed = subprocess.run(command, capture_output=True, text=True)

    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        error_output = completed.stderr.strip() or f"ssh exited with status {completed.returncode}"
        raise RuntimeError(error_output)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a file to a remote host using sshpass + scp."
    )
    parser.add_argument("source", type=Path, help="Local path to the file to upload.")
    parser.add_argument("destination", help="Remote destination path (e.g. /tmp/file.txt).")
    parser.add_argument(
        "remote",
        help="Remote address (hostname or hostname:port). Default port is 22.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run_scp(args.source.expanduser().resolve(), args.destination, args.remote)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
