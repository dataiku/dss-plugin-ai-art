import logging
import os
import subprocess

_CREDENTIAL_HELPER = (
    "!f() {\n"
    "  sleep 1\n"
    '  echo "username=${GIT_USER}"\n'
    '  echo "password=${GIT_PASSWORD}"\n'
    "}\n"
    "f"
)
"""Use a credential helper to pass credentials to Git via an env-var

This allows us to avoid passing the password through the command-line

Source: https://stackoverflow.com/a/43022442
"""


def shallow_clone(repo, dir_, *, branch, username, password):
    """Perform a shallow clone of a password-protected HTTP Git repo

    repo (str): Git repo to clone
    dir_ (str or path-like): Path that the repo will be cloned to
    branch (str): Branch that will be fetched
    username (str): Username used to log into the remote repo
    password (str): Password used to log into the remote repo
    """
    env = os.environ.copy()
    env["GIT_USER"] = username
    env["GIT_PASSWORD"] = password

    command = (
        "git",
        "-c",
        f"credential.helper={_CREDENTIAL_HELPER}",
        "clone",
        "--depth=1",
        f"--branch={branch}",
        "--",
        repo,
        dir_,
    )
    logging.info("Running command: %s", command)
    subprocess.run(command, check=True, env=env)


def check_lfs():
    """Assert that LFS is installed

    Raises `RuntimeError` if it's not installed
    """
    command = ("git", "config", "--get-regexp", r"^filter\.lfs\.")
    logging.info("Running command: %s", command)
    try:
        # The LFS config options are set when you run `git lfs install`
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        # TODO: improve error message
        raise RuntimeError("git-lfs isn't installed") from e
