import logging
import os
import re
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


def _run_git_with_auth(command, username, password, **kwargs):
    """Run the given Git command with authentication

    :param command: Partial command to run, e.g.
        `["clone", "https://REPO"]`
    :type command: Iterable[str]
    :param username: Username to authenticate with the HTTP Git server
    :type username: str
    :param password: Password to authenticate with the HTTP Git server
    :type password: str
    :param kwargs: Extra kwargs to pass to `subprocess.run()`
    :type kwargs: Any

    :return: Output of `subprocess.run()`
    :rtype: subprocess.CompletedProcess
    """
    if "env" in kwargs:
        # Add the auth env-vars to the user-provided env dict
        env = kwargs["env"].copy()
    else:
        # Inherit the parent env-vars when the env kwarg isn't set
        env = os.environ.copy()

    env["GIT_USER"] = username
    env["GIT_PASSWORD"] = password
    kwargs["env"] = env

    full_command = (
        "git",
        "-c",
        f"credential.helper={_CREDENTIAL_HELPER}",
    ) + tuple(command)

    logging.info("Running command: %s", full_command)
    return subprocess.run(full_command, **kwargs)


def shallow_clone(repo, dir_, *, branch, username, password):
    """Perform a shallow clone of a password-protected HTTP Git repo

    :param repo: Git repo to clone
    :type repo: str
    :param dir_: Path that the repo will be cloned to
    :type dir_: str | os.PathLike
    :param branch: Branch that will be fetched
    :type branch: str
    :param username: Username used to log into the remote repo
    :type username: str
    :param password: Password used to log into the remote repo
    :type password: str

    :return: None
    """
    command = (
        "clone",
        "--depth=1",
        # Don't fetch PyTorch checkpoint files. The files aren't needed
        # to run the model, and are very large
        "--config=lfs.fetchexclude=*.ckpt",
        f"--branch={branch}",
        "--",
        repo,
        dir_,
    )
    _run_git_with_auth(command, username, password, check=True)


_PARSE_BRANCH_REGEX = re.compile(
    r"""
    \b
    refs/heads/
    (?P<branch>
        [^\s/]+  # Match all chars except whitespace or '/'
    )
    (?:
        \s+|$  # Branch name must be followed by whitespace or end-of-line
    )
    """,
    re.VERBOSE,
)
"""Match the branch name from a line returned by `git ls-remote`

Example line:
    52b46db8e14744892bb7ee014fc1cbb8c408643f refs/heads/main
Result:
    main
"""


def _parse_branch_from_line(line):
    """Parse the branch name from a line from `git ls-remote`

    :param line: Output line from `git ls-remote`
    :type line: str

    :return: Branch name, or `None` if no branch name is found
    :rtype: str | None
    """
    result = _PARSE_BRANCH_REGEX.search(line)
    if result:
        return result.group("branch")

    logging.warning("Unable to parse branch line: %r", line)
    return None


def get_branches(repo, *, username, password):
    """Get the branches of a password-protected remote repo

    :param repo: Git repo to clone
    :type repo: str
    :param username: Username used to log into the remote repo
    :type username: str
    :param password: Password used to log into the remote repo
    :type password: str

    :return: Generator of branches
    :rtype: Generator[str, None, None]
    """
    # Ideally there would be a "--" arg before `repo` so that it doesn't
    # break if `repo` starts with a dash, but the version of Git that
    # RHEL 7 uses (1.8.3.1) is too old to support this syntax
    command = ("ls-remote", "--heads", repo)

    # Force the locale so the output doesn't change
    env = os.environ.copy()
    env["LC_ALL"] = "C"

    proc = _run_git_with_auth(
        command,
        username,
        password,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    for line in proc.stdout.splitlines():
        branch = _parse_branch_from_line(line)
        if branch:
            yield branch


def check_lfs():
    """Assert that LFS is installed

    :raises RuntimeError: LFS isn't installed

    :return: None
    """
    command = ("git", "config", "--get-regexp", r"^filter\.lfs\.")
    logging.info("Running command: %s", command)
    try:
        # The LFS config options are set when you run `git lfs install`
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("git-lfs isn't installed") from e
