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


# TODO: Manually check if LFS is enabled. Currently, it silently fails
# if it's not
def shallow_clone(repo, dir_, *, username, password):
    """Perform a shallow clone of a password-protected HTTP Git repo

    repo (str): Git repo to clone
    dir_ (str or path-like): Path that the repo will be cloned to
    username (str): Username used to log into the remote repo
    password (str): Password used to log into the remote repo
    """
    env = os.environ.copy()
    env["GIT_USER"] = username
    env["GIT_PASSWORD"] = password

    subprocess.run(
        (
            "git",
            "-c",
            f"credential.helper={_CREDENTIAL_HELPER}",
            "clone",
            "--depth=1",
            "--",
            repo,
            dir_,
        ),
        check=True,
        env=env,
    )
