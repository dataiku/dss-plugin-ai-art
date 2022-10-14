import logging

from ai_art import git
from ai_art.constants import DEFAULT_REVISIONS
from ai_art.params import resolve_model_repo


def _get_branches_from_remote(config):
    model_repo = resolve_model_repo(config)
    credentials = config["hugging_face_credentials"]
    hugging_face_username = credentials["username"]
    hugging_face_access_token = credentials["access_token"]

    branches = git.get_branches(
        model_repo,
        username=hugging_face_username,
        password=hugging_face_access_token,
    )
    # Exhaust the generator so that any errors are caught by the
    # try-block
    return tuple(branches)


def _sort_branches(branches):
    """Sort the given branches and remove duplicates

    Branches in DEFAULT_REVISIONS are listed first so that the user is
    more likely to pick them

    Returns a list of sorted branches
    """
    unsorted_branches = set(branches)
    preferred_branches = []

    for branch in DEFAULT_REVISIONS:
        if branch in branches:
            preferred_branches.append(branch)
            unsorted_branches.remove(branch)

    sorted_branches = preferred_branches + sorted(unsorted_branches)
    return sorted_branches


def do(payload, config, plugin_config, inputs):
    """Compute a list of Git branches for the "revision" param

    If we're unable to retrieve the branches for any reason (e.g. the
    Hugging Face credentials are incorrect), we fall back to the default
    revisions that are intended to be used
    """
    try:
        branches = _get_branches_from_remote(config)
    except Exception:
        logging.exception(
            "Unable to get Git revisions dynamically. "
            "Falling back to the default revisions",
        )
        branches = DEFAULT_REVISIONS

    sorted_branches = _sort_branches(branches)
    choices = [
        {"value": branch, "label": branch} for branch in sorted_branches
    ]

    return {"choices": choices}
