import logging

from dku_config import DkuConfig
from ai_art import git
from ai_art.constants import DEFAULT_REVISIONS
from ai_art.params import add_model_repo


def _get_branches_from_remote(config):
    """List available branches from a remote Git repo

    :param config: Macro config
    :type config: Mapping[str, Any]

    :raises dku_config.dss_parameter.DSSParameterError: Params needed to
        connect to the repo aren't set in the config

    :return: Branches from the Git repo
    :rtype: Generator[str, None, None]
    """
    dku_config = DkuConfig()
    add_model_repo(dku_config, config)

    branches = git.get_branches(dku_config.model_repo)
    return branches


def _sort_branches(branches):
    """Sort the given branches and remove duplicates

    :param branches: Branches to sort
    :type branches: Iterable[str]

    Branches in DEFAULT_REVISIONS are listed first so that the user is
    more likely to pick them

    :return: Sorted branches
    :rtype: list[str]
    """
    unsorted_branches = set(branches)
    preferred_branches = []

    for branch in DEFAULT_REVISIONS:
        if branch in unsorted_branches:
            preferred_branches.append(branch)
            unsorted_branches.remove(branch)

    sorted_branches = preferred_branches + sorted(unsorted_branches)
    return sorted_branches


def do(payload, config, plugin_config, inputs):
    """Compute a list of Git branches for the "revision" param

    If we're unable to retrieve the branches for any reason (e.g. the
    repo isn't accessible), we fall back to the default revisions that
    are intended to be used
    """
    try:
        # Convert the generator to a tuple so that any errors are caught
        branches = tuple(_get_branches_from_remote(config))
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
