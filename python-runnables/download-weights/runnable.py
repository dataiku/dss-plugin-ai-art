import html
import logging

import dataiku
from dataiku.runnables import Runnable

from ai_art import git
from ai_art.params import resolve_model_repo


class DownloadWeights(Runnable):
    """Download Hugging Face weights to a managed folder"""

    def __init__(self, project_key, config, plugin_config):
        """
        :param project_key: the project in which the runnable executes
        :param config: the dict of the configuration of the object
        :param plugin_config: contains the plugin settings
        """
        self.weights_folder = dataiku.Folder(config["weights_folder"])
        self.model_repo = resolve_model_repo(config)
        self.revision = config["revision"]

        credentials = config["hugging_face_credentials"]
        self.hugging_face_username = credentials["username"]
        self.hugging_face_access_token = credentials["access_token"]

        # If LFS isn't installed, `git clone` will quietly download fake
        # placeholder files, which is confusing
        git.check_lfs()

    def get_progress_target(self):
        """
        If the runnable will return some progress info, have this
        function return a tuple of (target, unit) where unit is one of:
        SIZE, FILES, RECORDS, NONE
        """
        return None

    def run(self, progress_callback):
        """
        Do stuff here. Can return a string or raise an exception.
        The progress_callback is a function expecting 1 value: current
        progress
        """
        file_path = self.weights_folder.get_path()

        logging.info("Cloning repo: %s", self.model_repo)
        git.shallow_clone(
            self.model_repo,
            file_path,
            branch=self.revision,
            username=self.hugging_face_username,
            password=self.hugging_face_access_token,
        )

        result = html.escape(
            f"Successfully downloaded weights from {self.model_repo}"
        )
        return result
