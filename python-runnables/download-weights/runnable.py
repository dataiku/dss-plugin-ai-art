import html
import logging
import shutil

import dataiku
from dataiku.runnables import Runnable

from ai_art import git
from ai_art.folder import get_file_path_or_temp, upload_folder
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
        file_path, temp_dir = get_file_path_or_temp(self.weights_folder)
        logging.info("Repo will be cloned to: %r", file_path)

        logging.info("Clearing weights folder: %r", self.weights_folder.name)
        self.weights_folder.clear()

        logging.info("Cloning repo: %r", self.model_repo)
        git.shallow_clone(
            self.model_repo,
            file_path,
            branch=self.revision,
            username=self.hugging_face_username,
            password=self.hugging_face_access_token,
        )

        self._rm_git_dir(file_path)

        # Upload the weights from the temp dir to the remote folder.
        # This is only needed if the managed folder is remote, since
        # local folders can be directly cloned to
        if temp_dir is not None:
            logging.info(
                "Uploading repo to weights folder: %r",
                self.weights_folder.name,
            )
            upload_folder(file_path, self.weights_folder)
            temp_dir.cleanup()

        result = html.escape(
            f"Successfully downloaded weights from {self.model_repo}"
        )
        return result

    @staticmethod
    def _rm_git_dir(repo_path):
        """Delete the .git dir of the given Git repo

        The .git dir is deleted because it doubles the size of the repo
        due to the large Git LFS files
        """
        git_dir = repo_path / ".git"

        logging.info("Deleting .git dir: %r", git_dir)
        shutil.rmtree(git_dir)
