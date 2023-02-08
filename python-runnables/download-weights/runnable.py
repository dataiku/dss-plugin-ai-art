import html
import logging
import shutil

import dataiku
from dataiku.runnables import Runnable

from ai_art import git
from ai_art.folder import get_file_path_or_temp, upload_folder
from ai_art.params import get_download_weights_config, WeightsFolderMode


class DownloadWeights(Runnable):
    """Download Hugging Face weights to a managed folder"""

    api_client = dataiku.api_client()

    def __init__(self, project_key, macro_config, plugin_config):
        """
        :param project_key: the project in which the runnable executes
        :param macro_config: the dict of the configuration of the object
        :param plugin_config: contains the plugin settings
        """
        self.project_key = project_key
        self.config = get_download_weights_config(macro_config)
        logging.info("Generated params: %r", self.config)

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
        weights_folder = self._get_weights_folder()
        logging.info("Using weights folder: %r", weights_folder.name)

        file_path, temp_dir = get_file_path_or_temp(weights_folder)
        logging.info("Repo will be cloned to: %r", file_path)

        if self.config.weights_folder_mode is WeightsFolderMode.USE_EXISTING:
            logging.info(
                "Clearing existing weights folder: %r", weights_folder.name
            )
            weights_folder.clear()

        logging.info("Cloning repo: %r", self.config.model_repo)
        git.shallow_clone(
            self.config.model_repo, file_path, branch=self.config.revision
        )

        self._rm_git_dir(file_path)

        # Upload the weights from the temp dir to the remote folder.
        # This is only needed if the managed folder is remote, since
        # local folders can be directly cloned to
        if temp_dir is not None:
            logging.info(
                "Uploading repo to weights folder: %r", weights_folder.name
            )
            upload_folder(file_path, weights_folder)
            temp_dir.cleanup()

        result = (
            "Successfully downloaded weights from "
            f"{self.config.model_repo} to {weights_folder.name}"
        )
        return html.escape(result)

    def _get_weights_folder(self):
        """Get the weights folder, creating it if needed

        The folder will be created in the default connection
        (filesystem_folders)

        :return: Weights folder
        :rtype: dataiku.Folder
        """
        if self.config.weights_folder_mode is WeightsFolderMode.CREATE_NEW:
            project = self.api_client.get_project(self.project_key)

            logging.info(
                "Creating new managed folder: %r",
                self.config.weights_folder_name,
            )
            dss_managed_folder = project.create_managed_folder(
                self.config.weights_folder_name
            )
            return dataiku.Folder(dss_managed_folder.id)
        else:  # Mode is USE_EXISTING
            return dataiku.Folder(self.config.weights_folder_name)

    @staticmethod
    def _rm_git_dir(repo_path):
        """Delete the .git dir of the given Git repo

        :param repo_path: Path to the Git repo
        :type repo_path: os.PathLike

        The .git dir is deleted because it doubles the size of the repo
        due to the large Git LFS files

        :return: None
        """
        git_dir = repo_path / ".git"

        logging.info("Deleting .git dir: %r", git_dir)
        shutil.rmtree(git_dir)
