import html
import logging
import shutil

from dataiku.runnables import Runnable

from ai_art import git
from ai_art.folder import get_file_path_or_temp, upload_folder
from ai_art.params import get_download_weights_config


class DownloadWeights(Runnable):
    """Download Hugging Face weights to a managed folder"""

    def __init__(self, project_key, macro_config, plugin_config):
        """
        :param project_key: the project in which the runnable executes
        :param macro_config: the dict of the configuration of the object
        :param plugin_config: contains the plugin settings
        """
        self.config = get_download_weights_config(macro_config)
        self._log_config()

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
        file_path, temp_dir = get_file_path_or_temp(self.config.weights_folder)
        logging.info("Repo will be cloned to: %r", file_path)

        logging.info(
            "Clearing weights folder: %r", self.config.weights_folder.name
        )
        self.config.weights_folder.clear()

        logging.info("Cloning repo: %r", self.config.model_repo)
        git.shallow_clone(
            self.config.model_repo,
            file_path,
            branch=self.config.revision,
            username=self.config.hugging_face_username,
            password=self.config.hugging_face_access_token,
        )

        self._rm_git_dir(file_path)

        # Upload the weights from the temp dir to the remote folder.
        # This is only needed if the managed folder is remote, since
        # local folders can be directly cloned to
        if temp_dir is not None:
            logging.info(
                "Uploading repo to weights folder: %r",
                self.config.weights_folder.name,
            )
            upload_folder(file_path, self.config.weights_folder)
            temp_dir.cleanup()

        result = html.escape(
            f"Successfully downloaded weights from {self.config.model_repo}"
        )
        return result

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

    def _log_config(self):
        """Log the config after redacting sensitive params

        :return: None
        """
        redacted_config = self.config.config.copy()
        if "hugging_face_access_token" in redacted_config:
            redacted_config["hugging_face_access_token"] = "<REDACTED>"

        logging.info("Generated params: %r", redacted_config)
