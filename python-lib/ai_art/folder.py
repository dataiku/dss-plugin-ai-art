import logging
import pathlib
import shutil
import tempfile


def get_file_path_or_temp(folder):
    """Attempt to get the local path to a folder

    If the folder isn't on the local filesystem, the path to a temporary
    dir will be returned instead

    :param folder: The folder to get the local path to
    :type folder: Dataiku.folder

    :return: Path to the local folder, and the temporary dir object if
        one was created
    :rtype: tuple[pathlib.Path, tempfile.TemporaryDirectory | None]
    """
    try:
        file_path = folder.get_path()
    except Exception:
        logging.warning(
            "Unable to access the folder %r directly because it's not on the "
            "local filesystem. The contents of the folder will be copied to a "
            "temporary local directory",
            folder.name,
        )
        # The temp dir will be automatically deleted when this
        # object is garbage-collected
        temp_dir = tempfile.TemporaryDirectory(
            prefix="dss-plugin-ai-art-weights-"
        )
        file_path = temp_dir.name
    else:
        temp_dir = None

    return pathlib.Path(file_path), temp_dir


def download_folder(remote_folder, local_path):
    """Download the contents of a managed folder to a local directory

    Empty directories are skipped

    :param remote_folder: Managed folder that will be downloaded
    :type remote_folder: Dataiku.folder
    :param local_path: Path to the local dir that ``remote_folder`` will
        be downloaded to
    :type local_path: pathlib.Path

    :return: None
    """
    for remote_path in remote_folder.list_paths_in_partition():
        rel_path = remote_path.lstrip("/")
        full_local_path = local_path / rel_path

        # Create parent dirs
        full_local_path.parent.mkdir(parents=True, exist_ok=True)

        with remote_folder.get_download_stream(remote_path) as remote_file:
            with open(full_local_path, "wb") as local_file:
                shutil.copyfileobj(remote_file, local_file)
