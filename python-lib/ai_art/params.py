from urllib.parse import urljoin

from ai_art.constants import HUGGING_FACE_BASE_URL


def resolve_model_repo(config):
    """Resolve the model_repo param to an absolute URL

    config (mapping): Config params of the recipe

    Returns the URL of the model repo
    """
    model_repo_path = config["model_repo"]
    if model_repo_path == "CUSTOM":
        model_repo_path = config.get("custom_model_repo")
        if not model_repo_path:
            raise ValueError("undefined parameter: Custom model repo")

    model_repo = urljoin(HUGGING_FACE_BASE_URL, model_repo_path)
    return model_repo
