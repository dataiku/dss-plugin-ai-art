HUGGING_FACE_BASE_URL = "https://huggingface.co"
"""Base URL that's used to download model weights"""

DEFAULT_REVISIONS = ("fp16", "main")
"""Default model revisions (Git branches)

These are used as a fallback by the macro when
`compute_model_revisions.py` is unable to get the branches dynamically
"""

DOCUMENTATION_URL = "https://www.dataiku.com/product/plugins/ai-art/"
"""URL leading to the plugin's documentation"""
