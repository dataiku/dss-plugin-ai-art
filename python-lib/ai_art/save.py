from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable

    import dataiku
    from PIL.Image import Image


def save_images(
    images: Iterable[Image], folder: dataiku.Folder, filename_prefix: str
) -> None:
    """Save images to a folder

    The images are named sequentially based on `filename_prefix`,
    e.g. 'prefix1.png', 'prefix2.png'
    """
    for i, image in enumerate(images):
        filename = f"{filename_prefix}{i+1}.png"

        logging.info("Saving image: %s", filename)
        with folder.get_writer(filename) as f:
            image.save(f, format="PNG")
