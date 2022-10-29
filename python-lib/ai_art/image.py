import enum
import logging

from PIL import Image


class _Dimension(enum.IntEnum):
    """Enum that corresponds to the dimension's index in `Image.size`"""

    WIDTH = 0
    HEIGHT = 1


def open_base_image(folder, image_path, resize):
    """Open image from a Dataiku folder to use with Stable Diffusion

    :param folder: Folder that contains the image
    :type folder: dataiku.Folder
    :param image_path: Filepath of the image within `folder`
    :type image_path: str
    :param resize: Whether or not to resize the image
    :type resize: bool

    :return: Image that was opened
    :rtype: PIL.Image.Image
    """
    with folder.get_download_stream(image_path) as file:
        image = Image.open(file)

    # Convert the image to RGB per the Diffusers documentation
    # https://huggingface.co/docs/diffusers/v0.6.0/using-diffusers/img2img
    if image.mode != "RGB":
        image = image.convert("RGB")

    if resize:
        image = _resize_image(image, min_size=512)

    return image


def _resize_image(image, min_size):
    """Resize the image so that the shorter dimension equals `min_size`

    :param image: Image to resize
    :type image: PIL.Image.Image
    :param min_size: The size that you want the shorter dimension of the
        image to be resized to
    :type min_size: int

    The aspect ratio of the image is maintained

    :return: Resized image
    :rtype: PIL.Image.Image
    """
    # `base_dimension` is the shorter dimension that will be resized to
    # `min_size`
    if image.width < image.height:
        base_dimension = _Dimension.WIDTH
        larger_dimension = _Dimension.HEIGHT
    else:
        base_dimension = _Dimension.HEIGHT
        larger_dimension = _Dimension.WIDTH

    current_base_size = image.size[base_dimension]
    current_larger_size = image.size[larger_dimension]

    if current_base_size == min_size:
        # Return an unchanged copy of the image if it's already the
        # right size.
        # A copy is returned so that it matches the behavior of
        # `Image.resize()`, which also creates a copy
        return image.copy()

    resized_larger_size = round(
        current_larger_size / current_base_size * min_size
    )

    if base_dimension is _Dimension.WIDTH:
        new_size = (min_size, resized_larger_size)
    else:
        new_size = (resized_larger_size, min_size)

    logging.info("Resizing base image from %r to %r", image.size, new_size)
    resized_image = image.resize(new_size, resample=Image.Resampling.LANCZOS)
    return resized_image
