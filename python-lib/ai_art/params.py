import logging

import torch

from dku_config import DkuConfig
from ai_art.folder import get_file_path_or_temp
from ai_art.image import open_base_image


def _cast_device_id(device_id):
    """Cast the `device_id` param to `None` if it's set to "auto"

    `_BaseImageGenerator` will auto-detect the device if it's set to
    `None`

    :param device_id: Device ID
    :type device_id: str | None

    :return: Casted device ID
    :rtype: str | None
    """
    if device_id == "auto":
        device_id = None
    return device_id


def _cast_torch_dtype(use_half_precision):
    """Get the `torch_dtype` param based on `use_half_precision`

    :param use_half_precision: Whether or not to use half-precison
        (16-bit) floats
    :type use_half_precision: bool

    :return: torch.dtype that will be used, or `None` to use the default
        torch.dtype of the model (float32)
    :rtype: torch.dtype | None
    """
    if use_half_precision:
        return torch.float16
    else:
        return None


def _cast_random_seed(random_seed):
    """Cast the `random_seed` param to an int, or set it to `None`

    The random seed will be casted to `None` if its value is `0`. This
    is needed because DSS sets "INT" params to `0` by default

    Setting the value to `None` will cause `_BaseImageGenerator` to
    generate its own random seed

    :param random_seed: Random seed
    :type random_seed: float | None

    :return: Casted random seed
    :rtype: int | None
    """
    if random_seed:
        return int(random_seed)
    else:
        return None


def _get_base_config(recipe_config, weights_folder, image_folder):
    """Create a DkuConfig instance that contains shared recipe params

    :param recipe_config: Recipe config
    :type recipe_config: Mapping[str, Any]
    :param weights_folder: Input weights_folder
    :type weights_folder: dataiku.Folder
    :param image_folder: Output image_folder
    :type image_folder: dataiku.Folder

    :return: Created DkuConfig instance
    :rtype: dku_config.DkuConfig
    """
    logging.info("Recipe config: %r", recipe_config)
    logging.info("Weights folder: %r", weights_folder.name)
    logging.info("Image folder: %r", image_folder.name)

    weights_path, temp_weights_dir = get_file_path_or_temp(weights_folder)

    config = DkuConfig()

    config.add_param(
        name="weights_folder",
        label="Weights folder",
        value=weights_folder,
        required=True,
    )
    config.add_param(name="weights_path", value=weights_path, required=True)
    config.add_param(
        name="temp_weights_dir", value=temp_weights_dir, required=False
    )
    config.add_param(
        name="image_folder",
        label="Image folder",
        value=image_folder,
        required=True,
    )

    config.add_param(
        name="prompt",
        label="Prompt",
        value=recipe_config.get("prompt"),
        required=True,
    )
    config.add_param(
        name="image_count",
        label="Image count",
        value=recipe_config.get("image_count"),
        default=1,
        cast_to=int,
        checks=(
            {
                "type": "sup_eq",
                "op": 1,
            },
        ),
    )
    config.add_param(
        name="batch_size",
        label="Batch size",
        value=recipe_config.get("batch_size"),
        default=1,
        cast_to=int,
        checks=(
            {
                "type": "sup_eq",
                "op": 1,
            },
        ),
    )
    config.add_param(
        name="filename_prefix",
        label="Filename prefix",
        value=recipe_config.get("filename_prefix"),
        default="image-",
    )
    config.add_param(
        name="device_id",
        label="CUDA device",
        value=recipe_config.get("device"),
        required=False,
        cast_to=_cast_device_id,
    )
    config.add_param(
        name="clear_folder",
        label="Clear folder",
        value=recipe_config.get("clear_folder"),
        default=True,
    )
    config.add_param(
        name="use_autocast",
        label="CUDA autocast",
        value=recipe_config.get("use_autocast"),
        default=False,
    )
    config.add_param(
        name="torch_dtype",
        label="Half precision",
        value=recipe_config.get("use_half_precision"),
        default=True,
        cast_to=_cast_torch_dtype,
    )
    config.add_param(
        name="enable_attention_slicing",
        label="Attention slicing",
        value=recipe_config.get("enable_attention_slicing"),
        default=True,
    )
    config.add_param(
        name="random_seed",
        label="Random seed",
        value=recipe_config.get("random_seed"),
        required=False,
        cast_to=_cast_random_seed,
    )
    config.add_param(
        name="num_inference_steps",
        label="Denoising steps",
        value=recipe_config.get("num_inference_steps"),
        default=50,
        cast_to=int,
        checks=(
            {
                "type": "sup_eq",
                "op": 1,
            },
        ),
    )
    config.add_param(
        name="guidance_scale",
        label="Guidance scale",
        value=recipe_config.get("guidance_scale"),
        default=7.5,
        cast_to=float,
        checks=(
            {
                "type": "sup_eq",
                "op": 0.0,
            },
        ),
    )

    return config


def get_text_to_image_config(recipe_config, weights_folder, image_folder):
    """Create a DkuConfig instance that contains the TextToImage params

    :param recipe_config: Recipe config
    :type recipe_config: Mapping[str, Any]
    :param weights_folder: Input weights_folder
    :type weights_folder: dataiku.Folder
    :param image_folder: Output image_folder
    :type image_folder: dataiku.Folder

    :return: Created DkuConfig instance
    :rtype: dku_config.DkuConfig
    """
    config = _get_base_config(recipe_config, weights_folder, image_folder)

    image_height = recipe_config.get("image_height")
    image_width = recipe_config.get("image_width")

    config.add_param(
        name="image_height",
        label="Image height",
        value=image_height,
        default=512,
        cast_to=int,
        checks=(
            {
                "type": "sup_eq",
                "op": 1,
            },
            {
                "type": "custom",
                "op": (image_height is not None) and (image_height % 64 == 0),
                "err_msg": (
                    "Should be a multiple of 64 "
                    f"(Currently {image_height!r})."
                ),
            },
        ),
    )
    config.add_param(
        name="image_width",
        label="Image width",
        value=image_width,
        default=512,
        cast_to=int,
        checks=(
            {
                "type": "sup_eq",
                "op": 1,
            },
            {
                "type": "custom",
                "op": (image_width is not None) and (image_width % 64 == 0),
                "err_msg": (
                    f"Should be a multiple of 64 (Currently {image_width!r})."
                ),
            },
        ),
    )

    return config


def get_text_guided_image_to_image_config(
    recipe_config, weights_folder, image_folder, base_image_folder
):
    """Create a DkuConfig instance that contains the
    TextGuidedImageToImage params

    :param recipe_config: Recipe config
    :type recipe_config: Mapping[str, Any]
    :param weights_folder: Input weights_folder
    :type weights_folder: dataiku.Folder
    :param image_folder: Output image_folder
    :type image_folder: dataiku.Folder
    :param base_image_folder: Input base_image_folder
    :type base_image_folder: dataiku.Folder

    :return: Created DkuConfig instance
    :rtype: dku_config.DkuConfig
    """
    config = _get_base_config(recipe_config, weights_folder, image_folder)

    logging.info("Base image folder: %r", base_image_folder.name)

    config.add_param(
        name="base_image_path",
        label="Base image",
        value=recipe_config.get("base_image_path"),
        required=True,
    )
    config.add_param(
        name="resize_base_image",
        label="Resize images",
        value=recipe_config.get("resize_base_image"),
        default=True,
    )
    config.add_param(
        name="resize_base_image_to",
        label="Image size",
        value=recipe_config.get("resize_base_image_to"),
        default=512,
        cast_to=int,
        checks=(
            {
                "type": "in",
                "op": frozenset((512, 768)),
            },
        ),
    )
    config.add_param(
        name="strength",
        label="Strength",
        value=recipe_config.get("strength"),
        default=0.8,
        cast_to=float,
        checks=(
            {
                "type": "between",
                "op": (0.0, 1.0),
            },
        ),
    )

    if config.resize_base_image:
        resize_to = config.resize_base_image_to
    else:
        resize_to = None

    logging.info("Opening base image: %r", config.base_image_path)
    base_image = open_base_image(
        folder=base_image_folder,
        image_path=config.base_image_path,
        resize_to=resize_to,
    )
    config.add_param(name="base_image", value=base_image, required=True)

    return config
