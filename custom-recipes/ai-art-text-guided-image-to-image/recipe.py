import logging

import dataiku
from dataiku.customrecipe import (
    get_input_names_for_role,
    get_output_names_for_role,
    get_recipe_config,
)

from ai_art.folder import download_folder
from ai_art.generate_image import TextGuidedImageToImage
from ai_art.params import get_text_guided_image_to_image_config
from ai_art.save import save_images

weights_folder_name = get_input_names_for_role("weights_folder")[0]
base_image_folder_name = get_input_names_for_role("base_image_folder")[0]
image_folder_name = get_output_names_for_role("image_folder")[0]
weights_folder = dataiku.Folder(weights_folder_name)
base_image_folder = dataiku.Folder(base_image_folder_name)
image_folder = dataiku.Folder(image_folder_name)
recipe_config = get_recipe_config()

params = get_text_guided_image_to_image_config(
    recipe_config,
    weights_folder,
    image_folder,
    base_image_folder,
)
logging.info("Generated params: %r", params)

# Download the weights folder to a local temp dir so that the pipeline
# can access them.
# This is only needed if the managed folder is remote, since local
# folders can be accessed directly
if params.temp_weights_dir is not None:
    logging.info(
        "Downloading weights to local folder: %r", params.weights_path
    )
    download_folder(params.weights_folder, params.weights_path)

generator = TextGuidedImageToImage(
    params.weights_path,
    device_id=params.device_id,
    torch_dtype=params.torch_dtype,
    enable_attention_slicing=params.enable_attention_slicing,
)

if params.clear_folder:
    logging.info("Clearing image folder: %r", params.image_folder.name)
    params.image_folder.clear()

images = generator.generate_images(
    params.prompt,
    params.base_image,
    params.image_count,
    params.batch_size,
    use_autocast=params.use_autocast,
    random_seed=params.random_seed,
    strength=params.strength,
    num_inference_steps=params.num_inference_steps,
    guidance_scale=params.guidance_scale,
)

save_images(images, params.image_folder, params.filename_prefix)

if params.temp_weights_dir is not None:
    params.temp_weights_dir.cleanup()
