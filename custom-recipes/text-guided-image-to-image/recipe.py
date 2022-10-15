import logging

from dataiku.customrecipe import (
    get_input_names_for_role,
    get_output_names_for_role,
    get_recipe_config,
)

from ai_art.generate_image import TextGuidedImageToImage
from ai_art.params import TextGuidedImageToImageParams
from ai_art.save import save_images

weights_folder_name = get_input_names_for_role("weights_folder")[0]
base_image_folder_name = get_input_names_for_role("base_image_folder")[0]
image_folder_name = get_output_names_for_role("image_folder")[0]
recipe_config = get_recipe_config()
params = TextGuidedImageToImageParams.from_config(
    recipe_config,
    weights_folder_name,
    image_folder_name,
    base_image_folder_name,
)
logging.info("Generated params: %r", params)

generator = TextGuidedImageToImage(
    params.weights_path,
    device_id=params.device_id,
    torch_dtype=params.torch_dtype,
    enable_attention_slicing=params.enable_attention_slicing,
)

if params.clear_folder:
    logging.info("Clearing image folder")
    params.image_folder.clear()

images = generator.generate_images(
    params.prompt,
    params.base_image,
    params.image_count,
    params.batch_size,
    use_autocast=params.use_autocast,
    strength=params.strength,
    num_inference_steps=params.num_inference_steps,
    guidance_scale=params.guidance_scale,
)

save_images(images, params.image_folder, params.filename_prefix)
