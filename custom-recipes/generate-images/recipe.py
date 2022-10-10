import logging

from dataiku.customrecipe import (
    get_input_names_for_role,
    get_output_names_for_role,
    get_recipe_config,
)

from ai_art.generate_image import ImageGenerator
from ai_art.params import GenerateImagesParams

weights_folder_name = get_input_names_for_role("weights_folder")[0]
image_folder_name = get_output_names_for_role("image_folder")[0]
recipe_config = get_recipe_config()
params = GenerateImagesParams.from_config(
    recipe_config, weights_folder_name, image_folder_name
)

generator = ImageGenerator(
    params.weights_path,
    device_id=params.device_id,
    torch_dtype=params.torch_dtype,
)

if params.clear_folder:
    logging.info("Clearing image folder")
    params.image_folder.clear()

images = generator.generate_images(
    params.prompt,
    params.image_count,
    params.batch_size,
    height=params.image_height,
    width=params.image_width,
    use_autocast=params.use_autocast,
)
for i, image in enumerate(images):
    filename = f"{params.filename_prefix}{i+1}.png"

    logging.info("Saving image: %s", filename)
    with params.image_folder.get_writer(filename) as f:
        # TODO: make the format configurable
        image.save(f, format="PNG")
