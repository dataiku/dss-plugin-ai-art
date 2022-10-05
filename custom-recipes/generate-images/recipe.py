import logging

import dataiku
from dataiku.customrecipe import (
    get_input_names_for_role,
    get_output_names_for_role,
    get_recipe_config,
)

from generate_image import ImageGenerator

weights_folder_name = get_input_names_for_role("weights_folder")[0]
image_folder_name = get_output_names_for_role("image_folder")[0]
weights_folder = dataiku.Folder(weights_folder_name)
image_folder = dataiku.Folder(image_folder_name)

recipe_config = get_recipe_config()
prompt = recipe_config["prompt"]
image_count = recipe_config["image_count"]
batch_size = recipe_config["batch_size"]
filename_prefix = recipe_config["filename_prefix"]
device = recipe_config["device"]
clear_folder = recipe_config["clear_folder"]
image_height = recipe_config["image_height"]
image_width = recipe_config["image_width"]

# TODO: figure out if there's a better way to store the weights. The
# current method only works if the folder is on the local FS.
# https://huggingface.co/docs/diffusers/v0.3.0/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained
weights_path = weights_folder.get_path()
if device == "auto":
    device_id = None
else:
    device_id = device

generator = ImageGenerator(weights_path, device_id)

if clear_folder:
    logging.info("Clearing image folder")
    image_folder.clear()

images = generator.generate_images(
    prompt, image_count, batch_size, height=image_height, width=image_width
)
for i, image in enumerate(images):
    filename = f"{filename_prefix}{i+1}.png"

    logging.info("Saving image: %s", filename)
    with image_folder.get_writer(filename) as f:
        # TODO: make the format configurable
        image.save(f, format="PNG")
