import logging

from dataiku.customrecipe import *
import dataiku

from generate_image import ImageGenerator

input_dataset_name = get_input_names_for_role('input_dataset')[0]
output_folder_name = get_output_names_for_role('output_folder')[0]

input_dataset = dataiku.Dataset(input_dataset_name)
output_folder = dataiku.Folder(output_folder_name)

input_dataset_df = input_dataset.get_dataframe()

recipe_config = get_recipe_config()
text_column = recipe_config['text_column']
filename_column = recipe_config.get('filename_column')
image_count = recipe_config['image_count']

# TODO: handle rows with empty values
for row_index, row in input_dataset_df.iterrows():
    description = str(row[text_column])

    if filename_column:
        base_filename = str(row[filename_column])
    else:
        base_filename = str(row_index)

    logging.info('Processing row: %s', base_filename)

    generator = ImageGenerator()
    images = generator.generate_images(description, image_count)

    for image_index, image in enumerate(images):
        filename = f'{base_filename}_{image_index}.png'
        logging.info('Saving image: %s', filename)
        with output_folder.get_writer(filename) as f:
            # TODO: make the format configurable
            image.save(f, format='PNG')
