import logging


def save_images(images, folder, filename_prefix):
    """Save images to a folder

    :param images: Images that will be saved
    :type images: Iterable[PIL.Image.Image]
    :param folder: Folder that the images will be saved to
    :type folder: dataiku.Folder
    :param filename_prefix: Images are named sequentially based on this,
        e.g. 'prefix1.png', 'prefix2.png'
    :type filename_prefix: str

    :return: None
    """
    for i, image in enumerate(images):
        filename = f"{filename_prefix}{i+1}.png"

        logging.info("Saving image: %s", filename)
        with folder.get_writer(filename) as f:
            image.save(f, format="PNG")
