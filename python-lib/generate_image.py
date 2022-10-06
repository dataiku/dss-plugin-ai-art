import logging
import math

import torch
from diffusers import StableDiffusionPipeline


class ImageGenerator:
    __slots__ = ("_pipe", "_device")

    def __init__(self, weights_path, *, device_id=None, torch_dtype=None):
        """
        weights_path (str or path-like): Path to a local folder that
            contains the Stable Diffusion weights
        device_id (str): PyTorch device id, e.g "cuda:0". If `None`,
            the default CUDA device will be used if available; otherwise
                the CPU will be used
        torch_dtype (torch.dtype | None): Override the default
            `torch.dtype` and load the model under this dtype
        """
        self._init_device(device_id)
        logging.info("Loading weights")
        pipe = StableDiffusionPipeline.from_pretrained(
            weights_path, torch_dtype=torch_dtype
        )
        self._pipe = pipe.to(self._device)

    def _init_device(self, device_id):
        # TODO: test this with multiple devices
        if device_id is None:
            # Auto-select the device
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                device_name = torch.cuda.get_device_name(self._device)
                logging.info("CUDA enabled. Device: %s", device_name)
            else:
                logging.warning("No CUDA device is available. Using the CPU")
                self._device = torch.device("cpu")
        else:
            logging.info("Using device: %s", device_id)
            self._device = torch.device(device_id)

    def _generate_image_batch(
        self, prompt, image_count, height, width, use_autocast
    ):
        """Generate a single batch of images

        Returns a list of images that were generated
        """
        # 1 image is generated per prompt
        prompts = [prompt] * image_count

        if use_autocast:
            with torch.autocast(self._device.type):
                # TODO: make the resolution and other params
                # configurable:
                # https://huggingface.co/docs/diffusers/v0.3.0/en/api/pipelines/stable_diffusion#diffusers.StableDiffusionPipeline.__call__
                output = self._pipe(prompts, height=height, width=width)
        else:
            output = self._pipe(prompts, height=height, width=width)

        return output.images

    # TODO: Add presets based on target VRAM usage.
    # See https://huggingface.co/docs/diffusers/optimization/fp16
    # TODO: Add support for Apple Metal:
    # https://huggingface.co/docs/diffusers/optimization/mps
    def generate_images(
        self,
        prompt,
        image_count,
        batch_size=None,
        *,
        height=512,
        width=512,
        use_autocast=True,
    ):
        """Generate images based on the text prompt

        prompt (str): Text description that will be used to generate the
            images
        image_count (int): Number of images to generate
        batch_size (int | None): Number of images to generate at once,
            or `None` to generate all images at once
        height (int): Height (in pixels) of the images. Must be
            a multiple of 64
        width (int): Width (in pixels) of the images. Must be a multiple
            of 64
        use_autocast (bool): Use `torch.autocast` when possible. Only
            available for CUDA devices

        The height and width must be a multiple of 64 due to this issue:
            https://github.com/CompVis/stable-diffusion/issues/60

        Yields a generator of images that were generated
        """
        image_processed_count = 0
        if batch_size:
            batch_count = math.ceil(image_count / batch_size)
        else:
            batch_count = 1

        logging.info(
            "Will generate %s total images in %s batches",
            image_count,
            batch_count,
        )

        # autocast only works with CUDA devices
        if use_autocast and (self._device.type == "cuda"):
            logging.info("autocast is enabled")
            use_autocast = True
        else:
            logging.info("autocast is disabled")
            use_autocast = False

        for i in range(batch_count):
            if image_processed_count + batch_size > image_count:
                current_batch_size = image_count - image_processed_count
            else:
                current_batch_size = batch_size

            logging.info("Generating batch %s", i + 1)
            images = self._generate_image_batch(
                prompt, current_batch_size, height, width, use_autocast
            )

            image_processed_count += current_batch_size
            yield from images
