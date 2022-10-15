import abc
import logging
import math

import torch
from diffusers import StableDiffusionPipeline


class _BaseImageGenerator(abc.ABC):
    """Abstract base class used by the image-generator classes"""

    __slots__ = ("_pipe", "_device")

    def __init__(
        self,
        weights_path,
        *,
        device_id=None,
        torch_dtype=None,
        enable_attention_slicing=False,
    ):
        """
        weights_path (str or path-like): Path to a local folder that
            contains the Stable Diffusion weights
        device_id (str): PyTorch device id, e.g "cuda:0". If `None`,
            the default CUDA device will be used if available; otherwise
                the CPU will be used
        torch_dtype (torch.dtype | None): Override the default
            `torch.dtype` and load the model under this dtype
        enable_attention_slicing (bool): Enable sliced attention
            computation when generating the images
        """
        self._init_device(device_id)

        logging.info("Loading weights")
        self._init_pipe(weights_path, torch_dtype)

        if enable_attention_slicing:
            self._pipe.enable_attention_slicing()

    def _init_device(self, device_id):
        """Load the PyTorch device

        If `device_id` is `None`, the device will be auto-detected based
        on whether a CUDA device is availabe
        """
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

    @abc.abstractmethod
    def _init_pipe(self, weights_path, torch_dtype):
        """Load the pipeline from the pretrained weights

        The pipeline must be assigned to the `_pipe` attribute
        """
        ...

    @abc.abstractmethod
    def generate_images(self):
        """Generate images using the pipeline

        This method must call `_generate_image_batches()`, which will
        call the pipeline

        It must accept the following kwargs and pass them to
        `_generate_image_batches()`:
            image_count, batch_size, use_autocast

        It may also accept other arguments. Any additional kwargs that
        are passed to `_generate_image_batches()` will be passed to the
        pipeline
        """
        ...

    def _generate_image_batch(self, use_autocast, **kwargs):
        """Generate a single batch of images

        All kwargs are passed to `_pipe()`

        Returns a list of images that were generated
        """
        if use_autocast:
            with torch.autocast(self._device.type):
                output = self._pipe(**kwargs)
        else:
            output = self._pipe(**kwargs)

        return output.images

    def _generate_image_batches(
        self, *, image_count, batch_size, use_autocast, **kwargs
    ):
        """Generic base method that is called by `generate_images()`

        All kwargs are passed to `_pipe()`

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
                use_autocast=use_autocast,
                num_images_per_prompt=current_batch_size,
                **kwargs,
            )

            image_processed_count += current_batch_size
            yield from images


class TextToImage(_BaseImageGenerator):
    """Generate images from a text prompt"""

    def _init_pipe(self, weights_path, torch_dtype):
        pipe = StableDiffusionPipeline.from_pretrained(
            weights_path, torch_dtype=torch_dtype
        )
        self._pipe = pipe.to(self._device)

    # TODO: Add all optimizations:
    #   https://huggingface.co/docs/diffusers/optimization/fp16
    def generate_images(
        self,
        prompt,
        image_count=1,
        batch_size=None,
        *,
        use_autocast=True,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
    ):
        """Generate images based on the text prompt

        prompt (str): Text description that will be used to generate the
            images
        image_count (int): Number of images to generate
        batch_size (int | None): Number of images to generate at once,
            or `None` to generate all images at once
        use_autocast (bool): Use `torch.autocast` when possible. Only
            available for CUDA devices
        height (int): Height (in pixels) of the images. Must be
            a multiple of 64
        width (int): Width (in pixels) of the images. Must be a multiple
            of 64
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): Guidance scale

        The height and width must be a multiple of 64 due to this issue:
            https://github.com/CompVis/stable-diffusion/issues/60

        Yields a generator of images that were generated
        """
        return self._generate_image_batches(
            prompt=prompt,
            image_count=image_count,
            batch_size=batch_size,
            use_autocast=use_autocast,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
