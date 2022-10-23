import abc
import logging
import math

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline


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
        :param weights_path: Path to a local folder that contains the
            Stable Diffusion weights
        :type weights_path: str | os.PathLike
        :param device_id: PyTorch device id, e.g "cuda:0". If `None`,
            the default CUDA device will be used if available; otherwise
            the CPU will be used
        :type device_id: str | None
        :param torch_dtype: Override the default `torch.dtype` and load
            the model under this dtype
        :type torch_dtype: torch.dtype | None
        :param enable_attention_slicing: Enable sliced attention
            computation when generating the images
        :type enable_attention_slicing: bool

        :return: None
        """
        self._init_device(device_id)

        if torch_dtype is torch.float16 and self._device.type == "cpu":
            # Running the pipeline will fail if half precison is enabled
            # when using the CPU
            logging.warning(
                "Half precision isn't supported when running on the CPU. "
                "Using full precision instead"
            )
            torch_dtype = None

        logging.info("Loading weights")
        self._init_pipe(weights_path, torch_dtype)

        if enable_attention_slicing:
            self._pipe.enable_attention_slicing()

    def _init_device(self, device_id):
        """Load the PyTorch device

        :param device_id: PyTorch device id, e.g "cuda:0". If `None`,
            the default CUDA device will be used if available; otherwise
            the CPU will be used
        :type device_id: str | None

        :return: None
        """
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

        :param weights_path: Path to a local folder that contains the
            Stable Diffusion weights
        :type weights_path: str | os.PathLike
        :param torch_dtype: Override the default `torch.dtype` and load
            the model under this dtype
        :type torch_dtype: torch.dtype | None

        :return: None
        """
        ...

    @abc.abstractmethod
    def generate_images(self):
        """Generate images using the pipeline

        This method must call `_generate_image_batches()`, which will
        call the pipeline

        It must accept the following kwargs and pass them to
        `_generate_image_batches()`:
            image_count, batch_size, use_autocast, random_seed

        It may also accept other arguments. Any additional kwargs that
        are passed to `_generate_image_batches()` will be passed to the
        pipeline
        """
        ...

    def _generate_image_batch(self, use_autocast, **kwargs):
        """Generate a single batch of images

        :param use_autocast: Use `torch.autocast` when possible. Only
            available for CUDA devices
        :type use_autocast: bool
        :param kwargs: kwargs to pass to `_pipe()`
        :type kwargs: Any

        :return: List of images that were generated
        :rtype: list[PIL.Image.Image]
        """
        if use_autocast:
            with torch.autocast(self._device.type):
                output = self._pipe(**kwargs)
        else:
            output = self._pipe(**kwargs)

        return output.images

    def _generate_image_batches(
        self, *, image_count, batch_size, use_autocast, random_seed, **kwargs
    ):
        """Generic base method that is called by `generate_images()`

        :param image_count: Number of images to generate
        :type image_count: int
        :param batch_size: Number of images to generate at once, or
            `None` to generate all images at once
        :type batch_size: int | None
        :param use_autocast: Use `torch.autocast` when possible. Only
            available for CUDA devices
        :type use_autocast: bool
        :param random_seed: Random seed that's used to generate the
            images
        :type random_seed: int | None
        :param kwargs: kwargs to pass to `_pipe()`
        :type kwargs: Any

        :return: Generator of images that were generated
        :rtype: Generator[PIL.Image.Image, None, None]
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

        if random_seed is None:
            # Set the generator to `None` so that PyTorch generates a
            # random seed for us
            generator = None
        else:
            generator = torch.Generator(self._device)
            generator = generator.manual_seed(random_seed)

        for i in range(batch_count):
            if image_processed_count + batch_size > image_count:
                current_batch_size = image_count - image_processed_count
            else:
                current_batch_size = batch_size

            logging.info("Generating batch %s", i + 1)
            images = self._generate_image_batch(
                use_autocast=use_autocast,
                num_images_per_prompt=current_batch_size,
                generator=generator,
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

    def generate_images(
        self,
        prompt,
        image_count=1,
        batch_size=None,
        *,
        use_autocast=True,
        random_seed=None,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
    ):
        """Generate images based on the text prompt

        :param prompt: Text description that will be used to generate
            the images
        :type prompt: str
        :param image_count: Number of images to generate
        :type image_count: int
        :param batch_size: Number of images to generate at once, or
            `None` to generate all images at once
        :type batch_size: int | None
        :param use_autocast: Use `torch.autocast` when possible. Only
            available for CUDA devices
        :type use_autocast: bool
        :param random_seed: Random seed that's used to generate the
            images
        :type random_seed: int | None
        :param height: Height (in pixels) of the images. Must be a
            multiple of 64
        :type height: int
        :param width: Width (in pixels) of the images. Must be a
            multiple of 64
        :type width: int
        :param num_inference_steps: Number of denoising steps
        :type num_inference_steps: int
        :param guidance_scale: Guidance scale
        :type guidance_scale: float

        The height and width must be a multiple of 64 due to this issue:
            https://github.com/CompVis/stable-diffusion/issues/60

        :return: Generator of images that were generated
        :rtype: Generator[PIL.Image.Image, None, None]
        """
        yield from self._generate_image_batches(
            prompt=prompt,
            image_count=image_count,
            batch_size=batch_size,
            use_autocast=use_autocast,
            random_seed=random_seed,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )


class TextGuidedImageToImage(_BaseImageGenerator):
    """Generate images from a base image, guided by a text prompt"""

    def _init_pipe(self, weights_path, torch_dtype):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            weights_path, torch_dtype=torch_dtype
        )
        self._pipe = pipe.to(self._device)

    def generate_images(
        self,
        prompt,
        init_image,
        image_count=1,
        batch_size=None,
        *,
        use_autocast=True,
        random_seed=None,
        strength=0.8,
        num_inference_steps=50,
        guidance_scale=7.5,
    ):
        """Generate images based on the init image, guided by the prompt

        :param prompt: Text description that will be used to generate
            the images
        :type prompt: str
        :param init_image: Base image that the output images will be
            based on
        :type init_image: PIL.Image.Image
        :param image_count: Number of images to generate
        :type image_count: int
        :param batch_size: Number of images to generate at once, or
            `None` to generate all images at once
        :type batch_size: int | None
        :param use_autocast: Use `torch.autocast` when possible. Only
            available for CUDA devices
        :type use_autocast: bool
        :param random_seed: Random seed that's used to generate the
            images
        :type random_seed: int | None
        :param strength: Indicates how much to transform `init_image`
        :type strength: float
        :param num_inference_steps: Number of denoising steps
        :type num_inference_steps: int
        :param guidance_scale: Guidance scale
        :type guidance_scale: float

        :return: Generator of images that were generated
        :rtype: Generator[PIL.Image.Image, None, None]
        """
        yield from self._generate_image_batches(
            prompt=prompt,
            init_image=init_image,
            image_count=image_count,
            batch_size=batch_size,
            use_autocast=use_autocast,
            random_seed=random_seed,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
