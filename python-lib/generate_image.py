import torch
from diffusers import StableDiffusionPipeline


class ImageGenerator:
    __slots__ = ("_pipe", "_device")

    def __init__(self, weights_path):
        """
        weights_path (str or path-like): Path to a local folder that
            contains the Stable Diffusion weights
        """
        # TODO: make the device configurable
        # https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
        self._device = torch.device("cuda")

        pipe = StableDiffusionPipeline.from_pretrained(weights_path)
        self._pipe = pipe.to(self._device)

    def generate_images(self, prompt, image_count):
        """Generate images based on the text prompt

        prompt (str): Text description that will be used to
            generate the images
        image_count (int): Number of images to generate

        Returns a list of images that were generated
        """
        # TODO: might need to generate the images separately so it
        # doesn't run out of VRAM when generating a lot of them
        prompts = [prompt] * image_count

        device_type = self._device.type
        with torch.autocast(device_type):
            # TODO: make the resolution and other params configurable
            # https://huggingface.co/docs/diffusers/v0.3.0/en/api/pipelines/stable_diffusion#diffusers.StableDiffusionPipeline.__call__
            output = self._pipe(prompts)

        return output.images
