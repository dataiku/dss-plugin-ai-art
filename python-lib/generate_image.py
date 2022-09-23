from functools import partial
import os
import random

from PIL import Image
from dalle_mini import DalleBart, DalleBartProcessor
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from vqgan_jax.modeling_flax_vqgan import VQModel
import jax
import jax.numpy as jnp
import numpy as np

# Prevent Weights & Biases from attempting to create an account when
# using a wandb model
os.environ['WANDB_MODE'] = 'disabled'


# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0, 4, 5, 6, 7))
def p_generate(
    model, tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# decode image
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(0,))
def p_decode(vqgan, indices, params):
    return vqgan.decode_code(indices, params=params)


class ImageGenerator:
    __slots__ = ('model', 'model_params', 'vqgan', 'vqgan_params', 'processor')

    # TODO: make this configurable
    # dalle-mini
    DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
    DALLE_COMMIT_ID = None
    # VQGAN model
    VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
    VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

    def __init__(self):
        # Load dalle-mini
        self.model, model_params = DalleBart.from_pretrained(
            # TODO: might need to set dtype to float16 when using a mega model
            self.DALLE_MODEL, revision=self.DALLE_COMMIT_ID, dtype=jnp.float32, _do_init=False
        )

        # Load VQGAN
        self.vqgan, vqgan_params = VQModel.from_pretrained(
            self.VQGAN_REPO, revision=self.VQGAN_COMMIT_ID, _do_init=False
        )

        self.model_params = replicate(model_params)
        self.vqgan_params = replicate(vqgan_params)

        self.processor = DalleBartProcessor.from_pretrained(self.DALLE_MODEL, revision=self.DALLE_COMMIT_ID)

    def generate_images(self, description, image_count):
        """Generate images based on the description

        description (str): Text description that will be used to
            generate the images
        image_count (int): Number of images to generate

        Returns a generator of images that were generated
        """
        prompts = [description]
        tokenized_prompts = self.processor(prompts)
        tokenized_prompt = replicate(tokenized_prompts)

        # Create a random key
        seed = random.randint(0, 2**32 - 1)
        key = jax.random.PRNGKey(seed)

        # We can customize generation parameters
        # (see https://huggingface.co/blog/how-to-generate)
        gen_top_k = None
        gen_top_p = None
        temperature = None
        cond_scale = 10.0

        for _ in range(max(image_count // jax.device_count(), 1)):
            # Get a new key
            key, subkey = jax.random.split(key)

            # Generate an image
            encoded_images = p_generate(
                self.model, tokenized_prompt, shard_prng_key(subkey), self.model_params,
                gen_top_k, gen_top_p, temperature, cond_scale
            )

            # Remove BOS
            encoded_images = encoded_images.sequences[..., 1:]

            # Decode the images
            decoded_images = p_decode(self.vqgan, encoded_images, self.vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for decoded_img in decoded_images:
                image = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                yield image
