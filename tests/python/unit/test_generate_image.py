import unittest.mock

import pytest
from PIL import Image

from ai_art.generate_image import TextGuidedImageToImage, TextToImage


def _exhaust(generator):
    """Exhaust the given generator without storing its output"""
    for _ in generator:
        pass


class TestTextToImage:
    @pytest.fixture(autouse=True)
    def setup_generator(self, mocker):
        """Create a generator instance with a mocked pipeline"""
        self.from_pretrained = mocker.patch(
            "diffusers.StableDiffusionPipeline.from_pretrained"
        )
        self.generator = TextToImage("/path/to/weights")
        self.pipe = self.from_pretrained.return_value.to.return_value

    def _assert_batches(self, expected_batch_sizes):
        """Assert that the pipe was called in the correct batches

        For example, if `image_count` is `5` and `batch_size` is `2`,
        the pipe should be called in the following batches::

            (2, 2, 1)

        :param expected_batch_sizes: Expected batch sizes
        :type expected_batch_sizes: tuple[int, ...]
        """
        assert self.pipe.call_count == len(expected_batch_sizes)

        actual_batch_sizes = tuple(
            call.kwargs["num_images_per_prompt"]
            for call in self.pipe.call_args_list
        )
        assert expected_batch_sizes == actual_batch_sizes

    def test_from_pretrained_called(self):
        self.from_pretrained.assert_called_once()

    def test_generate_images_default_args(self, mocker):
        _exhaust(self.generator.generate_images("PROMPT"))
        self.pipe.assert_called_once_with(
            num_images_per_prompt=1,
            generator=None,
            prompt="PROMPT",
            height=512,
            width=512,
            num_inference_steps=50,
            guidance_scale=7.5,
        )

    def test_generate_images_all_args(self, mocker):
        _exhaust(
            self.generator.generate_images(
                prompt="PROMPT",
                image_count=2,
                batch_size=4,
                use_autocast=False,
                random_seed=999999999,
                height=256,
                width=768,
                num_inference_steps=30,
                guidance_scale=6.0,
            )
        )
        self.pipe.assert_called_once_with(
            num_images_per_prompt=2,
            # The random generator is tested separately
            generator=unittest.mock.ANY,
            prompt="PROMPT",
            height=256,
            width=768,
            num_inference_steps=30,
            guidance_scale=6.0,
        )

    def test_generate_images_random_seed(self, mocker):
        """Assert that a random generator is created with the given seed"""
        random_seed = 999999999
        _exhaust(
            self.generator.generate_images("PROMPT", random_seed=random_seed)
        )
        self.pipe.assert_called_once()

        random_generator = self.pipe.call_args.kwargs["generator"]
        assert random_generator.initial_seed() == random_seed

    def test_generate_images_batch(self, mocker):
        _exhaust(
            self.generator.generate_images(
                "PROMPT", image_count=6, batch_size=2
            )
        )
        self._assert_batches((2, 2, 2))

    def test_generate_images_batch_remainder(self, mocker):
        _exhaust(
            self.generator.generate_images(
                "PROMPT", image_count=5, batch_size=2
            )
        )
        self._assert_batches((2, 2, 1))

    def test_generate_images_batch_one(self, mocker):
        _exhaust(
            self.generator.generate_images(
                "PROMPT", image_count=6, batch_size=1
            )
        )
        self._assert_batches((1, 1, 1, 1, 1, 1))

    def test_generate_images_batch_none(self, mocker):
        _exhaust(
            self.generator.generate_images(
                "PROMPT", image_count=6, batch_size=None
            )
        )
        self._assert_batches((6,))

    def test_generate_images_batch_large(self, mocker):
        _exhaust(
            self.generator.generate_images(
                "PROMPT", image_count=360, batch_size=65
            )
        )
        self._assert_batches((65, 65, 65, 65, 65, 35))


class TestTextGuidedImageToImage:
    """
    TextGuidedImageToImage has the same parent class as TextToImage,
    so we only need to test the parts that are different
    """

    @pytest.fixture(autouse=True)
    def setup_generator(self, mocker):
        """Create a generator instance with a mocked pipeline"""
        self.from_pretrained = mocker.patch(
            "diffusers.StableDiffusionImg2ImgPipeline.from_pretrained"
        )
        self.generator = TextGuidedImageToImage("/path/to/weights")
        self.pipe = self.from_pretrained.return_value.to.return_value

    @pytest.fixture
    def image(self):
        """Create a blank image for testing"""
        return Image.new("RGB", (512, 512))

    def test_from_pretrained_called(self):
        self.from_pretrained.assert_called_once()

    def test_generate_images_default_args(self, mocker, image):
        _exhaust(self.generator.generate_images("PROMPT", image))
        self.pipe.assert_called_once_with(
            init_image=image,
            num_images_per_prompt=1,
            generator=None,
            prompt="PROMPT",
            strength=0.8,
            num_inference_steps=50,
            guidance_scale=7.5,
        )

    def test_generate_images_all_args(self, mocker, image):
        _exhaust(
            self.generator.generate_images(
                prompt="PROMPT",
                init_image=image,
                image_count=2,
                batch_size=4,
                use_autocast=False,
                random_seed=999999999,
                strength=0.6,
                num_inference_steps=30,
                guidance_scale=6.0,
            )
        )
        self.pipe.assert_called_once_with(
            num_images_per_prompt=2,
            generator=unittest.mock.ANY,
            prompt="PROMPT",
            init_image=image,
            strength=0.6,
            num_inference_steps=30,
            guidance_scale=6.0,
        )
