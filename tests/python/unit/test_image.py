from PIL import Image

from ai_art.image import _resize_image


class TestResizeImage:
    @staticmethod
    def _create_image(width, height):
        """Create a new blank image of the given size"""
        image = Image.new("RGB", (width, height))
        return image

    def test_same_size(self):
        base_image = self._create_image(512, 512)
        resized_image = _resize_image(base_image, min_size=512)

        assert resized_image.size == (512, 512)
        # Assert that the new image is a copy of the original image
        assert resized_image is not base_image

    def test_smaller_square(self):
        base_image = self._create_image(256, 256)
        resized_image = _resize_image(base_image, min_size=512)

        assert resized_image.size == (512, 512)

    def test_larger_square(self):
        base_image = self._create_image(1024, 1024)
        resized_image = _resize_image(base_image, min_size=512)

        assert resized_image.size == (512, 512)

    def test_smaller_width_same_height(self):
        base_image = self._create_image(256, 512)
        resized_image = _resize_image(base_image, min_size=512)

        assert resized_image.size == (512, 1024)

    def test_same_width_smaller_height(self):
        base_image = self._create_image(512, 256)
        resized_image = _resize_image(base_image, min_size=512)

        assert resized_image.size == (1024, 512)

    def test_smaller_width_smallest_height(self):
        base_image = self._create_image(300, 200)
        resized_image = _resize_image(base_image, min_size=512)

        assert resized_image.size == (768, 512)

    def test_smallest_width_smaller_height(self):
        base_image = self._create_image(200, 300)
        resized_image = _resize_image(base_image, min_size=512)

        assert resized_image.size == (512, 768)

    def test_round_down(self):
        """Round floats down when less than 0.5"""
        base_image = self._create_image(618, 587)
        resized_image = _resize_image(base_image, min_size=512)

        assert resized_image.size == (539, 512)

    def test_round_up(self):
        """Round floats up when greater than 0.5"""
        base_image = self._create_image(619, 587)
        resized_image = _resize_image(base_image, min_size=512)

        assert resized_image.size == (540, 512)

    def test_different_min_size(self):
        base_image = self._create_image(512, 512)
        resized_image = _resize_image(base_image, min_size=256)

        assert resized_image.size == (256, 256)
