"""
Tests for mobile_use/utils/utils.py

Tests the utility functions including image encoding, message generation,
smart resize, and other helper functions.
"""

import pytest
import base64
from PIL import Image
import numpy as np

from mobile_use.utils.utils import (
    encode_image_url,
    generate_message,
    show_message,
    contains_non_ascii,
    contains_chinese,
    smart_resize,
    resize_image,
    remove_img_placeholder,
    compare_image,
    crop_top_by_ratio,
    is_same_image,
    draw_click_to_image,
)
from mobile_use.utils.constants import IMAGE_PLACEHOLDER


class TestEncodeImageUrl:
    """Tests for encode_image_url function."""

    def test_encode_basic_image(self, mock_image):
        """Test encoding a basic image to base64 URL."""
        url = encode_image_url(mock_image)
        assert url.startswith("data:image/png;base64,")
        # Verify it's valid base64
        base64_part = url.replace("data:image/png;base64,", "")
        decoded = base64.b64decode(base64_part)
        assert len(decoded) > 0

    def test_encode_with_resize(self, mock_image):
        """Test encoding with resize parameter."""
        url = encode_image_url(mock_image, resize=(100, 100))
        assert url.startswith("data:image/png;base64,")

    def test_encode_small_image(self, mock_image_small):
        """Test encoding a small image."""
        url = encode_image_url(mock_image_small)
        assert url.startswith("data:image/png;base64,")


class TestGenerateMessage:
    """Tests for generate_message function."""

    def test_text_only_message(self):
        """Test generating a text-only message."""
        message = generate_message("user", "Hello, how are you?")
        assert message["role"] == "user"
        assert len(message["content"]) == 1
        assert message["content"][0]["type"] == "text"
        assert message["content"][0]["text"] == "Hello, how are you?"

    def test_system_message(self):
        """Test generating a system message."""
        message = generate_message("system", "You are a helpful assistant.")
        assert message["role"] == "system"
        assert message["content"][0]["text"] == "You are a helpful assistant."

    def test_message_with_single_image(self, mock_image):
        """Test generating a message with a single image."""
        prompt = f"Describe this image: {IMAGE_PLACEHOLDER}"
        message = generate_message("user", prompt, images=[mock_image])
        
        assert message["role"] == "user"
        assert len(message["content"]) == 2
        # First should be text
        assert message["content"][0]["type"] == "text"
        assert "Describe this image:" in message["content"][0]["text"]
        # Second should be image
        assert message["content"][1]["type"] == "image_url"

    def test_message_with_multiple_images(self, mock_image, mock_image_small):
        """Test generating a message with multiple images."""
        prompt = f"Image 1: {IMAGE_PLACEHOLDER} Image 2: {IMAGE_PLACEHOLDER}"
        message = generate_message("user", prompt, images=[mock_image, mock_image_small])
        
        # Should have: text, image, text, image
        assert len(message["content"]) == 4
        assert message["content"][0]["type"] == "text"
        assert message["content"][1]["type"] == "image_url"
        assert message["content"][2]["type"] == "text"
        assert message["content"][3]["type"] == "image_url"

    def test_message_placeholder_count_mismatch_raises(self, mock_image):
        """Test that mismatched placeholder count raises error."""
        prompt = f"{IMAGE_PLACEHOLDER} {IMAGE_PLACEHOLDER}"  # 2 placeholders
        with pytest.raises(AssertionError):
            generate_message("user", prompt, images=[mock_image])  # Only 1 image


class TestContainsNonAscii:
    """Tests for contains_non_ascii function."""

    def test_ascii_only(self):
        """Test string with only ASCII characters."""
        assert contains_non_ascii("Hello World!") is False

    def test_with_non_ascii(self):
        """Test string with non-ASCII characters."""
        assert contains_non_ascii("你好世界") is True
        assert contains_non_ascii("Héllo") is True
        assert contains_non_ascii("日本語") is True

    def test_mixed(self):
        """Test string with mixed characters."""
        assert contains_non_ascii("Hello 世界") is True

    def test_empty_string(self):
        """Test empty string."""
        assert contains_non_ascii("") is False


class TestContainsChinese:
    """Tests for contains_chinese function."""

    def test_no_chinese(self):
        """Test string without Chinese characters."""
        assert contains_chinese("Hello World!") is False
        assert contains_chinese("Héllo") is False

    def test_with_chinese(self):
        """Test string with Chinese characters."""
        assert contains_chinese("你好") is True
        assert contains_chinese("世界") is True

    def test_mixed(self):
        """Test string with mixed characters."""
        assert contains_chinese("Hello 世界") is True

    def test_japanese_kanji(self):
        """Test that common Chinese/Japanese kanji is detected."""
        # Kanji in the CJK Unified Ideographs range
        assert contains_chinese("日本") is True

    def test_empty_string(self):
        """Test empty string."""
        assert contains_chinese("") is False


class TestSmartResize:
    """Tests for smart_resize function."""

    def test_basic_resize(self):
        """Test basic resize calculation."""
        height, width = smart_resize(1920, 1080)
        assert height > 0
        assert width > 0
        # Result should be divisible by factor (28)
        assert height % 28 == 0
        assert width % 28 == 0

    def test_small_image_resize(self):
        """Test resize for small image below min_pixels."""
        height, width = smart_resize(50, 50, min_pixels=3136)
        assert height * width >= 3136

    def test_large_image_resize(self):
        """Test resize for large image above max_pixels."""
        height, width = smart_resize(10000, 10000, max_pixels=1000000)
        assert height * width <= 1000000

    def test_custom_factor(self):
        """Test resize with custom factor."""
        height, width = smart_resize(100, 100, factor=14)
        assert height % 14 == 0
        assert width % 14 == 0

    def test_extreme_aspect_ratio_raises(self):
        """Test that extreme aspect ratio raises ValueError."""
        with pytest.raises(ValueError, match="aspect ratio"):
            smart_resize(10000, 1)  # Very extreme ratio


class TestResizeImage:
    """Tests for resize_image function."""

    def test_resize_large_image(self, mock_image):
        """Test resizing a large image."""
        # Create a large image
        large_image = Image.new('RGB', (2000, 2000), color='red')
        resized = resize_image(large_image, max_pixels=1000000)
        assert resized.width * resized.height <= 1000000

    def test_resize_small_image(self):
        """Test resizing a small image to min_pixels."""
        small_image = Image.new('RGB', (10, 10), color='green')
        resized = resize_image(small_image, min_pixels=3136)
        assert resized.width * resized.height >= 3136

    def test_convert_to_rgb(self):
        """Test that image is converted to RGB."""
        # Create an RGBA image
        rgba_image = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        resized = resize_image(rgba_image)
        assert resized.mode == "RGB"


class TestCompareImage:
    """Tests for compare_image function."""

    def test_identical_images(self, mock_image):
        """Test comparing identical images."""
        similarity = compare_image(mock_image, mock_image)
        assert similarity == 1.0

    def test_different_images(self):
        """Test comparing different images."""
        img1 = Image.new('RGB', (100, 100), color='white')
        img2 = Image.new('RGB', (100, 100), color='black')
        similarity = compare_image(img1, img2)
        assert similarity < 1.0


class TestCropTopByRatio:
    """Tests for crop_top_by_ratio function."""

    def test_no_crop(self, mock_image):
        """Test with ratio=0 (no crop)."""
        result = crop_top_by_ratio(mock_image, 0)
        assert result.size == mock_image.size

    def test_crop_half(self):
        """Test cropping half from top."""
        img = Image.new('RGB', (100, 200), color='red')
        result = crop_top_by_ratio(img, 0.5)
        assert result.height == 100
        assert result.width == 100

    def test_crop_small_ratio(self):
        """Test cropping with small ratio."""
        img = Image.new('RGB', (100, 100), color='blue')
        result = crop_top_by_ratio(img, 0.1)
        assert result.height == 90

    def test_invalid_ratio_raises(self):
        """Test that invalid ratio raises assertion."""
        img = Image.new('RGB', (100, 100))
        with pytest.raises(AssertionError):
            crop_top_by_ratio(img, 1.0)
        with pytest.raises(AssertionError):
            crop_top_by_ratio(img, -0.1)


class TestIsSameImage:
    """Tests for is_same_image function."""

    def test_identical_images(self, mock_image):
        """Test that identical images are detected as same."""
        assert is_same_image(mock_image, mock_image) is True

    def test_different_images(self):
        """Test that different images are detected as different."""
        img1 = Image.new('RGB', (100, 100), color='white')
        img2 = Image.new('RGB', (100, 100), color='black')
        assert is_same_image(img1, img2) is False

    def test_with_crop(self):
        """Test with crop_top_ratio."""
        # Create image with different top and bottom halves
        img = Image.new('RGB', (100, 100), color='white')
        # Same image should still be same after consistent cropping
        assert is_same_image(img, img, crop_top_ratio=0.2) is True


class TestDrawClickToImage:
    """Tests for draw_click_to_image function."""

    def test_draw_click(self, mock_image):
        """Test drawing a click point on image."""
        result = draw_click_to_image(mock_image, 540, 960)
        assert result.mode == "RGB"
        assert result.size == mock_image.size

    def test_draw_click_custom_params(self, mock_image):
        """Test drawing with custom transparency and radius."""
        result = draw_click_to_image(
            mock_image, 
            100, 100, 
            transparency=0.5, 
            radius=30
        )
        assert result is not None


class TestRemoveImgPlaceholder:
    """Tests for remove_img_placeholder function."""

    def test_remove_all_placeholders(self, mock_image):
        """Test removing image placeholders from messages."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": f"Look at this: {IMAGE_PLACEHOLDER}"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]}
        ]
        result = remove_img_placeholder(messages, num_latest_screenshot=0)
        # With 0 screenshots, all images should be removed
        for msg in result:
            for content in msg["content"]:
                assert "image" not in content.get("type", "")

    def test_keep_latest_screenshots(self, mock_image):
        """Test keeping only latest screenshots."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": f"Image: {IMAGE_PLACEHOLDER}"},
                {"type": "image_url", "image_url": {"url": "url1"}}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": f"Image: {IMAGE_PLACEHOLDER}"},
                {"type": "image_url", "image_url": {"url": "url2"}}
            ]}
        ]
        result = remove_img_placeholder(messages, num_latest_screenshot=1)
        # Count remaining images
        image_count = 0
        for msg in result:
            for content in msg["content"]:
                if "image" in content.get("type", ""):
                    image_count += 1
        assert image_count == 1


class TestShowMessage:
    """Tests for show_message function."""

    def test_show_message_text_only(self, caplog):
        """Test showing text-only messages."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]
        import logging
        with caplog.at_level(logging.INFO):
            show_message(messages, "Test")
        # Just verify it doesn't raise an error
        assert True

    def test_show_message_with_name(self, caplog):
        """Test showing messages with custom name."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "System prompt"}]}
        ]
        import logging
        with caplog.at_level(logging.INFO):
            show_message(messages, "CustomAgent")
        assert True

