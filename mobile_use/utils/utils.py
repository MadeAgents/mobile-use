import logging
import math
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from typing import Tuple, Union, List
import os
import subprocess
import shutil
import sys

from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

from mobile_use.utils.constants import IMAGE_PLACEHOLDER

logger = logging.getLogger(__name__)

def encode_image_url(image: Image.Image, resize: Union[Tuple, List]=None) -> str:
    """Encode an image to base64 string.
    """
    if resize:
        image = image.copy()
        image.thumbnail(resize)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_url = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{base64_url}"

def generate_message(
    role: str,
    prompt: str,
    images: List[Image.Image] = None,
) -> List[dict]:
    """
    Generate a message with role, prompt and images.
    """
    if images is None or len(images) == 0:
        return {"role": role, "content": [{"type": "text", "text": prompt}]}
    else:
        content = []
        prompts = prompt.split(IMAGE_PLACEHOLDER)
        assert len(prompts) - 1 == len(images), "The number of images must be equal to the number of placeholders."

        for i, p in enumerate(prompts):
            if p:
                content.append({"type": "text", "text": p})
            if i < len(prompts) - 1:
                content.append({"type": "image_url","image_url": {"url": encode_image_url(images[i])}})
        
        return {"role": role, "content": content}

def show_message(messages: List[dict], name: str = None):
    name = f"{name} " if name is not None else ""
    logger.info(f"==============={name}MESSAGE==============")
    for message in messages:
        logger.info(f"ROLE: {message['role']}")
        for content in message['content']:
            if content['type'] == 'text':
                logger.info(f"TEXT: {content['text']}")
            else:
                logger.info(f"{content['type']}: SKIP.")
    logger.info(f"==============={name}MESSAGE END==============")

def contains_non_ascii(text):
    for char in text:
        if ord(char) > 127:
            return True
    return False


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28
) -> tuple[int, int]:
    """
    Implemented by Qwen2.5-VL
    More detail see: https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    """
    MAX_RATIO = 200
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def resize_image(image: Image.Image, max_pixels: int=1024*1024, min_pixels=3136):
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    if isinstance(image, str):
        image = Image.open(image)
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image

def remove_img_placeholder(messages, num_latest_screenshot=None):
    # find all image content
    img_contents = []
    for msg in messages:
        for content in msg['content']:
            if "image" in content['type']:
                img_contents.append(content)
    start_idx = 0
    if num_latest_screenshot is not None:
        start_idx = max(0, len(img_contents) - num_latest_screenshot)

    img_idx = 0
    new_msgs = []
    for msg in messages:
        role = msg['role']
        new_contents = []
        for content in msg['content']:
            if "image" in content['type']:
                continue
            text = content['text'].split(IMAGE_PLACEHOLDER)
            if len(text) == 1:
                new_contents.append(content)
            else:
                for i, t in enumerate(text):
                    if t:
                        new_contents.append({'type': 'text','text': t})
                    if i < len(text) - 1:
                        if img_idx >= len(img_contents):
                            raise ValueError("Image content not match.")
                        if img_idx >= start_idx:
                            new_contents.append(img_contents[img_idx])
                        img_idx += 1
        if len(new_contents) > 0:
            new_msgs.append({'role': role, 'content': new_contents})
    assert img_idx == len(img_contents)
    return new_msgs

def compare_image(img1: Image.Image, img2: Image.Image):
    img1 = img1.convert('L')
    img2 = img2.convert('L')
    img1 = np.array(img1)
    img2 = np.array(img2)
    ssim_value = ssim(img1, img2)
    return ssim_value

def crop_top_by_ratio(img: Image.Image, ratio: float):
    assert ratio >=0 and ratio < 1, "ratio must be in the range [0, 1)"
    if ratio == 0:
        return img
    w, h = img.size
    cut = int(round(h * ratio))
    return img.crop((0, cut, w, h))

def is_same_image(img1: Image.Image, img2: Image.Image, crop_top_ratio: float = 0.0):
    if crop_top_ratio > 0:
        img1 = crop_top_by_ratio(img1, crop_top_ratio)
        img2 = crop_top_by_ratio(img2, crop_top_ratio)
    img1 = img1.convert('L')
    img2 = img2.convert('L')
    img1 = np.array(img1)
    img2 = np.array(img2)
    return np.array_equal(img1, img2)

def diff_image(
    img1: Image.Image,
    img2: Image.Image,
    pixel_threshold: int = 5,
    area_threshold: int = 1000,
    max_boxes: int = 2,
    merge_threshold: int = 20,
):
    import cv2
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    opencv_image1 = cv2.cvtColor(img1_array, cv2.COLOR_RGB2BGR)
    opencv_image2 = cv2.cvtColor(img2_array, cv2.COLOR_RGB2BGR)

    diff = cv2.absdiff(opencv_image1, opencv_image2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(diff_gray, pixel_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((merge_threshold, merge_threshold), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"DIFF IMAGE: Number of raw contours: {len(contours)}")
    contours = [c for c in contours if cv2.contourArea(c) > area_threshold]
    logger.info(f"DIFF IMAGE: Number of filtered contours: {len(contours)}")
    if len(contours) == 0:
        logger.info("DIFF IMAGE: No contours found.")
        return None, None
    if len(contours) == 1:
        x, y, w, h = cv2.boundingRect(contours[0])
        if (x, y, w, h) == (0, 0, img2.size[0], img2.size[1]):
            logger.info("DIFF IMAGE: The two images are exactly different.")
            return None, None
    if len(contours) > max_boxes:
        logger.info(f"DIFF IMAGE: Too many contours found: {len(contours)}")
        return None, None

    new_img1 = opencv_image1.copy()
    new_img2 = opencv_image2.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x = max(0, x-3)
        y = max(0, y-3)
        w = min(w + 6, img1.size[0] - x)
        h = min(h + 6, img1.size[1] - y)
        cv2.rectangle(new_img1, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.rectangle(new_img2, (x, y), (x + w, y + h), (0, 0, 255), 3)
    new_img1 = Image.fromarray(cv2.cvtColor(new_img1, cv2.COLOR_BGR2RGB))
    new_img2 = Image.fromarray(cv2.cvtColor(new_img2, cv2.COLOR_BGR2RGB))
    return new_img1, new_img2

def draw_click_to_image(image, x, y, transparency=0.75, radius=20):
    # Convert the image to RGBA for transparency handling
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))  # Transparent overlay

    # Draw the circle on the overlay
    draw = ImageDraw.Draw(overlay)
    color = (255, 0, 0, int(255 * transparency))
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius), fill=color
    )

    # Composite the overlay with the original image
    combined = Image.alpha_composite(image, overlay)

    # Convert back to RGB for the output
    return combined.convert("RGB")

def download_hf_model(repo_url: str, target_dir: str):
    # Check if target_dir already exists
    if os.path.exists(target_dir):
        print(f"Already exists: {target_dir}, skip download.")
        return

    # Check if git-lfs is installed
    try:
        subprocess.run(["git", "lfs", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("git-lfs is not installed. It may cause some files not downloaded correctly.")

    try:
        subprocess.run(
            ["git", "clone", repo_url, target_dir],
            check=True
        )
        logger.info(f"Finished downloading model {repo_url} to {target_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading model from {repo_url}: {e}")
