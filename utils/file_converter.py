import cv2
import numpy as np
from pdf2image import convert_from_path
from utils import get_file_name
import os
from PIL import Image, ImageEnhance, ImageFilter


SUPPORTED_IMAGE_TYPES = ['png', 'jpeg']


def pdf2image(
        pdf_path: str,
        output_path: str = None,
        image_type: str = 'png',
        return_images: bool = True
) -> list[Image.Image] | list[str]:
    """
    Convert a PDF file to images.

    :param pdf_path: original PDF file path
    :param output_path: output image path (if return_images is False, the images will be saved to this path)
    :param image_type: image type, 'png' or 'jpeg'
    :param return_images: if True, return the images, else save the images to the output path
    :return: list of images / list of image paths
    """

    def enhance_and_dilate(_image):
        # Convert to grayscale image
        gray_image = cv2.cvtColor(np.array(_image), cv2.COLOR_BGR2GRAY)

        # Binarize the image
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

        # Apply dilation to make black areas thicker
        kernel = np.ones((2, 2), np.uint8)
        dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

        # Invert the binary image
        dilated_image = cv2.bitwise_not(dilated_image)

        # Convert back to PIL image
        return Image.fromarray(dilated_image)

    # Check image_type
    if image_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError(f"Unsupported image type: {image_type}, supported types are 'png' and 'jpeg'")

    # Convert pdf pages to images
    raw_images = convert_from_path(pdf_path, dpi=200, fmt=image_type)

    # Process images
    images = []
    for image in raw_images:
        # Raise up contrast(对比度)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(3)

        # Raise up brightness(亮度)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.5)

        # Sharpen the image(锐化)
        image = image.filter(ImageFilter.SHARPEN)

        # Enhance and dilate the image(增强并膨胀)
        image = enhance_and_dilate(image)
        images.append(image)

    # if `return_images` is True, return the images
    if return_images:
        return images

    # else save the images to the output path, and return the image paths
    # Check the validity of `output_path`
    if output_path is None:
        output_path = os.path.join(os.getcwd(), 'output_pdf_images')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_paths = []
    base_file_name = get_file_name(pdf_path, with_extension=False)
    for page_idx, image in enumerate(images):
        cur_image_path = os.path.join(output_path, f'{base_file_name}_p{page_idx + 1}.{image_type}')
        cur_image_path = os.path.abspath(cur_image_path)
        image.save(cur_image_path, format=image_type)
        image_paths.append(cur_image_path)

    return image_paths
