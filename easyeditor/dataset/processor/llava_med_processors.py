import cv2
import numpy as np
from PIL import Image
from typing import Union, List
import torch
from transformers import CLIPImageProcessor


class LLaVAMedProcessor:
    def __init__(self):
        # LLaVA-Med 通常使用 CLIP 的图像处理器
        # 根据原始 LLaVA 的设置，使用 336x336 的图像尺寸
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14-336"
        )

    def __call__(self, file: Union[List[str], str], file_type):
        if file_type == "video":
            # LLaVA-Med 可能不支持视频，但为了兼容性保留此选项
            raise NotImplementedError("LLaVA-Med does not support video input")
        elif file_type in ["image", "single-image"]:
            process_data = self.process_single_image(file)
        elif file_type == "multi-image":
            process_data = self.process_multi_images(file)
        else:
            raise AssertionError(f"Not supported file type: {file_type}")

        return process_data

    def process_single_image(self, image_path: str):
        """处理单张医学图像"""
        image = Image.open(image_path).convert('RGB')
        # LLaVA-Med 可能需要特定的医学图像预处理
        # 这里使用标准的 CLIP 处理方式
        return image

    def process_multi_images(self, image_paths: List[str]):
        """处理多张医学图像"""
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            images.append(image)
        return images