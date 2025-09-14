"""
Advanced Image Preprocessing Pipeline for Plant Classification
Handles low-quality images, noise reduction, and enhancement
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
from skimage import exposure, restoration, filters
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class AdvancedImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def enhance_low_quality_image(self, image):
        """
        Enhance low-quality images using multiple techniques
        """
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply multiple enhancement techniques
        enhanced = self._apply_enhancement_pipeline(image)
        return enhanced
    
    def _apply_enhancement_pipeline(self, image):
        """
        Apply comprehensive enhancement pipeline
        """
        # Convert to numpy for OpenCV operations
        img_array = np.array(image)
        
        # 1. Noise reduction
        img_array = self._reduce_noise(img_array)
        
        # 2. Contrast enhancement
        img_array = self._enhance_contrast(img_array)
        
        # 3. Sharpening
        img_array = self._apply_sharpening(img_array)
        
        # 4. Color correction
        img_array = self._correct_colors(img_array)
        
        return Image.fromarray(img_array)
    
    def _reduce_noise(self, image):
        """
        Apply noise reduction techniques
        """
        # Bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Non-local means denoising for better results
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(denoised, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
        
        return denoised
    
    def _enhance_contrast(self, image):
        """
        Enhance contrast using multiple methods
        """
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
        
        # Gamma correction for better visibility
        enhanced = self._gamma_correction(enhanced, gamma=1.2)
        
        return enhanced
    
    def _gamma_correction(self, image, gamma=1.0):
        """
        Apply gamma correction
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def _apply_sharpening(self, image):
        """
        Apply sharpening filter
        """
        # Unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _correct_colors(self, image):
        """
        Apply color correction
        """
        # White balance correction
        result = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        
        return result
    
    def preprocess_for_model(self, image, model_type='efficientnet'):
        """
        Preprocess image for specific model architectures
        """
        # Resize to target size
        if isinstance(image, Image.Image):
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            img_array = np.array(image)
        else:
            img_array = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize based on model type
        if model_type == 'efficientnet':
            img_array = img_array.astype(np.float32) / 255.0
            # EfficientNet expects values in [0, 1]
        elif model_type == 'resnet':
            # ImageNet normalization
            img_array = img_array.astype(np.float32)
            img_array = (img_array - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
        elif model_type == 'vit':
            # Vision Transformer normalization
            img_array = img_array.astype(np.float32) / 255.0
            img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        else:
            # Default normalization
            img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def augment_for_training(self, image):
        """
        Apply data augmentation for training
        """
        augmentations = []
        
        # Original image
        augmentations.append(image)
        
        # Rotation
        for angle in [15, -15, 30, -30]:
            rotated = image.rotate(angle, expand=False, fillcolor=(255, 255, 255))
            augmentations.append(rotated)
        
        # Brightness adjustment
        enhancer = ImageEnhance.Brightness(image)
        for factor in [0.8, 1.2]:
            bright = enhancer.enhance(factor)
            augmentations.append(bright)
        
        # Contrast adjustment
        enhancer = ImageEnhance.Contrast(image)
        for factor in [0.8, 1.2]:
            contrast = enhancer.enhance(factor)
            augmentations.append(contrast)
        
        # Color adjustment
        enhancer = ImageEnhance.Color(image)
        for factor in [0.8, 1.2]:
            color = enhancer.enhance(factor)
            augmentations.append(color)
        
        # Horizontal flip
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        augmentations.append(flipped)
        
        # Gaussian blur
        blurred = image.filter(ImageFilter.GaussianBlur(radius=1))
        augmentations.append(blurred)
        
        return augmentations
    
    def detect_image_quality(self, image):
        """
        Assess image quality and return quality score
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Calculate various quality metrics
        
        # 1. Laplacian variance (sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Contrast measure
        contrast = gray.std()
        
        # 3. Brightness measure
        brightness = gray.mean()
        
        # 4. Noise estimation
        noise_level = self._estimate_noise(gray)
        
        # Combine metrics into quality score (0-100)
        sharpness_score = min(laplacian_var / 100, 1.0) * 30
        contrast_score = min(contrast / 50, 1.0) * 25
        brightness_score = (1 - abs(brightness - 127) / 127) * 20
        noise_score = max(0, 1 - noise_level / 50) * 25
        
        quality_score = sharpness_score + contrast_score + brightness_score + noise_score
        
        return {
            'overall_quality': quality_score,
            'sharpness': laplacian_var,
            'contrast': contrast,
            'brightness': brightness,
            'noise_level': noise_level,
            'needs_enhancement': quality_score < 60
        }
    
    def _estimate_noise(self, image):
        """
        Estimate noise level in image
        """
        # Use Laplacian to estimate noise
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        noise_level = laplacian.var()
        return noise_level

# Utility functions for batch processing
def process_image_batch(images, preprocessor):
    """
    Process a batch of images
    """
    processed_images = []
    for img in images:
        enhanced = preprocessor.enhance_low_quality_image(img)
        processed = preprocessor.preprocess_for_model(enhanced)
        processed_images.append(processed)
    
    return np.vstack(processed_images)

def adaptive_preprocessing(image, quality_threshold=60):
    """
    Apply adaptive preprocessing based on image quality
    """
    preprocessor = AdvancedImagePreprocessor()
    quality_info = preprocessor.detect_image_quality(image)
    
    if quality_info['needs_enhancement']:
        # Apply aggressive enhancement for low-quality images
        enhanced = preprocessor.enhance_low_quality_image(image)
        return enhanced, quality_info
    else:
        # Light processing for good quality images
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image, quality_info
