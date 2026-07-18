"""Core image processing pipeline module for raw and standard photography assets.

This module provides high-performance filters, 8-band HSL color manipulations,
geometric transformations, and dynamic range tonemapping using multi-threaded
stripe segmentation algorithms.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from scipy.interpolate import CubicSpline

import logging
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger("PhotoEditor.Core")

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    logger.info("Successfully registered pillow_heif decoder extension layout.")
except Exception as env_ex:
    logger.warning(f"pillow_heif extension disabled via system application control rule policy: {env_ex}")

def global_exception_handler(exc_type, exc_value, exc_traceback):
    # Allow standard Ctrl+C keyboard interrupts to kill the process normally
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Log the crash with the full traceback
    logger.critical("Uncaught exception occurred:", exc_info=(exc_type, exc_value, exc_traceback))

# Bind the custom handler to Python's global exception hook
sys.excepthook = global_exception_handler

try:
    import rawpy
    HAS_RAWPY = True
    logger.info("Successfully bound rawpy processing module to core pipeline layout.")
except ImportError as env_ex:
    HAS_RAWPY = False
    logger.warning(f"rawpy module not available in the current environment context: {env_ex}")

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.heic', '.JPG', '.JPEG', '.PNG', '.HEIC')
RAW_EXTENSIONS = ('.raf', '.cr2', '.nef', '.arw', '.dng', '.RAF', '.CR2', '.NEF', '.ARW', '.DNG')
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS + RAW_EXTENSIONS
FILM_PROFILES = {
    'Portra 400' : {
        'v': [0.04, 0.28, 0.52, 0.76, 0.97],
        'r': [0.05, 0.30, 0.55, 0.79, 0.98],
        'g': [0.04, 0.28, 0.52, 0.76, 0.97],
        'b': [0.06, 0.26, 0.48, 0.73, 0.94],
        'color_matrix': [
            [ 1.06, -0.04, -0.02], # Warms up skin tones (pushes red, pulls green)
            [-0.04,  1.02,  0.02], # Keeps greens natural but slightly muted
            [-0.02,  0.06,  0.96]  # Bleeds green into blue for the signature Portra "cyan sky"
        ]
    },
    'Velvia 50' : {
        'v': [0.00, 0.18, 0.50, 0.84, 1.00],
        'r': [0.00, 0.20, 0.54, 0.86, 1.00],
        'g': [0.00, 0.18, 0.51, 0.85, 1.00],
        'b': [0.02, 0.20, 0.48, 0.80, 0.98],
        'color_matrix': [
            [ 0.96,  0.06, -0.02], # Deepens reds slightly
            [-0.05,  1.08, -0.03], # Aggressively purifies and punches greens (great for foliage)
            [-0.02, -0.06,  1.08]  # Aggressively punches blues
        ]
    },
    'Kodachrome 64' : {
        'v': [0.01, 0.22, 0.51, 0.78, 0.96],
        'r': [0.02, 0.26, 0.56, 0.82, 0.98],
        'g': [0.01, 0.22, 0.51, 0.77, 0.95],
        'b': [0.01, 0.19, 0.46, 0.71, 0.90],
        'color_matrix': [
            [ 1.10, -0.08, -0.02], # Very strong, rich reds (classic National Geographic look)
            [-0.05,  1.05,  0.00], # Earthy, warm greens
            [-0.03, -0.02,  1.05]  # Deep, saturated analog blues
        ]
    },
    'Superia 400' : {
        'v': [0.03, 0.25, 0.52, 0.78, 0.98],
        'r': [0.03, 0.26, 0.52, 0.77, 0.98],
        'g': [0.04, 0.27, 0.54, 0.79, 0.99],
        'b': [0.05, 0.26, 0.49, 0.75, 0.96],
        'color_matrix': [
            [ 1.04, -0.02, -0.02], # Punchy consumer reds
            [-0.03,  1.06, -0.03], # Fuji's "4th color layer" - strong green presence
            [-0.05,  0.08,  0.97]  # Cool blues with a distinct green/cyan bleed
        ]
    },
    'Kodak Gold' : {
        'v': [0.04, 0.22, 0.50, 0.78, 0.98],
        'r': [0.00, 0.24, 0.52, 0.76, 1.00],
        'g': [0.00, 0.25, 0.50, 0.75, 1.00],
        'b': [0.05, 0.23, 0.48, 0.73, 0.95],
        'color_matrix': [
            [ 1.08, -0.06, -0.02], # Strong yellow/gold bias in the reds
            [ 0.04,  0.98, -0.02], # Bleeds red into green to push foliage warmer
            [-0.02, -0.04,  1.06]  # Isolates blue to let the yellows/golds dominate the image
        ]
    }
}


class PhotoEditor:
    """Handles core image processing operations for photo editing filters.

    Attributes:
        image_path (str): The absolute disk destination path to the targeted asset file.
        original_image (np.ndarray): High-precision floating-point source image matrix.
    """
    # TODO: push film profile defaults to all existing json files
    DEFAULT_PRESET = {
            "apply_temperature_adjustment": True,
            "values_multiplier": 1.0, "color_multiplier": 1.0, "color_adjustments_multiplier": 1.0,
            "hdr_compression": 0.0, "exposure": 0.0, "contrast": 0.0,
            "whites": 0.0, "blacks": 0.0,
            "highlights": 0.0, "shadows": 0.0, "texture": 0.0, "clarity": 0.0,
            "gaussian_blur": 0.0, "vibrance": 0.0, "saturation": 0.0, "grain": 0.0, "grain_size": 1.0,
            "temp_kelvin": 6500, "tint": 0.0,
            "color_adjustments": {
                "red": {"hue": 0.0, "sat": 0.0}, "orange": {"hue": 0.0, "sat": 0.0},
                "yellow": {"hue": 0.0, "sat": 0.0}, "green": {"hue": 0.0, "sat": 0.0},
                "aqua": {"hue": 0.0, "sat": 0.0}, "blue": {"hue": 0.0, "sat": 0.0},
                "purple": {"hue": 0.0, "sat": 0.0}, "magenta": {"hue": 0.0, "sat": 0.0}
            },
            "active_crop_variant" : "default",
            "crop_variants" : {
                "default": {
                    "crop_aspect_ratio": "Free",
                    "crop_aspect_ratio_flipped": False,
                    "crop_rotation": 0,
                    "crop_size": 100,
                    "crop_center_x": 50,
                    "crop_center_y": 50,
                    "crop_free_width": 100,
                    "crop_free_height": 100,
                    "add_white_border": True,
                    "white_border_width_pct": 2,
                    "resolution_percentage": 100,
                    "do_instagram_compression": True
                }
            },
            # 1. Tonality & Color Science
            "rgb_curves" : {
                "r" : [[0.0, 0.25, 0.50, 0.75, 1.0], [0.0, 0.25, 0.50, 0.75, 1.0]],
                "g" : [[0.0, 0.25, 0.50, 0.75, 1.0], [0.0, 0.25, 0.50, 0.75, 1.0]],
                "b" : [[0.0, 0.25, 0.50, 0.75, 1.0], [0.0, 0.25, 0.50, 0.75, 1.0]]
            },
            "color_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0,  0.0],
                [0.0, 0.0,  1.0]
            ],
            
            # 2. Optical Bloom
            "enable_bloom": True,
            "bloom_threshold": 0.70,
            "bloom_radius": 15.0,
            "bloom_strength": 0.20, 
            
            # 3. Selective Halation
            "enable_halation": True,
            "halation_threshold": 0.60,
            "halation_radius": 12.0,
            "halation_strength": 0.35,
            "halation_offset_x": 1.5,
            "halation_offset_y": 0.2,
            
            # 4. Smart Grain
            "enable_grain": True,
            "grain_strength": 0.05, 
            "grain_size": 0.08,      # >1.0 scales grain up for coarse emulsion
            "grain_chroma": 0.15,   # 0.0 = B&W grain, 1.0 = color grain
            
            # 5. Vignette & Optical Softness
            "enable_vignette": True,
            "vignette_strength": 0.35,
            "vignette_radius": 0.75,
            "vignette_softness": 0.45,
            "corner_blur_radius": 1.0
        }

    def __init__(self, image_path: str):
        """Initializes the PhotoEditor instance by reading the image matrix into memory.

        Args:
            image_path (str): The path to the image asset file.
        """
        self.image_path = image_path
        logger.info(f"Initializing PhotoEditor instance matrix for: {os.path.basename(image_path)}")
        self.original_image = self.load_image_matrix(image_path)

    @staticmethod
    def load_image_matrix(image_path: str, preview: bool = False, max_width: int = 1200) -> np.ndarray:
        """Loads image files into floating-point numpy structures with uniform color math."""
        ext = os.path.splitext(image_path)[1].lower()
        logger.info(f"Parsing image file matrix allocation layer: {os.path.basename(image_path)} (Preview Mode={preview})")

        if ext in RAW_EXTENSIONS:
            if not HAS_RAWPY:
                logger.error("Execution blocked: rawpy engine is not defined or installed in this environment.")
                raise ImportError("Cannot decode RAW files because 'rawpy' is not installed in this Python environment.")
            try:
                with rawpy.imread(image_path) as raw:
                    # Use half_size=True at the C++ decode level if preview is requested
                    rgb_16 = raw.postprocess(
                        use_camera_wb=True, 
                        half_size=preview,  # Massive speedup for previews
                        no_auto_bright=True, 
                        output_bps=16
                    )
                    
                    # Resize the 16-bit integer array BEFORE converting to float32
                    if preview and rgb_16.shape[1] > max_width:
                        scale = float(max_width) / rgb_16.shape[1]
                        new_height = int(rgb_16.shape[0] * scale)
                        rgb_16 = cv2.resize(rgb_16, (max_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # Perform float conversion and normalization only on the final pixel count
                    return rgb_16.astype(np.float32, copy=False) / 65535.0

            except Exception as e:
                logger.error(f"Critical error mapping raw image array layout structure: {e}")
                raise e
        else:
            try:
                # cv2.imread loads natively to a NumPy array (much faster than PIL -> NumPy)
                # IMREAD_COLOR ignores EXIF orientation by default, so we handle it via IMREAD_IGNORE_ORIENTATION or cv2
                img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
                
                if img_bgr is None:
                    raise ValueError(f"OpenCV failed to decode image buffer at path: {image_path}")

                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Resize the 8-bit integer array BEFORE float math
                if preview and img_rgb.shape[1] > max_width:
                    scale = float(max_width) / img_rgb.shape[1]
                    new_height = int(img_rgb.shape[0] * scale)
                    img_rgb = cv2.resize(img_rgb, (max_width, new_height), interpolation=cv2.INTER_AREA)

                # Convert to float32 and normalize
                return img_rgb.astype(np.float32, copy=False) / 255.0

            except Exception as e:
                logger.error(f"Failed to cleanly decode compressed standard image file layout: {e}")
                raise e    
    
    @staticmethod
    def _load_image_matrix_old(image_path: str, preview: bool = False, max_width=1200) -> np.ndarray:
        """Loads image files into floating-point numpy structures with uniform color math.

        Args:
            image_path (str): The absolute target string destination path.
            preview (bool, optional): Downsamples raw/standard configurations if True. Defaults to False.

        Returns:
            np.ndarray: A 3-channel floating-point RGB matrix normalized between 0.0 and 1.0.

        Raises:
            ImportError: If a RAW file is requested but rawpy is missing from the environment.
            ValueError: If the file format cannot be parsed by OpenCV or Pillow.
        """
        ext = os.path.splitext(image_path)[1].lower()
        logger.info(f"Parsing image file matrix allocation layer: {os.path.basename(image_path)} (Preview Mode={preview})")

        if ext in [e.lower() for e in RAW_EXTENSIONS]:
            if not HAS_RAWPY:
                logger.error("Execution blocked: rawpy engine is not defined or installed in this environment.")
                raise ImportError("Cannot decode RAW files because 'rawpy' is not installed in this Python environment.")
            try:
                with rawpy.imread(image_path) as raw:
                    rgb_16 = raw.postprocess(
                        use_camera_wb=True, 
                        half_size=False, 
                        no_auto_bright=True, 
                        output_bps=16
                    )
                    matrix = rgb_16.astype(np.float32) / 65535.0
                    if preview and matrix.shape[1] > max_width:
                        scale = float(max_width) / matrix.shape[1]
                        matrix = cv2.resize(matrix, (max_width, int(matrix.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                    return matrix
            except Exception as e:
                logger.error(f"Critical error mapping raw image array layout structure: {e}")
                raise e
        else:
            try:
                with Image.open(image_path) as img:
                    img = ImageOps.exif_transpose(img)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    matrix = np.array(img, dtype=np.float32) / 255.0
                    if preview and matrix.shape[1] > 1200:
                        scale = 1200.0 / matrix.shape[1]
                        matrix = cv2.resize(matrix, (1200, int(matrix.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                    return matrix
            except Exception as e:
                logger.error(f"Failed to cleanly decode compressed standard image file layout: {e}")
                raise e

    @staticmethod
    def _apply_sdr_preview(img: np.ndarray, intensity: float) -> np.ndarray:
        """
        Compresses HDR data into an SDR range (0.0 to 1.0), simulating Lightroom's 
        'Preview for SDR Display'.
        
        Args:
            img (np.ndarray): High precision source image (can contain values > 1.0).
            intensity (float): 0.0 represents standard SDR clipping (no HDR compression).
                               1.0 represents full ACES highlight roll-off.
        
        Returns:
            np.ndarray: Bounded SDR image matrix in range [0.0, 1.0].
        """
        # Clamp intensity to valid range
        intensity = float(np.clip(intensity, 0.0, 1.0))
        
        if intensity == 0.0:
            # At 0 intensity, simulate standard SDR display clipping without tonemapping
            return np.clip(img, 0.0, 1.0).astype(np.float32)

        # Narkowicz ACES approximation coefficients
        a = np.float32(2.51)
        b = np.float32(0.03)
        c = np.float32(2.43)
        d = np.float32(0.59)
        e = np.float32(0.14)

        # Apply ACES curve directly without artificial exposure pre-scaling
        tonemapped = (img * (a * img + b)) / (img * (c * img + d) + e)
        
        # Ensure mathematical precision errors don't drift outside [0, 1]
        np.clip(tonemapped, 0.0, 1.0, out=tonemapped)

        if intensity == 1.0:
            return tonemapped.astype(np.float32)

        # Blend between standard SDR clipping and full ACES compression.
        # Unlike blending with raw HDR, both inputs here are strictly <= 1.0, 
        # guaranteeing the output never blows out on an SDR display.
        sdr_clipped = np.clip(img, 0.0, 1.0)
        
        return cv2.addWeighted(
            tonemapped, intensity, 
            sdr_clipped, 1.0 - intensity, 
            0.0
        ).astype(np.float32)

    # TODO: this is getting heavier and heavier as we continue to add steps. We need to do lazy caching here.
    @classmethod
    def run_pipeline(cls, src_matrix: np.ndarray, preset: Dict[str, Any]) -> np.ndarray:
        """Executes core image filters sequentially using in-place operations.

        Args:
            src_matrix (np.ndarray): The source image chunk to apply filter logic upon.
            preset (Dict[str, Any]): Dictionary containing configuration presets.

        Returns:
            np.ndarray: Fully processed floating-point RGB matrix layout.
        """
        img = np.copy(src_matrix).astype(np.float32)
        h, w, _ = img.shape
        max_dim = max(h, w)

        v_mult = np.clip(preset.get('values_multiplier', 1.0), 0.0, 1.0)
        c_mult = np.clip(preset.get('color_multiplier', 1.0), 0.0, 1.0)
        ca_mult = np.clip(preset.get('color_adjustments_multiplier', 1.0), 0.0, 1.0)

        # 1. White Balance (Temperature & Tint via Multiplicative Gain)
        apply_temp = preset.get('apply_temperature_adjustment', True)
        temp_kelvin = preset.get('temp_kelvin', 6500.0)
        tint = preset.get('tint', 0.0)  
        
        if apply_temp and temp_kelvin != 6500.0:
            if temp_kelvin > 6500.0:
                temp_factor = min(temp_kelvin - 6500.0, 5500.0) / 5500.0
                img[:, :, 0] *= np.float32(1.0 + temp_factor * 0.25)
                img[:, :, 2] *= np.float32(1.0 - temp_factor * 0.20)
            else:
                temp_factor = max(6500.0 - temp_kelvin, 4500.0) / 4500.0
                img[:, :, 2] *= np.float32(1.0 + temp_factor * 0.30)
                img[:, :, 0] *= np.float32(1.0 - temp_factor * 0.15)

        if tint != 0.0:
            img[:, :, 1] *= np.float32(1.0 + tint * 0.15)

        # 2. Exposure (1 EV = 2x multiplier)
        exposure = preset.get('exposure', 0.0)
        if exposure != 0.0:
            img *= np.float32(2.0 ** exposure)

        return img

        # 3. Contrast (Pivot around middle gray 0.18)
        contrast = preset.get('contrast', 0.0) * v_mult
        if contrast != 0.0:
            pivot = np.float32(0.18) 
            img = (img - pivot) * np.float32(1.0 + contrast) + pivot
            img = np.maximum(img, np.float32(0.0))

        # 4. Whites & Blacks (Soft shoulder/toe)
        whites = preset.get('whites', 0.0) * v_mult
        blacks = preset.get('blacks', 0.0) * v_mult
        
        if whites != 0.0 or blacks != 0.0:
            lum = np.float32(0.2126) * img[:, :, 0] + np.float32(0.7152) * img[:, :, 1] + np.float32(0.0722) * img[:, :, 2]
            lum_safe = np.maximum(lum, np.float32(1e-6))
            new_lum = lum.copy()

            if whites != 0.0:
                w_mask = np.clip((lum - np.float32(0.5)) * np.float32(2.0), 0.0, 1.0)
                new_lum += (whites * np.float32(0.25)) * (w_mask ** 2) * lum
                new_lum = np.maximum(new_lum, np.float32(0.0))

            if blacks != 0.0:
                b_mask = np.clip((np.float32(0.3) - lum) * np.float32(3.33), 0.0, 1.0)
                new_lum += (blacks * np.float32(0.15)) * (b_mask ** 2)
                new_lum = np.maximum(new_lum, np.float32(0.0))

            img *= np.expand_dims(new_lum / lum_safe, axis=2)

        # 5. Highlights & Shadows
        highlights = preset.get('highlights', 0.0) * v_mult
        shadows = preset.get('shadows', 0.0) * v_mult

        if highlights != 0.0 or shadows != 0.0:
            lum = np.float32(0.2126) * img[:, :, 0] + np.float32(0.7152) * img[:, :, 1] + np.float32(0.0722) * img[:, :, 2]
            lum_safe = np.maximum(lum, np.float32(1e-6))
            new_lum = lum.copy()

            if highlights != 0.0:
                hl_range = np.clip((lum - np.float32(0.5)) * np.float32(2.0), 0.0, 1.0)
                hl_mask = hl_range * hl_range * (np.float32(3.0) - np.float32(2.0) * hl_range)
                if highlights < 0.0:
                    new_lum *= (np.float32(1.0) + (highlights * hl_mask * np.float32(0.5)))
                else:
                    safe_headroom = np.maximum(np.float32(0.0), np.float32(1.0) - lum)
                    new_lum += highlights * hl_mask * safe_headroom * np.float32(0.7)

            if shadows != 0.0:
                sh_range = np.clip((np.float32(0.5) - lum) * np.float32(2.0), 0.0, 1.0)
                sh_mask = sh_range * sh_range * (np.float32(3.0) - np.float32(2.0) * sh_range)
                if shadows > 0.0:
                    new_lum += shadows * sh_mask * (np.sqrt(lum_safe) - lum) * np.float32(0.8)
                else:
                    new_lum *= (np.float32(1.0) + (shadows * sh_mask * np.float32(0.5)))

            img *= np.expand_dims(new_lum / lum_safe, axis=2)
        
        # 6. Texture (High-Frequency Local Detail - Dynamic Kernel Scaling)
        texture = preset.get('texture', 0.0)
        if texture != 0.0:
            t_k = max(3, int(max_dim * 0.005) | 1)
            low_pass_tex = cv2.GaussianBlur(img, (t_k, t_k), 0)
            high_freq = img - low_pass_tex
            img += np.float32(texture * 0.5) * high_freq

        # 7. Clarity (Mid-Frequency Local Contrast - Dynamic Wide Kernel)
        clarity = preset.get('clarity', 0.0)
        if clarity != 0.0:
            c_k = max(5, int(max_dim * 0.05) | 1)
            low_pass_clarity = cv2.GaussianBlur(img, (c_k, c_k), 0)
            mid_freq = img - low_pass_clarity
            img += np.float32(clarity * 0.4) * mid_freq

        # 8. Saturation & Vibrance (HDR-Compatible Grayscale Interpolation)
        vibrance = preset.get('vibrance', 0.0) * c_mult
        saturation = preset.get('saturation', 0.0) * c_mult
        
        if saturation != 0.0 or vibrance != 0.0:
            lum_matrix = (np.float32(0.2126) * img[:, :, 0] + 
                          np.float32(0.7152) * img[:, :, 1] + 
                          np.float32(0.0722) * img[:, :, 2])
            grayscale = np.expand_dims(lum_matrix, axis=2)
            
            if vibrance != 0.0:
                max_rgb = np.max(img, axis=2, keepdims=True)
                min_rgb = np.min(img, axis=2, keepdims=True)
                sat_mask = np.where(max_rgb > 1e-5, (max_rgb - min_rgb) / max_rgb, np.float32(0.0))
                vib_factor = np.float32(vibrance) * (np.float32(1.0) - sat_mask)
                img = grayscale + (img - grayscale) * (np.float32(1.0) + vib_factor)

            if saturation != 0.0:
                sat_factor = np.maximum(np.float32(1.0 + saturation), np.float32(0.0))
                img = grayscale + (img - grayscale) * sat_factor
        
        # 9. 8-Band Color Adjustments (Luminance-Preserved HDR Chroma Mapping)
        color_adj = preset.get('color_adjustments', {})
        if color_adj and ca_mult > 0.0:
            # Save original unbounded HDR luminance to prevent highlight clipping during HSV math
            orig_lum = (np.float32(0.2126) * img[:, :, 0] + 
                        np.float32(0.7152) * img[:, :, 1] + 
                        np.float32(0.0722) * img[:, :, 2])
            orig_lum_safe = np.maximum(orig_lum, np.float32(1e-6))

            # Normalize to [0, 1] purely for safe OpenCV HSV conversion
            safe_rgb = np.clip(img, 0.0, 1.0)
            hsv = cv2.cvtColor(safe_rgb, cv2.COLOR_RGB2HSV)
            
            bands_config = {
                'red': {'center': 0.0, 'width': 22.0},
                'orange': {'center': 30.0, 'width': 15.0},
                'yellow': {'center': 60.0, 'width': 20.0},
                'green': {'center': 120.0, 'width': 40.0},
                'aqua': {'center': 175.0, 'width': 25.0},
                'blue': {'center': 225.0, 'width': 35.0},
                'purple': {'center': 275.0, 'width': 25.0},
                'magenta': {'center': 315.0, 'width': 25.0}
            }

            h_matrix = hsv[:, :, 0]
            s_matrix = hsv[:, :, 1]
            total_h_delta = np.zeros_like(h_matrix)
            total_s_mod = np.zeros_like(s_matrix)
            has_adjustments = False

            for band, cfg in bands_config.items():
                adjustments = color_adj.get(band, {"hue": 0.0, "sat": 0.0})
                h_shift = float(adjustments.get('hue', 0.0) * 180.0 * ca_mult)
                s_shift = float(adjustments.get('sat', 0.0) * ca_mult)

                if h_shift == 0.0 and s_shift == 0.0:
                    continue
                
                has_adjustments = True
                diff = np.abs(h_matrix - cfg['center'])
                diff = np.minimum(diff, 360.0 - diff)
                weight = np.clip(1.0 - (diff / cfg['width']), 0.0, 1.0)
                weight = 0.5 * (1.0 - np.cos(weight * np.pi))

                if h_shift != 0.0:
                    total_h_delta += (weight * h_shift)
                if s_shift != 0.0:
                    total_s_mod += (weight * s_shift)

            if has_adjustments:
                hsv[:, :, 0] = (hsv[:, :, 0] + total_h_delta) % 360.0
                pos_mask = (total_s_mod >= 0.0)
                neg_mask = ~pos_mask
                
                s_matrix[pos_mask] = s_matrix[pos_mask] * (1.0 + total_s_mod[pos_mask]) + (total_s_mod[pos_mask] * 0.2)
                s_matrix[neg_mask] = s_matrix[neg_mask] * (1.0 + total_s_mod[neg_mask])
                hsv[:, :, 1] = np.clip(s_matrix, 0.0, 1.0)

                # Convert back to RGB and scale by original HDR luminance ratio!
                new_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
                new_lum = (np.float32(0.2126) * new_rgb[:, :, 0] + 
                           np.float32(0.7152) * new_rgb[:, :, 1] + 
                           np.float32(0.0722) * new_rgb[:, :, 2])
                new_lum_safe = np.maximum(new_lum, np.float32(1e-6))
                
                img = new_rgb * np.expand_dims(orig_lum / new_lum_safe, axis=2)
        
        # 10. Profile RGB Curves (Procedural 1D LUTs)
        # Interpolates channel values rapidly for split-toning effects.
        rgb_curves = preset.get('rgb_curves', None)
        if rgb_curves:
            for i, channel in enumerate(['r', 'g', 'b']):
                if channel in rgb_curves:
                    xp, yp = rgb_curves[channel]
                    # Note: xp and yp must be monotonically increasing.
                    img[:, :, i] = np.interp(img[:, :, i], xp, yp)

        # Film Simulation methods
        img = apply_color_matrix(img, preset)
        img = apply_bloom(img, preset)
        img = apply_halation(img, preset)
        img = apply_smart_grain(img, preset)
        img = apply_vignette_and_softness(img, preset)

        # 11. Optional Smoothing
        # blur_radius = preset.get('gaussian_blur', 0.0)
        # if blur_radius > 0:
        #     sigma = float(blur_radius * 1.5)
        #     img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # 12. Display Preparation: Tonemapping at the end of float processing
        hdr_comp = preset.get('hdr_compression', 0.0)
        if hdr_comp > 0.0:
            img = cls._apply_sdr_preview(img, hdr_comp)

        # 13. Final In-Place Hard Clip to SDR Monitor Bounds [0.0, 1.0]
        np.clip(img, 0.0, 1.0, out=img)
        return img

    @classmethod
    def run_parallel_pipeline(cls, src_matrix: np.ndarray, preset: Dict[str, Any]) -> np.ndarray:
        """Scales pre-cropped image matrices to match compression dimensions, then processes tiles concurrently.

        Args:
            src_matrix (np.ndarray): Bounded pre-cropped image matrix array layer.
            preset (Dict[str, Any]): Dictionary containing configuration presets.

        Returns:
            np.ndarray: Multi-core rendered full raster color grid matrix output.
        """
        h, w, c = src_matrix.shape
        active_crop_variant = preset.get('active_crop_variant', 'default')
        crop_variant_data = preset['crop_variants'][active_crop_variant]
        do_instagram_compression = crop_variant_data.get('do_instagram_compression', True)

        if do_instagram_compression:
            target_w = 1080
            if w != target_w:
                aspect_ratio = h / w
                src_matrix = cv2.resize(src_matrix, (target_w, int(target_w * aspect_ratio)), interpolation=cv2.INTER_LANCZOS4)
        else:
            pct = int(crop_variant_data.get('resolution_percentage', 100)) / 100.0
            if pct < 1.0:
                src_matrix = cv2.resize(src_matrix, (int(w * pct), int(h * pct)), interpolation=cv2.INTER_LANCZOS4)
        
        h, w, c = src_matrix.shape
        if h < 32 or w < 32:
            return cls.run_pipeline(src_matrix, preset)

        output_matrix = np.empty_like(src_matrix)
        cores = os.cpu_count() or 4
        stripe_height = h // cores
        margin = max(16, min(64, h // 8))

        def process_stripe(stripe_idx):
            y_start = stripe_idx * stripe_height
            y_end = h if stripe_idx == (cores - 1) else (stripe_idx + 1) * stripe_height

            pad_start = max(0, y_start - margin)
            pad_end = min(h, y_end + margin)

            stripe_input = src_matrix[pad_start:pad_end, :, :]
            processed_stripe = cls.run_pipeline(stripe_input, preset)

            offset = y_start - pad_start
            slice_length = y_end - y_start
            output_matrix[y_start:y_end, :, :] = processed_stripe[offset : offset + slice_length, :, :]

        with ThreadPoolExecutor(max_workers=cores) as executor:
            executor.map(process_stripe, range(cores))

        return output_matrix

    @staticmethod
    def apply_crop(img: np.ndarray, crop_data: Dict[str, Any]) -> np.ndarray:
        """Applies center rotation skew corrections and slices image bounding boxes.

        Args:
            img (np.ndarray): High precision source image matrix layout.
            crop_data (Dict[str, Any]): Dictionary containing crop data.

        Returns:
            np.ndarray: Cropped and modified bounding box sub-matrix configuration.
        """
        h, w, _ = img.shape
        
        rotation = float(crop_data.get('crop_rotation', 0.0))
        if rotation != 0.0:
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rotation, 1.0)
            fill_color = (0.0, 0.0, 0.0) if img.dtype == np.float32 else (0, 0, 0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=fill_color)
            
        ratio_mode = crop_data.get('crop_aspect_ratio', 'Free')
        
        if ratio_mode == 'Free':
            box_w = int(max(10, min(100, crop_data.get('crop_free_width', 100))) / 100.0 * w)
            box_h = int(max(10, min(100, crop_data.get('crop_free_height', 100))) / 100.0 * h)
            if crop_data.get('crop_aspect_ratio_flipped', False):
                box_w, box_h = box_h, box_w
        else:
            if ratio_mode == 'Original': target_ratio = w / h
            elif ratio_mode == '1:1': target_ratio = 1.0
            elif ratio_mode == '4:5': target_ratio = 5.0 / 4.0 if w >= h else 4.0 / 5.0
            elif ratio_mode == '5:7': target_ratio = 7.0 / 5.0 if w >= h else 5.0 / 7.0
            elif ratio_mode == '8:10': target_ratio = 10.0 / 8.0 if w >= h else 8.0 / 10.0
            elif ratio_mode == '16:9': target_ratio = 16.0 / 9.0 if w >= h else 9.0 / 16.0
            else: target_ratio = w / h
                
            if crop_data.get('crop_aspect_ratio_flipped', False):
                target_ratio = 1.0 / target_ratio
                
            if w / h >= target_ratio:
                max_h = h
                max_w = int(h * target_ratio)
            else:
                max_w = w
                max_h = int(w / target_ratio)
                
            size_scale = max(10, min(100, crop_data.get('crop_size', 100))) / 100.0
            box_w = int(max_w * size_scale)
            box_h = int(max_h * size_scale)
            
        cx_pct = crop_data.get('crop_center_x', 50) / 100.0
        cy_pct = crop_data.get('crop_center_y', 50) / 100.0
        
        ideal_cx = int(cx_pct * w)
        ideal_cy = int(cy_pct * h)
        
        x_min = ideal_cx - box_w // 2
        y_min = ideal_cy - box_h // 2
        
        if x_min < 0: x_min = 0
        if x_min + box_w > w: x_min = w - box_w
        if y_min < 0: y_min = 0
        if y_min + box_h > h: y_min = h - box_h
            
        x_max = x_min + box_w
        y_max = y_min + box_h
        
        x_min = max(0, min(w - 1, x_min))
        x_max = max(x_min + 1, min(w, x_max))
        y_min = max(0, min(h - 1, y_min))
        y_max = max(y_min + 1, min(h, y_max))
        
        return img[y_min:y_max, x_min:x_max, :]

    @staticmethod
    def apply_white_border(img: np.ndarray, preset: Dict[str, Any]) -> np.ndarray:
        """Appends a white border by downscaling the inner frame to keep overall dimensions exact.

        Args:
            img (np.ndarray): Image array matrix to map.
            preset (Dict[str, Any]): Dictionary containing configuration presets.

        Returns:
            np.ndarray: Modified bounded border canvas matrix configuration layer.
        """
        active_crop_variant = preset.get("active_crop_variant", "default")
        crop_data = preset["crop_variants"][active_crop_variant]

        if not crop_data.get('add_white_border', False):
            return img
        h, w, _ = img.shape
        border_pct = crop_data.get('white_border_width_pct', 5)
        border_pixels = int(max(w, h) * (border_pct / 100.0))
        
        if border_pixels > 0 and w > 2 * border_pixels and h > 2 * border_pixels:
            inner_w = w - 2 * border_pixels
            inner_h = h - 2 * border_pixels
            img_resized = cv2.resize(img, (inner_w, inner_h), interpolation=cv2.INTER_LANCZOS4)
            fill_val = [1.0, 1.0, 1.0] if img.dtype == np.float32 else [255, 255, 255]
            return cv2.copyMakeBorder(
                img_resized, border_pixels, border_pixels, border_pixels, border_pixels,
                cv2.BORDER_CONSTANT, value=fill_val
            )
        return img
    
    @classmethod
    def calculate_auto_preset(cls, img: np.ndarray, is_linear: bool = True) -> Dict[str, Any]:
        # 1. Fast 512px downscaled proxy
        h, w, _ = img.shape
        scale = min(1.0, 512.0 / max(h, w))
        small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small = np.clip(np.float32(small), 0.0, 1.0)
        
        lum = (np.float32(0.2126) * small[:, :, 0] + 
               np.float32(0.7152) * small[:, :, 1] + 
               np.float32(0.0722) * small[:, :, 2])
        lum_safe = np.maximum(lum, np.float32(1e-6))
        
        valid_mask = (lum > 0.01) & (lum < 2.5)

        # ==========================================================
        # 1. AUTO-EXPOSURE (+0.125 EV Stylistic Bias)
        # ==========================================================
        target_display = np.float32(0.47) if is_linear else np.float32(0.48)
        current_linear_median = np.median(lum[valid_mask]) if np.sum(valid_mask) > 0 else np.float32(0.18)
        
        ev_shift = float(np.log2(0.24 / max(current_linear_median, 1e-4)))
        
        for _ in range(4):
            test_img = small * (np.float32(2.0 ** ev_shift))
            tonemapped = cls._apply_sdr_preview(test_img, 1.0)
            
            t_lum = (np.float32(0.2126) * tonemapped[:, :, 0] + 
                     np.float32(0.7152) * tonemapped[:, :, 1] + 
                     np.float32(0.0722) * tonemapped[:, :, 2])
            
            display_median = np.median(t_lum[valid_mask]) if np.sum(valid_mask) > 0 else target_display
            ratio = target_display / max(display_median, 1e-4)
            ev_shift += float(np.log2(ratio) * 0.6)
            
        stylistic_ev_bias = 0.125
        auto_exposure = float(np.clip(ev_shift + stylistic_ev_bias, -2.5, 2.5))

        # ==========================================================
        # 2. TONAL RECOVERY (Refined Highlights & Shadows)
        # ==========================================================
        final_test_img = small * (np.float32(2.0 ** auto_exposure))
        final_tonemapped = cls._apply_sdr_preview(final_test_img, 1.0)
        
        t_lum_final = (np.float32(0.2126) * final_tonemapped[:, :, 0] + 
                       np.float32(0.7152) * final_tonemapped[:, :, 1] + 
                       np.float32(0.0722) * final_tonemapped[:, :, 2])

        t_p10 = np.percentile(t_lum_final, 10)
        t_p20 = np.percentile(t_lum_final, 20)
        t_p80 = np.percentile(t_lum_final, 80)
        t_p95 = np.percentile(t_lum_final, 95)
        t_p99 = np.percentile(t_lum_final, 99)
        
        # Highlights
        auto_highlights = 0.0
        if t_p95 > 0.80:
            auto_highlights = float(np.clip((0.80 - t_p95) * 2.5, -0.55, 0.0))
            
        # Whites
        auto_whites = 0.0
        if t_p99 > 0.96:
            auto_whites = float(np.clip((0.96 - t_p99) * 4.0, -0.40, 0.0))
        elif t_p99 < 0.85:
            auto_whites = float(np.clip((0.85 - t_p99) * 2.0, 0.0, 0.30))
            
        # Shadows
        auto_shadows = 0.0
        if t_p10 < 0.12:
            auto_shadows = float(np.clip((0.12 - t_p10) * 7.0, 0.0, 0.75))

        # Dynamic Contrast
        midtone_range = float(t_p80 - t_p20)
        auto_contrast = float(np.clip(0.20 - (midtone_range * 0.25), 0.05, 0.20))
        
        # Blacks
        predicted_lum = lum * (2.0 ** auto_exposure)
        p005 = np.percentile(predicted_lum, 0.5)
        auto_blacks = float(np.clip((0.003 - p005) * 3.0, -0.10, 0.0)) if p005 > 0.003 else 0.0

        # ==========================================================
        # 3. LANDSCAPE-AWARE AUTO WHITE BALANCE & TINT
        # ==========================================================
        max_c = np.max(small, axis=2)
        min_c = np.min(small, axis=2)
        chroma = max_c - min_c
        sat = np.where(max_c > 1e-5, chroma / max_c, 0.0)
        
        # Convert to HSV early so hue and sat are available for both WB and heuristics
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        
        lum_valid = (lum >= 0.10) & (lum <= 0.95)
        
        # Hierarchical Neutral Detection based on absolute RGB color purity (Chroma)
        # Targets true achromatic references: clouds, snow, white water foam, gray rocks, asphalt
        strict_neutrals = lum_valid & (chroma < 0.04)
        relaxed_neutrals = lum_valid & (chroma < 0.08)
        
        if np.sum(strict_neutrals) > 100:
            wb_mask = strict_neutrals
        elif np.sum(relaxed_neutrals) > 100:
            wb_mask = relaxed_neutrals
        else:
            # Fallback: Top 5% least chromatic pixels, capped at 0.12 to prevent
            # color-casting scenes that genuinely lack neutral reference points
            threshold = min(0.12, float(np.percentile(chroma[lum_valid], 5))) if np.sum(lum_valid) > 0 else 0.05
            wb_mask = lum_valid & (chroma <= threshold)
            
        auto_kelvin = 6500.0
        auto_tint = 0.0
        
        if np.sum(wb_mask) > 50:
            mean_r = float(np.mean(small[:, :, 0][wb_mask]))
            mean_g = float(np.mean(small[:, :, 1][wb_mask]))
            mean_b = float(np.mean(small[:, :, 2][wb_mask]))
            
            # 1. CONTINUOUS KELVIN SHIFT
            rb_ratio = mean_r / max(mean_b, 1e-5)
            kelvin_shift = float(-np.log2(rb_ratio) * 2800.0)
            auto_kelvin = float(np.clip(6500.0 + kelvin_shift, 4800.0, 8000.0))
            
            # 2. TINT CORRECTION
            # Negative tint = magenta, Positive tint = green.
            # If the scene has a green cast (mean_g > mean_rb), rg_ratio > 1.0.
            # (1.0 - rg_ratio) yields a NEGATIVE value, shifting tint towards magenta to neutralize green.
            # If the scene has a magenta cast (mean_g < mean_rb), rg_ratio < 1.0.
            # (1.0 - rg_ratio) yields a POSITIVE value, shifting tint towards green to neutralize magenta.
            mean_rb = (mean_r + mean_b) * 0.5
            rg_ratio = mean_g / max(mean_rb, 1e-5)
            auto_tint = float(np.clip((1.0 - rg_ratio) * 1.5, -0.25, 0.25))

        # Global Saturation scaling
        current_median_sat = float(np.median(sat[valid_mask])) if np.sum(valid_mask) > 0 else 0.22
        auto_vibrance = float(np.clip((0.30 - current_median_sat) * 0.65, -0.05, 0.18))
        auto_saturation = float(np.clip((0.24 - current_median_sat) * 0.20, -0.05, 0.06))

        # ==========================================================
        # 4. CONTENT HEURISTICS (Foliage, Warm Tones, Sky Blues)
        # ==========================================================
        # Green / Foliage
        grass_mask = (hue >= 45.0) & (hue <= 135.0) & (sat > 0.12) & (lum > 0.05)
        grass_ratio = float(np.mean(grass_mask))
        green_sat_boost = float(np.clip(grass_ratio * 0.4, 0.0, 0.15)) if grass_ratio > 0.03 else 0.0

        # Orange / Warm tones
        orange_mask = (hue >= 10.0) & (hue < 45.0) & (sat > 0.15) & (lum > 0.08)
        orange_ratio = float(np.mean(orange_mask))
        orange_sat_boost = float(np.clip(orange_ratio * 0.3, 0.0, 0.12)) if orange_ratio > 0.03 else 0.0

        # Blue / Sky desaturation
        # sky_mask = (hue >= 190.0) & (hue <= 240.0) & (sat > 0.15) & (lum > 0.20)
        # sky_ratio = float(np.mean(sky_mask))
        blue_sat_adjust = float(np.clip(0.5, -0.22, -0.04))

        return {
            "exposure": round(auto_exposure, 2) + 0.025,  # add a little of over-exposure
            "contrast": round(auto_contrast, 2),
            "highlights": round(auto_highlights, 2),
            "shadows": round(auto_shadows, 2),
            "whites": round(auto_whites, 2),
            "blacks": round(auto_blacks, 2),
            "temp_kelvin": round(auto_kelvin, -1) + 250.0,  # slightly warmer tint
            "tint": round(auto_tint, 2),
            "vibrance": round(auto_vibrance, 2),
            "saturation": round(auto_saturation, 2),
            "texture": -0.05,
            "clarity": -0.05,
            "hdr_compression": 1.0,
            "color_adjustments": {
                "green": {"hue": 0.0, "sat": round(green_sat_boost, 2)},
                "orange": {"hue": 0.0, "sat": round(orange_sat_boost, 2)},
                "blue": {"hue": 0.0, "sat": round(blue_sat_adjust, 2)},
                "red" : {"hue" : 0.0, "sat" : 0.15}  # always increase any reds
            }
        }

    def apply_presets(self, preset: Dict[str, Any]) -> np.ndarray:
        """Unified instance abstraction method executing the full pipeline pass.

        Args:
            preset (Dict[str, Any]): Target metrics properties layout map.

        Returns:
            np.ndarray: Rendered image matrix.
        """

        active_crop_variant = preset.get("active_crop_variant", "default")
        crop_data = preset["crop_variants"][active_crop_variant]

        cropped = self.apply_crop(self.original_image, crop_data)
        return self.run_parallel_pipeline(cropped, preset)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite polynomial smoothstep for seamless mask thresholds."""
    x_scaled = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x_scaled * x_scaled * (3.0 - 2.0 * x_scaled)

def get_luminance(image: np.ndarray) -> np.ndarray:
    """Calculates Rec.709 relative luminance."""
    return np.dot(image, [0.2126, 0.7152, 0.0722])

def apply_lut_fast(channel: np.ndarray, points: list, lut_size: int = 4096) -> np.ndarray:
    """
    Generates a high-precision 1D LUT using CubicSpline and maps pixel values 
    via direct integer indexing rather than per-pixel interpolation.
    """
    if not points or len(points) < 2:
        return channel
    
    pts = np.array(points)
    pts = pts[np.argsort(pts[:, 0])] # Ensure monotonicity
    
    cs = CubicSpline(pts[:, 0], pts[:, 1], bc_type='natural')
    x_lut = np.linspace(0.0, 1.0, lut_size)
    y_lut = np.clip(cs(x_lut), 0.0, 1.0).astype(np.float32)
    
    # Map [0.0, 1.0] floats to integer indices [0, lut_size - 1]
    indices = np.clip(channel * (lut_size - 1), 0, lut_size - 1).astype(np.int32)
    return y_lut[indices]

def apply_color_matrix(image: np.ndarray, params: dict) -> np.ndarray:
    result = image.copy()
    
    # Subtractive Color Matrix (Channel Crosstalk)
    matrix = np.array(params.get("color_matrix", np.eye(3)), dtype=np.float32)
    if not np.array_equal(matrix, np.eye(3)):
        # Matrix dot product across the RGB channel axis
        result = np.dot(result, matrix.T)
        
    return np.clip(result, 0.0, 1.0)

def apply_bloom(image: np.ndarray, params: dict) -> np.ndarray:
    # if not params.get("enable_bloom", False) or params.get("bloom_strength", 0.0) <= 0:
    #     return image
        
    threshold = params.get("bloom_threshold", 0.70)
    radius = params.get("bloom_radius", 15.0)
    strength = params.get("bloom_strength", 0.15)
    
    # 1. Generate smooth luminance mask
    lum = get_luminance(image)
    mask = smoothstep(threshold, np.clip(threshold + 0.25, 0.0, 1.0), lum)
    
    # 2. Extract highlights and apply Gaussian diffusion
    highlights = image * mask[..., np.newaxis]
    
    # Ensure kernel size is odd and positive
    k_size = int(radius * 2) | 1
    blurred_highlights = cv2.GaussianBlur(highlights, (k_size, k_size), sigmaX=radius, sigmaY=radius)
    
    # 3. Screen blend composite
    bloom_layer = blurred_highlights * strength
    screen_blend = 1.0 - (1.0 - image) * (1.0 - bloom_layer)
    
    return np.clip(screen_blend, 0.0, 1.0)

def apply_halation(image: np.ndarray, params: dict) -> np.ndarray:
    if not params.get("enable_halation", False) or params.get("halation_strength", 0.0) <= 0:
        return image
        
    threshold = params.get("halation_threshold", 0.60)
    radius = params.get("halation_radius", 8.0)
    strength = params.get("halation_strength", 0.30)
    offset_x = params.get("halation_offset_x", 0.0)
    offset_y = params.get("halation_offset_y", 0.0)
    
    # 1. Calculate Red Dominance + Luminance mask
    lum = get_luminance(image)
    red_dom = np.clip((image[..., 0] * 2.0 - image[..., 1] - image[..., 2]), 0.0, 1.0)
    
    # Halation occurs where light is bright AND warm/neutral
    halation_mask = smoothstep(threshold, 1.0, lum) * smoothstep(0.1, 0.5, red_dom + lum)
    
    # 2. Isolate red channel highlights and blur
    red_highlights = image[..., 0] * halation_mask
    k_size = int(radius * 2) | 1
    blurred_red = cv2.GaussianBlur(red_highlights, (k_size, k_size), sigmaX=radius, sigmaY=radius)
    
    # 3. Apply spatial affine translation (emulsion scatter / chromatic shift)
    if offset_x != 0.0 or offset_y != 0.0:
        rows, cols = image.shape[:2]
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        blurred_red = cv2.warpAffine(blurred_red, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        
    # 4. Composite exclusively onto the Red channel via Screen blend
    result = image.copy()
    red_glow = blurred_red * strength
    result[..., 0] = 1.0 - (1.0 - result[..., 0]) * (1.0 - red_glow)
    
    return np.clip(result, 0.0, 1.0)

def apply_smart_grain(image: np.ndarray, params: dict) -> np.ndarray:
    if not params.get("enable_grain", False) or params.get("grain_strength", 0.0) <= 0:
        return image
        
    strength = params.get("grain_strength", 0.05)
    size = max(0.5, params.get("grain_size", 1.0))
    chroma_weight = np.clip(params.get("grain_chroma", 0.1), 0.0, 1.0)
    
    h, w = image.shape[:2]
    
    # 1. Calculate dimensions for grain crystal scaling
    gh, gw = int(h / size), int(w / size)
    
    # 2. Generate luma and chroma noise textures
    luma_noise = np.random.normal(0.0, 1.0, (gh, gw)).astype(np.float32)
    
    if chroma_weight > 0:
        chroma_noise = np.random.normal(0.0, 1.0, (gh, gw, 3)).astype(np.float32)
        # Combine mono and color noise based on chroma weight
        noise = (chroma_noise * chroma_weight) + (luma_noise[..., np.newaxis] * (1.0 - chroma_weight))
    else:
        noise = luma_noise[..., np.newaxis]
        
    # 3. Scale noise up to image resolution using bicubic interpolation for organic softness
    if size != 1.0:
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        if noise.ndim == 2:
            noise = noise[..., np.newaxis]
            
    # 4. Parabolic luminance mask: peak grain at 50% gray (0.5), zero at 0.0 and 1.0
    lum = get_luminance(image)[..., np.newaxis]
    luma_mask = 4.0 * lum * (1.0 - lum)
    
    # 5. Apply grain via linear addition modulated by the luma mask
    grain_layer = noise * strength * luma_mask
    return np.clip(image + grain_layer, 0.0, 1.0)

def apply_vignette_and_softness(image: np.ndarray, params: dict) -> np.ndarray:
    if not params.get("enable_vignette", False) or params.get("vignette_strength", 0.0) <= 0:
        return image
        
    strength = np.clip(params.get("vignette_strength", 0.3), 0.0, 1.0)
    radius = params.get("vignette_radius", 0.75)
    softness = max(0.01, params.get("vignette_softness", 0.45))
    blur_radius = params.get("corner_blur_radius", 0.0)
    
    h, w = image.shape[:2]
    
    # 1. Generate aspect-ratio corrected radial distance grid [0.0 to ~1.4]
    y, x = np.ogrid[-1.0:1.0:h*1j, -1.0:1.0:w*1j]
    aspect = w / float(h)
    x_corrected = x * aspect
    r = np.sqrt(x_corrected**2 + y**2) / np.sqrt(aspect**2 + 1.0)
    
    # 2. Compute smooth vignette falloff mask
    inner = max(0.0, radius - softness)
    outer = radius + softness
    vignette_mask = 1.0 - (smoothstep(inner, outer, r) * strength)
    vignette_mask = vignette_mask[..., np.newaxis]
    
    result = image.copy()
    
    # 3. Apply corner optical softness (vintage wide-angle lens emulation)
    if blur_radius > 0:
        k_size = int(blur_radius * 2) | 1
        blurred_img = cv2.GaussianBlur(image, (k_size, k_size), sigmaX=blur_radius)
        
        # Blur mask increases toward the corners
        blur_mask = smoothstep(radius * 0.5, outer, r)[..., np.newaxis]
        result = (result * (1.0 - blur_mask)) + (blurred_img * blur_mask)
        
    # 4. Apply exposure falloff
    return np.clip(result * vignette_mask, 0.0, 1.0)

def export_photo(img_array: np.ndarray, output_path: str, preset: Dict[str, Any], max_mb: float = 8.0):
    """Applies film noise grain overlays and frames pre-cropped, processed image matrices.

    Args:
        img_array (np.ndarray): Pre-rendered structural pipeline array reference.
        output_path (str): The target location path value to write to disk layout.
        preset (Dict[str, Any]): Presets mapping properties map dictionary layout.
        max_mb (float, optional): Maximum compressed file boundary. Defaults to 8.0.
    """
    final_img_array = (np.clip(img_array, 0.0, 1.0) * 255.0).astype(np.uint8)

    active_crop_variant = preset.get('active_crop_variant', 'default')
    crop_variant_data = preset['crop_variants'][active_crop_variant]
    
    # grain = preset.get('grain', 0.0)
    # grain_size = preset.get('grain_size', 1.0)
    # if grain > 0.0:
    #     fh, fw, fc = final_img_array.shape
    #     g_size = max(0.1, grain_size)
    #     noise_h, noise_w = max(1, int(fh / g_size)), max(1, int(fw / g_size))
    #     noise = np.random.normal(0, grain * 12.7, (noise_h, noise_w, fc)).astype(np.float32)
    #     if g_size != 1.0:
    #         noise = cv2.resize(noise, (fw, fh), interpolation=cv2.INTER_LINEAR)
    #     final_img_array = np.clip(final_img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if crop_variant_data.get('add_white_border', False):
        final_img_array = PhotoEditor.apply_white_border(final_img_array, preset)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_pil = Image.fromarray(final_img_array)
    do_instagram_compression = crop_variant_data.get('do_instagram_compression', True)

    if do_instagram_compression:
        quality = 95
        while quality >= 70:
            img_pil.save(output_path, format='JPEG', quality=quality)
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            if file_size_mb <= max_mb:
                break
            quality -= 5
    else:
        img_pil.save(output_path, format='JPEG', quality=92)

def playbook_single_file(file_path: str, output_dir: str, preset_data: Dict[str, Any]):
    _, input_filename = os.path.split(file_path)
    filename_wo_ext, _ = os.path.splitext(input_filename)
    output_filename = f"{filename_wo_ext}_edit.jpg"
    final_output_path = os.path.join(output_dir, output_filename)

    editor = PhotoEditor(file_path)
    processed_array = editor.apply_presets(preset_data)
    export_photo(processed_array, final_output_path, preset_data)


def main():
    parser = argparse.ArgumentParser(description="Multi-threaded batch processing engine CLI.")
    parser.add_argument("-i", "--input", required=True, help="Path to input file or folder.")
    parser.add_argument("-p", "--preset", required=True, help="Path to JSON preset.")
    args = parser.parse_args()

    with open(args.preset, "r") as f:
        preset_data = json.load(f)

    if os.path.isdir(args.input):
        output_dir = os.path.join(args.input, "edits")
        target_images = [f for f in os.listdir(args.input) if f.lower().endswith(SUPPORTED_EXTENSIONS) and not f.lower().endswith("_edit.jpg")]
        for filename in target_images:
            playbook_single_file(os.path.join(args.input, filename), output_dir, preset_data)
    else:
        playbook_single_file(args.input, os.path.join(os.path.dirname(args.input), "edits"), preset_data)


if __name__ == "__main__":
    main()