"""Core image processing pipeline module for raw and standard photography assets.

This module provides high-performance filters, 8-band HSL color manipulations,
geometric transformations, and dynamic range tonemapping using multi-threaded
stripe segmentation algorithms.
"""

import argparse
import json
import os
import sys



import logging
from typing import Dict, Any, List
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


class PhotoEditor:
    """Handles core image processing operations for photo editing filters.

    Attributes:
        image_path (str): The absolute disk destination path to the targeted asset file.
        original_image (np.ndarray): High-precision floating-point source image matrix.
    """

    def __init__(self, image_path: str):
        """Initializes the PhotoEditor instance by reading the image matrix into memory.

        Args:
            image_path (str): The path to the image asset file.
        """
        self.image_path = image_path
        logger.info(f"Initializing PhotoEditor instance matrix for: {os.path.basename(image_path)}")
        self.original_image = self.load_image_matrix(image_path)
        
    @staticmethod
    def load_image_matrix(image_path: str, preview: bool = False) -> np.ndarray:
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
                    if preview and matrix.shape[1] > 1200:
                        scale = 1200.0 / matrix.shape[1]
                        matrix = cv2.resize(matrix, (1200, int(matrix.shape[0] * scale)), interpolation=cv2.INTER_AREA)
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

            if blacks != 0.0:
                b_mask = np.clip((np.float32(0.3) - lum) * np.float32(3.33), 0.0, 1.0)
                new_lum += (blacks * np.float32(0.15)) * (b_mask ** 2)

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

        # 10. Optional Smoothing
        blur_radius = preset.get('gaussian_blur', 0.0)
        if blur_radius > 0:
            sigma = float(blur_radius * 1.5)
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # 11. Display Preparation: Tonemapping at the end of float processing
        hdr_comp = preset.get('hdr_compression', 0.0)
        if hdr_comp > 0.0:
            img = cls._apply_sdr_preview(img, hdr_comp)

        # 12. Final In-Place Hard Clip to SDR Monitor Bounds [0.0, 1.0]
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
        crop_variant_data = preset.get(active_crop_variant, dict())
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
        current_crop_variant = preset.get("active_crop_variant", "default")
        crop_data = preset["crop_variants"].get(current_crop_variant, dict())

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

    def apply_presets(self, preset: Dict[str, Any]) -> np.ndarray:
        """Unified instance abstraction method executing the full pipeline pass.

        Args:
            preset (Dict[str, Any]): Target metrics properties layout map.

        Returns:
            np.ndarray: Rendered image matrix.
        """

        current_crop_variant = preset.get("active_crop_variant", "default")
        crop_data = preset["crop_variants"].get(current_crop_variant)

        cropped = self.apply_crop(self.original_image, crop_data)
        return self.run_parallel_pipeline(cropped, preset)


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
    crop_variant_data = preset.get(active_crop_variant, dict())
    
    grain = preset.get('grain', 0.0)
    grain_size = preset.get('grain_size', 1.0)
    if grain > 0.0:
        fh, fw, fc = final_img_array.shape
        g_size = max(0.1, grain_size)
        noise_h, noise_w = max(1, int(fh / g_size)), max(1, int(fw / g_size))
        noise = np.random.normal(0, grain * 12.7, (noise_h, noise_w, fc)).astype(np.float32)
        if g_size != 1.0:
            noise = cv2.resize(noise, (fw, fh), interpolation=cv2.INTER_LINEAR)
        final_img_array = np.clip(final_img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)

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