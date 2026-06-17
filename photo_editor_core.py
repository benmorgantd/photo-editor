"""A headless photo editing utility mimicking basic Adobe Lightroom functionality.

High-performance variation implementing a padded-stripe multi-threaded rendering
pipeline with isolated grain, center rotation skew fixes, and white print borders.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageOps
import pillow_heif
import rawpy

pillow_heif.register_heif_opener()

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.heic')
RAW_EXTENSIONS = ('.raf', '.cr2', '.nef', '.arw', '.dng')
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS + RAW_EXTENSIONS


class PhotoEditor:
    """Handles core image processing operations for photo editing filters."""

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original_image = self.load_image_matrix(image_path)
        
    @staticmethod
    def load_image_matrix(image_path: str, preview: bool = False) -> np.ndarray:
        """Loads image files into floating-point numpy structures with uniform color math."""
        ext = os.path.splitext(image_path)[1].lower()

        if ext in RAW_EXTENSIONS:
            with rawpy.imread(image_path) as raw:
                # Always decode RAW files with identical 16-bit demosaicing and gamma spaces
                rgb_16 = raw.postprocess(
                    use_camera_wb=True, 
                    half_size=False, 
                    no_auto_bright=True, 
                    output_bps=16
                )
                matrix = rgb_16.astype(np.float32) / 65535.0
                
                # Downsample via clean pixel interpolation ONLY after uniform demosaicing
                if preview and matrix.shape[1] > 1200:
                    scale = 1200.0 / matrix.shape[1]
                    matrix = cv2.resize(matrix, (1200, int(matrix.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                return matrix
        else:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                matrix = np.array(img, dtype=np.float32) / 255.0
                if preview and matrix.shape[1] > 1200:
                    scale = 1200.0 / matrix.shape[1]
                    matrix = cv2.resize(matrix, (1200, int(matrix.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                return matrix

    @staticmethod
    def _apply_aces_tonemap(img: np.ndarray, intensity: float) -> np.ndarray:
        if intensity <= 0.0:
            return img

        a = np.float32(2.51)
        b = np.float32(0.03)
        c = np.float32(2.43)
        d = np.float32(0.59)
        e = np.float32(0.14)

        scaled_img = img * np.float32(1.0 + intensity * 1.5)
        tonemapped = (scaled_img * (a * scaled_img + b)) / (scaled_img * (c * scaled_img + d) + e)
        np.clip(tonemapped, 0.0, 1.0, out=tonemapped)

        out = cv2.addWeighted(tonemapped, float(intensity), img, float(1.0 - intensity), 0)
        return out.astype(np.float32)

    @classmethod
    def run_pipeline(cls, src_matrix: np.ndarray, preset: Dict[str, Any]) -> np.ndarray:
        """Executes core image filters sequentially using in-place operations."""
        img = np.copy(src_matrix).astype(np.float32)

        v_mult = np.clip(preset.get('values_multiplier', 1.0), 0.0, 1.0)
        c_mult = np.clip(preset.get('color_multiplier', 1.0), 0.0, 1.0)
        ca_mult = np.clip(preset.get('color_adjustments_multiplier', 1.0), 0.0, 1.0)

        # 1. ACES Tonemapping
        hdr_comp = preset.get('hdr_compression', 0.0)
        if hdr_comp > 0.0:
            img = cls._apply_aces_tonemap(img, hdr_comp)

        # 2. White Balance
        apply_temp = preset.get('apply_temperature_adjustment', True)
        temp_kelvin = preset.get('temp_kelvin', 6500.0)
        tint = preset.get('tint', 0.0)  
        
        if apply_temp and temp_kelvin != 6500.0:
            if temp_kelvin > 6500.0:
                clipped_kelvin = min(temp_kelvin, 12000.0)
                temp_factor = (clipped_kelvin - 6500.0) / (12000.0 - 6500.0)
            else:
                clipped_kelvin = max(temp_kelvin, 2000.0)
                temp_factor = (clipped_kelvin - 6500.0) / (6500.0 - 2000.0)
            
            img[:, :, 0] += np.float32(temp_factor * 0.1)  
            img[:, :, 2] -= np.float32(temp_factor * 0.1)  

        if tint != 0.0:
            img[:, :, 1] += np.float32(tint * 0.05)
            img[:, :, 0] += np.float32(tint * 0.025)
            img[:, :, 2] += np.float32(tint * 0.025)

        np.clip(img, 0.0, 1.0, out=img)

        # 3. Exposure
        exposure = preset.get('exposure', 0.0)
        if exposure != 0.0:
            img *= np.float32(2.0 ** exposure)
            np.clip(img, 0.0, 1.0, out=img)

        # 4. Contrast
        contrast = preset.get('contrast', 0.0) * v_mult
        if contrast != 0.0:
            img -= np.float32(0.5)
            img *= np.float32(1.0 + contrast)
            img += np.float32(0.5)
            np.clip(img, 0.0, 1.0, out=img)

        # 5. Whites & Blacks
        whites = preset.get('whites', 0.0) * v_mult
        blacks = preset.get('blacks', 0.0) * v_mult
        if whites != 0.0 or blacks != 0.0:
            w_thr = np.float32(1.0 - (whites * 0.15))
            b_thr = np.float32(0.0 - (blacks * 0.15))
            if w_thr <= b_thr:
                w_thr = b_thr + np.float32(0.01)
            
            img = (img - b_thr) / (w_thr - b_thr)
            img = img.astype(np.float32)
            np.clip(img, 0.0, 1.0, out=img)

        # 6. Highlights and Shadows
        highlights = preset.get('highlights', 0.0) * v_mult
        shadows = preset.get('shadows', 0.0) * v_mult

        if highlights != 0.0 or shadows != 0.0:
            luminance = np.float32(0.299) * img[:, :, 0] + np.float32(0.587) * img[:, :, 1] + np.float32(0.114) * img[:, :, 2]
            luminance = np.expand_dims(luminance, axis=2)

            if highlights != 0.0:
                hl_mask = np.power(luminance, 2)
                img += (np.float32(highlights) * hl_mask * (np.float32(1.0) - img) * np.float32(0.5))
                del hl_mask

            if shadows != 0.0:
                sh_mask = np.power(np.float32(1.0) - luminance, 2)
                img += (np.float32(shadows) * sh_mask * img * np.float32(0.5))
                del sh_mask
            
            del luminance
            np.clip(img, 0.0, 1.0, out=img)

        # 7. Texture and Clarity
        texture = preset.get('texture', 0.0)
        if texture != 0.0:
            low_pass = cv2.GaussianBlur(img, (5, 5), 0)
            img += (np.float32(texture) * (img - low_pass) * np.float32(0.4))
            del low_pass

        clarity = preset.get('clarity', 0.0)
        if clarity != 0.0:
            h, w, _ = img.shape
            if w > 2200:
                sf = 4
                small_img = cv2.resize(img, (w // sf, h // sf), interpolation=cv2.INTER_AREA)
                small_blur = cv2.GaussianBlur(small_img, (31 // sf | 1, 31 // sf | 1), 0)
                large_blur = cv2.resize(small_blur, (w, h), interpolation=cv2.INTER_LINEAR)
                del small_img, small_blur
            else:
                large_blur = cv2.GaussianBlur(img, (31, 31), 0)
            
            img += (np.float32(clarity) * (img - large_blur) * np.float32(0.3))
            del large_blur

        np.clip(img, 0.0, 1.0, out=img)

        # 8. Gaussian Blur
        blur_radius = preset.get('gaussian_blur', 0.0)
        if blur_radius > 0:
            k_size = int(blur_radius * 4) | 1
            if k_size > 1:
                img = cv2.GaussianBlur(img, (k_size, k_size), 0)

        # 9. High-Precision Color Engine
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        vibrance = preset.get('vibrance', 0.0) * c_mult
        if vibrance != 0.0:
            hsv[:, :, 1] *= (np.float32(1.0) + np.float32(vibrance) * (np.float32(1.0) - hsv[:, :, 1]))

        saturation = preset.get('saturation', 0.0) * c_mult
        if saturation != 0.0:
            hsv[:, :, 1] *= (np.float32(1.0) + np.float32(saturation))

        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0.0, 1.0)

        color_adj = preset.get('color_adjustments', {})
        hue_ranges = {
            'red': [(0.0, 20.0), (340.0, 360.0)], 'orange': [(20.0, 45.0)],
            'yellow': [(45.0, 70.0)], 'green': [(70.0, 160.0)], 'blue': [(160.0, 260.0)]
        }

        for color, adjustments in color_adj.items():
            if color not in hue_ranges:
                continue
            h_shift = adjustments.get('hue', 0.0) * 15.0 * ca_mult     
            s_shift = adjustments.get('sat', 0.0) * ca_mult
            
            if h_shift == 0.0 and s_shift == 0.0:
                continue

            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for r in hue_ranges[color]:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv[:, :, 0], r[0], r[1]))

            h_channel = hsv[:, :, 0].copy()
            s_channel = hsv[:, :, 1].copy()
            
            if h_shift != 0.0:
                h_channel = np.where(mask > 0, (h_channel + np.float32(h_shift)) % np.float32(360.0), h_channel)
            if s_shift != 0.0:
                s_channel = np.where(mask > 0, np.clip(s_channel * (np.float32(1.0) + np.float32(s_shift)), 0.0, 1.0), s_channel)
                
            hsv[:, :, 0], hsv[:, :, 1] = h_channel, s_channel

        final_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return np.clip(final_rgb, 0.0, 1.0).astype(np.float32)

    @classmethod
    def run_parallel_pipeline(cls, src_matrix: np.ndarray, preset: Dict[str, Any]) -> np.ndarray:
        """Scales pre-cropped image matrices to match compression dimensions, then processes tiles concurrently."""
        h, w, c = src_matrix.shape
        do_instagram_compression = preset.get('do_instagram_compression', True)

        if do_instagram_compression:
            target_w = 1080
            if w != target_w:
                aspect_ratio = h / w
                src_matrix = cv2.resize(src_matrix, (target_w, int(target_w * aspect_ratio)), interpolation=cv2.INTER_LANCZOS4)
        else:
            pct = int(preset.get('resolution_percentage', 100)) / 100.0
            if pct < 1.0:
                src_matrix = cv2.resize(src_matrix, (int(w * pct), int(h * pct)), interpolation=cv2.INTER_LANCZOS4)
        
        h, w, c = src_matrix.shape
        if h < 100 or w < 100:
            return cls.run_pipeline(src_matrix, preset)

        output_matrix = np.empty_like(src_matrix)
        cores = os.cpu_count() or 4
        stripe_height = h // cores
        margin = 64  

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
    def apply_crop(img: np.ndarray, preset: Dict[str, Any]) -> np.ndarray:
        """Applies center rotation skew corrections and slices image bounding boxes."""
        h, w, _ = img.shape
        
        rotation = float(preset.get('crop_rotation', 0.0))
        if rotation != 0.0:
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rotation, 1.0)
            fill_color = (0.0, 0.0, 0.0) if img.dtype == np.float32 else (0, 0, 0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=fill_color)
            
        ratio_mode = preset.get('crop_aspect_ratio', 'Free')
        
        if ratio_mode == 'Free':
            box_w = int(max(10, min(100, preset.get('crop_free_width', 100))) / 100.0 * w)
            box_h = int(max(10, min(100, preset.get('crop_free_height', 100))) / 100.0 * h)
            if preset.get('crop_aspect_ratio_flipped', False):
                box_w, box_h = box_h, box_w
        else:
            if ratio_mode == 'Original':
                target_ratio = w / h
            elif ratio_mode == '1:1':
                target_ratio = 1.0
            elif ratio_mode == '4:5':
                target_ratio = 5.0 / 4.0 if w >= h else 4.0 / 5.0
            elif ratio_mode == '5:7':
                target_ratio = 7.0 / 5.0 if w >= h else 5.0 / 7.0
            elif ratio_mode == '8:10':
                target_ratio = 10.0 / 8.0 if w >= h else 8.0 / 10.0
            elif ratio_mode == '16:9':
                target_ratio = 16.0 / 9.0 if w >= h else 9.0 / 16.0
            else:
                target_ratio = w / h
                
            if preset.get('crop_aspect_ratio_flipped', False):
                target_ratio = 1.0 / target_ratio
                
            if w / h >= target_ratio:
                max_h = h
                max_w = int(h * target_ratio)
            else:
                max_w = w
                max_h = int(w / target_ratio)
                
            size_scale = max(10, min(100, preset.get('crop_size', 100))) / 100.0
            box_w = int(max_w * size_scale)
            box_h = int(max_h * size_scale)
            
        cx_pct = preset.get('crop_center_x', 50) / 100.0
        cy_pct = preset.get('crop_center_y', 50) / 100.0
        
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
        """Appends a white border by downscaling the inner frame to keep overall dimensions exact."""
        if not preset.get('add_white_border', False):
            return img
        h, w, _ = img.shape
        border_pct = preset.get('white_border_width_pct', 5)
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
        cropped = self.apply_crop(self.original_image, preset)
        return self.run_parallel_pipeline(cropped, preset)


def export_photo(img_array: np.ndarray, output_path: str, preset: Dict[str, Any], max_mb: float = 8.0):
    """Applies film noise grain overlays and frames pre-cropped, processed image matrices."""
    final_img_array = (np.clip(img_array, 0.0, 1.0) * 255.0).astype(np.uint8)
    
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

    if preset.get('add_white_border', False):
        final_img_array = PhotoEditor.apply_white_border(final_img_array, preset)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_pil = Image.fromarray(final_img_array)
    do_instagram_compression = preset.get('do_instagram_compression', True)

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