"""A headless photo editing utility mimicking basic Adobe Lightroom functionality.

This script processes individual images or entire directories of JPEG and HEIC
images using adjustments defined in a JSON preset file. It handles adjustments
like HDR-to-SDR filmic tone mapping, exposure, global contrast, global vibrance,
global saturation, highlights, shadows, clarity, texture, customizable grain
structures, and selective color adjustments before exporting Instagram-optimized JPEGs.
It supports macro group multipliers and toggles for isolating specific adjustments.
"""

import argparse
import json
import os
from typing import Dict, Any, List
import cv2
import numpy as np
from PIL import Image, ImageOps
import pillow_heif

# Register HEIF opener with Pillow to natively support .heic files
pillow_heif.register_heif_opener()

# Supported file extensions for batch processing
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".heic")


class PhotoEditor:
    """Handles core image processing operations for photo editing filters."""

    def __init__(self, image_path: str):
        """Initializes the PhotoEditor with an image path.

        Args:
            image_path: Path to the input JPEG or HEIC image.
        """
        self.image_path = image_path
        self.original_image = self._load_image()
        
    def _load_image(self) -> np.ndarray:
        """Loads HEIC or JPEG image and converts it to a float32 RGB NumPy array.

        This method accounts for smartphone EXIF orientation metadata, transforming
        the raw pixel data to match its intended visual orientation prior to conversion.

        Returns:
            A NumPy array of shape (H, W, 3) normalized to values between 0.0 and 1.0.
        """
        with Image.open(self.image_path) as img:
            # Transpose the pixel array based on EXIF data to lock in correct orientation
            img = ImageOps.exif_transpose(img)

            if img.mode in ("I;16", "I;16B", "I;16L", "RGBA"):
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            
            return np.array(img, dtype=np.float32) / 255.0

    def _apply_aces_tonemap(self, img: np.ndarray, intensity: float) -> np.ndarray:
        """Compresses high dynamic range data to SDR using an ACES filmic curve.

        Args:
            img: Float32 RGB image array.
            intensity: Slider value from 0.0 (none) to 1.0 (maximum compression).

        Returns:
            The tone-mapped float32 RGB image array.
        """
        if intensity <= 0.0:
            return img

        # Force constants to float32 to prevent implicit upcasting to float64
        a = np.float32(2.51)
        b = np.float32(0.03)
        c = np.float32(2.43)
        d = np.float32(0.59)
        e = np.float32(0.14)

        scaled_img = img * np.float32(1.0 + intensity * 1.5)
        tonemapped = (scaled_img * (a * scaled_img + b)) / (scaled_img * (c * scaled_img + d) + e)
        tonemapped = np.clip(tonemapped, 0.0, 1.0).astype(np.float32)

        # Ensure weights are standard floats for OpenCV's native signature mapping
        out = cv2.addWeighted(tonemapped, float(intensity), img, float(1.0 - intensity), 0)
        return out.astype(np.float32)

    def apply_presets(self, preset: Dict[str, Any]) -> np.ndarray:
        """Applies a dictionary of artistic sliders to the loaded image array.

        This method executes an image processing pipeline in a strict structural
        sequence to ensure filters do not introduce unnatural artifacts. All operations
        run in 32-bit float precision.

        Args:
            preset: A configuration dictionary mapping processing settings to 
                their slider scales. Expected top-level control keys include:
                
                - apply_temperature_adjustment (bool): Toggles color temperature 
                  processing. If False, the original image temperature is retained.
                - values_multiplier (float): Scales contrast, highlights, and shadows.
                - color_multiplier (float): Scales global vibrance and saturation.
                - color_adjustments_multiplier (float): Scales targeted color bands.

        Returns:
            A fully processed float32 numpy.ndarray with dimensions (H, W, 3) normalized [0.0-1.0].
        """
        img = np.copy(self.original_image).astype(np.float32)

        # Retrieve global group multipliers
        v_mult = np.clip(preset.get("values_multiplier", 1.0), 0.0, 1.0)
        c_mult = np.clip(preset.get("color_multiplier", 1.0), 0.0, 1.0)
        ca_mult = np.clip(preset.get("color_adjustments_multiplier", 1.0), 0.0, 1.0)

        # 1. HDR to SDR Dynamic Range Compression
        hdr_comp = preset.get("hdr_compression", 0.0)
        if hdr_comp > 0.0:
            img = self._apply_aces_tonemap(img, hdr_comp)

        # 2. White Balance (Color Temperature in Kelvin and Global Tint)
        apply_temp = preset.get("apply_temperature_adjustment", True)
        temp_kelvin = preset.get("temp_kelvin", 6500.0)
        tint = preset.get("tint", 0.0)  
        
        # Only compute color temperature transformations if explicitly enabled
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

        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        # 3. Exposure
        exposure = preset.get("exposure", 0.0)
        if exposure != 0.0:
            img = img * np.float32(2.0 ** exposure)

        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        # 4. Contrast
        contrast = preset.get("contrast", 0.0) * v_mult
        if contrast != 0.0:
            img = (img - np.float32(0.5)) * np.float32(1.0 + contrast) + np.float32(0.5)

        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        # 5. Highlights and Shadows
        highlights = preset.get("highlights", 0.0) * v_mult
        shadows = preset.get("shadows", 0.0) * v_mult

        if highlights != 0.0 or shadows != 0.0:
            luminance = np.float32(0.299) * img[:, :, 0] + np.float32(0.587) * img[:, :, 1] + np.float32(0.114) * img[:, :, 2]
            luminance = np.expand_dims(luminance, axis=2)

            if highlights != 0.0:
                hl_mask = np.power(luminance, 2)
                img = img + (np.float32(highlights) * hl_mask * (np.float32(1.0) - img) * np.float32(0.5))

            if shadows != 0.0:
                sh_mask = np.power(np.float32(1.0) - luminance, 2)
                img = img + (np.float32(shadows) * sh_mask * img * np.float32(0.5))

        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        # 6. Texture and Clarity (Local Contrast)
        texture = preset.get("texture", 0.0)
        if texture != 0.0:
            low_pass = cv2.GaussianBlur(img, (5, 5), 0)
            high_pass = img - low_pass
            img = img + (np.float32(texture) * high_pass * np.float32(0.4))

        clarity = preset.get("clarity", 0.0)
        if clarity != 0.0:
            low_pass_large = cv2.GaussianBlur(img, (31, 31), 0)
            mid_pass = img - low_pass_large
            img = img + (np.float32(clarity) * mid_pass * np.float32(0.3))

        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        # 7. Gaussian Blur
        blur_radius = preset.get("gaussian_blur", 0.0)
        if blur_radius > 0:
            k_size = int(blur_radius * 4) | 1
            if k_size > 1:
                img = cv2.GaussianBlur(img, (k_size, k_size), 0)

        # Explicit safety cast to prevent any implicit float64 type promotion from breaking cv2.cvtColor
        img = img.astype(np.float32)

        # 8. High-Precision Float32 Color Engine
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        vibrance = preset.get("vibrance", 0.0) * c_mult
        if vibrance != 0.0:
            hsv[:, :, 1] = hsv[:, :, 1] * (np.float32(1.0) + np.float32(vibrance) * (np.float32(1.0) - hsv[:, :, 1]))

        saturation = preset.get("saturation", 0.0) * c_mult
        if saturation != 0.0:
            hsv[:, :, 1] = hsv[:, :, 1] * (np.float32(1.0) + np.float32(saturation))

        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0.0, 1.0).astype(np.float32)

        color_adj = preset.get("color_adjustments", {})
        hue_ranges = {
            "red": [(0.0, 20.0), (340.0, 360.0)],
            "orange": [(20.0, 45.0)],
            "yellow": [(45.0, 70.0)],
            "green": [(70.0, 160.0)],
            "blue": [(160.0, 260.0)]
        }

        for color, adjustments in color_adj.items():
            if color not in hue_ranges:
                continue
                
            h_shift = adjustments.get("hue", 0.0) * 15.0 * ca_mult     
            s_shift = adjustments.get("sat", 0.0) * ca_mult
            
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for r in hue_ranges[color]:
                curr_mask = cv2.inRange(hsv[:, :, 0], r[0], r[1])
                mask = cv2.bitwise_or(mask, curr_mask)

            if h_shift != 0.0 or s_shift != 0.0:
                h_channel = hsv[:, :, 0].copy()
                s_channel = hsv[:, :, 1].copy()

                if h_shift != 0.0:
                    h_channel[mask > 0] = (h_channel[mask > 0] + np.float32(h_shift)) % np.float32(360.0)
                if s_shift != 0.0:
                    s_channel[mask > 0] = np.clip(s_channel[mask > 0] * (np.float32(1.0) + np.float32(s_shift)), 0.0, 1.0)

                hsv[:, :, 0] = h_channel
                hsv[:, :, 1] = s_channel

        final_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return np.clip(final_rgb, 0.0, 1.0).astype(np.float32)


def export_photo(img_array: np.ndarray, output_path: str, preset: Dict[str, Any], max_width: int = 1080, max_mb: float = 8.0):
    """Resizes down only if the width exceeds max_width, maintaining orientation and aspect ratio.

    Args:
        img_array: Processed float32 RGB NumPy array normalized between 0.0 and 1.0.
        output_path: Target path destination including file name.
        preset: Config parsing parameters container for checking grain details.
        max_width: Maximum width boundary allowed in pixels.
        max_mb: Ceiling target threshold for target file size on disk.
    """
    h, w, c = img_array.shape

    if w > max_width:
        aspect_ratio = h / w
        new_height = int(max_width * aspect_ratio)
        img_resized = cv2.resize(img_array, (max_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    else:
        img_resized = np.copy(img_array)

    final_img_array = (np.clip(img_resized, 0.0, 1.0) * 255.0).astype(np.uint8)

    grain = preset.get("grain", 0.0)
    grain_size = preset.get("grain_size", 1.0)

    if grain > 0.0:
        fh, fw, fc = final_img_array.shape
        g_size = max(0.1, grain_size)

        noise_h = max(1, int(fh / g_size))
        noise_w = max(1, int(fw / g_size))
        
        noise = np.random.normal(0, grain * 12.7, (noise_h, noise_w, fc)).astype(np.float32)
        
        if g_size != 1.0:
            noise = cv2.resize(noise, (fw, fh), interpolation=cv2.INTER_LINEAR)
            
        final_img_array = np.clip(final_img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_bgr = cv2.cvtColor(final_img_array, cv2.COLOR_RGB2BGR)

    quality = 100
    while quality >= 70:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        cv2.imwrite(output_path, img_bgr, params)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        if file_size_mb <= max_mb:
            break
        quality -= 5


def playbook_single_file(file_path: str, output_dir: str, preset_data: Dict[str, Any]):
    """Processes a single file and outputs it to the target edits folder.

    Args:
        file_path: Absolute or relative path to the image file.
        output_dir: Destination directory where the file will be saved.
        preset_data: Configuration slider mappings.
    """
    _, input_filename = os.path.split(file_path)
    filename_wo_ext, _ = os.path.splitext(input_filename)
    output_filename = f"{filename_wo_ext}_edit.jpg"
    final_output_path = os.path.join(output_dir, output_filename)

    editor = PhotoEditor(file_path)
    processed_array = editor.apply_presets(preset_data)
    export_photo(processed_array, final_output_path, preset_data)


def main():
    """Main parsing mechanism handles CLI context execution for files or folders."""
    parser = argparse.ArgumentParser(
        description="A lightweight pipeline engine for batching image adjustments."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to an input image file OR directory.")
    parser.add_argument("-p", "--preset", required=True, help="Path to JSON preset file configuration.")
    args = parser.parse_args()

    with open(args.preset, "r") as f:
        preset_data = json.load(f)

    if os.path.isdir(args.input):
        all_files = os.listdir(args.input)
        target_images = [
            f for f in all_files 
            if f.lower().endswith(SUPPORTED_EXTENSIONS) and not f.lower().endswith("_edit.jpg")
        ]
        
        total_images = len(target_images)
        if total_images == 0:
            print(f"No valid images found in directory: {args.input}")
            return

        output_dir = os.path.join(args.input, "edits")
        print(f"Found {total_images} images. Processing batch directory target...")

        for idx, filename in enumerate(target_images):
            full_input_path = os.path.join(args.input, filename)
            print(f"[{idx + 1}/{total_images}] Processing: {filename}...", end="", flush=True)
            
            try:
                playbook_single_file(full_input_path, output_dir, preset_data)
                print(" Done.")
            except Exception as e:
                print(f" Failed.\nError details: {e}")

        print(f"\nBatch processing complete. Target output destination: {output_dir}")

    elif os.path.isfile(args.input):
        if not args.input.lower().endswith(SUPPORTED_EXTENSIONS):
            print(f"Unsupported file format provided: {args.input}")
            return

        input_dir, _ = os.path.split(args.input)
        output_dir = os.path.join(input_dir, "edits")
        
        print(f"Processing single target image: {os.path.basename(args.input)}...", end="", flush=True)
        try:
            playbook_single_file(args.input, output_dir, preset_data)
            print(" Done.")
        except Exception as e:
            print(f" Failed.\nError details: {e}")
            
    else:
        print(f"Provided path does not exist or is inaccessible: {args.input}")


if __name__ == "__main__":
    main()