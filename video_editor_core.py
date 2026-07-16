import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2  # Lightweight way to grab exact source FPS
from photo_editor_core import PhotoEditor, export_photo


class VideoEditor:
    def __init__(self, photo_editor_instance=None):
        """
        Accepts an existing PhotoEditor instance or creates a new one.
        """
        self.editor = PhotoEditor

    def _get_framerate(self, video_path: Path) -> float:
        """Extracts the exact frames-per-second (FPS) from the source video."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 30.0  # Fallback to 30 FPS if metadata is missing

    def _process_single_frame(self, input_path: Path, output_path: Path, preset : Dict[str, Any], crop_settings : Dict[str, Any]):
        """Worker function to run a single frame through the PhotoEditor pipeline."""
        matrix = self.editor.load_image_matrix(str(input_path))
        
        if crop_settings:
            matrix = self.editor.apply_crop(matrix, crop_settings)
            
        processed_matrix = self.editor.run_parallel_pipeline(matrix, preset)
        
        # Exporting directly to the output temp path
        export_photo(processed_matrix, str(output_path), preset)
        return output_path

    def process_video(self, video_path: str | Path, preset : Dict[str, Any]):
        source_path = Path(video_path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source video not found: {source_path}")

        # 1. Setup output directory structure: original_dir/edits/original_name_edit.mp4
        output_dir = source_path.parent / "edits"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = output_dir / f"{source_path.stem}_edit.mp4"

        # Grab exact FPS to prevent audio/video drift during stitching
        fps = self._get_framerate(source_path)

        # Use a temporary directory for extracting thousands of intermediate frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_frames_dir = temp_path / "raw"
            processed_frames_dir = temp_path / "processed"
            raw_frames_dir.mkdir()
            processed_frames_dir.mkdir()

            # 2. Extract frames as JPEGs (lossless to avoid compression artifacts before editing)
            print(f"Extracting frames from {source_path.name}...")
            extract_cmd = [
                "ffmpeg", "-y",
                "-i", str(source_path),
                "-c:v", "mjpeg",
                "-q:v", "2",  # don't compress the jpg
                str(raw_frames_dir / "frame_%07d.jpg")
            ]
            subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Gather all extracted frame paths
            frame_files = sorted(list(raw_frames_dir.glob("frame_*.jpg")))
            total_frames = len(frame_files)
            print(f"Extracted {total_frames} frames. Starting processing...")

            # 3. Process frames (Parallelized across CPU cores for performance)
            # Note: If run_parallel_pipeline is already heavily utilizing multi-threading internally,
            # you may want to adjust max_workers to avoid CPU thrashing.
            with ProcessPoolExecutor() as executor:
                futures = []
                for frame_path in frame_files:
                    out_frame_path = processed_frames_dir / frame_path.name
                    futures.append(
                        executor.submit(self._process_single_frame, frame_path, out_frame_path, preset, preset['crop_variants'][preset['active_crop_variant']])
                    )

                for i, future in enumerate(as_completed(futures), 1):
                    if i % 25 == 0 or i == total_frames:
                        print(f"Processed {i}/{total_frames} frames ({i/total_frames*100:.1f}%)")

            # 4. Stitch frames back together and mux with original audio
            print("Stitching frames and mixing original audio...")
            stitch_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(processed_frames_dir / "frame_%07d.jpg"),  # Processed video stream (Input 0)
                "-i", str(source_path),                              # Source audio stream (Input 1)
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",  # Highly recommended for smartphone/mobile playback compatibility
                "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",  # Shave off 1px to ensure width and height are both even numbers to avoid a failure
                "-crf", "18",           # Visually lossless H.264 encoding quality
                "-c:a", "copy",         # Copy audio directly without re-encoding
                "-map", "0:v:0",        # Take video from input 0
                "-map", "1:a:0?",       # Take audio from input 1 (trailing '?' ignores if video has no audio)
                "-shortest",            # Match duration to the shortest stream
                str(final_output_path)
            ]
            try:
                subprocess.run(stitch_cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print("\n" + "="*50)
                print("FFMPEG STITCH FAILED - ERROR LOG:")
                print("="*50)
                print(e.stderr)  # This will print the exact reason FFmpeg died
                print("="*50 + "\n")
                raise
        print(f"Finished! Output saved to: {final_output_path}")
        return final_output_path
    
    def get_auto_preset_from_video(self, video_path: str | Path, is_linear: bool = True) -> Dict[str, Any]:
        """
        Seeks to the exact middle frame of a video, loads it through PhotoEditor's `
        matrix loader for pipeline consistency, and generates the auto preset.
        """
        source_path = Path(video_path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Video file not found: {source_path}")

        cap = cv2.VideoCapture(str(source_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate the middle frame index
        mid_frame_idx = max(0, total_frames // 2)
        
        # Seek directly to the middle frame without decoding prior frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise RuntimeError(f"Failed to extract frame at index {mid_frame_idx} from {source_path.name}")

        # Write to a temporary lossless PNG to guarantee load_image_matrix applies its standard color/dtype formatting
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            cv2.imwrite(temp_path, frame)
            matrix = self.editor.load_image_matrix(temp_path)
            
            # Since calculate_auto_preset is a class method (@classmethod), 
            # we can call it directly on the class or via the instance
            preset = self.editor.calculate_auto_preset(matrix, is_linear=is_linear)
            return preset
        finally:
            # Clean up the temporary frame file
            Path(temp_path).unlink(missing_ok=True)

if __name__ == '__main__':
    video_file = r"C:\Users\bmorgan\Pictures\phone_videos\gem lake july 2026\20260714_081453.mp4"
    
    # 1. Initialize the video editor (which initializes or wraps PhotoEditor)
    v_editor = VideoEditor()
    
    # 2. Extract the middle frame and calculate the auto preset
    print("Calculating auto preset from middle frame...")
    auto_edit_results = v_editor.get_auto_preset_from_video(video_file, is_linear=True)

    preset = PhotoEditor.DEFAULT_PRESET.copy()

    for key, value in auto_edit_results.items():
        if isinstance(value, dict):
            for _k, _v in value.items():
                preset[key][_k] = _v
        else:
            preset[key] = value
    
    preset['crop_variants'][preset['active_crop_variant']]['add_white_border'] = False
    
    print(f"Generated Preset: {preset}")
    
    # 3. Process the entire video sequence using that preset
    print("Starting video processing pipeline...")
    output_path = v_editor.process_video(video_file, preset)