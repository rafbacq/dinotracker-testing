"""
Extract frames from ultrasound MP4 videos with grayscale→RGB conversion.

DINO-Tracker expects RGB (3-channel) input frames since DINOv2 was pretrained
on ImageNet. This script handles:
  1. Reading grayscale ultrasound MP4 frames
  2. Converting single-channel → 3-channel by triplicating the grayscale channel
  3. Temporal subsampling to keep the frame count tractable
  4. Saving as JPEG files in the DINO-Tracker expected directory structure

Why triplicate rather than colormap?
  Triplication preserves the original intensity structure without injecting
  artificial color patterns that could confuse DINO's learned features.
  DINOv2 applies ImageNet normalization (per-channel mean/std subtraction),
  which still works well with identical channels — the model just sees a
  "monochrome" image in its feature space.
"""

import argparse
import os
import sys
import numpy as np
import imageio

# Ensure project root is on sys.path for subprocess execution
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from preprocessing.crop_ultrasound_roi import detect_ultrasound_roi, crop_to_roi


def extract_frames(
    video_path: str,
    output_folder: str,
    subsample_rate: int = 1,
    max_frames: int = None,
    crop_roi: bool = True,
    roi_threshold: int = 10,
    roi_pad: int = 5,
):
    """
    Extract and convert frames from a grayscale ultrasound MP4.

    Args:
        video_path: Path to the input MP4 file.
        output_folder: Directory to save extracted JPEG frames.
        subsample_rate: Take every Nth frame. Higher = fewer frames.
        max_frames: Maximum number of frames to extract. None = no limit
                    (after subsampling).
        crop_roi: If True, detect and crop to the ultrasound sector ROI.
        roi_threshold: Intensity threshold (0-255) for ROI detection.
        roi_pad: Padding pixels around the detected ROI.
    """
    os.makedirs(output_folder, exist_ok=True)

    # First pass: count total frames
    reader = imageio.get_reader(video_path)
    total_raw_frames = reader.count_frames()
    reader.close()

    # Compute which frame indices to extract
    frame_indices = list(range(0, total_raw_frames, subsample_rate))

    # Apply max_frames cap
    if max_frames is not None and len(frame_indices) > max_frames:
        # Uniformly subsample to max_frames
        step = len(frame_indices) / max_frames
        frame_indices = [frame_indices[int(i * step)] for i in range(max_frames)]

    frame_indices_set = set(frame_indices)

    print(f"Video: {video_path}")
    print(f"  Total raw frames: {total_raw_frames}")
    print(f"  Subsample rate: {subsample_rate}")
    print(f"  Frames to extract: {len(frame_indices)}")

    # Second pass: read and save selected frames
    reader = imageio.get_reader(video_path)
    saved_count = 0
    out_idx = 0
    frame_rgb = None
    roi_bbox = None  # Detected once from the first frame, reused for all

    for frame_idx, frame in enumerate(reader):
        if frame_idx not in frame_indices_set:
            continue

        # Handle different frame shapes
        if len(frame.shape) == 2:
            # Grayscale: H x W → H x W x 3
            frame_rgb = np.stack([frame, frame, frame], axis=-1)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            # Grayscale with channel dim: H x W x 1 → H x W x 3
            frame_rgb = np.concatenate([frame, frame, frame], axis=-1)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            # Already RGB (some codecs decode grayscale as RGB)
            frame_rgb = frame
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            # RGBA → RGB
            frame_rgb = frame[:, :, :3]
        else:
            raise ValueError(
                f"Unexpected frame shape: {frame.shape} at index {frame_idx}"
            )

        # Ensure uint8
        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)

        # ROI cropping: detect from first frame, apply to all
        if crop_roi:
            if roi_bbox is None:
                roi_bbox = detect_ultrasound_roi(
                    frame_rgb,
                    threshold=roi_threshold,
                    pad=roi_pad,
                )
                if roi_bbox is not None:
                    y0, y1, x0, x1 = roi_bbox
                    print(f"  ROI detected: y=[{y0}:{y1}], x=[{x0}:{x1}] "
                          f"({x1 - x0}x{y1 - y0} from {frame_rgb.shape[1]}x{frame_rgb.shape[0]})")
                else:
                    print("  WARNING: ROI detection failed, saving uncropped frames")
            if roi_bbox is not None:
                frame_rgb = crop_to_roi(frame_rgb, roi_bbox)

        out_path = os.path.join(output_folder, f"{out_idx:05d}.jpg")
        imageio.imwrite(out_path, frame_rgb)
        out_idx += 1
        saved_count += 1

    reader.close()

    print(f"  Saved {saved_count} frames to {output_folder}")
    if frame_rgb is not None:
        print(f"  Frame shape: {frame_rgb.shape}")
    return saved_count


def compute_subsample_rate(video_path: str, target_frames: int = 150) -> int:
    """
    Compute the subsampling rate to reach approximately target_frames.

    Args:
        video_path: Path to the input MP4 file.
        target_frames: Desired number of output frames.

    Returns:
        Subsampling rate (integer >= 1).
    """
    reader = imageio.get_reader(video_path)
    total_frames = reader.count_frames()
    reader.close()

    rate = max(1, total_frames // target_frames)
    return rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ultrasound frames with grayscale→RGB conversion"
    )
    parser.add_argument(
        "--video-path", type=str, required=True, help="Path to input MP4 file"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Directory to save extracted frames",
    )
    parser.add_argument(
        "--subsample-rate",
        type=int,
        default=1,
        help="Take every Nth frame (default: 1 = all frames)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: no limit)",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=None,
        help="Auto-compute subsample rate to reach this many frames",
    )
    parser.add_argument(
        "--crop-roi",
        action="store_true",
        default=True,
        help="Crop to ultrasound fan sector ROI (default: True)",
    )
    parser.add_argument(
        "--no-crop-roi",
        action="store_true",
        help="Disable ROI cropping",
    )
    parser.add_argument(
        "--roi-threshold",
        type=int,
        default=10,
        help="Intensity threshold for ROI detection (default: 10)",
    )
    parser.add_argument(
        "--roi-pad",
        type=int,
        default=5,
        help="Padding pixels around detected ROI (default: 5)",
    )
    args = parser.parse_args()

    subsample_rate = args.subsample_rate
    if args.target_frames is not None:
        subsample_rate = compute_subsample_rate(args.video_path, args.target_frames)
        print(f"Auto-computed subsample rate: {subsample_rate}")

    do_crop = args.crop_roi and not args.no_crop_roi

    extract_frames(
        video_path=args.video_path,
        output_folder=args.output_folder,
        subsample_rate=subsample_rate,
        max_frames=args.max_frames,
        crop_roi=do_crop,
        roi_threshold=args.roi_threshold,
        roi_pad=args.roi_pad,
    )
