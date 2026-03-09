"""
DINO-Tracker Pipeline Orchestrator for Liver Ultrasound Videos

Automated pipeline that processes all ultrasound MP4 videos through the
complete DINO-Tracker workflow:
  1. Frame extraction (grayscale → RGB, temporal subsampling)
  2. Preprocessing (optical flow, DINO embeddings, FG masks, best-buddies)
  3. Per-video self-supervised training
  4. Grid-based trajectory inference
  5. Trajectory visualization

Usage:
    python run_ultrasound_pipeline.py [OPTIONS]

    # Process all videos with defaults:
    python run_ultrasound_pipeline.py

    # Process a single video:
    python run_ultrasound_pipeline.py --videos volunteer01.mp4

    # Skip training (if already trained):
    python run_ultrasound_pipeline.py --skip-training
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ─── Configuration ───────────────────────────────────────────────────────
DEFAULT_INPUT_DIR = "usliverseq-mp4"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_PREPROCESS_CONFIG = "config/ultrasound_preprocessing.yaml"
DEFAULT_TRAIN_CONFIG = "config/ultrasound_train.yaml"
DEFAULT_TARGET_FRAMES = 150     # Target frames per video after subsampling
DEFAULT_VIS_FPS = 10            # Visualization output framerate
DEFAULT_GRID_INTERVAL = 10     # Pixel interval for grid query points

# Project root (this file lives at the root of dino-tracker)
PROJECT_ROOT = Path(__file__).resolve().parent


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to both console and file."""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    logger = logging.getLogger("ultrasound_pipeline")
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(fh)

    logger.info(f"Logging to {log_file}")
    return logger


def run_step(cmd: list, step_name: str, logger: logging.Logger, cwd: str = None) -> bool:
    """
    Run a subprocess step with logging. Returns True on success, False on failure.
    """
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info(f"[{step_name}] Running: {cmd_str}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per step
        )
        elapsed = time.time() - start

        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                logger.debug(f"  [stdout] {line}")
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                logger.debug(f"  [stderr] {line}")

        if result.returncode != 0:
            logger.error(
                f"[{step_name}] FAILED (return code {result.returncode}) "
                f"after {elapsed:.1f}s"
            )
            logger.error(f"  stderr: {result.stderr[-500:] if result.stderr else 'none'}")
            return False

        logger.info(f"[{step_name}] Completed in {elapsed:.1f}s")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"[{step_name}] TIMEOUT after 7200s")
        return False
    except Exception as e:
        logger.error(f"[{step_name}] Exception: {e}")
        return False


def discover_videos(input_dir: str, specific_videos: list = None) -> list:
    """Find all MP4 files in the input directory."""
    if specific_videos:
        return [os.path.join(input_dir, v) for v in specific_videos]
    videos = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    return videos


def get_video_name(video_path: str) -> str:
    """Extract volunteer name from video path, e.g. 'volunteer01'."""
    return Path(video_path).stem


def process_video(
    video_path: str,
    output_dir: str,
    preprocess_config: str,
    train_config: str,
    target_frames: int,
    vis_fps: int,
    grid_interval: int,
    logger: logging.Logger,
    skip_extraction: bool = False,
    skip_preprocessing: bool = False,
    skip_training: bool = False,
    skip_inference: bool = False,
    skip_visualization: bool = False,
) -> dict:
    """
    Process a single video through the complete DINO-Tracker pipeline.

    Returns a dict with status for each step.
    """
    video_name = get_video_name(video_path)
    data_path = os.path.join(output_dir, video_name)
    video_frames_dir = os.path.join(data_path, "video")
    results = {"video": video_name, "steps": {}}

    logger.info(f"{'='*60}")
    logger.info(f"Processing: {video_name}")
    logger.info(f"  Input: {video_path}")
    logger.info(f"  Output: {data_path}")
    logger.info(f"{'='*60}")

    # ── Step 1: Frame extraction ──────────────────────────────────────
    if not skip_extraction:
        success = run_step(
            [
                sys.executable,
                "preprocessing/extract_ultrasound_frames.py",
                "--video-path", video_path,
                "--output-folder", video_frames_dir,
                "--target-frames", str(target_frames),
            ],
            step_name=f"{video_name}/extract_frames",
            logger=logger,
        )
        results["steps"]["extraction"] = "success" if success else "failed"
        if not success:
            logger.error(f"Frame extraction failed for {video_name}, skipping.")
            return results
    else:
        results["steps"]["extraction"] = "skipped"

    # Verify frames exist
    frame_count = len(list(Path(video_frames_dir).glob("*.jpg")))
    if frame_count == 0:
        logger.error(f"No frames found in {video_frames_dir}")
        results["steps"]["extraction"] = "no_frames"
        return results
    logger.info(f"  Frame count: {frame_count}")

    # ── Step 2: Preprocessing ─────────────────────────────────────────
    if not skip_preprocessing:
        success = run_step(
            [
                sys.executable,
                "preprocessing/main_preprocessing.py",
                "--config", preprocess_config,
                "--data-path", data_path,
            ],
            step_name=f"{video_name}/preprocess",
            logger=logger,
        )
        results["steps"]["preprocessing"] = "success" if success else "failed"
        if not success:
            logger.error(f"Preprocessing failed for {video_name}, skipping.")
            return results
    else:
        results["steps"]["preprocessing"] = "skipped"

    # ── Step 3: Training ──────────────────────────────────────────────
    if not skip_training:
        success = run_step(
            [
                sys.executable,
                "train.py",
                "--config", train_config,
                "--data-path", data_path,
            ],
            step_name=f"{video_name}/train",
            logger=logger,
        )
        results["steps"]["training"] = "success" if success else "failed"
        if not success:
            logger.error(f"Training failed for {video_name}, continuing to next.")
            return results
    else:
        results["steps"]["training"] = "skipped"

    # ── Step 4: Inference ─────────────────────────────────────────────
    if not skip_inference:
        success = run_step(
            [
                sys.executable,
                "inference_grid.py",
                "--config", train_config,
                "--data-path", data_path,
                "--interval", str(grid_interval),
                "--use-segm-mask",
            ],
            step_name=f"{video_name}/inference",
            logger=logger,
        )
        results["steps"]["inference"] = "success" if success else "failed"
        if not success:
            logger.error(f"Inference failed for {video_name}, continuing.")
            return results
    else:
        results["steps"]["inference"] = "skipped"

    # ── Step 5: Visualization ─────────────────────────────────────────
    if not skip_visualization:
        # Read the video resolution from the config for correct scaling
        import yaml
        with open(train_config, "r") as f:
            tcfg = yaml.safe_load(f)
        infer_h = tcfg["video_resh"]
        infer_w = tcfg["video_resw"]

        success = run_step(
            [
                sys.executable,
                "visualization/visualize_rainbow.py",
                "--data-path", data_path,
                "--infer-res-size", str(infer_h), str(infer_w),
                "--of-res-size", str(infer_h), str(infer_w),
                "--fps", str(vis_fps),
            ],
            step_name=f"{video_name}/visualize",
            logger=logger,
        )
        results["steps"]["visualization"] = "success" if success else "failed"
    else:
        results["steps"]["visualization"] = "skipped"

    return results


def main():
    parser = argparse.ArgumentParser(
        description="DINO-Tracker Pipeline for Ultrasound Videos"
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing MP4 videos (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--preprocess-config",
        default=DEFAULT_PREPROCESS_CONFIG,
        help="Preprocessing config YAML",
    )
    parser.add_argument(
        "--train-config",
        default=DEFAULT_TRAIN_CONFIG,
        help="Training config YAML",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help="Specific video filenames to process (e.g., volunteer01.mp4)",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=DEFAULT_TARGET_FRAMES,
        help=f"Target frame count per video (default: {DEFAULT_TARGET_FRAMES})",
    )
    parser.add_argument(
        "--vis-fps",
        type=int,
        default=DEFAULT_VIS_FPS,
        help=f"Visualization output FPS (default: {DEFAULT_VIS_FPS})",
    )
    parser.add_argument(
        "--grid-interval",
        type=int,
        default=DEFAULT_GRID_INTERVAL,
        help=f"Grid interval for query points (default: {DEFAULT_GRID_INTERVAL})",
    )
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    logger.info("DINO-Tracker Ultrasound Pipeline")
    logger.info(f"  Input dir:  {args.input_dir}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Target frames per video: {args.target_frames}")

    # Discover videos
    videos = discover_videos(args.input_dir, args.videos)
    if not videos:
        logger.error(f"No MP4 videos found in {args.input_dir}")
        sys.exit(1)
    logger.info(f"  Found {len(videos)} videos: {[Path(v).name for v in videos]}")

    # Process each video
    all_results = []
    for i, video_path in enumerate(videos):
        logger.info(f"\n{'#'*60}")
        logger.info(f"VIDEO {i+1}/{len(videos)}: {Path(video_path).name}")
        logger.info(f"{'#'*60}")

        try:
            result = process_video(
                video_path=video_path,
                output_dir=args.output_dir,
                preprocess_config=args.preprocess_config,
                train_config=args.train_config,
                target_frames=args.target_frames,
                vis_fps=args.vis_fps,
                grid_interval=args.grid_interval,
                logger=logger,
                skip_extraction=args.skip_extraction,
                skip_preprocessing=args.skip_preprocessing,
                skip_training=args.skip_training,
                skip_inference=args.skip_inference,
                skip_visualization=args.skip_visualization,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Unhandled exception processing {video_path}: {e}")
            all_results.append(
                {"video": get_video_name(video_path), "steps": {"error": str(e)}}
            )

    # ── Summary ───────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE SUMMARY")
    logger.info(f"{'='*60}")

    summary_path = os.path.join(args.output_dir, "pipeline_summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "target_frames": args.target_frames,
        "results": all_results,
    }

    for r in all_results:
        steps_str = ", ".join(f"{k}: {v}" for k, v in r["steps"].items())
        status = "✓" if all(v == "success" or v == "skipped" for v in r["steps"].values()) else "✗"
        logger.info(f"  {status} {r['video']}: {steps_str}")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")

    # Check for failures
    any_failures = any(
        any(v == "failed" for v in r["steps"].values()) for r in all_results
    )
    if any_failures:
        logger.warning("Some videos had failures. Check logs for details.")
        sys.exit(1)
    else:
        logger.info("All videos processed successfully!")


if __name__ == "__main__":
    main()
