"""
Flow Matching Pipeline Orchestrator for Liver Ultrasound Videos.

Runs the Flow Matching tracker on preprocessed ultrasound data.
Assumes preprocessing (frame extraction, DINO embeddings, optical flow
trajectories, masks) has already been done by the DinoTracker pipeline.

Pipeline:
    1. [SKIP] Frame extraction (done by DinoTracker pipeline)
    2. [SKIP] Preprocessing (done by DinoTracker pipeline)
    3. Flow Matching training
    4. Grid-based trajectory inference
    5. Trajectory visualization
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_FM_CONFIG = "config/ultrasound_flow_matching.yaml"
DEFAULT_TRAIN_CONFIG = "config/ultrasound_train.yaml"  # For visualization res
DEFAULT_VIS_FPS = 10
DEFAULT_GRID_INTERVAL = 10


def setup_logging(output_dir: str) -> logging.Logger:
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"fm_pipeline_{timestamp}.log")

    logger = logging.getLogger("fm_pipeline")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Logging to {log_file}")
    return logger


def run_step(cmd, step_name, logger, cwd=None):
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info(f"[{step_name}] Running: {cmd_str}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd, cwd=cwd or str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=7200,
        )
        elapsed = time.time() - start

        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                logger.debug(f"  [stdout] {line}")
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                logger.debug(f"  [stderr] {line}")

        if result.returncode != 0:
            logger.error(f"[{step_name}] FAILED (rc {result.returncode}) after {elapsed:.1f}s")
            if result.stdout:
                logger.error(f"  stdout: {result.stdout[-1000:]}")
            if result.stderr:
                logger.error(f"  stderr: {result.stderr[-1000:]}")
            return False

        logger.info(f"[{step_name}] Completed in {elapsed:.1f}s")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"[{step_name}] TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"[{step_name}] Exception: {e}")
        return False


def process_video(
    video_name, output_dir, fm_config, train_config,
    vis_fps, grid_interval, logger,
    skip_training=False, skip_inference=False, skip_visualization=False,
    wandb_entity=None, wandb_project=None, wandb_group=None,
):
    data_path = os.path.join(output_dir, video_name)
    results = {"video": video_name, "steps": {}}

    logger.info(f"{'='*60}")
    logger.info(f"FM Processing: {video_name}")
    logger.info(f"  Data path: {data_path}")
    logger.info(f"{'='*60}")

    # Verify preprocessing exists
    video_frames_dir = os.path.join(data_path, "video")
    if not os.path.exists(video_frames_dir) or len(list(Path(video_frames_dir).glob("*.jpg"))) == 0:
        logger.error(f"No preprocessed frames found in {video_frames_dir}. Run DinoTracker pipeline first.")
        results["steps"]["prereq"] = "missing_preprocessing"
        return results

    # Step 1: Training
    if not skip_training:
        train_cmd = [
            sys.executable, os.path.join("scripts", "train_flow_matching.py"),
            "--config", fm_config,
            "--data-path", data_path,
        ]
        if wandb_entity:
            train_cmd += ["--wandb-entity", wandb_entity]
        if wandb_project:
            train_cmd += ["--wandb-project", wandb_project]
        if wandb_group:
            train_cmd += ["--wandb-group", wandb_group]

        success = run_step(train_cmd, f"{video_name}/fm_train", logger)
        results["steps"]["training"] = "success" if success else "failed"
        if not success:
            return results
    else:
        results["steps"]["training"] = "skipped"

    # Step 2: Inference
    if not skip_inference:
        success = run_step([
            sys.executable, os.path.join("scripts", "inference_grid_fm.py"),
            "--config", fm_config,
            "--data-path", data_path,
            "--interval", str(grid_interval),
            "--use-segm-mask",
        ], f"{video_name}/fm_inference", logger)
        results["steps"]["inference"] = "success" if success else "failed"
        if not success:
            return results
    else:
        results["steps"]["inference"] = "skipped"

    # Step 3: Visualization (uses same visualize_rainbow.py but with FM trajectories)
    if not skip_visualization:
        import yaml
        with open(train_config, "r") as f:
            tcfg = yaml.safe_load(f)
        infer_h = tcfg["video_resh"]
        infer_w = tcfg["video_resw"]

        # We need to temporarily swap grid_trajectories to fm_grid_trajectories
        # Create symlinks or copy for visualization
        fm_traj_dir = os.path.join(data_path, "fm_grid_trajectories")
        fm_occ_dir = os.path.join(data_path, "fm_grid_occlusions")
        orig_traj_dir = os.path.join(data_path, "grid_trajectories")
        orig_occ_dir = os.path.join(data_path, "grid_occlusions")

        # Backup original if exists, swap in FM ones
        backup_traj = os.path.join(data_path, "grid_trajectories_dino_backup")
        backup_occ = os.path.join(data_path, "grid_occlusions_dino_backup")

        if os.path.exists(orig_traj_dir) and not os.path.exists(backup_traj):
            os.rename(orig_traj_dir, backup_traj)
        if os.path.exists(orig_occ_dir) and not os.path.exists(backup_occ):
            os.rename(orig_occ_dir, backup_occ)

        # Copy FM results to standard paths
        import shutil
        if os.path.exists(fm_traj_dir):
            if os.path.exists(orig_traj_dir):
                shutil.rmtree(orig_traj_dir)
            shutil.copytree(fm_traj_dir, orig_traj_dir)
        if os.path.exists(fm_occ_dir):
            if os.path.exists(orig_occ_dir):
                shutil.rmtree(orig_occ_dir)
            shutil.copytree(fm_occ_dir, orig_occ_dir)

        # Run visualization
        fm_vis_dir = os.path.join(data_path, "fm_visualizations")
        os.makedirs(fm_vis_dir, exist_ok=True)

        success = run_step([
            sys.executable, "visualization/visualize_rainbow.py",
            "--data-path", data_path,
            "--infer-res-size", str(infer_h), str(infer_w),
            "--of-res-size", str(infer_h), str(infer_w),
            "--fps", str(vis_fps),
        ], f"{video_name}/fm_visualize", logger)
        results["steps"]["visualization"] = "success" if success else "failed"

        # Move visualization results to FM-specific directory
        vis_dir = os.path.join(data_path, "visualizations")
        if os.path.exists(vis_dir):
            for f in os.listdir(vis_dir):
                src = os.path.join(vis_dir, f)
                dst = os.path.join(fm_vis_dir, f"fm_{f}")
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        # Restore original grid trajectories
        if os.path.exists(orig_traj_dir):
            shutil.rmtree(orig_traj_dir)
        if os.path.exists(orig_occ_dir):
            shutil.rmtree(orig_occ_dir)
        if os.path.exists(backup_traj):
            os.rename(backup_traj, orig_traj_dir)
        if os.path.exists(backup_occ):
            os.rename(backup_occ, orig_occ_dir)
    else:
        results["steps"]["visualization"] = "skipped"

    return results


def main():
    parser = argparse.ArgumentParser(description="Flow Matching Pipeline for Ultrasound Videos")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fm-config", default=DEFAULT_FM_CONFIG)
    parser.add_argument("--train-config", default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--videos", nargs="+", default=None,
                        help="Specific video names (e.g., volunteer01)")
    parser.add_argument("--vis-fps", type=int, default=DEFAULT_VIS_FPS)
    parser.add_argument("--grid-interval", type=int, default=DEFAULT_GRID_INTERVAL)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")

    parser.add_argument("--wandb-entity", type=str, default="multincde_daml")
    parser.add_argument("--wandb-project", type=str, default="flowmatchingtesting")
    parser.add_argument("--wandb-group", type=str, default="flow-matching")
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    logger.info("Flow Matching Ultrasound Pipeline")

    # Discover videos from existing output dirs
    if args.videos:
        video_names = args.videos
    else:
        video_names = sorted([
            d for d in os.listdir(args.output_dir)
            if os.path.isdir(os.path.join(args.output_dir, d))
            and os.path.exists(os.path.join(args.output_dir, d, "video"))
        ])

    if not video_names:
        logger.error("No preprocessed videos found. Run DinoTracker pipeline first.")
        sys.exit(1)

    logger.info(f"  Found {len(video_names)} videos: {video_names}")

    all_results = []
    for i, video_name in enumerate(video_names):
        logger.info(f"\n{'#'*60}")
        logger.info(f"FM VIDEO {i+1}/{len(video_names)}: {video_name}")
        logger.info(f"{'#'*60}")

        wandb_entity = None if args.no_wandb else args.wandb_entity
        wandb_project = None if args.no_wandb else args.wandb_project
        wandb_group = None if args.no_wandb else args.wandb_group

        try:
            result = process_video(
                video_name=video_name,
                output_dir=args.output_dir,
                fm_config=args.fm_config,
                train_config=args.train_config,
                vis_fps=args.vis_fps,
                grid_interval=args.grid_interval,
                logger=logger,
                skip_training=args.skip_training,
                skip_inference=args.skip_inference,
                skip_visualization=args.skip_visualization,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                wandb_group=wandb_group,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error processing {video_name}: {e}")
            all_results.append({"video": video_name, "steps": {"error": str(e)}})

    # Summary
    summary_path = os.path.join(args.output_dir, "fm_pipeline_summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "output_dir": args.output_dir,
        "results": all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    for r in all_results:
        steps_str = ", ".join(f"{k}: {v}" for k, v in r["steps"].items())
        status = "✓" if all(v in ("success", "skipped") for v in r["steps"].values()) else "✗"
        logger.info(f"  {status} {r['video']}: {steps_str}")

    logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
