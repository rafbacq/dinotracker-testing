"""
Generate comparison results between DinoTracker and Flow Matching models.

Produces matplotlib plots comparing:
    1. Training loss curves (from saved logs)
    2. Trajectory quality visualizations
    3. Side-by-side model performance summary

Results are saved to outputs/comparison/
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (14, 8),
    'figure.dpi': 150,
})


def parse_loss_logs(log_dir, prefix="pipeline_"):
    """Parse loss values from pipeline log files."""
    losses = {"iterations": [], "total": [], "components": {}}

    log_files = sorted(Path(log_dir).glob(f"{prefix}*.log"))
    if not log_files:
        return None

    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                # Parse DinoTracker logs
                if "loss_of:" in line:
                    try:
                        parts = line.split("loss_of:")[1]
                        loss_of = float(parts.split(",")[0].strip())
                        loss_total = float(parts.split("loss_total:")[1].strip()) if "loss_total:" in parts else None

                        losses["total"].append(loss_total or loss_of)
                        if "optical_flow" not in losses["components"]:
                            losses["components"]["optical_flow"] = []
                        losses["components"]["optical_flow"].append(loss_of)
                    except (ValueError, IndexError):
                        continue

                # Parse Flow Matching logs
                elif "[FM]" in line and "total=" in line:
                    try:
                        parts = line.split("total=")[1]
                        total = float(parts.split(",")[0].strip())
                        fm = float(parts.split("fm=")[1].split(",")[0].strip()) if "fm=" in parts else total

                        losses["total"].append(total)
                        if "flow_matching" not in losses["components"]:
                            losses["components"]["flow_matching"] = []
                        losses["components"]["flow_matching"].append(fm)
                    except (ValueError, IndexError):
                        continue

    if losses["total"]:
        losses["iterations"] = list(range(0, len(losses["total"]) * 100, 100))
    return losses if losses["total"] else None


def plot_training_curves(dino_losses, fm_losses, output_dir, video_name):
    """Plot training loss curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Total loss comparison
    ax = axes[0]
    if dino_losses and dino_losses["total"]:
        iters = dino_losses["iterations"][:len(dino_losses["total"])]
        ax.plot(iters, dino_losses["total"],
                label="DinoTracker", color="#2196F3", linewidth=2, alpha=0.8)

        # Smoothed
        if len(dino_losses["total"]) > 10:
            window = min(20, len(dino_losses["total"]) // 3)
            smoothed = np.convolve(dino_losses["total"],
                                   np.ones(window)/window, mode='valid')
            ax.plot(iters[:len(smoothed)], smoothed,
                    color="#2196F3", linewidth=3, linestyle='--', alpha=0.5)

    if fm_losses and fm_losses["total"]:
        iters = fm_losses["iterations"][:len(fm_losses["total"])]
        ax.plot(iters, fm_losses["total"],
                label="Flow Matching", color="#FF5722", linewidth=2, alpha=0.8)

        if len(fm_losses["total"]) > 10:
            window = min(20, len(fm_losses["total"]) // 3)
            smoothed = np.convolve(fm_losses["total"],
                                   np.ones(window)/window, mode='valid')
            ax.plot(iters[:len(smoothed)], smoothed,
                    color="#FF5722", linewidth=3, linestyle='--', alpha=0.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.set_title(f"Training Loss — {video_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Component losses
    ax = axes[1]
    colors = {'optical_flow': '#4CAF50', 'flow_matching': '#FF9800',
              'contrastive': '#9C27B0', 'cycle_consistency': '#00BCD4'}

    for model_name, losses in [("DinoTracker", dino_losses), ("Flow Matching", fm_losses)]:
        if losses is None:
            continue
        for comp_name, comp_vals in losses["components"].items():
            iters = losses["iterations"][:len(comp_vals)]
            color = colors.get(comp_name, '#607D8B')
            ax.plot(iters, comp_vals,
                    label=f"{model_name}: {comp_name}",
                    color=color, linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Component Loss")
    ax.set_title(f"Component Losses — {video_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"training_curves_{video_name}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


def plot_trajectory_comparison(output_dir, video_names):
    """Plot trajectory statistics comparison across videos."""
    fig, axes = plt.subplots(1, len(video_names), figsize=(8*len(video_names), 6))
    if len(video_names) == 1:
        axes = [axes]

    for idx, video_name in enumerate(video_names):
        ax = axes[idx]
        data_path = os.path.join(output_dir, video_name)

        # Load DinoTracker trajectories
        dino_traj_path = os.path.join(data_path, "grid_trajectories", "grid_trajectories.npy")
        fm_traj_path = os.path.join(data_path, "fm_grid_trajectories", "grid_trajectories.npy")

        stats = {}
        for name, path in [("DinoTracker", dino_traj_path), ("Flow Matching", fm_traj_path)]:
            if os.path.exists(path):
                trajs = np.load(path)  # (N, T, 2)
                # Compute displacement magnitudes
                displacements = np.sqrt(((trajs[:, 1:] - trajs[:, :-1]) ** 2).sum(axis=-1))
                stats[name] = {
                    "mean_disp": displacements.mean(),
                    "std_disp": displacements.std(),
                    "max_disp": displacements.max(),
                    "n_tracks": trajs.shape[0],
                    "n_frames": trajs.shape[1],
                    "total_displacement": np.sqrt(((trajs[:, -1] - trajs[:, 0]) ** 2).sum(axis=-1)),
                }

        if stats:
            models = list(stats.keys())
            x = np.arange(3)
            width = 0.35

            for i, model in enumerate(models):
                s = stats[model]
                values = [s["mean_disp"], s["std_disp"], s["max_disp"]]
                color = "#2196F3" if model == "DinoTracker" else "#FF5722"
                ax.bar(x + i * width, values, width, label=model, color=color, alpha=0.8)

            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(["Mean Δ", "Std Δ", "Max Δ"])
            ax.set_ylabel("Displacement (pixels)")
            ax.set_title(f"Trajectory Stats — {video_name}")
            ax.legend()

            # Add track count annotation
            for model in models:
                s = stats[model]
                ax.text(0.02, 0.98, f"{model}: {s['n_tracks']} tracks, {s['n_frames']} frames",
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    out_path = os.path.join(output_dir, "comparison", "trajectory_comparison.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


def plot_summary_table(output_dir, video_names):
    """Create a summary comparison table as a figure."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # Gather data
    rows = []
    for video_name in video_names:
        data_path = os.path.join(output_dir, video_name)

        for model, traj_subdir in [("DinoTracker", "grid_trajectories"), ("Flow Matching", "fm_grid_trajectories")]:
            traj_path = os.path.join(data_path, traj_subdir, "grid_trajectories.npy")
            if os.path.exists(traj_path):
                trajs = np.load(traj_path)
                disps = np.sqrt(((trajs[:, 1:] - trajs[:, :-1]) ** 2).sum(axis=-1))
                total_disp = np.sqrt(((trajs[:, -1] - trajs[:, 0]) ** 2).sum(axis=-1))
                rows.append([
                    video_name, model,
                    f"{trajs.shape[0]}", f"{trajs.shape[1]}",
                    f"{disps.mean():.2f}", f"{disps.std():.2f}",
                    f"{total_disp.mean():.2f}",
                ])
            else:
                rows.append([video_name, model, "N/A", "N/A", "N/A", "N/A", "N/A"])

    if rows:
        headers = ["Video", "Model", "# Tracks", "# Frames",
                    "Mean Δ/step", "Std Δ/step", "Mean Total Δ"]
        table = ax.table(cellText=rows, colLabels=headers,
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Style header
        for j in range(len(headers)):
            table[0, j].set_facecolor('#37474F')
            table[0, j].set_text_props(color='white', weight='bold')

        # Alternate row colors
        for i in range(1, len(rows) + 1):
            color = '#E3F2FD' if rows[i-1][1] == "DinoTracker" else '#FBE9E7'
            for j in range(len(headers)):
                table[i, j].set_facecolor(color)

    ax.set_title("Model Comparison Summary", fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "comparison", "summary_table.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Comparison Results")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--videos", nargs="+", default=None)
    args = parser.parse_args()

    comparison_dir = os.path.join(args.output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Find videos
    if args.videos:
        video_names = args.videos
    else:
        video_names = sorted([
            d for d in os.listdir(args.output_dir)
            if os.path.isdir(os.path.join(args.output_dir, d))
            and d.startswith("volunteer")
        ])

    if not video_names:
        print("No videos found in output directory.")
        return

    print(f"Generating comparison results for: {video_names}")

    # 1. Training curves
    print("\n1. Training Loss Curves")
    log_dir = os.path.join(args.output_dir, "logs")
    for video_name in video_names:
        dino_losses = parse_loss_logs(log_dir, prefix="pipeline_")
        fm_losses = parse_loss_logs(log_dir, prefix="fm_pipeline_")
        plot_training_curves(dino_losses, fm_losses, comparison_dir, video_name)

    # 2. Trajectory comparison
    print("\n2. Trajectory Statistics")
    plot_trajectory_comparison(args.output_dir, video_names)

    # 3. Summary table
    print("\n3. Summary Table")
    plot_summary_table(args.output_dir, video_names)

    print(f"\n✓ All comparison results saved to {comparison_dir}")


if __name__ == "__main__":
    main()
