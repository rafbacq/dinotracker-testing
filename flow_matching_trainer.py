"""
Flow Matching Trainer for liver ultrasound point tracking.

Analogous to DINOTracker but uses Conditional Flow Matching loss instead of
direct regression. Shares the same preprocessed data (trajectories, masks,
DINO embeddings) and produces compatible outputs (grid trajectories, visualizations).

Training loss:
    L_total = λ_fm * L_fm + λ_emb_norm * L_emb_norm + λ_angle * L_angle

Where:
    L_fm = E_{t,x_0,x_1} ||v_θ(x_t, t, c) - (x_1 - x_0)||²
    L_emb_norm = regularization on refined embedding norms
    L_angle = regularization on refined embedding angles
"""

import logging
import os
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

from models.utils import fix_random_seeds, get_last_ckpt_iter, get_feature_cos_sims
from models.flow_tracker import FlowTracker
from data.data_utils import load_video
from data.dataset import DinoTrackerSampler, RangeNormalizer
from preprocessing.split_trajectories_to_fg_bg import load_masks
from utils import add_config_paths

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class FlowMatchingTrainer:
    """
    Trainer for the Flow Matching point tracker.

    Uses the same data pipeline and preprocessing as DINOTracker but
    trains a conditional flow matching velocity network.
    """

    def __init__(self, args):
        self.load_config(args.config)
        self.set_paths(args.data_path)
        self.data_path = args.data_path

        # Wandb configuration
        self.wandb_config = getattr(args, 'wandb_config', None)
        self.use_wandb = (self.wandb_config is not None and WANDB_AVAILABLE)

        self.orig_video_res_h, self.orig_video_res_w, video_rest = self.get_original_video_res(self.video_path)
        self.range_normalizer = RangeNormalizer(
            shapes=(self.config["video_resw"], self.config["video_resh"], video_rest)
        ).to(device)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f.read())

    def set_paths(self, data_path):
        config_paths = add_config_paths(data_path, {})
        self.video_path = config_paths["video_folder"]
        self.fg_masks_path = config_paths["masks_path"]
        self.dino_embed_path = config_paths["dino_embed_video_path"]
        self.fg_trajectories_path = config_paths["fg_trajectories_file"]
        self.bg_trajectories_path = config_paths["bg_trajectories_file"]
        # Use a separate checkpoint folder for FM model
        self.ckpt_folder = os.path.join(data_path, "models", "flow_matching")
        self.trajectories_dir = config_paths['trajectories_dir']
        self.occlusions_dir = config_paths['occlusions_dir']
        self.grid_trajectories_dir = os.path.join(data_path, "fm_grid_trajectories")
        self.grid_occlusions_dir = os.path.join(data_path, "fm_grid_occlusions")
        os.makedirs(self.ckpt_folder, exist_ok=True)

    def get_original_video_res(self, video_path):
        from PIL import Image
        video_frames_list = sorted(
            list(Path(video_path).glob("*.jpg")) + list(Path(video_path).glob("*.png"))
        )
        video_rest = len(video_frames_list)
        frame = Image.open(video_frames_list[0])
        video_res_hw = frame.size[::-1]
        return video_res_hw + (video_rest,)

    def load_fg_masks(self):
        self.fg_masks = torch.from_numpy(
            load_masks(
                self.fg_masks_path,
                h_resize=self.config["video_resh"],
                w_resize=self.config["video_resw"],
            )
        ).to(device)

    def load_trajectories(self):
        assert os.path.exists(self.fg_trajectories_path) & os.path.exists(self.bg_trajectories_path)
        trj_device = torch.device('cpu') if self.config.get('keep_traj_in_cpu', False) else device
        train_fg = torch.load(self.fg_trajectories_path, map_location=trj_device)
        train_bg = torch.load(self.bg_trajectories_path, map_location=trj_device)
        return train_fg, train_bg

    def get_sampler(self):
        train_fg, train_bg = self.load_trajectories()
        sampler = DinoTrackerSampler(
            fg_trajectories=train_fg,
            bg_trajectories=train_bg,
            fg_traj_ratio=self.config.get("fg_traj_ratio", 0.5),
            batch_size=self.config.get("train_batch_size", 512),
            range_normalizer=self.range_normalizer,
            dst_range=(-1, 1),
            num_frames=self.config.get("batch_n_frames", 4),
            keep_in_cpu=self.config.get('keep_traj_in_cpu', False),
        )
        return sampler

    def get_model(self):
        video = load_video(
            video_folder=self.video_path,
            resize=(self.config["video_resh"], self.config["video_resw"]),
        ).to(device)

        model = FlowTracker(
            video=video,
            device=device,
            dino_embed_path=self.dino_embed_path,
            dino_patch_size=self.config.get("dino_patch_size", 14),
            stride=self.config.get("stride", 7),
            ckpt_path=self.ckpt_folder,
            hidden_dim=self.config.get("hidden_dim", 512),
            time_embed_dim=self.config.get("time_embed_dim", 64),
            num_velocity_blocks=self.config.get("num_velocity_blocks", 6),
            ode_steps=self.config.get("ode_steps", 10),
        ).to(device)

        self.init_iter = get_last_ckpt_iter(self.ckpt_folder)
        if self.init_iter > 0:
            model.load_weights(self.init_iter)

        return model

    def get_inputs_and_labels(self, sampler):
        sample = sampler()
        # source: (B, 3) with (x, y, t) in pixel space
        source_coords = sample["t1_points"]  # (B, 3), t is already normalized time
        target_coords_normalized = sample["t2_points_normalized"][:, :-1]  # (B, 2) in [-1, 1]
        source_frame_indices = sample["source_frame_indices"]
        target_frame_indices = sample["target_frame_indices"]
        frames_set_t = sample["frames_set_t"]

        # We need both source and target in pixel space for FM
        # target in pixel space from unnormalized
        t2_points_raw = self.range_normalizer.unnormalize(
            sample["t2_points_normalized"], src=(-1, 1), dims=[0, 1, 2]
        )

        return source_coords, t2_points_raw, source_frame_indices, target_frame_indices, frames_set_t

    def get_emb_norm_regularization_loss(self, model):
        refined_emb_norm = model.frame_embeddings.norm(dim=1)
        dino_emb_norm = model.raw_embeddings.norm(dim=1)
        norm_ratio = refined_emb_norm / dino_emb_norm
        return (norm_ratio - 1).abs().mean()

    def get_emb_angle_regularization_loss(self, model):
        refined_emb = model.frame_embeddings
        dino_emb = model.raw_embeddings
        cos_sims = get_feature_cos_sims(refined_emb, dino_emb)
        return (cos_sims - 1).abs().mean()

    def _init_wandb(self):
        if not self.use_wandb:
            return
        video_name = Path(self.data_path).name
        try:
            wandb.init(
                entity=self.wandb_config.get("entity"),
                project=self.wandb_config.get("project", "flowmatchingtesting"),
                name=f"fm-{video_name}",
                group=self.wandb_config.get("group", "flow-matching"),
                config={
                    "model_type": "flow_matching",
                    "video_name": video_name,
                    "data_path": str(self.data_path),
                    **self.config,
                },
                reinit=True,
            )
            logging.info(f"Wandb run initialized: {wandb.run.name} ({wandb.run.url})")
        except Exception as e:
            logging.warning(f"Wandb init failed: {e}")
            self.use_wandb = False

    def _finish_wandb(self):
        if self.use_wandb and wandb.run is not None:
            wandb.finish()

    def train(self):
        self.load_fg_masks()
        total_iterations = self.config.get("total_iterations", 4000)
        checkpoint_interval = self.config.get("checkpoint_interval", 2000)
        log_interval = self.config.get("log_interval", 100)

        self._init_wandb()

        train_sampler = self.get_sampler()
        model = self.get_model()
        model.train()

        # Optimizer for both delta_dino and velocity_net
        params = [
            {"params": model.delta_dino.parameters(), "lr": self.config.get("lr_delta_dino", 0.001)},
            {"params": model.velocity_net.parameters(), "lr": self.config.get("lr_velocity_net", 0.001)},
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=self.config.get("weight_decay", 1e-4))

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_iterations, eta_min=1e-6
        )

        if self.init_iter > 0:
            for _ in range(self.init_iter):
                scheduler.step()

        # Running loss accumulators
        running = {
            "total": 0.0, "fm": 0.0, "emb_norm": 0.0, "angle": 0.0,
            "v_pred_norm": 0.0, "u_t_norm": 0.0,
        }

        print(f"--- Flow Matching Training ---")
        print(f"  Total iterations: {total_iterations}")
        print(f"  Init iter: {self.init_iter}")
        print(f"  ODE steps: {self.config.get('ode_steps', 10)}")

        for i in tqdm(range(self.init_iter, total_iterations)):
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            source_coords, target_coords, src_idx, tgt_idx, frames_set_t = \
                self.get_inputs_and_labels(train_sampler)

            # Flow matching loss
            fm_loss, loss_dict = model.compute_fm_loss(
                source_coords, target_coords, src_idx, tgt_idx, frames_set_t
            )

            loss = fm_loss

            # Embedding regularization
            lambda_emb_norm = self.config.get("lambda_emb_norm", 0.0001)
            lambda_angle = self.config.get("lambda_angle", 0.0001)

            emb_norm_loss = self.get_emb_norm_regularization_loss(model)
            angle_loss = self.get_emb_angle_regularization_loss(model)

            loss += lambda_emb_norm * emb_norm_loss + lambda_angle * angle_loss

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Accumulate losses
            running["total"] += loss.item()
            running["fm"] += fm_loss.item()
            running["emb_norm"] += emb_norm_loss.item()
            running["angle"] += angle_loss.item()
            running["v_pred_norm"] += loss_dict["v_pred_norm"]
            running["u_t_norm"] += loss_dict["u_t_norm"]

            # Log
            if i % log_interval == 0 and i > 0:
                avg = {k: v / log_interval for k, v in running.items()}
                log_str = (
                    f"[FM] iter={i}, "
                    f"total={avg['total']:.4f}, fm={avg['fm']:.4f}, "
                    f"emb_norm={avg['emb_norm']:.4f}, angle={avg['angle']:.4f}, "
                    f"v_norm={avg['v_pred_norm']:.4f}, u_norm={avg['u_t_norm']:.4f}"
                )
                logging.info(log_str)
                print(log_str)

                if self.use_wandb:
                    wandb.log({
                        "iteration": i,
                        "loss/total": avg["total"],
                        "loss/flow_matching": avg["fm"],
                        "loss/emb_norm_reg": avg["emb_norm"],
                        "loss/angle_reg": avg["angle"],
                        "diagnostics/v_pred_norm": avg["v_pred_norm"],
                        "diagnostics/u_t_norm": avg["u_t_norm"],
                        "lr": optimizer.param_groups[0]["lr"],
                    }, step=i)

                running = {k: 0.0 for k in running}

            # Checkpoint
            if i == total_iterations - 1 or i % checkpoint_interval == 0:
                model.save_weights(i)

        model.save_weights(total_iterations)
        self._finish_wandb()
        print("Flow Matching training complete!")
