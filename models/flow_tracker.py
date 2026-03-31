"""
Flow Matching Tracker for liver ultrasound point tracking.

Uses Conditional Flow Matching (CFM) to learn a velocity field that
transports noise → target point coordinates, conditioned on DINO visual features.

Architecture:
    1. DINO feature backbone (shared with DinoTracker, pre-extracted)
    2. DeltaDINO feature refinement (learned residual on DINO features)
    3. FlowVelocityNet: v_θ(z_t, t, c) predicts velocity conditioned
       on fused visual context c
    4. ODE integration at inference (Euler solver)

Training:
    x_1 = target coordinates (from optical flow trajectories)
    x_0 ~ N(0, I) or source coordinates
    x_t = (1-t)*x_0 + t*x_1
    Loss = ||v_θ(x_t, t, c) - (x_1 - x_0)||²

Inference:
    Sample x_0 from noise, integrate ODE from t=0 to t=1
    Multiple samples → uncertainty estimation
"""

import os
import gc
import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path

from models.networks.flow_velocity_net import FlowVelocityNet
from models.networks.delta_dino import DeltaDINO
from models.utils import load_pre_trained_model, get_last_ckpt_iter
from data.dataset import RangeNormalizer
from utils import bilinear_interpolate_video


EPS = 1e-08


class FlowTracker(nn.Module):
    """
    Flow Matching-based point tracker for ultrasound videos.

    Uses the same DINO feature extraction as DinoTracker but replaces
    the correlation-map + soft-argmax approach with conditional flow matching.
    """

    def __init__(
        self,
        video=None,
        ckpt_path="",
        dino_embed_path="",
        dino_patch_size=14,
        stride=7,
        device="cuda:0",
        # Flow matching specific
        hidden_dim=512,
        time_embed_dim=64,
        num_velocity_blocks=6,
        ode_steps=10,
        use_noise_source=True,
    ):
        super().__init__()

        self.stride = stride
        self.dino_patch_size = dino_patch_size
        self.device = device
        self.refined_features = None
        self.dino_embed_path = dino_embed_path
        self.ckpt_path = ckpt_path
        self.ode_steps = ode_steps
        self.use_noise_source = use_noise_source

        self.video = video

        # DINO embed
        self.load_dino_embed_video()
        dino_embed_dim = self.dino_embed_video.shape[1]

        # Delta-DINO (feature refinement, same as DinoTracker)
        self.delta_dino = DeltaDINO(
            vit_stride=self.stride,
            channels=[3, 64, 128, 256, dino_embed_dim],
        ).to(device)

        # Context dimension: source_embedding + correlation_features + interpolated_embedding
        # source_emb: dino_embed_dim
        # We flatten correlation map features into a fixed representation
        t, c, h, w = self.video.shape
        self.range_normalizer = RangeNormalizer(shapes=(w, h, self.video.shape[0]))

        # Context = source_embedding (dino_embed_dim) + target_frame_pooled (dino_embed_dim)
        context_dim = dino_embed_dim * 2

        # Velocity network
        self.velocity_net = FlowVelocityNet(
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            num_blocks=num_velocity_blocks,
        ).to(device)

    @torch.no_grad()
    def load_dino_embed_video(self):
        assert os.path.exists(self.dino_embed_path)
        self.dino_embed_video = torch.load(
            self.dino_embed_path, map_location=self.device
        )

    def get_dino_embed_video(self, frames_set_t):
        dino_emb = (
            self.dino_embed_video[frames_set_t.to(self.dino_embed_video.device)]
            if frames_set_t.device != self.dino_embed_video.device
            else self.dino_embed_video[frames_set_t]
        )
        return dino_emb

    def normalize_points_for_sampling(self, points):
        t, c, vid_h, vid_w = self.video.shape
        h, w = vid_h, vid_w
        patch_size = self.dino_patch_size
        stride = self.stride

        last_coord_h = ((h - patch_size) // stride) * stride + (patch_size / 2)
        last_coord_w = ((w - patch_size) // stride) * stride + (patch_size / 2)
        ah = 2 / (last_coord_h - (patch_size / 2))
        aw = 2 / (last_coord_w - (patch_size / 2))
        bh = 1 - last_coord_h * 2 / (last_coord_h - (patch_size / 2))
        bw = 1 - last_coord_w * 2 / (last_coord_w - (patch_size / 2))

        a = torch.tensor([[aw, ah, 1]]).to(self.device)
        b = torch.tensor([[bw, bh, 0]]).to(self.device)
        normalized_points = a * points + b
        return normalized_points

    def sample_embeddings(self, embeddings, source_points):
        """
        embeddings: T x C x H x W
        source_points: B x 3, (x, y, t) in [-1, 1] normalized
        """
        t, c, h, w = embeddings.shape
        sampled = bilinear_interpolate_video(
            video=rearrange(embeddings, "t c h w -> 1 c t h w"),
            points=source_points,
            h=h, w=w, t=t,
            normalize_w=False, normalize_h=False, normalize_t=True,
        )
        sampled = sampled.squeeze()
        if len(sampled.shape) == 1:
            sampled = sampled.unsqueeze(1)
        sampled = sampled.permute(1, 0)
        return sampled

    def get_refined_embeddings(self, frames_set_t, return_raw_embeddings=False):
        frames_dino_embeddings = self.get_dino_embed_video(frames_set_t=frames_set_t)
        refiner_input_frames = self.video[frames_set_t]

        batch_size = 8
        n_frames = frames_set_t.shape[0]
        residual_embeddings = torch.zeros_like(frames_dino_embeddings)
        for i in range(0, n_frames, batch_size):
            end_idx = min(i + batch_size, n_frames)
            residual_embeddings[i:end_idx] = self.delta_dino(
                refiner_input_frames[i:end_idx], frames_dino_embeddings[i:end_idx]
            )

        refined_embeddings = frames_dino_embeddings + residual_embeddings

        if return_raw_embeddings:
            return refined_embeddings, residual_embeddings, frames_dino_embeddings
        return refined_embeddings, residual_embeddings

    def cache_refined_embeddings(self, move_dino_to_cpu=False):
        refined_features, _ = self.get_refined_embeddings(
            torch.arange(0, self.video.shape[0])
        )
        self.refined_features = refined_features
        if move_dino_to_cpu:
            self.dino_embed_video = self.dino_embed_video.to("cpu")

    def uncache_refined_embeddings(self, move_dino_to_gpu=False):
        self.refined_features = None
        torch.cuda.empty_cache()
        gc.collect()
        if move_dino_to_gpu:
            self.dino_embed_video = self.dino_embed_video.to("cuda")

    def get_context(self, source_points_normalized, source_frame_indices, target_frame_indices, frame_embeddings):
        """
        Build the conditioning context c for the velocity network.

        c = [source_embedding, target_frame_correlation_features]

        Args:
            source_points_normalized: (B, 3) normalized (x, y, t)
            source_frame_indices: (B,) index into frames_set
            target_frame_indices: (B,) index into frames_set
            frame_embeddings: (N_frames, C, H, W) DINO features

        Returns:
            context: (B, context_dim) fused conditioning features
        """
        # Sample source embeddings
        source_emb = self.sample_embeddings(
            frame_embeddings,
            torch.cat([source_points_normalized[:, :2], source_frame_indices[:, None].float()], dim=1),
        )  # (B, C)

        # For target frame context, pool the target frame embeddings
        # Use correlation between source embedding and target frame
        target_emb_frames = frame_embeddings[target_frame_indices.long()]  # (B, C, H, W)
        # Global average of target frame features
        target_pooled = target_emb_frames.mean(dim=(-2, -1))  # (B, C)

        context = torch.cat([source_emb, target_pooled], dim=-1)  # (B, 2*C)
        return context

    def compute_fm_loss(self, source_coords, target_coords, source_frame_indices, target_frame_indices, frames_set_t):
        """
        Compute the Conditional Flow Matching loss.

        L_CFM = E_{t, x0, x1} ||v_θ(x_t, t, c) - (x_1 - x_0)||²

        Args:
            source_coords: (B, 3) source point coordinates (x, y, t) in pixel space
            target_coords: (B, 3) target point coordinates (x, y, t) in pixel space
            source_frame_indices: (B,) frame indices
            target_frame_indices: (B,) frame indices
            frames_set_t: (N,) frame set

        Returns:
            loss: scalar FM loss
            loss_dict: dict of loss components for logging
        """
        # Get frame embeddings
        if self.refined_features is not None:
            frame_embeddings = self.refined_features[frames_set_t]
            raw_embeddings = self.dino_embed_video[frames_set_t.to(self.dino_embed_video.device)]
        else:
            frame_embeddings, _, raw_embeddings = self.get_refined_embeddings(
                frames_set_t, return_raw_embeddings=True
            )
        self.frame_embeddings = frame_embeddings
        self.raw_embeddings = raw_embeddings

        # Normalize coordinates to [-1, 1]
        t_dim, c_dim, vid_h, vid_w = self.video.shape
        x1 = source_coords[:, :2].clone()  # source points (B, 2)
        x1[:, 0] = x1[:, 0] / (vid_w - 1) * 2 - 1  # normalize x
        x1[:, 1] = x1[:, 1] / (vid_h - 1) * 2 - 1  # normalize y

        target_xy = target_coords[:, :2].clone()  # target points (B, 2)
        target_xy[:, 0] = target_xy[:, 0] / (vid_w - 1) * 2 - 1
        target_xy[:, 1] = target_xy[:, 1] / (vid_h - 1) * 2 - 1

        # In FM: x_0 is noise/source, x_1 is target
        # We use: z_0 = source coords, z_1 = target coords
        if getattr(self, "use_noise_source", True):
            z_0 = torch.randn_like(x1)
        else:
            z_0 = x1  # source point (B, 2) in [-1, 1]
            
        z_1 = target_xy  # target point (B, 2) in [-1, 1]

        # Sample time
        B = z_0.shape[0]
        t = torch.rand(B, device=z_0.device)  # (B,) ~ Uniform(0, 1)

        # OT interpolation
        z_t = (1 - t[:, None]) * z_0 + t[:, None] * z_1  # (B, 2)

        # Target velocity (constant for OT path)
        u_t = z_1 - z_0  # (B, 2)

        # Build conditioning context
        source_points_normalized = self.normalize_points_for_sampling(source_coords)
        context = self.get_context(
            source_points_normalized, source_frame_indices, target_frame_indices, frame_embeddings
        )

        # Predict velocity
        v_pred = self.velocity_net(z_t, t, context)  # (B, 2)

        # FM loss
        fm_loss = ((v_pred - u_t) ** 2).mean()

        loss_dict = {
            "fm_loss": fm_loss.item(),
            "v_pred_norm": v_pred.detach().norm(dim=-1).mean().item(),
            "u_t_norm": u_t.detach().norm(dim=-1).mean().item(),
        }

        return fm_loss, loss_dict

    @torch.no_grad()
    def predict_flow(self, source_coords, source_frame_indices, target_frame_indices, frames_set_t, num_steps=None):
        """
        Predict target coordinates using ODE integration.

        Args:
            source_coords: (B, 3) source coordinates in pixel space
            source_frame_indices: (B,)
            target_frame_indices: (B,)
            frames_set_t: (N,) frame set
            num_steps: ODE steps (default: self.ode_steps)

        Returns:
            predicted_coords: (B, 2) predicted target coordinates in pixel space
        """
        num_steps = num_steps or self.ode_steps

        # Get frame embeddings
        if self.refined_features is not None:
            frame_embeddings = self.refined_features[frames_set_t]
        else:
            frame_embeddings, _ = self.get_refined_embeddings(frames_set_t)

        # Normalize source coordinates
        t_dim, c_dim, vid_h, vid_w = self.video.shape
        if getattr(self, "use_noise_source", True):
            z = torch.randn_like(source_coords[:, :2])
        else:
            z = source_coords[:, :2].clone()
            z[:, 0] = z[:, 0] / (vid_w - 1) * 2 - 1
            z[:, 1] = z[:, 1] / (vid_h - 1) * 2 - 1

        # Build context (same for all ODE steps)
        source_points_normalized = self.normalize_points_for_sampling(source_coords)
        context = self.get_context(
            source_points_normalized, source_frame_indices, target_frame_indices, frame_embeddings
        )

        # Euler integration
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((z.shape[0],), step * dt, device=z.device)
            v = self.velocity_net(z, t, context)
            z = z + v * dt

        # Unnormalize back to pixel space
        pred = z.clone()
        pred[:, 0] = (pred[:, 0] + 1) / 2 * (vid_w - 1)
        pred[:, 1] = (pred[:, 1] + 1) / 2 * (vid_h - 1)

        return pred

    def save_weights(self, iter):
        torch.save(
            self.velocity_net.state_dict(),
            Path(self.ckpt_path) / f"flow_velocity_net_{iter}.pt",
        )
        torch.save(
            self.delta_dino.state_dict(),
            Path(self.ckpt_path) / f"fm_delta_dino_{iter}.pt",
        )

    def load_weights(self, iter):
        vel_path = os.path.join(self.ckpt_path, f"flow_velocity_net_{iter}.pt")
        dd_path = os.path.join(self.ckpt_path, f"fm_delta_dino_{iter}.pt")
        if os.path.exists(vel_path):
            self.velocity_net = load_pre_trained_model(
                torch.load(vel_path), self.velocity_net
            )
        if os.path.exists(dd_path):
            self.delta_dino = load_pre_trained_model(
                torch.load(dd_path), self.delta_dino
            )

    def forward(self, inp, use_raw_features=False):
        """
        Forward pass for inference — predicts target coordinates.

        inp: (source_points, source_frame_indices, target_frame_indices, frames_set_t)
        """
        source_points, source_frame_indices, target_frame_indices, frames_set_t = inp

        if use_raw_features:
            frame_embeddings = self.get_dino_embed_video(frames_set_t=frames_set_t)
        elif self.refined_features is not None:
            frame_embeddings = self.refined_features[frames_set_t]
        else:
            frame_embeddings, _, _ = self.get_refined_embeddings(
                frames_set_t, return_raw_embeddings=True
            )
        self.frame_embeddings = frame_embeddings
        self.raw_embeddings = self.get_dino_embed_video(frames_set_t=frames_set_t)

        # Use ODE integration for prediction
        pred_coords = self.predict_flow(
            source_points, source_frame_indices, target_frame_indices, frames_set_t
        )

        # Normalize to [-1, 1] for compatibility with DinoTracker's output format
        t_dim, c_dim, vid_h, vid_w = self.video.shape
        normalized = pred_coords.clone()
        normalized[:, 0] = normalized[:, 0] / (vid_w - 1) * 2 - 1
        normalized[:, 1] = normalized[:, 1] / (vid_h - 1) * 2 - 1

        return normalized
