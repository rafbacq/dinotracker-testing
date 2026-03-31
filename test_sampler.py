import torch
import os
import sys
from flow_matching_trainer import FlowMatchingTrainer
import json

class Args:
    config = "c:/Users/tdc65/dinotracker-testing/config/ultrasound_flow_matching.yaml"
    data_path = "c:/Users/tdc65/dinotracker-testing/outputs/volunteer01"
    wandb_config = None

args = Args()
trainer = FlowMatchingTrainer(args)
sampler = trainer.get_sampler()
trainer.load_fg_masks()

src, tgt, s_idx, t_idx, frames_set = trainer.get_inputs_and_labels(sampler)

res = {
    "src": src[:4].tolist(),
    "tgt": tgt[:4].tolist(),
    "s_idx": s_idx[:4].tolist(),
    "t_idx": t_idx[:4].tolist()
}

with open('debug_output.json', 'w') as f:
    json.dump(res, f, indent=2)

print("done")
