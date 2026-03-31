"""
Entry point for Flow Matching model training.
Analogous to train.py but trains the Flow Matching tracker.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from flow_matching_trainer import FlowMatchingTrainer
from models.utils import fix_random_seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flow Matching Tracker")

    parser.add_argument("--config", default="./config/ultrasound_flow_matching.yaml", type=str)
    parser.add_argument("--data-path", default="./dataset/libby", type=str)
    parser.add_argument("--seed", default=2, type=int)

    # Weights & Biases arguments
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)

    args = parser.parse_args()

    # Build wandb config
    if args.wandb_entity or args.wandb_project:
        args.wandb_config = {
            "entity": args.wandb_entity,
            "project": args.wandb_project or "flowmatchingtesting",
            "group": args.wandb_group or "flow-matching",
        }
    else:
        args.wandb_config = None

    fix_random_seeds(args.seed)
    trainer = FlowMatchingTrainer(args)
    trainer.train()
