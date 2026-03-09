import argparse
from dino_tracker import DINOTracker
from models.utils import fix_random_seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="./config/train.yaml", type=str)
    parser.add_argument("--data-path", default="./dataset/libby", type=str)
    parser.add_argument("--seed", default=2, type=int)

    # Weights & Biases arguments (optional)
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Wandb entity (team/org name)")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Wandb project name")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="Wandb group name for grouping runs")

    args = parser.parse_args()

    # Build wandb config if entity/project provided
    if args.wandb_entity or args.wandb_project:
        args.wandb_config = {
            "entity": args.wandb_entity,
            "project": args.wandb_project or "dino-tracker",
            "group": args.wandb_group or "ultrasound-pipeline",
        }
    else:
        args.wandb_config = None

    fix_random_seeds(args.seed)
    dino_tracker = DINOTracker(args)
    dino_tracker.train()
