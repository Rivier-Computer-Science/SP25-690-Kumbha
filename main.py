import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time

from config import SEED
from utils.utils import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Selective Classification with Reject Option"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "train_cnn_baseline", "train_cnn_selective", "train_vit_selective", "evaluate"],
        help="Execution mode",
    )
    parser.add_argument("--skip_training", action="store_true", help="Skip training, only evaluate")
    return parser.parse_args()


def run_training_phase():
    print("=" * 70)
    print("PHASE 1: Training CNN Baseline (no rejection)")
    print("=" * 70)
    t0 = time.time()
    from scripts.train_cnn_baseline import train_cnn_baseline
    train_cnn_baseline()
    print(f"CNN Baseline training completed in {(time.time()-t0)/60:.1f} min\n")

    print("=" * 70)
    print("PHASE 2: Training CNN with Selective Classification")
    print("=" * 70)
    t0 = time.time()
    from scripts.train_cnn_selective import train_cnn_selective
    train_cnn_selective()
    print(f"CNN Selective training completed in {(time.time()-t0)/60:.1f} min\n")

    print("=" * 70)
    print("PHASE 3: Training Vision Transformer with Selective Classification")
    print("=" * 70)
    t0 = time.time()
    from scripts.train_vit_selective import train_vit_selective
    train_vit_selective()
    print(f"ViT Selective training completed in {(time.time()-t0)/60:.1f} min\n")


def run_evaluation_phase():
    print("=" * 70)
    print("PHASE 4: Threshold Sweep and Full Evaluation")
    print("=" * 70)
    t0 = time.time()
    from scripts.evaluate_thresholds import evaluate_all_thresholds
    evaluate_all_thresholds()
    print(f"Evaluation completed in {(time.time()-t0)/60:.1f} min\n")


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()

    print("=" * 70)
    print("Selective Classification with Reject Option")
    print("CNN and Vision Transformers for Reliable Image Recognition")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print()

    total_start = time.time()

    if args.mode == "all":
        if not args.skip_training:
            run_training_phase()
        run_evaluation_phase()

    elif args.mode == "train_cnn_baseline":
        from scripts.train_cnn_baseline import train_cnn_baseline
        train_cnn_baseline()

    elif args.mode == "train_cnn_selective":
        from scripts.train_cnn_selective import train_cnn_selective
        train_cnn_selective()

    elif args.mode == "train_vit_selective":
        from scripts.train_vit_selective import train_vit_selective
        train_vit_selective()

    elif args.mode == "evaluate":
        from scripts.evaluate_thresholds import evaluate_all_thresholds
        evaluate_all_thresholds()

    total_time = (time.time() - total_start) / 60
    print("=" * 70)
    print(f"All tasks completed in {total_time:.1f} minutes.")
    print("Outputs saved to: outputs/")
    print("  - outputs/checkpoints/  : model checkpoints")
    print("  - outputs/plots/        : training curves, risk-coverage plots")
    print("  - outputs/results/      : JSON evaluation results")
    print("=" * 70)


if __name__ == "__main__":
    main()