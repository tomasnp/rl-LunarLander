#!/usr/bin/env python3
"""
A2C Baseline - Configuration EXACTE du run réussi (74.8% success)
"""

import sys
sys.path.insert(0, '.')

from A2C import Config, train, setup_logging

if __name__ == "__main__":
    # Setup logging
    log_filepath, tee_logger = setup_logging(log_dir="logs", experiment_name="a2c_baseline")
    sys.stdout = tee_logger

    try:
        cfg = Config()

        # Hyperparamètres optimaux (ne pas changer)
        cfg.lr_policy = 5e-4
        cfg.lr_value = 1e-3
        cfg.entropy_coef_start = 0.05
        cfg.entropy_coef_final = 0.005
        cfg.value_coef = 0.5
        cfg.rollout_steps = 2048
        cfg.max_updates = 10000
        cfg.eval_every = 50
        cfg.eval_episodes = 30
        cfg.grad_clip = 0.5

        # Network size du run réussi
        cfg.hidden_size = 256  

        # Checkpoint
        cfg.save_name = f"a2c_r_{cfg.rollout_steps}_u_{cfg.max_updates}_h_{cfg.hidden_size}.pt"
        cfg.plot_name = f"a2c_r_{cfg.rollout_steps}_u_{cfg.max_updates}_h_{cfg.hidden_size}.png"

        print("="*80)
        print("🎯 A2C BASELINE - Configuration du Run Réussi")
        print("="*80)
        print(f"Hidden size: {cfg.hidden_size}")
        print(f"LR policy: {cfg.lr_policy}")
        print(f"LR value: {cfg.lr_value}")
        print(f"Entropy final: {cfg.entropy_coef_final}")
        print(f"Eval episodes: {cfg.eval_episodes}")
        print()
        print("="*80 + "\n")

        # Entraîner
        history = train(cfg)

        print("\n" + "="*80)
        print("✅ Entraînement terminé!")

        print("="*80)

    finally:
        # Restore stdout and close log
        sys.stdout = tee_logger.terminal
        tee_logger.close()
        print(f"\n✅ Log saved to: {log_filepath}")
