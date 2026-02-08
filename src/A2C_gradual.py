#!/usr/bin/env python3
"""
A2C Gradual - Améliorations PROGRESSIVES (moins agressives)
Commence avec le baseline et ajoute UNE amélioration à la fois
"""

import sys
sys.path.insert(0, '.')

from A2C import Config, train, setup_logging

if __name__ == "__main__":
    # Setup logging
    log_filepath, tee_logger = setup_logging(log_dir="logs", experiment_name="a2c_gradual")
    sys.stdout = tee_logger

    try:
        cfg = Config()

        # ============================================================
        # AMÉLIORATION GRADUELLE #1: Network plus large SEULEMENT
        # ============================================================

        # Baseline hyperparamètres (prouvés)
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

        # AMÉLIORATION #1: Network plus large (moins risqué)
        cfg.hidden_size = 512  # ← Compromis entre 256 et 512

        # Autres améliorations DÉSACTIVÉES pour l'instant
        cfg.normalize_obs = False     # ← Peut causer instabilité
        cfg.reward_clip = None        # ← Peut perturber signal
        cfg.weight_decay = 1e-6       # ← Très faible pour commencer

        # Checkpoint
        cfg.save_name = f"a2c_r_{cfg.rollout_steps}_u_{cfg.max_updates}_h_{cfg.hidden_size}.pt"
        cfg.plot_name = f"a2c_r_{cfg.rollout_steps}_u_{cfg.max_updates}_h_{cfg.hidden_size}.png"

        print("="*80)
        print("🔬 A2C GRADUAL - Améliorations Progressives")
        print("="*80)
        print(f"Hidden size: {cfg.hidden_size} (baseline=256, full=512)")
        print(f"Weight decay: {cfg.weight_decay} (très faible)")
        print()
        print("✅ AMÉLIORATIONS ACTIVÉES:")
        print(f"  • Hidden size augmenté: 256 → {cfg.hidden_size}")
        print(f"  • Weight decay minimal: {cfg.weight_decay}")
        print()
        print("❌ AMÉLIORATIONS EN ATTENTE:")
        print(f"  • Observation normalization: {cfg.normalize_obs}")
        print(f"  • Reward clipping: {cfg.reward_clip}")
        print()
        print("🎯 OBJECTIF: 220+ reward, 80-85% success")
        print("="*80 + "\n")

        # Entraîner
        history = train(cfg)

        print("\n" + "="*80)
        print("✅ Entraînement terminé!")
        print()
        print("📊 Si résultats > baseline (200, 74.8%):")
        print("   → Activez normalisation dans prochaine itération")
        print()
        print("📊 Si résultats ≈ baseline:")
        print("   → Hidden size 384 ne change pas grand chose")
        print("   → Essayez 512 ou activez normalisation")
        print()
        print("📊 Si résultats < baseline:")
        print("   → Revenez à hidden=256")
        print("   → Problème ailleurs (seed, env, etc.)")
        print("="*80)

    finally:
        # Restore stdout and close log
        sys.stdout = tee_logger.terminal
        tee_logger.close()
        print(f"\n✅ Log saved to: {log_filepath}")
