#!/usr/bin/env python3
"""
Script de test pour A2C amélioré avec normalisation et stabilité
Test rapide (100 updates) pour vérifier que tout fonctionne
"""

import sys
sys.path.insert(0, '.')

from A2C import Config, train, setup_logging

if __name__ == "__main__":
    # Setup logging
    log_filepath, tee_logger = setup_logging(log_dir="logs", experiment_name="a2c_improved_test")
    sys.stdout = tee_logger

    try:
        cfg = Config()

        # Configuration pour test rapide (5-10 minutes environ)
        cfg.max_updates = 100
        cfg.rollout_steps = 2048
        cfg.eval_every = 25

        # AMÉLIORATIONS ACTIVÉES
        cfg.normalize_obs = True      # ← Normalisation des observations
        cfg.reward_clip = 10.0         # ← Clipping des rewards
        cfg.hidden_size = 512          # ← Réseau plus large
        cfg.weight_decay = 1e-5        # ← Régularisation L2

        print("="*80)
        print("🧪 TEST RAPIDE A2C AMÉLIORÉ - 100 updates (~5-10 minutes)")
        print("="*80)
        print(f"Rollout steps: {cfg.rollout_steps}")
        print(f"Max updates: {cfg.max_updates}")
        print(f"Total steps: {cfg.rollout_steps * cfg.max_updates:,}")
        print()
        print("🚀 AMÉLIORATIONS ACTIVÉES:")
        print(f"  ✓ Observation normalization (clip=±{cfg.obs_clip})")
        print(f"  ✓ Reward clipping (clip=±{cfg.reward_clip})")
        print(f"  ✓ Larger network (hidden={cfg.hidden_size})")
        print(f"  ✓ AdamW optimizer (weight_decay={cfg.weight_decay})")
        print("="*80 + "\n")

        # Entraîner
        history = train(cfg)

        print("\n" + "="*80)
        print("✅ Test terminé ! Vérifiez:")
        print("  1. Observation normalizer actif")
        print("  2. Reward clipping appliqué")
        print("  3. Réseau 512 hidden units")
        print("  4. AdamW optimizer utilisé")
        print("  5. training_performance_a2c.png généré")
        print("="*80)

    finally:
        # Restore stdout and close log
        sys.stdout = tee_logger.terminal
        tee_logger.close()
        print(f"\n✅ Log saved to: {log_filepath}")
