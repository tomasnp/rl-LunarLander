#!/usr/bin/env python3
"""
Script de test rapide pour A2C avec GAE
Entraînement court pour vérifier que tout fonctionne
"""

import sys
sys.path.insert(0, '.')

from A2C import Config, train

if __name__ == "__main__":
    cfg = Config()

    # Configuration pour test rapide (10 minutes environ)
    cfg.max_updates = 100
    cfg.rollout_steps = 2048
    cfg.eval_every = 20

    print("="*70)
    print("🧪 TEST RAPIDE A2C - 100 updates (~5 minutes)")
    print("="*70)
    print(f"Rollout steps: {cfg.rollout_steps}")
    print(f"Max updates: {cfg.max_updates}")
    print(f"Total steps: {cfg.rollout_steps * cfg.max_updates:,}")
    print("="*70 + "\n")

    # Entraîner
    history = train(cfg)

    print("\n" + "="*70)
    print("✅ Test terminé ! Vérifiez:")
    print("  1. adv_mean proche de 0")
    print("  2. adv_std proche de 1.0")
    print("  3. entropy décroît graduellement")
    print("  4. training_performance_a2c.png généré")
    print("="*70)
