# 📝 Système de Logging Ajouté à REINFORCE

## 🎯 Améliorations Appliquées

J'ai importé le système de logging complet de A2C.py vers reinforce.py pour avoir une traçabilité complète des entraînements.

---

## ✅ Ce Qui a Été Ajouté

### **1. TeeLogger Class**

```python
class TeeLogger:
    """Duplique stdout vers terminal ET fichier log."""
    def __init__(self, filepath, mode='a'):
        self.terminal = sys.stdout
        self.log = open(filepath, mode, encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)  # Affiche dans terminal
        self.log.write(message)       # Sauvegarde dans fichier
        self.log.flush()
```

**Utilité:**
- Tout ce qui s'affiche dans le terminal est automatiquement sauvegardé
- Pas besoin de rediriger manuellement avec `> logfile.txt`
- Le log est disponible même si l'entraînement crash

---

### **2. setup_logging() Function**

```python
def setup_logging(log_dir="logs", experiment_name="reinforce"):
    """Crée un fichier log avec timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    tee = TeeLogger(log_filepath, mode='a')
    return log_filepath, tee
```

**Génère:**
```
logs/
├── reinforce_20260208_143052.log
├── reinforce_20260208_150234.log
└── reinforce_test_20260208_162845.log
```

**Avantages:**
- Timestamp automatique (jamais de conflit de noms)
- Organisation dans dossier `logs/`
- Nom indique l'expérience (reinforce, reinforce_test, etc.)

---

### **3. log_config() Function**

```python
def log_config(cfg: Config):
    """Log tous les hyperparamètres."""
    print("\n" + "=" * 80)
    print("⚙️  CONFIGURATION")
    print("=" * 80)
    for field_name in cfg.__dataclass_fields__:
        value = getattr(cfg, field_name)
        print(f"  {field_name:25s} = {value}")
    print("=" * 80)
```

**Génère dans le log:**
```
================================================================================
⚙️  CONFIGURATION
================================================================================
  env_id                    = LunarLander-v3
  seed                      = 42
  gamma                     = 0.99
  lr_policy                 = 0.0003
  lr_value                  = 0.001
  entropy_coef              = 0.05
  value_coef                = 0.5
  max_episodes              = 5000
  eval_every                = 50
  eval_episodes             = 10
  hidden_size               = 128
  ...
================================================================================
```

**Utilité:**
- Savoir exactement quels hyperparamètres ont été utilisés
- Reproductibilité des expériences
- Comparaison facile entre runs

---

### **4. Enhanced Training Summary**

```python
# À la fin de train()
print("\n" + "=" * 80)
print("📊 TRAINING SUMMARY")
print("=" * 80)
print(f"Training time:        {training_time:.1f}s ({training_time/60:.1f} minutes)")
print(f"Total episodes:       {len(reward_history)}")
print(f"Best eval reward:     {best_eval:.1f}")
print(f"Final mean (100 ep):  {final_mean:.1f} ± {final_std:.1f}")
print(f"Solved:               {'✅ YES' if solved else '❌ NO'}")
print(f"Checkpoint:           {save_path}")
print("=" * 80)
```

**Génère:**
```
================================================================================
📊 TRAINING SUMMARY
================================================================================
Training time:        2847.3s (47.5 minutes)
Total episodes:       2456
Best eval reward:     215.3
Final mean (100 ep):  210.6 ± 18.2
Solved:               ✅ YES
Checkpoint:           checkpoints/reinforce_baseline_lunar.pt
================================================================================
```

---

### **5. Main Block avec try/finally**

```python
if __name__ == "__main__":
    # Setup logging
    log_filepath, tee_logger = setup_logging(log_dir="logs", experiment_name="reinforce")
    sys.stdout = tee_logger

    try:
        cfg = Config()

        # Print header
        print("=" * 80)
        print("🚀 REINFORCE with Baseline - Lunar Lander Training")
        print("=" * 80)

        # Log configuration
        log_config(cfg)

        # Train
        history = train(cfg)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        traceback.print_exc()
    finally:
        # TOUJOURS sauvegarder le log
        sys.stdout = tee_logger.terminal
        tee_logger.close()
        print(f"\n✅ Log saved to: {log_filepath}")
```

**Avantages:**
- Log sauvegardé même si Ctrl+C ou crash
- Erreurs et stack traces capturés dans le log
- Message de confirmation à la fin

---

## 📊 Exemple de Log Complet

**Fichier:** `logs/reinforce_20260208_143052.log`

```
================================================================================
🗂️  Logging started: 2026-02-08 14:30:52
📝 Log file: logs/reinforce_20260208_143052.log
================================================================================

================================================================================
🚀 REINFORCE with Baseline - Lunar Lander Training
================================================================================
Max episodes:       5000
Eval every:         50 episodes
Hidden size:        128
Learning rates:     policy=0.0003, value=0.001
Entropy coef:       0.05
================================================================================

================================================================================
⚙️  CONFIGURATION
================================================================================
  env_id                    = LunarLander-v3
  seed                      = 42
  gamma                     = 0.99
  lr_policy                 = 0.0003
  lr_value                  = 0.001
  entropy_coef              = 0.05
  value_coef                = 0.5
  max_episodes              = 5000
  eval_every                = 50
  eval_episodes             = 10
  hidden_size               = 128
  ...
================================================================================

[INFO] Device: cpu
[INFO] PyTorch version: 2.1.0
[INFO] Gymnasium version: 0.29.1

Ep   10 | return=  -145.3 | mean(100)= -145.3 | loss=2.456 | ...
Ep   20 | return=   -98.7 | mean(100)=  -98.7 | loss=2.123 | ...
...
[EVAL] Ep   50 | avg_return over 10 eps = -85.4
...
Ep  500 | return=   45.2 | mean(100)=   42.3 | loss=1.234 | ...
[EVAL] Ep  500 | avg_return over 10 eps = 52.3
...
Ep 2000 | return=  210.6 | mean(100)=  205.3 | loss=0.456 | ...
[EVAL] Ep 2000 | avg_return over 10 eps = 215.3
[SAVE] New best model saved to: checkpoints/reinforce_baseline_lunar.pt
[DONE] Solved! rolling_mean=205.3 >= 200.0

================================================================================
📊 TRAINING SUMMARY
================================================================================
Training time:        2847.3s (47.5 minutes)
Total episodes:       2456
Best eval reward:     215.3
Final mean (100 ep):  210.6 ± 18.2
Solved:               ✅ YES
Checkpoint:           checkpoints/reinforce_baseline_lunar.pt
================================================================================

📊 Graphique sauvegardé : training_performance.png

✅ Log saved to: logs/reinforce_20260208_143052.log
```

---

## 🧪 Comment Utiliser

### **Test Rapide (100 episodes)**
```bash
python test_reinforce.py
```

**Génère:**
- `logs/reinforce_test_YYYYMMDD_HHMMSS.log`
- Terminal output visible en temps réel
- Log sauvegardé automatiquement

---

### **Entraînement Complet**
```bash
cd src
python reinforce.py
```

**Génère:**
- `logs/reinforce_YYYYMMDD_HHMMSS.log`
- `training_performance.png`
- `checkpoints/reinforce_baseline_lunar.pt`

---

## 📈 Analyser les Logs

### **Voir le dernier log**
```bash
ls -t logs/reinforce_*.log | head -1 | xargs cat
```

### **Suivre en temps réel**
```bash
tail -f logs/reinforce_20260208_143052.log
```

### **Comparer plusieurs runs**
```bash
# Extraire best eval de tous les logs
grep "Best eval reward" logs/reinforce_*.log

# Sortie:
# logs/reinforce_20260208_143052.log:Best eval reward:     215.3
# logs/reinforce_20260208_150234.log:Best eval reward:     -23.2
# logs/reinforce_20260208_162845.log:Best eval reward:     198.7
```

### **Chercher si résolu**
```bash
grep "Solved:" logs/reinforce_*.log
```

---

## 🔍 Différences A2C vs REINFORCE

| Feature | A2C | REINFORCE |
|---------|-----|-----------|
| **Logging System** | ✅ Identique | ✅ Identique |
| **Config Logging** | ✅ Identique | ✅ Identique |
| **Training Summary** | ✅ Identique | ✅ Identique |
| **try/finally** | ✅ Identique | ✅ Identique |
| **Timestamp** | ✅ a2c_gae_YYYYMMDD_HHMMSS.log | ✅ reinforce_YYYYMMDD_HHMMSS.log |
| **Updates** | Per update (batch) | Per episode |
| **Experiment name** | "a2c_gae" | "reinforce" |

---

## ✅ Avantages du Logging

### **1. Traçabilité**
- Savoir exactement ce qui s'est passé
- Configuration complète enregistrée
- Reproductibilité garantie

### **2. Debugging**
- Stack traces sauvegardées
- Valeurs intermédiaires enregistrées
- Facile de retrouver où ça a échoué

### **3. Comparaison**
- Comparer facilement plusieurs runs
- `grep` pour extraire métriques
- Analyse post-mortem

### **4. Historique**
- Garder trace de tous les essais
- Voir l'évolution des hyperparamètres
- Pas de perte d'information

### **5. Collaboration**
- Partager logs avec équipe
- Montrer résultats sans re-run
- Documentation automatique

---

## 🎓 Best Practices

### **DO ✅**
1. Laissez le logging activé par défaut
2. Archivez les runs réussis
3. Utilisez `grep` pour analyser
4. Gardez les logs importants
5. Comparez configs entre runs

### **DON'T ❌**
1. Ne désactivez pas le logging
2. Ne modifiez pas les logs manuellement
3. Ne commitez pas les logs dans Git (trop gros)
4. Ne supprimez pas immédiatement
5. N'ignorez pas les warnings dans les logs

---

## 📚 Fichiers Créés/Modifiés

### **Modifié:**
- `src/reinforce.py` - Ajout système de logging complet

### **Créé:**
- `test_reinforce.py` - Script de test avec logging
- `REINFORCE_LOGGING_ADDED.md` - Ce document

### **Généré automatiquement:**
- `logs/reinforce_*.log` - Logs d'entraînement
- `logs/reinforce_test_*.log` - Logs de test

---

## 🎯 Prochaines Étapes

1. **Testez le logging:**
   ```bash
   python test_reinforce.py
   ```

2. **Vérifiez le log créé:**
   ```bash
   ls -lh logs/
   cat logs/reinforce_test_*.log
   ```

3. **Lancez entraînement complet:**
   ```bash
   cd src
   python reinforce.py
   ```

4. **Comparez avec A2C:**
   ```bash
   # Logs A2C
   cat logs/a2c_gae_*.log

   # Logs REINFORCE
   cat logs/reinforce_*.log
   ```

---

**Le système de logging est maintenant identique entre A2C et REINFORCE!** 🎉

Tous les entraînements sont automatiquement enregistrés avec timestamp et configuration complète.
