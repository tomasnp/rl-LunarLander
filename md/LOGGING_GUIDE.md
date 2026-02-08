# 📝 Guide du Système de Logging

## 🎯 Fonctionnalités

Le système de logging automatique enregistre **toutes** les sorties du terminal dans un fichier avec timestamp. Chaque exécution crée un nouveau fichier de log.

### **Ce qui est enregistré :**
- ✅ Tous les `print()` statements
- ✅ Configuration complète (hyperparamètres)
- ✅ Métriques d'entraînement (chaque update)
- ✅ Résultats d'évaluation
- ✅ Résumé final (temps, best eval, etc.)
- ✅ Erreurs et stack traces
- ✅ Interruptions (Ctrl+C)

---

## 📁 Structure des Fichiers de Log

### **Emplacement:**
```
logs/
├── a2c_gae_20260207_143052.log    # Timestamp: YYYYMMDD_HHMMSS
├── a2c_gae_20260207_150234.log
├── a2c_gae_20260207_162845.log
└── ...
```

### **Nom de Fichier:**
```
a2c_gae_20260207_143052.log
│       │        │      │
│       │        │      └─ Secondes
│       │        └─ Heure:Minute
│       └─ Date (Année/Mois/Jour)
└─ Nom de l'expérience
```

---

## 📊 Exemple de Contenu d'un Log

```
================================================================================
🗂️  Logging started: 2026-02-07 14:30:52
📝 Log file: logs/a2c_gae_20260207_143052.log
================================================================================

================================================================================
🚀 A2C with GAE - Lunar Lander Training
================================================================================
Rollout steps:      2048
Max updates:        3000
GAE lambda:         0.95
Entropy annealing:  0.05 → 0.001
Gradient clipping:  0.5
================================================================================

================================================================================
⚙️  CONFIGURATION
================================================================================
  env_id                    = LunarLander-v3
  seed                      = 42
  gamma                     = 0.99
  gae_lambda                = 0.95
  lr_policy                 = 0.0003
  lr_value                  = 0.0003
  entropy_coef_start        = 0.05
  entropy_coef_final        = 0.001
  value_coef                = 0.5
  rollout_steps             = 2048
  max_updates               = 3000
  eval_every                = 50
  eval_episodes             = 10
  hidden_size               = 128
  grad_clip                 = 0.5
  save_dir                  = checkpoints
  save_name                 = a2c_2048_5000.pt
  render_eval_human         = False
  record_video              = False
  video_dir                 = videos_record
  solved_mean_reward        = 200.0
  solved_window             = 100
================================================================================

[INFO] Device: cpu
[INFO] Training mode: A2C with GAE (rollout_steps=2048)
[INFO] PyTorch version: 2.1.0
[INFO] Gymnasium version: 0.29.1

Update   10 | return=  -145.3 (n=8) | loss=2.456 | policy=1.234 | value=1.222 | entropy=1.183 (coef=0.0499) | adv: μ=0.002 σ=0.998
Update   20 | return=  -98.7 (n=18) | loss=2.123 | policy=0.987 | value=1.136 | entropy=1.045 (coef=0.0498) | adv: μ=-0.001 σ=1.001
...
[EVAL] Update   50 | avg_return over 10 eps = 125.4
[SAVE] New best model saved to: checkpoints/a2c_2048_5000.pt
...
Update 2000 | return= 210.6 (n=16450) | ...
[EVAL] Update 2000 | avg_return over 10 eps = 215.3
[DONE] Solved! rolling_mean=205.3 >= 200.0

================================================================================
📊 TRAINING SUMMARY
================================================================================
Training time:        2847.3s (47.5 minutes)
Total updates:        2000
Total episodes:       16450
Best eval reward:     215.3
Final mean (100 ep):  210.6 ± 18.2
Solved:               ✅ YES
Checkpoint:           checkpoints/a2c_2048_5000.pt
================================================================================

📊 Graphique sauvegardé : training_performance_a2c.png

✅ Training completed at: 2026-02-07 15:18:39

✅ Log saved to: logs/a2c_gae_20260207_143052.log
```

---

## 🔍 Utilisation des Logs

### **1. Consulter un Log Récent**
```bash
# Log le plus récent
tail -f logs/a2c_gae_*.log | tail -1

# Ou spécifiquement
cat logs/a2c_gae_20260207_143052.log
```

### **2. Suivre un Entraînement en Cours**
```bash
tail -f logs/a2c_gae_20260207_143052.log
```

### **3. Comparer Plusieurs Runs**
```bash
# Extraire les résultats finaux de tous les logs
grep "Best eval reward" logs/*.log

# Exemple de sortie:
# logs/a2c_gae_20260207_143052.log:Best eval reward:     215.3
# logs/a2c_gae_20260207_150234.log:Best eval reward:     -23.2
# logs/a2c_gae_20260207_162845.log:Best eval reward:     198.7
```

### **4. Chercher des Patterns**
```bash
# Tous les eval results
grep "\[EVAL\]" logs/a2c_gae_20260207_143052.log

# Tous les updates 100, 200, 300, etc.
grep "Update  [0-9]*00 |" logs/a2c_gae_20260207_143052.log

# Chercher si l'entraînement a été interrompu
grep "interrupted" logs/*.log
```

### **5. Extraire Métriques pour Plot**
```bash
# Extraire toutes les entropy values
grep "entropy=" logs/a2c_gae_20260207_143052.log | sed 's/.*entropy=\([0-9.]*\).*/\1/'

# Returns moyens
grep "return=" logs/a2c_gae_20260207_143052.log | sed 's/.*return=\s*\([0-9.-]*\).*/\1/'
```

---

## 🛠️ Gestion des Logs

### **Nettoyer les Vieux Logs**
```bash
# Supprimer logs plus vieux que 7 jours
find logs/ -name "*.log" -mtime +7 -delete

# Garder seulement les 10 derniers
ls -t logs/*.log | tail -n +11 | xargs rm
```

### **Archiver les Logs Importants**
```bash
# Créer un dossier d'archives
mkdir -p logs/archive

# Archiver un run réussi
cp logs/a2c_gae_20260207_143052.log logs/archive/a2c_SOLVED_20260207.log
```

### **Compresser les Logs**
```bash
# Compresser tous les logs
gzip logs/*.log

# Décompresser pour lire
gunzip logs/a2c_gae_20260207_143052.log.gz
```

---

## 📈 Analyse Automatique des Logs

### **Script Python pour Analyser les Logs**

```python
import re
import glob
from pathlib import Path

def analyze_log(log_path):
    """Extract key metrics from log file."""
    with open(log_path, 'r') as f:
        content = f.read()

    # Extract metrics
    best_eval = re.search(r'Best eval reward:\s+([-0-9.]+)', content)
    training_time = re.search(r'Training time:\s+([-0-9.]+)s', content)
    solved = re.search(r'Solved:\s+(.*)', content)

    return {
        'file': Path(log_path).name,
        'best_eval': float(best_eval.group(1)) if best_eval else None,
        'time_sec': float(training_time.group(1)) if training_time else None,
        'solved': '✅' in solved.group(1) if solved else False
    }

# Analyser tous les logs
logs = glob.glob('logs/a2c_gae_*.log')
results = [analyze_log(log) for log in logs]

# Trier par best eval
results.sort(key=lambda x: x['best_eval'] if x['best_eval'] else -1e9, reverse=True)

# Afficher tableau
print("Rank | File                              | Best Eval | Time (min) | Solved")
print("-----|-----------------------------------|-----------|------------|-------")
for i, r in enumerate(results, 1):
    time_min = r['time_sec'] / 60 if r['time_sec'] else 0
    solved = "✅" if r['solved'] else "❌"
    print(f"{i:4d} | {r['file']:33s} | {r['best_eval']:9.1f} | {time_min:10.1f} | {solved}")
```

**Usage:**
```bash
python analyze_logs.py

# Sortie:
# Rank | File                              | Best Eval | Time (min) | Solved
# -----|-----------------------------------|-----------|------------|-------
#    1 | a2c_gae_20260207_143052.log       |     215.3 |       47.5 | ✅
#    2 | a2c_gae_20260207_162845.log       |     198.7 |       51.2 | ❌
#    3 | a2c_gae_20260207_150234.log       |     -23.2 |       28.3 | ❌
```

---

## 🎯 Best Practices

### **DO ✅**
1. Laissez le logging activé par défaut
2. Archivez les runs réussis (best eval > 200)
3. Utilisez `grep` pour chercher patterns dans les logs
4. Comparez différents hyperparamètres via les logs
5. Gardez les logs des expériences importantes

### **DON'T ❌**
1. Ne supprimez pas les logs immédiatement après run
2. Ne modifiez pas manuellement les logs (corrompt timestamp)
3. Ne commitez pas les logs dans Git (trop gros)

---

## 🐛 Troubleshooting

### **Problème: Log file vide**
- Vérifiez que `sys.stdout = tee_logger` est appelé
- Vérifiez les permissions du dossier `logs/`

### **Problème: Encoding errors**
- Le TeeLogger utilise `encoding='utf-8'` par défaut
- Si problème, changez dans la classe TeeLogger

### **Problème: Log ne se ferme pas**
- Le `finally` block garantit que le log est fermé
- Même si Ctrl+C ou erreur, le fichier est fermé proprement

---

## 📚 Références

### **Fichiers Liés**
- `A2C.py` : Code principal avec logging intégré
- `logs/` : Dossier contenant tous les logs
- `checkpoints/` : Checkpoints sauvegardés

### **Classes et Fonctions**
- `TeeLogger` : Classe qui duplique stdout vers fichier
- `setup_logging()` : Configure le logging avec timestamp
- `log_config()` : Log tous les hyperparamètres

---

## ✅ Checklist

Après chaque entraînement:
- [ ] Log créé dans `logs/` avec timestamp ✅
- [ ] Configuration complète enregistrée ✅
- [ ] Résultat final (best eval) présent ✅
- [ ] Si résolu (≥200), archiver le log ✅
- [ ] Comparer avec runs précédents ✅

---

## 🎉 Avantages du Logging

1. **Traçabilité complète** : Savoir exactement ce qui s'est passé
2. **Comparaisons faciles** : `grep` pour comparer runs
3. **Debugging** : Stack traces complètes en cas d'erreur
4. **Reproductibilité** : Configuration exacte enregistrée
5. **Historique** : Voir l'évolution des expériences

Tous vos runs sont maintenant enregistrés automatiquement ! 🚀
