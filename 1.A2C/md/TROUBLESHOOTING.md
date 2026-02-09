# 🔧 Guide de Dépannage - A2C Lunar Lander

## ⚠️ "Les résultats sont pires qu'avant"

### **Diagnostic Rapide**

#### **Scénario 1: Vous testez l'ancien checkpoint (256 hidden)**
**Symptômes:**
```bash
RuntimeError: size mismatch for net.0.weight
```
ou performances dégradées avec l'ancien checkpoint.

**Cause:**
- Code actuel utilise `hidden_size=512`
- Ancien checkpoint a `hidden_size=256`
- OU normalizer activé mais pas dans le checkpoint

**Solution:**
```bash
# Option A: Utiliser le baseline exact
python A2C_baseline.py

# Option B: Tester l'ancien checkpoint correctement
# Le load_policy() devrait gérer ça automatiquement maintenant
```

---

#### **Scénario 2: Nouvel entraînement pire que baseline**
**Symptômes:**
- Returns stagnent à ~50-100 (vs 200 avant)
- Entropy reste > 1.0 après 1000 updates
- Value loss > 10 après 3000 updates

**Causes possibles:**

##### **A. Normalizer Instable en Début**
Le normalizer démarre avec mean=0, var=1 (données insuffisantes).

**Solution:**
```python
# Désactivez temporairement
cfg.normalize_obs = False
```

##### **B. Reward Clipping Trop Agressif**
Clip à ±10 peut trop réduire le signal d'apprentissage.

**Solution:**
```python
# Désactivez ou augmentez
cfg.reward_clip = None  # Ou 20.0
```

##### **C. Network 512 Trop Large**
Nécessite plus de données pour converger.

**Solution:**
```python
# Réduisez à 384 (compromis)
cfg.hidden_size = 384
```

##### **D. Weight Decay Trop Fort**
1e-5 peut trop régulariser en début.

**Solution:**
```python
# Réduisez
cfg.weight_decay = 1e-6  # Ou 0.0
```

---

## 📊 Comparaison des Configurations

| Config | Hidden | Normalize | Reward Clip | Weight Decay | Attendu |
|--------|--------|-----------|-------------|--------------|---------|
| **Baseline** | 256 | ❌ | ❌ | 0.0 | 200, 74.8% ✅ PROUVÉ |
| **Gradual** | 384 | ❌ | ❌ | 1e-6 | 210, 78-80% |
| **Full** | 512 | ✅ | 10.0 | 1e-5 | 220, 90%+ |

---

## 🧪 Plan de Test Systématique

### **Étape 1: Reproduire Baseline (OBLIGATOIRE)**
```bash
python A2C_baseline.py
```

**Résultat attendu:**
- Best eval: ~220
- Final: ~200
- Success: 74-76%

**Si ça ne marche PAS:**
→ Problème ailleurs (environment, seed, PyTorch version)
→ STOP et debug

**Si ça marche:**
→ Continuez étape 2

---

### **Étape 2: Tester Hidden Size Augmenté**
```bash
python A2C_gradual.py  # hidden=384
```

**Résultat attendu:**
- Best eval: 220-230
- Final: 205-215
- Success: 78-82%

**Si meilleur que baseline:**
→ Hidden size aide! Continuez étape 3

**Si pareil:**
→ Hidden size ne change rien, essayez normalisation

**Si pire:**
→ Revenez à 256, problème de convergence

---

### **Étape 3: Ajouter Normalisation**

Modifiez `A2C_gradual.py`:
```python
cfg.hidden_size = 384  # ou 512 si étape 2 réussie
cfg.normalize_obs = True  # ← NOUVEAU
cfg.reward_clip = None    # Pas encore
cfg.weight_decay = 1e-6   # Minimal
```

**Résultat attendu:**
- Best eval: 230-240
- Final: 215-225
- Success: 85-90%

**Si meilleur:**
→ Normalisation aide! Continuez étape 4

**Si instable (NaN, diverge):**
→ Problème avec normalizer, désactivez

---

### **Étape 4: Ajouter Reward Clipping**

```python
cfg.normalize_obs = True
cfg.reward_clip = 15.0  # ← Moins agressif que 10.0
```

**Résultat attendu:**
- Best eval: 235-250
- Final: 220-230
- Success: 88-93%

---

### **Étape 5: Full Config (Si Tout Marche)**

```python
cfg.hidden_size = 512
cfg.normalize_obs = True
cfg.reward_clip = 10.0
cfg.weight_decay = 1e-5
```

**Résultat attendu:**
- Best eval: 250+
- Final: 230+
- Success: 90-95%

---

## 🚨 Signaux d'Alerte

### **CRITIQUE - STOP Immédiatement**
```
Update 100 | return= NaN  ❌ DIVERGENCE
Update 500 | value=45.2   ❌ Value loss explose
[EVAL] avg_return = -2000 ❌ Eval catastrophique
```

**Action:**
1. Ctrl+C pour arrêter
2. Revenir au baseline
3. Désactiver TOUTES les améliorations
4. Chercher bug dans le code

---

### **WARNING - Surveiller**
```
Update 2000 | entropy=1.1  ⚠️ Entropy trop haute
Update 3000 | return=50    ⚠️ Stagnation
[EVAL STATS] std=150.2     ⚠️ Variance très haute
```

**Action:**
1. Laisser entraîner jusqu'à update 5000
2. Si pas d'amélioration → revert une amélioration
3. Essayer learning rate plus élevé

---

## 🔍 Debugging Checklist

Si performances pires qu'attendu:

- [ ] **Vérifier seed:** Même seed (42) utilisé?
- [ ] **Vérifier env:** LunarLander-v3 (pas v2)?
- [ ] **Vérifier PyTorch:** Version compatible?
- [ ] **Vérifier logs:**
  - `[INFO] Observation normalization: ENABLED` si normalize_obs=True
  - `[INFO] Reward clipping: ENABLED` si reward_clip défini
- [ ] **Comparer configs:**
  ```python
  # Dans le log, section CONFIGURATION
  # Comparez avec baseline réussi
  ```
- [ ] **Vérifier gradients:**
  ```
  adv: μ≈0.000 σ≈1.000  ✅ OK
  adv: μ=2.5 σ=15.2     ❌ PAS OK (advantages non normalisés)
  ```

---

## 💡 Optimisations Alternatives

Si AUCUNE amélioration ne marche:

### **Option 1: Tuning Hyperparamètres**
```python
# Essayez des LR différents
cfg.lr_policy = 7e-4  # Au lieu de 5e-4
cfg.lr_value = 1.5e-3  # Au lieu de 1e-3

# Ou entropy annealing plus lent
cfg.entropy_coef_final = 0.01  # Au lieu de 0.005
```

### **Option 2: Plus d'Entraînement**
```python
cfg.max_updates = 15000  # Au lieu de 10000
```

### **Option 3: Changer Algo**
- Passer à **PPO** (plus stable)
- Ou **SAC** (state-of-the-art)

---

## 📈 Résultats de Référence

### **Baseline (PROUVÉ)**
```
Config: 256 hidden, no improvements
Updates: 8484
Time: 62.4 min
Best eval: 220.2
Final: 200.1 ± 99.9
Success: 74.8%
Status: ✅ SOLVED
```

### **Objectif avec Améliorations**
```
Config: 512 hidden, all improvements
Updates: 6000-7000
Time: 50-60 min
Best eval: 250+
Final: 220 ± 50
Success: 90-95%
Status: ✅ SOLVED STABLE
```

**Si vous n'atteignez pas ces résultats:**
1. Revenez au baseline
2. Vérifiez que baseline fonctionne
3. Ajoutez améliorations UNE par UNE
4. Testez chaque changement séparément

---

## 🆘 Support

**Si tout échoue:**

1. Partagez votre log file complet
2. Partagez votre configuration exacte
3. Indiquez:
   - Résultats attendus vs obtenus
   - Étape où ça bloque
   - Messages d'erreur

**Commandes utiles pour debug:**
```bash
# Comparer deux logs
diff logs/a2c_baseline_*.log logs/a2c_gradual_*.log

# Voir dernières 50 lignes
tail -50 logs/a2c_*.log

# Chercher "SOLVED"
grep "SOLVED" logs/*.log

# Extraire best eval de tous les runs
grep "Best eval reward" logs/*.log
```

---

## ✅ Rappel: Commencez Simple!

**N'essayez PAS d'appliquer toutes les améliorations d'un coup.**

**Workflow recommandé:**
1. ✅ Baseline (256, no improvements) → **74.8% success**
2. ✅ +Hidden size (384) → **78-82% success**
3. ✅ +Normalization → **85-90% success**
4. ✅ +Reward clip → **90-95% success**

Chaque étape doit AMÉLIORER les résultats, sinon STOP et debug!

---

**Bonne chance! 🚀**
