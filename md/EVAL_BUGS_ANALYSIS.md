# 🐛 Analyse des Bugs d'Évaluation - A2C Lunar Lander

## 📊 Résultats Observés (Log 20260207_132338)

### **Progression Générale**
```
Training time: 448.4 minutes (7.5 heures)
Total updates: 7500
Best eval: 172.0
Final mean: 125.1 ± 112.7
Status: ❌ NOT SOLVED (target: 200)
```

### **Evals Catastrophiques en Début**
```
Update   50 | train=-127.2 → eval=-1527.9  ❌ (12x pire!)
Update  100 | train=-82.9  → eval=-3077.8  ❌ (37x pire!)
Update  150 | train=-57.8  → eval=-396.8   ❌ (7x pire!)
Update  200 | train=-30.2  → eval=-278.7   ❌ (9x pire!)
Update  400 | train=+8.4   → eval=-216.2   ❌ (26x pire!)
...
Update 7450 | train=134.2  → eval=172.0    ✅ (meilleur)
Update 7500 | train=125.1  → eval=121.8    ✅ (cohérent)
```

---

## 🔍 BUG #1: Évaluations Extrêmement Négatives

### **Problème Identifié**

Les returns d'éval en début d'entraînement sont 10-37x plus mauvais que le training. Cela suggère que certains épisodes d'évaluation accumulent des pénalités ÉNORMES.

### **Cause Probable: Time Limit**

LunarLander-v3 a un time limit de **1000 steps** par défaut. Si la policy est très mauvaise au début:

1. Agent crash rapidement → -100 reward (normal)
2. **OU** Agent reste en vol sans atterrir → accumule des pénalités chaque step

**Exemple de catastrophe:**
```python
# Policy très mauvaise qui fait juste hover
# Chaque step: reward ≈ -0.3 (carburant) - 0.3 (distance) = -0.6
# Sur 1000 steps: -0.6 * 1000 = -600 reward ❌

# Pire: si plusieurs épisodes comme ça
# 10 épisodes x -600 = -6000 / 10 = -600 moyenne ❌
# Mais si 1 épisode à -3000 (bug?) + 9 autres à -100:
# (-3000 + 9*(-100)) / 10 = -3900 / 10 = -390 ❌
```

### **Vérification dans le Code**

La fonction `evaluate()` ne semble PAS avoir de protection contre les épisodes qui ne terminent jamais:

```python
# A2C.py ligne 546-565
def evaluate(cfg: Config, policy: PolicyNet, device: torch.device = torch.device('cpu')) -> float:
    env = make_eval_env(cfg)  # ← Crée env avec render_mode
    returns = []
    for _ in range(cfg.eval_episodes):
        obs, _ = env.reset()  # ← PAS de seed fixe (OK maintenant)
        done = False
        ep_return = 0.0
        while not done:  # ← Peut boucler jusqu'à time limit
            # ...
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)  # ← Accumule toutes les pénalités
        returns.append(ep_return)
    return float(np.mean(returns))
```

**Le problème**: Si la policy au début est mauvaise et fait juste hover sans jamais atterrir, l'épisode va jusqu'au time limit (1000 steps) et accumule -600 à -1000 de pénalités.

### **Solutions Proposées**

#### **Solution 1: Augmenter eval_episodes (recommandé)**
```python
cfg.eval_episodes = 30  # Au lieu de 10
```
- Réduit la variance causée par 1-2 épisodes catastrophiques
- Plus représentatif de la vraie performance

#### **Solution 2: Clipper les returns d'eval**
```python
# Dans evaluate()
returns.append(max(ep_return, -500))  # Clip les catastrophes
```
- Empêche un seul épisode catastrophique de dominer la moyenne

#### **Solution 3: Logger les stats détaillées**
```python
# Dans evaluate()
print(f"[EVAL DEBUG] Episode returns: {returns}")
print(f"[EVAL DEBUG] Min: {min(returns):.1f}, Max: {max(returns):.1f}, Std: {np.std(returns):.1f}")
```
- Permet d'identifier si le problème vient de quelques outliers

#### **Solution 4: Early stopping des épisodes pathologiques**
```python
# Dans evaluate()
MAX_STEPS = 1000  # Limit explicite
step_count = 0
while not done and step_count < MAX_STEPS:
    # ...
    step_count += 1
if step_count >= MAX_STEPS:
    print(f"[EVAL WARNING] Episode truncated at {MAX_STEPS} steps")
```

---

## 🔍 BUG #2: Entropy Reste Haute

### **Observation**
```
Update    0: entropy=1.381 (max, policy uniforme)
Update 7500: entropy=0.740 (toujours haute!)
  entropy_coef=0.001 (minimal)
```

**Attendu**: Entropy devrait descendre vers 0.3-0.4 pour une policy décisive.

### **Problème**

L'entropy NE DESCEND PAS malgré:
- 7500 updates d'entraînement
- entropy_coef annealed à 0.001 (quasi nul)
- Training return plateau à ~125

**Cela signifie:** La policy reste très stochastique (indécise).

### **Causes Possibles**

1. **Annealing trop rapide**
   - entropy_coef passe de 0.05 à 0.001 sur 7500 updates
   - Progress = update / 7500
   - À update 3750: coef ≈ 0.025 (déjà moitié)
   - La policy n'a pas eu assez de temps pour explorer avant que le bonus entropy disparaisse

2. **Policy gradient trop faible vs entropy**
   - Si les advantages sont petits, le gradient de policy est faible
   - L'entropy domine et empêche la convergence

3. **Learning rate trop bas**
   - lr_policy = 3e-4 pourrait être trop bas
   - La policy n'apprend pas assez vite

### **Solutions Proposées**

#### **Solution 1: Ralentir l'annealing (RECOMMANDÉ)**
```python
# Config
entropy_coef_start: float = 0.05
entropy_coef_final: float = 0.005  # Au lieu de 0.001
max_updates: int = 10000  # Au lieu de 7500

# Ou changer la formule d'annealing
# Linéaire → Exponentiel (reste haut plus longtemps)
entropy_coef = max(
    cfg.entropy_coef_final,
    cfg.entropy_coef_start * (0.995 ** update_idx)  # Décroissance exp
)
```

#### **Solution 2: Augmenter lr_policy**
```python
lr_policy: float = 5e-4  # Au lieu de 3e-4
```

#### **Solution 3: Augmenter value_coef**
```python
value_coef: float = 1.0  # Au lieu de 0.5
```
- Force le critique à converger plus vite
- Advantages plus stables → policy gradient plus fort

---

## 🔍 BUG #3: Value Loss Reste Élevée

### **Observation**
```
Update    0: value_loss=32.224 (très haute)
Update 7500: value_loss=7.322  (toujours haute)
```

**Attendu**: value_loss < 2.0 pour convergence

### **Problème**

Le critique ne converge PAS. Cela signifie:
- Les predictions de valeur sont mauvaises
- Les advantages sont bruités
- Le policy gradient est instable

### **Causes Possibles**

1. **Learning rate trop bas**
   - lr_value = 3e-4 pourrait être trop bas
   - Le critique n'apprend pas assez vite

2. **Targets instables (GAE)**
   - Si les returns calculés par GAE sont bruyants
   - Le critique ne peut pas apprendre

3. **Network trop petit**
   - hidden_size = 128 pourrait être insuffisant
   - Le critique ne peut pas approximer la value function

### **Solutions Proposées**

#### **Solution 1: Augmenter lr_value**
```python
lr_value: float = 1e-3  # Au lieu de 3e-4
```

#### **Solution 2: Augmenter hidden_size**
```python
hidden_size: int = 256  # Au lieu de 128
```

#### **Solution 3: Réduire GAE lambda**
```python
gae_lambda: float = 0.9  # Au lieu de 0.95
```
- Moins de variance dans les returns
- Mais plus de bias

---

## 🔍 BUG #4: Plateau à ~120-140

### **Observation**
```
Update 6000-7500: returns oscillent entre 60-140
Best eval: 172.0 (update 7450)
Final: 121.8 (update 7500)
Jamais atteint 200
```

### **Problème**

L'agent plateau et ne progresse plus après update ~5000.

### **Causes Cumulatives**

1. Entropy trop haute → policy indécise
2. Value loss haute → gradients bruités
3. Annealing trop rapide → exploration arrêtée trop tôt
4. Network trop petit → capacité limitée

### **Solutions**

Appliquer TOUTES les corrections précédentes:
1. Ralentir entropy annealing
2. Augmenter lr_policy et lr_value
3. Augmenter hidden_size
4. Augmenter eval_episodes
5. Plus d'updates (10000 au lieu de 7500)

---

## 📋 PLAN D'ACTION RECOMMANDÉ

### **🔥 Corrections Urgentes (Priorité 1)**

```python
@dataclass
class Config:
    # ... autres params ...

    # 1. Ralentir entropy annealing
    entropy_coef_start: float = 0.05
    entropy_coef_final: float = 0.005  # ← Au lieu de 0.001

    # 2. Plus d'updates
    max_updates: int = 10000  # ← Au lieu de 7500

    # 3. Plus d'épisodes d'eval (réduire variance)
    eval_episodes: int = 30  # ← Au lieu de 10

    # 4. Augmenter LR du critique
    lr_value: float = 1e-3  # ← Au lieu de 3e-4
```

### **⚡ Corrections Importantes (Priorité 2)**

```python
@dataclass
class Config:
    # 5. Network plus large
    hidden_size: int = 256  # ← Au lieu de 128

    # 6. Augmenter LR de la policy
    lr_policy: float = 5e-4  # ← Au lieu de 3e-4
```

### **🔧 Debug/Monitoring (Priorité 3)**

Ajouter dans `evaluate()`:
```python
def evaluate(cfg: Config, policy: PolicyNet, device: torch.device = torch.device('cpu')) -> float:
    env = make_eval_env(cfg)
    returns = []
    for ep_idx in range(cfg.eval_episodes):
        # ... code existant ...
        returns.append(ep_return)

        # DEBUG: Log outliers
        if ep_return < -500:
            print(f"[EVAL WARNING] Episode {ep_idx+1} extreme: {ep_return:.1f}")

    env.close()

    # DEBUG: Log stats
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    min_ret = float(np.min(returns))
    max_ret = float(np.max(returns))

    if update_idx % 100 == 0:  # Toutes les 100 updates
        print(f"[EVAL STATS] mean={mean_ret:.1f} std={std_ret:.1f} min={min_ret:.1f} max={max_ret:.1f}")

    return mean_ret
```

---

## 📊 Résultats Attendus Après Corrections

### **Avant (Actuel)**
```
Updates: 7500
Training time: 7.5 heures
Best eval: 172.0
Final: 121.8
Entropy (final): 0.74
Value loss (final): 7.3
Status: ❌ NOT SOLVED
```

### **Après (Attendu)**
```
Updates: ~5000-8000
Training time: ~5-6 heures
Best eval: > 220
Final: > 200
Entropy (final): 0.3-0.4
Value loss (final): < 2.0
Status: ✅ SOLVED
```

---

## ✅ Checklist de Vérification

Après réentraînement avec corrections:

- [ ] Eval episodes augmenté à 30
- [ ] Pas d'evals < -500 dans les logs
- [ ] Entropy descend < 0.5 vers update 5000
- [ ] Value loss < 3.0 vers update 3000
- [ ] Train et eval concordent (écart < 30)
- [ ] Plateau dépassé (returns > 150)
- [ ] Solved (eval > 200)

---

## 🎯 Conclusion

Les problèmes principaux sont:
1. **Eval instable** due à variance élevée (10 épisodes) et outliers catastrophiques
2. **Entropy annealing trop rapide** empêche convergence
3. **Value loss haute** indique critique qui n'apprend pas bien
4. **Network/LR sous-dimensionnés** pour la complexité du problème

Avec les corrections proposées, vous devriez atteindre 200+ reward en ~5000-8000 updates. 🚀
