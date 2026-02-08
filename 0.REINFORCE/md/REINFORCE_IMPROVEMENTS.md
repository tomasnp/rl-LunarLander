# REINFORCE - Améliorations Appliquées

## 📋 Résumé

Ce document récapitule les améliorations apportées au code REINFORCE pour résoudre les problèmes identifiés dans l'analyse (0% de taux de succès, value loss ~290, entropy stagnante à 1.10).

**Toutes les améliorations gardent la logique REINFORCE originale** : Monte Carlo returns + policy gradient + baseline.

---

## 🔍 Problèmes Identifiés (Analyse Précédente)

1. **High Variance** : Monte Carlo returns avec épisodes longs → gradients très bruyants
2. **Learning Rates Trop Élevées** : 3e-4 (policy) et 1e-3 (value) inadaptés aux gradients bruités
3. **Value Loss Instable** : ~290 vs ~3 pour A2C → baseline inefficace
4. **Entropy Stagnante** : Reste à 1.10 → politique jamais convergée (actions aléatoires)
5. **Pas de Batching** : Update après chaque épisode → amplification du bruit

---

## ✅ Améliorations Appliquées

### 1. **Réduction des Learning Rates** ⭐⭐⭐
**Amélioration critique pour stabilité avec haute variance**

```python
# AVANT
lr_policy: float = 3e-4
lr_value: float = 1e-3

# APRÈS
lr_policy: float = 1e-4         # ↓ 3x plus lent
lr_value: float = 5e-4          # ↓ 2x plus lent
```

**Pourquoi ?**
- Les gradients REINFORCE ont une variance élevée (∝ longueur épisode)
- Des LR élevés amplifient le bruit et empêchent la convergence
- LR plus faibles permettent une progression stable malgré les gradients bruités

---

### 2. **Augmentation de la Capacité du Réseau** ⭐⭐
**Meilleure approximation de la fonction de valeur**

```python
# AVANT
hidden_size: int = 128

# APRÈS
hidden_size: int = 256          # ↑ 2x plus de capacité
```

**Pourquoi ?**
- Plus de paramètres = meilleure approximation des fonctions complexes
- Permet au value network de mieux estimer le baseline
- Réduit l'erreur de la baseline → variance réduite

---

### 3. **Architecture Améliorée : LayerNorm + ReLU** ⭐⭐⭐
**Stabilisation des activations et meilleur flow de gradients**

```python
# AVANT (PolicyNet et ValueNet)
nn.Sequential(
    nn.Linear(obs_dim, hidden),
    nn.Tanh(),
    nn.Linear(hidden, hidden),
    nn.Tanh(),
    nn.Linear(hidden, output_dim),
)

# APRÈS (PolicyNet et ValueNet)
nn.Sequential(
    nn.Linear(obs_dim, hidden),
    nn.LayerNorm(hidden),        # Normalise activations
    nn.ReLU(),                   # Meilleur que Tanh pour gradients
    nn.Linear(hidden, hidden),
    nn.LayerNorm(hidden),
    nn.ReLU(),
    nn.Linear(hidden, output_dim),
)
```

**Pourquoi ?**
- **LayerNorm** : Stabilise les activations → réduit la variance interne
- **ReLU** : Pas de saturation des gradients (vs Tanh qui sature à ±1)
- Meilleure propagation des gradients → apprentissage plus stable
- **Impact attendu** : Value loss devrait descendre de ~290 vers ~10-30

---

### 4. **Batching d'Épisodes** ⭐⭐⭐
**Réduction de la variance via moyennage**

```python
# NOUVEAU
batch_episodes: int = 4         # Accumule 4 épisodes avant update
```

**Algorithme** :
```python
# Accumule losses sur batch_episodes épisodes
for ep in range(1, max_episodes + 1):
    run_episode()
    compute_losses()

    # Update seulement tous les 4 épisodes
    if ep % batch_episodes == 0:
        avg_loss = mean(accumulated_losses)  # Moyenne des losses
        avg_loss.backward()
        optimizer.step()
```

**Pourquoi ?**
- Moyenne de N épisodes → variance réduite d'un facteur √N
- Gradients plus stables sans changer la logique REINFORCE
- Plus efficace que update à chaque épisode

---

### 5. **Gradient Clipping Renforcé** ⭐⭐
**Prévention des explosions de gradients**

```python
# AVANT
nn.utils.clip_grad_norm_(..., max_norm=1.0)

# APRÈS
grad_clip: float = 0.5
nn.utils.clip_grad_norm_(..., max_norm=0.5)  # ↓ 2x plus strict
```

**Pourquoi ?**
- Haute variance des returns → risque de gradients extrêmes
- Clipping plus strict empêche les mises à jour destructrices
- Préserve la stabilité de l'entraînement

---

### 6. **Entropy Coefficient Decay** ⭐⭐
**Exploration → Exploitation progressive**

```python
# NOUVEAU
entropy_coef: float = 0.01          # Départ (↓ de 0.05)
entropy_coef_decay: float = 0.995   # Multiplicateur par épisode
entropy_coef_min: float = 0.001     # Limite basse

# Dans train()
current_entropy_coef = max(entropy_coef_min,
                          current_entropy_coef * entropy_coef_decay)
```

**Évolution** :
- Épisode 1 : ent_coef = 0.010
- Épisode 500 : ent_coef ≈ 0.0062
- Épisode 1000 : ent_coef ≈ 0.0038
- Épisode 2000 : ent_coef ≈ 0.0014
- Épisode 3000+ : ent_coef = 0.001 (min atteint)

**Pourquoi ?**
- Début : Exploration élevée pour découvrir stratégies
- Milieu : Réduction progressive → favorise convergence
- Fin : Exploitation pure de la meilleure politique
- **Résout le problème** : Entropy stagnante à 1.10 → devrait décroître vers ~0.5

---

### 7. **Logging Amélioré** ⭐
**Meilleure visibilité sur l'entraînement**

```python
# AVANT
print(f"loss={loss.item():.3f} | policy={policy_loss.item():.3f} | value={value_loss.item():.3f}")

# APRÈS
print(f"loss={loss:.3f} | policy={policy_loss:.3f} | value={value_loss:.3f} | "
      f"ent_coef={current_entropy_coef:.4f} | ent={avg_entropy:.3f}")
```

**Nouvelles métriques affichées** :
- `ent_coef` : Coefficient d'entropie actuel (pour suivre le decay)
- `ent` : Entropie moyenne de l'épisode (doit décroître)

---

## 📊 Résultats Attendus

### Comparaison Avant/Après

| Métrique | AVANT (Original) | APRÈS (Amélioré) | Explication |
|----------|------------------|-------------------|-------------|
| **Taux de Succès Final** | 0.0% | 30-60% | Politique converge grâce aux améliorations |
| **Best Eval Return** | -55.0 | 50-150 | Meilleure approximation, gradients stables |
| **Value Loss** | ~290 | ~10-30 | LayerNorm + architecture améliorée |
| **Entropy Finale** | 1.10 | ~0.4-0.6 | Decay force la convergence |
| **Convergence** | Jamais | Vers 2000-3000 ep | LR adaptés + batching |

### Métriques Clés à Surveiller

1. **Entropy** : Doit décroître progressivement de 1.38 → 0.4-0.6
   - Si stagne à >1.0 : Politique n'apprend pas
   - Si descend à <0.2 : Sur-exploitation (trop de certitude)

2. **Value Loss** : Doit descendre de ~290 → ~10-30
   - Si reste élevé : Baseline inefficace (considérer plus d'epochs pour value)

3. **Mean Reward** : Doit progresser de -68 → +50 → +150 → +200
   - Progression lente normale (REINFORCE est plus lent que A2C)

4. **Gradient Norm** : Devrait rester < 0.5 (grâce au clipping)

---

## 🎯 Logique REINFORCE Préservée

**Toutes les améliorations sont des optimisations d'hyperparamètres et d'architecture.**

La logique fondamentale reste **exactement la même** :

```python
# 1. Monte Carlo Returns (inchangé)
returns_t = compute_returns(rewards, gamma)

# 2. Baseline (inchangé)
values_t = value(states_t)
advantages = returns_t - values_t.detach()

# 3. Policy Gradient (inchangé)
policy_loss = -(log_probs * advantages).mean() - entropy_coef * entropy

# 4. Value Update (inchangé)
value_loss = 0.5 * (returns_t - values_t).pow(2).mean()
```

**Ce qui a changé** :
- ❌ PAS de bootstrapping (reste Monte Carlo pur)
- ❌ PAS de GAE (reste avantage simple)
- ❌ PAS de n-step returns
- ✅ OUI aux meilleurs hyperparamètres
- ✅ OUI à une meilleure architecture de réseau
- ✅ OUI au batching (moyenne sans changer la logique)

---

## 🚀 Utilisation

### Entraînement
```bash
python src/reinforce.py
```

### Test Rapide (100 épisodes)
```bash
python test_reinforce.py
```

### Fichiers Générés
- **Checkpoint** : `checkpoints/reinforce_baseline_lunar.pt`
- **Log** : `logs/reinforce_YYYYMMDD_HHMMSS.log`
- **Graphique** : `training_performance_reinforce.png`

---

## 📈 Prédictions de Performance

### Timeline Attendue

**Épisodes 1-500** : Phase d'Exploration
- Mean reward : -100 → -20
- Entropy : 1.38 → 1.1
- Value loss : 290 → 150
- Status : Découverte des actions

**Épisodes 500-1500** : Phase de Transition
- Mean reward : -20 → +50
- Entropy : 1.1 → 0.8
- Value loss : 150 → 50
- Status : Émergence de patterns

**Épisodes 1500-3000** : Phase de Convergence
- Mean reward : +50 → +150
- Entropy : 0.8 → 0.5
- Value loss : 50 → 20
- Status : Politique se stabilise

**Épisodes 3000+** : Phase de Raffinement
- Mean reward : +150 → +200
- Entropy : 0.5 → 0.4
- Value loss : 20 → 10
- Status : Approche de la solution (200+)

---

## 🔬 Expériences Possibles

Si les résultats ne sont toujours pas satisfaisants, ajustements possibles :

### Option 1 : Learning Rates Encore Plus Bas
```python
lr_policy: float = 5e-5   # Au lieu de 1e-4
lr_value: float = 2e-4    # Au lieu de 5e-4
```

### Option 2 : Plus de Batching
```python
batch_episodes: int = 8   # Au lieu de 4
```

### Option 3 : Plus d'Entraînement du Value Network
```python
value_train_epochs: int = 5  # Entraîner value sur 5 passes par épisode
```

### Option 4 : Entropy Decay Plus Lent
```python
entropy_coef_decay: float = 0.998  # Au lieu de 0.995
```

---

## 📚 Comparaison REINFORCE vs A2C

| Aspect | REINFORCE (Amélioré) | A2C |
|--------|---------------------|-----|
| **Returns** | Monte Carlo (attendre fin épisode) | Bootstrapping (n-step) |
| **Variance** | Haute (∝ longueur épisode) | Basse (grâce au bootstrapping) |
| **Sample Efficiency** | Plus faible | Plus élevée |
| **Convergence** | Plus lente (~3000 ep) | Plus rapide (~1000 ep) |
| **Complexité** | Simple | Moyenne |
| **Stabilité** | Nécessite tuning précis | Plus robuste |

**REINFORCE reste pertinent pour** :
- Comprendre les fondamentaux du policy gradient
- Épisodes courts où le bootstrapping n'aide pas
- Recherche académique sur la variance des gradients

**A2C est préférable pour** :
- Atteindre rapidement de bonnes performances
- Environnements avec épisodes longs (comme Lunar Lander)
- Applications pratiques en production

---

## 🎓 Références Théoriques

**REINFORCE (Williams, 1992)** :
```
∇J(θ) = E[∑_t ∇log π(a_t|s_t) * G_t]
```

**REINFORCE with Baseline** :
```
∇J(θ) = E[∑_t ∇log π(a_t|s_t) * (G_t - V(s_t))]
```

**Variance Reduction via Batching** :
```
Var(mean(X_1, ..., X_n)) = Var(X) / n
```

---

## ✅ Checklist de Vérification

Après l'entraînement, vérifier :

- [ ] Entropy décroît progressivement (1.38 → 0.4-0.6)
- [ ] Value loss descend sous 50
- [ ] Mean reward progresse vers valeurs positives
- [ ] Pas d'explosion de gradients (loss devient NaN)
- [ ] Best eval reward s'améliore régulièrement
- [ ] Graphique montre tendance croissante (même avec bruit)
- [ ] Log sauvegardé correctement

---

**Date de création** : 2026-02-08
**Auteur** : Claude Code (Sonnet 4.5)
**Basé sur** : Analyse REINFORCE_SUMMARY.md
