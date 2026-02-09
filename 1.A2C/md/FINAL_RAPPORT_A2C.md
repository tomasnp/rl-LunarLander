# Rapport Final - A2C (Advantage Actor-Critic) avec GAE sur LunarLander-v3

---

## Table des matières

1. [Vue d'ensemble de la méthode](#1-vue-densemble-de-la-méthode)
2. [Architecture des réseaux (Weights)](#2-architecture-des-réseaux-weights)
3. [Collecte des données (Rollout)](#3-collecte-des-données-rollout)
4. [Calcul des cibles (Ground Truth / Returns)](#4-calcul-des-cibles-ground-truth--returns)
5. [Fonctions de perte (Losses)](#5-fonctions-de-perte-losses)
6. [Techniques d'optimisation](#6-techniques-doptimisation)
7. [Techniques de stabilisation](#7-techniques-de-stabilisation)
8. [Variantes testées (256 / 384)](#8-variantes-testées-256--384)
9. [Pipeline complet pas-à-pas](#9-pipeline-complet-pas-à-pas)
10. [Evaluation et arrêt anticipé](#10-evaluation-et-arrêt-anticipé)
11. [Récapitulatif des hyperparamètres](#11-récapitulatif-des-hyperparamètres)
12. [Forces et limites de l'implémentation](#12-forces-et-limites-de-limplémentation)

---

## 1. Vue d'ensemble de la méthode

### REINFORCE vs A2C : positionnement

Cette implémentation n'est **pas** du REINFORCE pur (Monte-Carlo policy gradient). C'est un **A2C (Advantage Actor-Critic)**, une évolution directe de REINFORCE qui corrige son principal défaut : la **haute variance** des estimations de gradient.

| Aspect | REINFORCE pur | A2C (cette implémentation) |
|---|---|---|
| Estimation du retour | Monte-Carlo (retour complet G_t) | **GAE** (Generalized Advantage Estimation) |
| Baseline | Aucune ou moyenne simple | **Critique V(s)** appris par un réseau dédié |
| Signal de gradient | G_t * log pi(a\|s) | **A(s,a) * log pi(a\|s)** avec A = avantage |
| Mise à jour | Fin d'épisode uniquement | **Tous les K pas** (rollout de taille fixe) |
| Variance | Très haute | **Réduite** grâce à la baseline + GAE |
| Biais | Aucun | **Contrôlé** via lambda du GAE |

### Principe fondamental

L'idée centrale est le **théorème du gradient de la politique** :

```
nabla J(theta) = E[ A(s,a) * nabla log pi_theta(a|s) ]
```

Où :
- `pi_theta(a|s)` est la politique paramétrisée par theta (réseau acteur)
- `A(s,a)` est la **fonction d'avantage** : "à quel point l'action `a` est meilleure que la moyenne dans l'état `s`"
- Le gradient pousse la politique à **augmenter la probabilité** des actions ayant un avantage positif et **diminuer** celles ayant un avantage négatif

---

## 2. Architecture des réseaux (Weights)

L'implémentation utilise **deux réseaux séparés** (pas de partage de poids) :

### 2.1 Réseau Acteur (PolicyNet)

```
Input(8) -> Linear(8, H) -> Tanh -> Linear(H, H) -> Tanh -> Linear(H, 4) -> logits
```

- **Entrée** : 8 dimensions (état LunarLander : position x/y, vitesse x/y, angle, vitesse angulaire, contact jambe gauche/droite)
- **Couches cachées** : 2 couches de taille `H` avec activation **Tanh**
- **Sortie** : 4 logits (scores bruts) pour les 4 actions discrètes (rien, moteur gauche, moteur principal, moteur droit)
- Les logits sont transformés en **distribution catégorielle** via softmax implicite dans `Categorical(logits=...)`

### 2.2 Réseau Critique (ValueNet)

```
Input(8) -> Linear(8, H) -> Tanh -> Linear(H, H) -> Tanh -> Linear(H, 1) -> V(s)
```

- **Même architecture** que l'acteur mais avec une **sortie scalaire** : V(s)
- Rôle : estimer la **valeur d'un état** (retour espéré à partir de cet état)
- Le squeeze(-1) final transforme la sortie de forme [batch, 1] en [batch]

### 2.3 Choix de Tanh comme activation

- Tanh borne les activations dans [-1, 1], ce qui **limite les gradients explosifs**
- Souvent préféré à ReLU en RL pour la stabilité (pas de neurones morts, sorties centrées)

### 2.4 Tailles testées

| Variante | H (hidden_size) | Params acteur (approx.) | Params critique (approx.) |
|---|---|---|---|
| A2C_256 | 256 | ~68K | ~67K |
| A2C_384 | 384 | ~151K | ~150K |

---

## 3. Collecte des données (Rollout)

### 3.1 Principe du rollout à taille fixe

La fonction `collect_rollout()` collecte exactement **K = 2048 pas** d'interaction avec l'environnement, indépendamment des frontières d'épisodes. C'est une différence majeure avec REINFORCE qui attend la fin d'un épisode.

### 3.2 Fonctionnement détaillé

```
Pour chaque pas t dans [0, K-1] :
    1. Normaliser l'observation (si activé)
    2. Passer l'observation dans PolicyNet -> logits
    3. Échantillonner une action ~ Categorical(logits)
    4. Passer l'observation dans ValueNet -> V(s_t)
    5. Exécuter l'action dans l'environnement -> (obs', reward, terminated, truncated)
    6. Stocker : (s_t, a_t, r_t, terminated_t, V(s_t))
    7. Si l'épisode est terminé -> reset et continuer
```

### 3.3 Gestion des frontières d'épisodes

Point subtil : le rollout **traverse les frontières d'épisodes**. Si un épisode se termine au pas 500 du rollout, l'environnement est reset et la collecte continue. Cela signifie qu'un rollout de 2048 pas peut contenir **plusieurs épisodes partiels ou complets**.

Les retours des épisodes complets sont collectés dans `episode_returns` pour le suivi des performances.

### 3.4 Distinction terminated vs truncated

```python
terminated = True  # L'agent a crashé ou atterri -> état véritablement terminal
truncated = True   # Limite de temps atteinte -> l'épisode est coupé mais pas "fini"
```

Seul le flag `terminated` est stocké dans les données du rollout. Cette distinction est **cruciale pour le bootstrapping** dans le calcul du GAE : on ne bootstrap **pas** après un état terminal (V=0), mais on **bootstrap** après une troncature (V=V(s')).

### 3.5 Données collectées

| Donnée | Forme | Description |
|---|---|---|
| states | (2048, 8) | Observations (normalisées si activé) |
| actions | (2048,) | Actions entières [0-3] |
| rewards | (2048,) | Récompenses (clippées si activé) |
| terminateds | (2048,) | 1.0 si terminal, 0.0 sinon |
| values | (2048,) | V(s_t) estimé par le critique |

---

## 4. Calcul des cibles (Ground Truth / Returns)

### 4.1 GAE - Generalized Advantage Estimation

C'est le coeur de l'algorithme. Le GAE calcule l'avantage A(s_t, a_t) en faisant un **compromis biais/variance** contrôlé par lambda.

### 4.2 Formule mathématique

L'erreur TD (Temporal Difference) à chaque pas :

```
delta_t = r_t + gamma * (1 - terminated_t) * V(s_{t+1}) - V(s_t)
```

L'avantage GAE est une **somme exponentiellement pondérée** des erreurs TD futures :

```
A_t^GAE = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}
```

Calculé efficacement en **parcours inverse** :

```python
for t in reversed(range(T)):
    delta = r_t + gamma * (1 - terminated_t) * V(s_{t+1}) - V(s_t)
    gae = delta + gamma * lambda * (1 - terminated_t) * gae
    advantages[t] = gae
```

### 4.3 Rôle de lambda

| Lambda | Comportement | Biais | Variance |
|---|---|---|---|
| 0.0 | TD(0) pur : A = r + gamma*V(s') - V(s) | **Haut** (dépend de V) | **Basse** |
| 1.0 | Monte-Carlo pur : A = G_t - V(s) | **Bas** | **Haute** |
| **0.95** (utilisé) | Compromis optimal | Modéré | Modérée |

### 4.4 Returns (cibles du critique)

Les cibles pour entraîner le critique sont :

```
returns_t = advantages_t + V(s_t)
```

Ce sont les **Ground Truth** du critique : la valeur "réelle" estimée par GAE que V(s) doit apprendre à prédire.

### 4.5 Bootstrapping de la valeur finale

À la fin du rollout, on a besoin de V(s_{T+1}) pour calculer le dernier delta :

```python
if dernière_étape_est_terminale:
    next_value = 0.0          # Pas de futur après un état terminal
else:
    next_value = V(s_{T+1})   # Bootstrap avec le critique
```

---

## 5. Fonctions de perte (Losses)

La loss totale combine **trois termes** :

```
L_total = L_policy + value_coef * L_value
```

(Le terme d'entropie est intégré dans L_policy)

### 5.1 Policy Loss (Acteur)

```python
policy_loss = -(log_probs * advantages.detach()).mean() - entropy_coef * entropy
```

**Décomposition :**

- `log_probs` : log pi_theta(a_t | s_t) -- la log-probabilité de l'action prise
- `advantages.detach()` : avantages GAE **détachés du graphe** (pas de gradient à travers le critique)
- Le signe **négatif** transforme la maximisation en minimisation (convention PyTorch)
- `entropy_coef * entropy` : **bonus d'entropie** pour encourager l'exploration (soustrait = encourage une entropie haute)

**Intuition :** Si A > 0 (bonne action), on **augmente** log pi(a|s). Si A < 0 (mauvaise action), on la **diminue**.

### 5.2 Value Loss (Critique)

```python
value_loss = SmoothL1Loss(V(s_t), returns_t.detach())
```

- **SmoothL1Loss** (Huber Loss) au lieu de MSE : combine le meilleur de L1 et L2
  - Se comporte comme **L2** pour les petites erreurs (gradient lisse)
  - Se comporte comme **L1** pour les grandes erreurs (robuste aux outliers)
- `returns_t.detach()` : les cibles sont **fixes** (pas de gradient à travers le calcul des returns)
- Pondéré par `value_coef = 0.5` dans la loss totale

### 5.3 Bonus d'entropie

```python
entropy = Categorical(logits=logits).entropy().mean()
```

- Mesure le **désordre** de la distribution de politique
- Entropie haute = politique exploratoire (probabilités uniformes)
- Entropie basse = politique déterministe (une action dominante)
- Ajouté à la loss de politique comme terme de **régularisation**

---

## 6. Techniques d'optimisation

### 6.1 Optimiseur : AdamW

```python
opt_policy = AdamW(policy.parameters(), lr=5e-4, eps=1e-5, weight_decay=0)
opt_value  = AdamW(value.parameters(), lr=1e-3, eps=1e-5, weight_decay=0)
```

- **AdamW** : Adam avec découplage du weight decay (meilleure régularisation que Adam classique)
- **eps=1e-5** : stabilité numérique dans la division par la racine du second moment
- **Learning rates séparés** : le critique apprend **2x plus vite** (1e-3 vs 5e-4) car il doit fournir des estimations fiables rapidement pour guider l'acteur
- **weight_decay=0** par défaut (désactivé) mais configurable

### 6.2 Gradient Clipping

```python
nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
nn.utils.clip_grad_norm_(value.parameters(), max_norm=0.5)
```

- Limite la **norme L2** du vecteur gradient à 0.5
- Empêche les mises à jour catastrophiques quand un batch contient des avantages extrêmes
- Appliqué **séparément** à l'acteur et au critique

### 6.3 Normalisation des avantages

```python
advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
```

- Centre les avantages autour de 0 et les met à l'échelle (variance unitaire)
- `unbiased=False` : diviseur N au lieu de N-1 (plus stable pour la normalisation)
- Réduit la variance des mises à jour de la politique
- Empêche que des rewards très élevées/basses dominent le gradient

### 6.4 Décroissance linéaire du coefficient d'entropie

```python
progress = update_idx / max_updates
entropy_coef = max(entropy_coef_final, entropy_coef_start * (1.0 - progress))
```

- **Début** : `entropy_coef = 0.05` (forte exploration)
- **Fin** : `entropy_coef = 0.005` (exploitation dominante)
- **Décroissance linéaire** avec un plancher (ne descend jamais sous 0.005)
- Stratégie classique : explorer beaucoup au début, exploiter ensuite

---

## 7. Techniques de stabilisation

### 7.1 Normalisation des observations (Welford en ligne)

Classe `RunningMeanStd` -- **désactivée par défaut** (`normalize_obs=False`) :

```
obs_normalisée = clip( (obs - mean_courante) / sqrt(var_courante + 1e-8), -10, +10 )
```

- **Algorithme de Welford** : calcul incrémental de la moyenne et variance, numériquement stable
- Mise à jour après chaque rollout avec les observations brutes collectées
- Clipping dans [-10, +10] pour éviter les valeurs extrêmes

### 7.2 Clipping des récompenses

- **Désactivé par défaut** (`reward_clip=None`)
- Si activé : `reward = clip(reward, -reward_clip, +reward_clip)`
- Utile pour les environnements avec des récompenses très dispersées

### 7.3 Séparation acteur/critique

- Deux réseaux **totalement indépendants** avec leurs propres optimiseurs
- Avantage : pas d'interférence entre les gradients de la politique et de la valeur
- Inconvénient : pas de partage de représentation (mais plus stable)

### 7.4 Détachement des gradients

- `advantages.detach()` dans la policy loss : le gradient ne remonte **pas** dans le critique via les avantages
- `returns.detach()` dans la value loss : les cibles sont traitées comme des constantes
- Évite les boucles de gradient instables entre acteur et critique

---

## 8. Variantes testées (256 / 384)

Les trois fichiers partagent **exactement le même code** via import de `A2C.py`. La seule différence est la taille du réseau :

### Structure du code

- **A2C.py** (256) : Fichier principal contenant toute l'implémentation. `hidden_size=256` par défaut.
- **A2C_256.py** : Script lanceur qui importe depuis `A2C.py` et fixe `hidden_size=256`.
- **A2C_384.py** : Script lanceur qui importe depuis `A2C.py` et fixe `hidden_size=384`.

### Comparaison

| Paramètre | A2C_256 | A2C_384 |
|---|---|---|---|
| hidden_size | 256 | 384 |
| lr_policy | 5e-4 | 5e-4 |
| lr_value | 1e-3 | 1e-3 |
| rollout_steps | 2048 | 2048 |
| max_updates | 10000 | 10000 |
| entropy_coef_start | 0.05 | 0.05 |
| entropy_coef_final | 0.005 | 0.005 |
| grad_clip | 0.5 | 0.5 |
| eval_episodes | 30 | 30 |
| Résultat noté | **74.8% success** | **80% success** |


---

## 9. Pipeline complet pas-à-pas

Voici le déroulement complet d'une itération d'entraînement :

```
INITIALISATION
    |
    v
[1] Seed (numpy + torch) pour reproductibilité
    |
    v
[2] Créer env LunarLander-v3 (sans rendu)
    |
    v
[3] Créer PolicyNet(8 -> H -> H -> 4) et ValueNet(8 -> H -> H -> 1)
    |
    v
[4] Créer optimiseurs AdamW séparés (lr_actor=5e-4, lr_critic=1e-3)
    |
    v
=== BOUCLE D'ENTRAÎNEMENT (max 10000 updates) ===
    |
    v
[5] COLLECTE : rollout de 2048 pas
    |   - Pour chaque pas :
    |     - obs -> PolicyNet -> logits -> Categorical -> action échantillonnée
    |     - obs -> ValueNet -> V(s)
    |     - env.step(action) -> (obs', r, terminated, truncated)
    |     - Stocker (s, a, r, terminated, V(s))
    |     - Si épisode fini : reset env, enregistrer le retour
    |
    v
[6] BOOTSTRAP : calculer V(s_{2049}) pour le dernier état
    |   - Si terminal : next_value = 0
    |   - Sinon : next_value = ValueNet(s_{2049})
    |
    v
[7] GAE : calculer avantages et returns (parcours inverse)
    |   - delta_t = r_t + gamma * (1-term) * V(s_{t+1}) - V(s_t)
    |   - A_t = delta_t + gamma * lambda * (1-term) * A_{t+1}
    |   - returns_t = A_t + V(s_t)
    |
    v
[8] NORMALISATION des avantages : (A - mean) / (std + eps)
    |
    v
[9] FORWARD PASS :
    |   - PolicyNet(states) -> logits -> Categorical -> log_prob(actions), entropy
    |   - ValueNet(states) -> V_pred
    |
    v
[10] CALCUL DES LOSSES :
    |   - L_policy = -mean(log_prob * A.detach()) - entropy_coef * entropy
    |   - L_value  = SmoothL1(V_pred, returns.detach())
    |   - L_total  = L_policy + 0.5 * L_value
    |
    v
[11] BACKPROP + OPTIMISATION :
    |   - zero_grad sur les deux optimiseurs
    |   - loss.backward()
    |   - clip_grad_norm (max=0.5) sur acteur et critique
    |   - step() sur les deux optimiseurs
    |
    v
[12] LOGGING (tous les 10 updates) + EVAL (tous les 50 updates)
    |   - Eval : 30 épisodes en mode argmax (déterministe)
    |   - Sauvegarde du meilleur modèle si nouveau record
    |
    v
[13] ARRET ANTICIPÉ si moyenne glissante >= 200 sur 100 épisodes
    |
    v
=== FIN ===
```

---

## 10. Evaluation et arrêt anticipé

### 10.1 Evaluation (tous les 50 updates)

```python
action = argmax( Categorical(logits=policy(obs)).probs )
```

- Mode **déterministe** : on prend l'action la plus probable (argmax), pas d'échantillonnage
- Sur **30 épisodes** pour avoir une estimation fiable
- Le réseau est mis en `eval()` mode puis remis en `train()`

### 10.2 Arrêt anticipé

```python
if mean(derniers_100_retours) >= 200.0:
    STOP -> "Résolu !"
```

- Fenêtre glissante de 100 épisodes d'entraînement (pas d'évaluation)
- Seuil de 200 (score standard pour considérer LunarLander comme résolu)

### 10.3 Sauvegarde du meilleur modèle

Le checkpoint contient :
- `policy_state_dict` : poids de l'acteur
- `value_state_dict` : poids du critique
- `cfg` : configuration complète
- `best_eval` : meilleur score d'évaluation
- `update` : numéro de l'update
- `obs_normalizer` : statistiques du normalisateur (si activé)

---

## 11. Récapitulatif des hyperparamètres

| Hyperparamètre | Valeur | Rôle |
|---|---|---|
| `gamma` | 0.99 | Facteur d'actualisation (vision long terme) |
| `gae_lambda` | 0.95 | Compromis biais/variance du GAE |
| `lr_policy` | 5e-4 | Learning rate de l'acteur |
| `lr_value` | 1e-3 | Learning rate du critique (2x plus rapide) |
| `entropy_coef_start` | 0.05 | Exploration initiale |
| `entropy_coef_final` | 0.005 | Exploration minimale |
| `value_coef` | 0.5 | Poids du critique dans la loss totale |
| `rollout_steps` | 2048 | Taille du buffer de collecte |
| `max_updates` | 10000 | Budget max d'itérations |
| `hidden_size` | 256/384 | Capacité des réseaux |
| `grad_clip` | 0.5 | Norme max des gradients |
| `seed` | 42 | Reproductibilité |

---

## 12. Forces et limites de l'implémentation

### Forces

1. **GAE bien implémenté** : le compromis biais/variance via lambda=0.95 est optimal pour LunarLander
2. **Réseaux séparés** : pas d'interférence acteur/critique, plus stable
3. **SmoothL1Loss** : robuste aux outliers dans les retours, meilleur que MSE
4. **Gradient clipping** : protection contre les updates catastrophiques
5. **Normalisation des avantages** : réduit la sensibilité aux échelles de reward
6. **Décroissance de l'entropie** : transition exploration -> exploitation bien gérée
7. **Code modulaire** : séparation claire collecte/GAE/update/eval
8. **Gestion terminated vs truncated** : bootstrapping correct aux frontières d'épisodes
9. **Logging complet** : TeeLogger, visualisations, checkpointsls


### Limites

1. **Pas de mini-batches** : tout le rollout est utilisé en un seul batch (contrairement à PPO)
2. **Un seul environnement** : pas de parallélisation (A2C bénéficie normalement de N envs en parallèle)
3. **Pas de learning rate scheduling** : les LR sont constants tout au long de l'entraînement
4. **Normalisation obs désactivée** : `normalize_obs=False` par défaut, pourrait aider
5. **Pas de clipping du ratio** de politique (ce serait PPO)
6. **Single update par rollout** : les données sont utilisées une seule fois puis jetées (sample-inefficient)
7. **Activation Tanh** partout : pourrait bénéficier de ReLU/GELU dans certains cas

---

*Rapport généré à partir de l'analyse de `A2C.py`, `A2C_256.py` et `A2C_384.py`*
