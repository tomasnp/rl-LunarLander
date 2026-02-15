# 2.DQN — Deep Q-Network (LunarLander-v3)

Ce dossier contient l'implémentation **DQN** et les artefacts associés (checkpoints, logs, courbes) pour l'environnement `LunarLander-v3`.

## Structure

- `src/dqn.py` : entraînement + évaluation + sauvegarde checkpoint
- `checkpoints/` : modèles sauvegardés (`.pth`)
- `logs/` : métriques exportées (`.csv`/`.json`)
- `png/` : courbes exportées (`.png`)

## Multi-trainings (sweep)

Le script `src/dqn_sweep.py` lance plusieurs entraînements avec des paramétrages pertinents (exploration, taille du replay buffer, fréquence du target network, learning rate).
Chaque run sauvegarde :

- un checkpoint dans `checkpoints/`
- un CSV + JSON dans `logs/`
- une figure synthèse dans `png/` avec :
	- score par épisode + moyenne mobile (100)
	- exploration (epsilon)
	- taux de succès (score $\ge 200$, fenêtre 100)
	- distribution des scores

## Lancer

Depuis la racine du projet :

- Entraînement : exécuter `src/dqn.py` (voir options CLI dans le fichier)
- Évaluation : utiliser l'option d'évaluation en pointant vers un checkpoint dans `checkpoints/`

> Remarque : cette version est une **copie** de `rl-LunarLander/src/dqn.py` pour un rendu final organisé par méthode.
