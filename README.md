# rl-LunarLander — Rendu final

Ce dépôt regroupe trois approches d'apprentissage par renforcement sur `LunarLander-v3`.

## Organisation (rendu)

- `0.REINFORCE/` : méthode REINFORCE (+ baseline)
- `1.A2C/` : méthode A2C (+ GAE)
- `2.DQN/` : méthode DQN

Chaque dossier contient :
- `src/` : code (train/eval)
- `checkpoints/` : modèles sauvegardés
- `logs/` : logs et métriques
- `png/` : figures/courbes

## Rapport

Le rapport LaTeX est dans `report_latex.tex`.
Les images que le rapport inclut sont dans `figures/`.

## Notes

- Les dossiers racines historiques (`src/`, `checkpoints/`, `logs/`, `png/`) ont été réorganisés vers les 3 dossiers de méthodes.
