# rl-LunarLander — Rendu final

Ce dépôt regroupe trois approches d'apprentissage par renforcement sur `LunarLander-v3`.

## Vidéo

<video controls width="100%">
    <source src="./visual.evaluation.mp4" type="video/mp4" />
    Votre navigateur ne supporte pas la lecture de vidéos via la balise HTML <code>&lt;video&gt;</code>.
    <a href="./visual.evaluation.mp4">Télécharger la vidéo</a>.
</video>

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

Le rapport LaTeX est dans `Report_RL.pdf`.

## Notes

- Les dossiers racines historiques (`src/`, `checkpoints/`, `logs/`, `png/`) ont été réorganisés vers les 3 dossiers de méthodes.
