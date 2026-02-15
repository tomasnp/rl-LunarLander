from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from dqn import Config, evaluate


@dataclass
class EvalSummary:
    checkpoint: str
    n_episodes: int
    mean: float
    std: float
    min_score: float
    max_score: float
    success_ge_200: int
    success_rate: float


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _format_header(title: str) -> str:
    line = "=" * 80
    return f"{line}\n{title}\n{line}\n"


def run_eval(checkpoint: Path, n_episodes: int, seed: int, no_render: bool) -> EvalSummary:
    cfg = Config(seed=seed)
    # On force le rendu désactivé par défaut (plus robuste en run batch / CI)
    res = evaluate(cfg, checkpoint, n_episodes=n_episodes, render_human=not no_render)

    scores: List[float] = [float(s) for s in res["scores"]]
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=0))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    success_ge_200 = int(np.sum(np.array(scores) >= 200.0))
    success_rate = float(success_ge_200 / n_episodes)

    return EvalSummary(
        checkpoint=str(checkpoint),
        n_episodes=n_episodes,
        mean=mean,
        std=std,
        min_score=min_score,
        max_score=max_score,
        success_ge_200=success_ge_200,
        success_rate=success_rate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation DQN sur 100 épisodes (avec log style REINFORCE)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Chemin vers le fichier .pth (réseau Q) à évaluer",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Nombre d'épisodes de test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Dossier où écrire le fichier .log (relatif à 2.DQN/)",
    )
    parser.add_argument(
        "--log-name",
        type=str,
        default=None,
        help="Nom du fichier .log (sinon auto: dqn_test_YYYYmmdd_HHMMSS.log)",
    )

    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {ckpt}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_name = args.log_name or f"dqn_test_{_timestamp()}.log"
    if not log_name.endswith(".log"):
        log_name += ".log"
    log_path = log_dir / log_name

    summary = run_eval(ckpt, n_episodes=args.episodes, seed=args.seed, no_render=args.no_render)

    # Affichage console + sauvegarde log
    content = []
    content.append(_format_header(f"Logging : {_timestamp()}\nFichier : {log_path}"))
    content.append(_format_header("CONFIGURATION"))
    content.append(f"  env_id                    = {Config().env_id}\n")
    content.append(f"  seed                      = {args.seed}\n")
    content.append(f"  checkpoint                = {ckpt}\n")
    content.append(f"  eval_episodes             = {args.episodes}\n")
    content.append(f"  render_human              = {not args.no_render}\n")

    content.append("\n" + "=" * 60 + "\n")
    content.append(f"Test de la politique sur {args.episodes} épisodes\n")
    content.append("=" * 60 + "\n\n")

    content.append(f"Politique chargée : {ckpt}\n\n")

    # NOTE: dqn.evaluate imprime déjà le score épisode par épisode.
    # On garde ces prints à l'écran, et on log seulement le résumé final ici.

    content.append("\n" + "=" * 60 + "\n")
    content.append("Résultats :\n")
    content.append(f"  Moyenne :  {summary.mean:7.2f} +/- {summary.std:5.2f}\n")
    content.append(f"  Min :     {summary.min_score:8.2f}\n")
    content.append(f"  Max :     {summary.max_score:8.2f}\n")
    content.append(
        f"  Succès :  {summary.success_ge_200}/{summary.n_episodes} (>= 200)\n"
    )
    content.append("=" * 60 + "\n\n")
    content.append(f"Log sauvegardé : {log_path}\n")

    final_text = "".join(content)
    print("\n" + final_text)
    log_path.write_text(final_text, encoding="utf-8")


if __name__ == "__main__":
    main()
