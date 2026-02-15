"""Hyperparameter sweep runner for DQN.

Goal: run multiple short/medium trainings with different (reasonable) settings,
store each run in its own folder, and export a summary plot.

Outputs (per run_id):
- logs/<run_id>/metrics.csv + config.json
- checkpoints/<run_id>.pth
- png/<run_id>.png  (learning curves + epsilon + success rate + score distribution)

This is deliberately simple and folder-based, similar to the REINFORCE/A2C folders.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from dqn import Config, train


@dataclass
class SweepConfig:
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    png_dir: str = "png"

    # Evaluation definition for success rate from training rewards
    success_threshold: float = 200.0
    success_window: int = 100


def _ensure_clean_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def plot_run(metrics_csv: Path, out_png: Path, title: str, *, success_threshold: float = 200.0, success_window: int = 100):
    import csv

    import matplotlib.pyplot as plt

    episodes: list[int] = []
    scores: list[float] = []
    avg100: list[float] = []
    eps: list[float] = []

    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            scores.append(float(row["score"]))
            avg100.append(float(row["avg_score_100"]))
            eps.append(float(row["epsilon"]))

    scores_np = np.array(scores, dtype=float)

    # Success rate (moving) — keep same length as episodes, even for short runs
    success = (scores_np >= success_threshold).astype(float)
    win = max(1, min(int(success_window), len(success)))
    if len(success) == 0:
        success_rate = np.array([], dtype=float)
    elif win == 1:
        success_rate = success
    else:
        kernel = np.ones(win, dtype=float) / float(win)
        success_rate = np.convolve(success, kernel, mode="same")

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes, scores, alpha=0.35, label="score")
    ax1.plot(episodes, avg100, linewidth=2.0, label="moyenne mobile (100)")
    ax1.axhline(success_threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label=f"seuil succès = {success_threshold:.0f}")
    ax1.set_title("Récompense cumulée")
    ax1.set_xlabel("Épisode")
    ax1.set_ylabel("Score")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episodes, eps, color="#d95f02")
    ax2.set_title("Exploration (epsilon)")
    ax2.set_xlabel("Épisode")
    ax2.set_ylabel("epsilon")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(episodes, success_rate * 100.0, color="#1b9e77")
    ax3.set_title(f"Taux de succès (fenêtre {win})")
    ax3.set_xlabel("Épisode")
    ax3.set_ylabel("Succès (%)")
    ax3.set_ylim(0, 100)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(scores_np, bins=30, color="#7570b3", alpha=0.85)
    ax4.set_title("Distribution des scores")
    ax4.set_xlabel("Score")
    ax4.set_ylabel("Count")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _run_id(cfg: Config, suffix: str) -> str:
    # Compact run-id with key hyperparams
    return (
        f"dqn_ep{cfg.n_episodes}_lr{cfg.learning_rate:g}_epsd{cfg.epsilon_decay:g}_"
        f"buf{cfg.memory_size}_tgt{cfg.target_update_freq}_hid{cfg.hidden_size}_{suffix}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run multiple DQN trainings with different settings and save plots.")
    parser.add_argument("--episodes", type=int, default=1500, help="Episodes per run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", type=str, default="v1", help="Suffix for run ids")

    args = parser.parse_args()

    sweep_cfg = SweepConfig()

    root = Path(__file__).resolve().parents[1]  # 2.DQN/
    ckpt_dir = root / sweep_cfg.checkpoints_dir
    logs_dir = root / sweep_cfg.logs_dir
    png_dir = root / sweep_cfg.png_dir

    _ensure_clean_dir(ckpt_dir)
    _ensure_clean_dir(logs_dir)
    _ensure_clean_dir(png_dir)

    base = Config()
    base.n_episodes = int(args.episodes)
    base.seed = int(args.seed)

    variants: list[Config] = [
        # baseline
        base,
        # slower epsilon decay
        replace(base, epsilon_decay=0.997),
        # larger replay buffer
        replace(base, memory_size=50_000),
        # more frequent target updates
        replace(base, target_update_freq=5),
        # lower learning rate
        replace(base, learning_rate=5e-4),
    ]

    results = []

    for i, cfg in enumerate(variants, start=1):
        rid = _run_id(cfg, f"{args.suffix}_{i}")

        # set output names for this run
        cfg.save_dir = str(ckpt_dir)
        cfg.save_name = f"{rid}.pth"
        cfg.log_dir = str(logs_dir)

        # train writes logs with timestamp; we copy/rename into run_id for consistency
        out = train(cfg)

        src_csv = Path(out["log_csv"])
        dst_csv = logs_dir / f"{rid}.csv"
        _safe_copy(src_csv, dst_csv)

        with open(logs_dir / f"{rid}.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

        out_png = png_dir / f"{rid}.png"
        plot_run(
            dst_csv,
            out_png,
            title=f"DQN sweep — {rid}",
            success_threshold=sweep_cfg.success_threshold,
            success_window=sweep_cfg.success_window,
        )

        results.append({"run_id": rid, "checkpoint": str(ckpt_dir / cfg.save_name), "csv": str(dst_csv), "png": str(out_png)})

    # print summary
    print("\n=" * 40)
    print("✅ Sweep terminé. Artefacts:")
    for r in results:
        print(f"- {r['run_id']}\n  ckpt: {r['checkpoint']}\n  csv : {r['csv']}\n  png : {r['png']}")


if __name__ == "__main__":
    main()
