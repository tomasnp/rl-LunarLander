"""Run a single DQN training with *visible* exploration.

Why this exists:
- The sweep runs use `epsilon_decay` per *gradient update* (every replay step), so epsilon can
  collapse to epsilon_min within only a few episodes.
- This helper runs ONE training with a slower, easier-to-see schedule, and saves logs + a PNG.

Outputs:
- logs/<run_id>.csv + logs/<run_id>.json
- checkpoints/<run_id>.pth
- png/<run_id>.png

Usage (from rl-LunarLander/2.DQN):
  ../../.venv/bin/python src/run_one_training.py --episodes 300 --eps-decay 0.99995

Tip:
- If you want epsilon to stay high longer, use a decay close to 1 (e.g. 0.99995) and/or increase
  --eps-min.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from dqn import Config, train
from dqn_sweep import SweepConfig, plot_run


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run ONE DQN training with a controllable epsilon schedule.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=str, default=None, help="Explicit run id (optional)")

    # Exploration knobs
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-min", type=float, default=0.05, help="Keep exploration non-trivial")
    # IMPORTANT: in this repo implementation epsilon decays *per replay step*, so use values VERY close to 1.
    parser.add_argument("--eps-decay", type=float, default=0.99995, help="Multiplicative decay applied frequently")

    # Keep the rest compatible with the existing DQN code
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--buffer", type=int, default=50_000)
    parser.add_argument("--target", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]  # 2.DQN/
    ckpt_dir = root / "checkpoints"
    logs_dir = root / "logs"
    png_dir = root / "png"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    cfg.n_episodes = int(args.episodes)
    cfg.seed = int(args.seed)

    cfg.learning_rate = float(args.lr)
    cfg.memory_size = int(args.buffer)
    cfg.target_update_freq = int(args.target)
    cfg.hidden_size = int(args.hidden)

    cfg.epsilon_start = float(args.eps_start)
    cfg.epsilon_min = float(args.eps_min)
    cfg.epsilon_decay = float(args.eps_decay)

    # Stable, readable run id
    rid = args.run_id
    if rid is None:
        rid = (
            f"dqn_one_ep{cfg.n_episodes}_lr{cfg.learning_rate:g}_"
            f"eps({cfg.epsilon_start:g}->{cfg.epsilon_min:g},d={cfg.epsilon_decay:g})_"
            f"buf{cfg.memory_size}_tgt{cfg.target_update_freq}_hid{cfg.hidden_size}_seed{cfg.seed}"
        )

    cfg.save_dir = str(ckpt_dir)
    cfg.save_name = f"{rid}.pth"
    cfg.log_dir = str(logs_dir)

    out = train(cfg)

    # The trainer writes timestamp-based files; we keep them as-is but also copy into a predictable name.
    src_csv = Path(out["log_csv"])
    dst_csv = logs_dir / f"{rid}.csv"
    if src_csv.resolve() != dst_csv.resolve():
        dst_csv.write_bytes(src_csv.read_bytes())

    with open(logs_dir / f"{rid}.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    sweep_cfg = SweepConfig()
    out_png = png_dir / f"{rid}.png"
    plot_run(
        dst_csv,
        out_png,
        title=f"DQN one-run — {rid}",
        success_threshold=sweep_cfg.success_threshold,
        success_window=sweep_cfg.success_window,
    )

    print("\n✅ ONE training terminé:")
    print(f"- ckpt: {ckpt_dir / cfg.save_name}")
    print(f"- csv : {dst_csv}")
    print(f"- png : {out_png}")


if __name__ == "__main__":
    main()
