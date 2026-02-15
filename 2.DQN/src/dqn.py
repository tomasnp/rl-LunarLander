from __future__ import annotations

import csv
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Config (simple)
# -----------------------------
@dataclass
class Config:
    env_id: str = "LunarLander-v3"
    seed: int = 42

    # Training
    n_episodes: int = 1_000
    max_steps: int = 1_000

    # DQN hyperparams
    gamma: float = 0.99
    learning_rate: float = 1e-3

    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995

    batch_size: int = 64
    memory_size: int = 10_000

    target_update_freq: int = 10  # épisodes

    # Network
    hidden_size: int = 128

    # Output
    save_dir: str = "checkpoints"
    save_name: str = "dqn_lunar_lander.pth"

    log_dir: str = "logs"
    log_every: int = 50


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# -----------------------------
# Model
# -----------------------------
class DQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size: int, action_size: int, cfg: Config, device: torch.device):
        self.state_size = state_size
        self.action_size = action_size
        self.cfg = cfg
        self.device = device

        # Hyperparams
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon_start
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay
        self.learning_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size

        # Nets
        self.q_network = DQNetwork(state_size, action_size, cfg.hidden_size).to(device)
        self.target_network = DQNetwork(state_size, action_size, cfg.hidden_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay memory
        from collections import deque

        self.memory = deque(maxlen=cfg.memory_size)

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(q_values.argmax().item())

    def replay(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convertir en numpy array avant torch pour éviter le warning "slow list -> tensor"
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(np.array(dones), dtype=torch.float32, device=self.device)

        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_network(next_states_t).max(1)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Décroissance epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())


# -----------------------------
# Logging
# -----------------------------
def open_metrics_writer(log_path: Path):
    f = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "episode",
            "score",
            "avg_score_100",
            "epsilon",
            "episode_steps",
            "mean_loss",
            "duration_s",
        ],
    )
    writer.writeheader()
    return f, writer


# -----------------------------
# Train
# -----------------------------
def train(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = resolve_device()

    ensure_dir(cfg.save_dir)
    ensure_dir(cfg.log_dir)

    env = gym.make(cfg.env_id)

    run_name = f"dqn_{timestamp()}"
    log_path = Path(cfg.log_dir) / f"{run_name}.csv"
    ckpt_path = Path(cfg.save_dir) / cfg.save_name
    f_csv = None

    try:
        state_size = int(env.observation_space.shape[0])
        action_size = int(env.action_space.n)

        agent = DQNAgent(state_size, action_size, cfg, device)

        f_csv, writer = open_metrics_writer(log_path)

        # Petit fichier config à côté
        with open(Path(cfg.log_dir) / f"{run_name}.json", "w", encoding="utf-8") as f:
            import json

            json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

        print("=" * 80)
        print("🎯 DQN LunarLander (simple, style notebook)")
        print("=" * 80)
        print(f"Env: {cfg.env_id}")
        print(f"Device: {device}")
        print(f"Episodes: {cfg.n_episodes} | Batch: {cfg.batch_size} | Buffer: {cfg.memory_size}")
        print(f"Checkpoint -> {ckpt_path}")
        print(f"Logs -> {log_path}")
        print("=" * 80)

        scores: list[float] = []

        for ep in range(cfg.n_episodes):
            t0 = time.time()
            state, _ = env.reset(seed=cfg.seed + ep)
            done = False
            total_reward = 0.0
            steps = 0
            losses: list[float] = []

            while not done and steps < cfg.max_steps:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)

                agent.remember(state, int(action), float(reward), next_state, bool(done))
                loss = agent.replay()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                total_reward += float(reward)
                steps += 1

            if ep % cfg.target_update_freq == 0:
                agent.update_target_network()

            scores.append(total_reward)
            avg_score_100 = float(np.mean(scores[-100:]))
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            dt = time.time() - t0

            writer.writerow(
                {
                    "episode": ep + 1,
                    "score": total_reward,
                    "avg_score_100": avg_score_100,
                    "epsilon": agent.epsilon,
                    "episode_steps": steps,
                    "mean_loss": mean_loss,
                    "duration_s": dt,
                }
            )

            if (ep + 1) % cfg.log_every == 0:
                print(
                    f"Ep {ep+1:5d}/{cfg.n_episodes} | score={total_reward:8.2f} | "
                    f"avg100={avg_score_100:7.2f} | eps={agent.epsilon:6.3f} | "
                    f"loss={mean_loss:8.4f} | steps={steps:4d} | {dt:5.2f}s"
                )

        # Save checkpoint
        torch.save(agent.q_network.state_dict(), ckpt_path)
        print(f"\n✅ Checkpoint sauvegardé: {ckpt_path}")

        return {
            "scores": scores,
            "last_avg_100": float(np.mean(scores[-100:])),
            "log_csv": str(log_path),
            "checkpoint": str(ckpt_path),
        }

    except KeyboardInterrupt:
        # Permettre d'arrêter un run long tout en conservant un checkpoint exploitable.
        partial_path = ckpt_path.with_name(ckpt_path.stem + "_INTERRUPTED" + ckpt_path.suffix)
        try:
            torch.save(agent.q_network.state_dict(), partial_path)
            print(f"\n🟡 Interruption: checkpoint partiel sauvegardé -> {partial_path}")
            print(f"🟡 Log CSV conservé -> {log_path}")
        except Exception as e:
            print(f"\n⚠️ Interruption: impossible de sauvegarder le checkpoint partiel ({e})")
        raise

    finally:
        if f_csv is not None:
            try:
                f_csv.close()
            except Exception:
                pass
        env.close()


def load_agent_for_eval(cfg: Config, checkpoint_path: str | Path, render_human: bool = True):
    device = resolve_device()
    env = gym.make(cfg.env_id, render_mode="human" if render_human else None)

    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)

    agent = DQNAgent(state_size, action_size, cfg, device)
    agent.q_network.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.q_network.eval()
    agent.epsilon = 0.0

    return env, agent


def evaluate(cfg: Config, checkpoint_path: str | Path, n_episodes: int = 10, render_human: bool = True) -> dict:
    env, agent = load_agent_for_eval(cfg, checkpoint_path, render_human=render_human)

    scores: list[float] = []
    try:
        for ep in range(n_episodes):
            state, _ = env.reset(seed=cfg.seed + 10_000 + ep)
            done = False
            total_reward = 0.0
            steps = 0

            while not done and steps < cfg.max_steps:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                state = next_state
                total_reward += float(reward)
                steps += 1

            scores.append(total_reward)
            print(f"Test ep {ep+1}/{n_episodes} | score={total_reward:.2f}")

        print(f"\n✅ Score moyen ({n_episodes} eps): {float(np.mean(scores)):.2f}")
        return {"scores": scores, "mean": float(np.mean(scores))}

    finally:
        env.close()


# -----------------------------
# CLI minimal
# -----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="DQN LunarLander")
    parser.add_argument("--train", action="store_true", help="Lancer l'entraînement")
    parser.add_argument("--eval", action="store_true", help="Eval d'un checkpoint")
    parser.add_argument("--episodes", type=int, default=None, help="Nombre d'épisodes d'entraînement")
    parser.add_argument("--checkpoint", type=str, default=None, help="Chemin checkpoint (eval)")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")

    args = parser.parse_args()

    cfg = Config()
    if args.episodes is not None:
        cfg.n_episodes = args.episodes

    if args.train:
        train(cfg)

    if args.eval:
        ckpt = args.checkpoint or (Path(cfg.save_dir) / cfg.save_name)
        evaluate(cfg, ckpt, n_episodes=args.eval_episodes, render_human=not args.no_render)


if __name__ == "__main__":
    main()
