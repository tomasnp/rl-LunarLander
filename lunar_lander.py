import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    env_id: str = "LunarLander-v3"
    seed: int = 42

    gamma: float = 0.99
    lr_policy: float = 3e-4
    lr_value: float = 1e-3

    entropy_coef: float = 0.05      # encourages exploration
    value_coef: float = 0.5         # critic loss weight

    max_episodes: int = 10000
    eval_every: int = 50            # evaluate every N episodes
    eval_episodes: int = 10

    hidden_size: int = 128

    save_dir: str = "checkpoints"
    save_name: str = "reinforce_baseline_lunar.pt"

    # Rendering / video
    render_eval_human: bool = False  # set True if you want a window during eval
    record_video: bool = False       # set True to save mp4 videos during eval
    video_dir: str = "videos_record"

    # Early stop if solved (common threshold ~200+)
    solved_mean_reward: float = 200.0
    solved_window: int = 100


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    """Compute discounted returns G_t."""
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


# -----------------------------
# Networks
# -----------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # shape: [batch]


# -----------------------------
# Rollout (one episode)
# -----------------------------
@torch.no_grad()
def select_action(policy: PolicyNet, obs: np.ndarray) -> Tuple[int, float, float]:
    """
    Returns:
      action (int)
      log_prob (float)
      entropy (float)
    """
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, obs_dim]
    logits = policy(obs_t)  # [1, act_dim]
    dist = Categorical(logits=logits)
    action_t = dist.sample()
    action = action_t.item()
    log_prob = dist.log_prob(action_t).item()
    entropy = dist.entropy().item()

    return action, log_prob, entropy


def run_episode(env: gym.Env, policy: PolicyNet) -> Tuple[List[np.ndarray], List[int], List[float], List[float], List[float], float]:
    """
    Collect one full episode (for REINFORCE).
    Returns lists: states, actions, rewards, log_probs, entropies, episode_return
    """
    states, actions, rewards, log_probs, entropies = [], [], [], [], []
    obs, _ = env.reset()
    done = False
    ep_return = 0.0

    while not done:
        action, logp, ent = select_action(policy, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(obs)
        actions.append(action)
        rewards.append(float(reward))
        log_probs.append(float(logp))
        entropies.append(float(ent))

        ep_return += float(reward)
        obs = next_obs

    return states, actions, rewards, log_probs, entropies, ep_return


# -----------------------------
# Evaluation
# -----------------------------
def make_eval_env(cfg: Config):
    render_mode = "human" if cfg.render_eval_human else "rgb_array"
    env = gym.make(cfg.env_id, render_mode=render_mode)

    if cfg.record_video:
        from gymnasium.wrappers import RecordVideo
        os.makedirs(cfg.video_dir, exist_ok=True)
        # record every episode
        env = RecordVideo(env, video_folder=cfg.video_dir, episode_trigger=lambda ep: True)

    return env


@torch.no_grad()
def evaluate(cfg: Config, policy: PolicyNet) -> float:
    env = make_eval_env(cfg)
    returns = []
    for _ in range(cfg.eval_episodes):
        obs, _ = env.reset(seed=cfg.seed)
        done = False
        ep_return = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_t)
            dist = Categorical(logits=logits)
            action = torch.argmax(dist.probs, dim=-1).item()  # deterministic
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)
        returns.append(ep_return)

    env.close()
    return float(np.mean(returns))


# -----------------------------
# Training
# -----------------------------
def train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    set_seed(cfg.seed)

    # Training env (no rendering for speed)
    env = gym.make(cfg.env_id)
    env.reset(seed=cfg.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    policy = PolicyNet(obs_dim, act_dim, cfg.hidden_size).to(device)
    value = ValueNet(obs_dim, cfg.hidden_size).to(device)

    opt_policy = optim.Adam(policy.parameters(), lr=cfg.lr_policy)
    opt_value = optim.Adam(value.parameters(), lr=cfg.lr_value)

    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, cfg.save_name)

    reward_history: List[float] = []
    best_eval = -1e9

    t0 = time.time()

    for ep in range(1, cfg.max_episodes + 1):
        states, actions, rewards, log_probs, entropies, ep_return = run_episode(env, policy)
        reward_history.append(ep_return)

        # Prepare tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)  # [T, obs_dim]
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device)            # [T]
        returns_t = compute_returns(rewards, cfg.gamma).to(device)                     # [T]

        # Critic baseline
        values_t = value(states_t)                                                    # [T]
        advantages = returns_t - values_t.detach()

        # Advantage normalization (stability)
        adv_std = advantages.std(unbiased=False) + 1e-8
        advantages = (advantages - advantages.mean()) / adv_std

        # Recompute log_probs with current policy (better practice than storing floats)
        logits = policy(states_t)                                                     # [T, act_dim]
        dist = Categorical(logits=logits)
        logp_t = dist.log_prob(actions_t)                                             # [T]
        entropy_t = dist.entropy().mean()

        # Losses
        policy_loss = -(logp_t * advantages).mean() - cfg.entropy_coef * entropy_t
        value_loss = 0.5 * (returns_t - values_t).pow(2).mean()
        loss = policy_loss + cfg.value_coef * value_loss

        # Optimize
        opt_policy.zero_grad()
        opt_value.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value.parameters()), max_norm=1.0)
        opt_policy.step()
        opt_value.step()

        # Logging
        if ep >= cfg.solved_window:
            rolling_mean = float(np.mean(reward_history[-cfg.solved_window:]))
        else:
            rolling_mean = float(np.mean(reward_history))

        if ep % 10 == 0:
            print(
                f"Ep {ep:4d} | return={ep_return:8.1f} | "
                f"mean({cfg.solved_window})={rolling_mean:7.1f} | "
                f"loss={loss.item():.3f} | policy={policy_loss.item():.3f} | value={value_loss.item():.3f}"
            )

        # Periodic evaluation + checkpoint
        if ep % cfg.eval_every == 0:
            policy.eval()
            avg_eval = evaluate(cfg, policy)
            policy.train()
            print(f"[EVAL] Ep {ep:4d} | avg_return over {cfg.eval_episodes} eps = {avg_eval:.1f}")

            if avg_eval > best_eval:
                best_eval = avg_eval
                torch.save(
                    {
                        "policy_state_dict": policy.state_dict(),
                        "value_state_dict": value.state_dict(),
                        "cfg": cfg.__dict__,
                        "best_eval": best_eval,
                        "episode": ep,
                    },
                    save_path,
                )
                print(f"[SAVE] New best model saved to: {save_path}")

        # Early stop if solved
        if rolling_mean >= cfg.solved_mean_reward and ep >= cfg.solved_window:
            print(f"[DONE] Solved! rolling_mean={rolling_mean:.1f} >= {cfg.solved_mean_reward}")
            break

    env.close()
    print(f"[INFO] Training finished in {time.time() - t0:.1f}s")
    print(f"[INFO] Best eval: {best_eval:.1f}")
    print(f"[INFO] Checkpoint: {save_path}")


def load_policy(cfg: Config) -> PolicyNet:
    ckpt_path = os.path.join(cfg.save_dir, cfg.save_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Train first!")

    # Need obs/action dims to build the net. Create a temp env.
    env = gym.make(cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    policy = PolicyNet(obs_dim, act_dim, cfg.hidden_size)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


def play(cfg: Config):
    # Human rendering for playing
    cfg.render_eval_human = True
    cfg.record_video = False

    policy = load_policy(cfg)

    env = gym.make(cfg.env_id, render_mode="human")
    obs, _ = env.reset(seed=cfg.seed)

    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = policy(obs_t)
        dist = Categorical(logits=logits)
        action = torch.argmax(dist.probs, dim=-1).item()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    # env.close()  # unreachable


if __name__ == "__main__":
    cfg = Config()

    # Toggle these if you want:
    # cfg.record_video = True
    # cfg.video_dir = "videos_reinforce"
    # cfg.eval_every = 100

    train(cfg)

    # After training, uncomment to watch:
    play(cfg)
