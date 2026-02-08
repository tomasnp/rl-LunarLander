import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# -----------------------------
# Logging Utility
# -----------------------------
class TeeLogger:
    """
    Duplicates stdout/stderr to both terminal and log file.
    Usage: sys.stdout = TeeLogger('logfile.txt')
    """
    def __init__(self, filepath, mode='a'):
        self.terminal = sys.stdout
        self.log = open(filepath, mode, encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Flush immediately for real-time logging

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup_logging(log_dir: str = "logs", experiment_name: str = "a2c") -> tuple:
    """
    Set up logging to file with timestamp.

    Returns:
        (log_filepath, tee_logger): Path to log file and TeeLogger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # Create TeeLogger
    tee = TeeLogger(log_filepath, mode='a')

    # Log header
    print("=" * 80)
    print(f"🗂️  Logging started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 Log file: {log_filepath}")
    print("=" * 80)
    print()

    return log_filepath, tee


def log_config(cfg: 'Config'):
    """Log configuration parameters."""
    print("\n" + "=" * 80)
    print("⚙️  CONFIGURATION")
    print("=" * 80)
    for field_name in cfg.__dataclass_fields__:
        value = getattr(cfg, field_name)
        print(f"  {field_name:25s} = {value}")
    print("=" * 80)
    print()


# -----------------------------
# Observation Normalization
# -----------------------------
class RunningMeanStd:
    """
    Computes running mean and std for observation normalization.
    Uses Welford's online algorithm for numerical stability.
    """
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        """Update running statistics with new batch of observations."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x, clip=10.0):
        """Normalize observation and clip to [-clip, +clip]."""
        x_normalized = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(x_normalized, -clip, clip)


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    env_id: str = "LunarLander-v3"
    seed: int = 42

    gamma: float = 0.99
    gae_lambda: float = 0.95        # GAE lambda parameter
    lr_policy: float = 5e-4         # FIXED: increased from 3e-4 for faster learning
    lr_value: float = 1e-3          # FIXED: increased from 3e-4 for better critic convergence

    entropy_coef_start: float = 0.05   # starting entropy coefficient
    entropy_coef_final: float = 0.005  # FIXED: was 0.001 (too low, entropy stayed high)
    value_coef: float = 0.5            # critic loss weight

    rollout_steps: int = 2048       # steps per rollout batch
    max_updates: int = 10000        # FIXED: increased from 7500 (need more training)
    eval_every: int = 50            # evaluate every N updates
    eval_episodes: int = 30         # FIXED: increased from 10 (reduce eval variance)

    hidden_size: int = 512          # IMPROVED: increased from 256 for better capacity
    grad_clip: float = 0.5          # gradient clipping threshold

    # STABILITY IMPROVEMENTS (NEW)
    normalize_obs: bool = True      # normalize observations with running mean/std
    reward_clip: float = 10.0       # clip rewards to [-clip, +clip]
    weight_decay: float = 1e-5      # L2 regularization for AdamW
    obs_clip: float = 10.0          # clip normalized observations

    save_dir: str = "checkpoints"
    save_name: str = f"a2c_{rollout_steps}_{max_updates}_{hidden_size}.pt"
    plot_name: str = f"a2c_{rollout_steps}_{max_updates}_{hidden_size}.png"

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


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, terminateds: torch.Tensor,
                next_value: float, gamma: float, gae_lambda: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    CRITICAL: Uses 'terminateds' (not 'dones') for correct bootstrapping:
    - terminated=1: True terminal (crash/success) -> don't bootstrap (V_next=0)
    - truncated: Time limit -> DO bootstrap (use V_next)

    Args:
        rewards: [T] tensor of rewards
        values: [T] tensor of value predictions V(s_t)
        terminateds: [T] tensor of TERMINAL flags (1 if truly terminal, 0 if truncated or ongoing)
        next_value: V(s_{T+1}) for bootstrapping (0 if last state was terminal)
        gamma: discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: [T] tensor of advantages
        returns: [T] tensor of value targets (advantages + values)
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0.0

    # Append next_value for bootstrapping
    values_extended = torch.cat([values, torch.tensor([next_value], dtype=torch.float32)])

    # Backward computation of GAE
    for t in reversed(range(T)):
        # TD error: delta_t = r_t + gamma * (1 - terminated_t) * V_{t+1} - V_t
        # If terminated=1 (true terminal), V_{t+1} is masked out
        # If terminated=0 (truncated or ongoing), bootstrap with V_{t+1}
        delta = rewards[t] + gamma * (1 - terminateds[t]) * values_extended[t + 1] - values[t]

        # GAE: A_t = delta_t + gamma * lambda * (1 - terminated_t) * A_{t+1}
        gae = delta + gamma * gae_lambda * (1 - terminateds[t]) * gae
        advantages[t] = gae

    # Returns = advantages + values (value targets for critic)
    returns = advantages + values

    return advantages, returns


def plot_performance(history: Dict, save_path: str = f"training_performance.png"):
    """
    Crée une figure à 4 sous-graphiques pour visualiser les performances d'entraînement.

    Args:
        history: Dictionnaire contenant:
            - 'episode_rewards': Liste des récompenses par épisode
            - 'episode_entropies': Liste des entropies moyennes par épisode
        save_path: Chemin pour sauvegarder la figure
    """
    episode_rewards = history['episode_rewards']
    episode_entropies = history.get('episode_entropies', [])

    # Configuration du style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # ============================================================
    # HAUT-GAUCHE : Évolution des récompenses
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])
    episodes = np.arange(1, len(episode_rewards) + 1)

    # Scores bruts (bleu clair)
    ax1.plot(episodes, episode_rewards, color='lightblue', alpha=0.6,
             linewidth=0.8, label='Score brut')

    # Moyenne mobile (fenêtre de 100)
    if len(episode_rewards) >= 100:
        moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        ax1.plot(np.arange(100, len(episode_rewards) + 1), moving_avg,
                color='orange', linewidth=2.5, label='Moyenne mobile (100 ép.)')

    # Ligne objectif à 200
    ax1.axhline(y=200, color='green', linestyle='--', linewidth=2,
                label='Objectif (200)', alpha=0.8)

    ax1.set_xlabel('Épisodes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Évolution des Récompenses au Cours de l\'Entraînement',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax1.set_xlim(0, len(episode_rewards))

    # ============================================================
    # HAUT-DROITE : Décroissance de l'Entropy (Exploration)
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])

    if episode_entropies:
        # Detect if entropies are per-episode or per-update (A2C case)
        if len(episode_entropies) == len(episode_rewards):
            # Per-episode case (REINFORCE)
            x_axis = episodes
            x_label = 'Épisodes'
            title = 'Évolution de l\'Entropy (Exploration)'
        else:
            # Per-update case (A2C with rollout batching)
            x_axis = np.arange(1, len(episode_entropies) + 1)
            x_label = 'Updates'
            title = 'Évolution de l\'Entropy par Update (Exploration)'

        ax2.plot(x_axis, episode_entropies, color='purple', linewidth=2,
                marker='o', markersize=2, alpha=0.7)
        ax2.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax2.set_ylabel('Entropy Moyenne', fontsize=12, fontweight='bold')
        ax2.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax2.set_xlim(0, len(x_axis))

        # Annotations
        if len(episode_entropies) > 0:
            start_ent = episode_entropies[0]
            end_ent = episode_entropies[-1]
            ax2.annotate(f'Départ: {start_ent:.3f}',
                        xy=(x_axis[0], start_ent), xytext=(x_axis[0]+10, start_ent),
                        fontsize=9, color='purple', alpha=0.8)
            if len(x_axis) > 50:
                ax2.annotate(f'Fin: {end_ent:.3f}',
                            xy=(x_axis[-1], end_ent),
                            xytext=(x_axis[-1]-len(x_axis)*0.1, end_ent),
                            fontsize=9, color='purple', alpha=0.8)
    else:
        ax2.text(0.5, 0.5, 'Données d\'entropy non disponibles',
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title('Évolution de l\'Entropy (Exploration)',
                     fontsize=14, fontweight='bold', pad=15)

    # ============================================================
    # BAS-GAUCHE : Distribution des scores
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Histogramme
    n, bins, patches = ax3.hist(episode_rewards, bins=50, color='skyblue',
                                 edgecolor='black', alpha=0.7, linewidth=0.8)

    # Ligne verticale pour la moyenne
    mean_reward = np.mean(episode_rewards)
    ax3.axvline(x=mean_reward, color='red', linestyle='--', linewidth=2.5,
                label=f'Moyenne: {mean_reward:.1f}')

    # Colorer les barres selon le seuil de 200
    for i, patch in enumerate(patches):
        if bins[i] >= 200:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.8)

    ax3.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Fréquence', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution des Scores', fontsize=14, fontweight='bold', pad=15)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.8, axis='y')

    # ============================================================
    # BAS-DROITE : Taux de succès (fenêtre glissante)
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 1])

    window_size = 50
    if len(episode_rewards) >= window_size:
        success_rates = []
        episodes_success = []

        for i in range(window_size, len(episode_rewards) + 1):
            window = episode_rewards[i-window_size:i]
            success_rate = sum(1 for r in window if r >= 200) / window_size * 100
            success_rates.append(success_rate)
            episodes_success.append(i)

        ax4.plot(episodes_success, success_rates, color='green',
                linewidth=2.5, marker='o', markersize=3, alpha=0.8)
        ax4.fill_between(episodes_success, success_rates, alpha=0.3, color='green')

        # Ligne horizontale à 100%
        ax4.axhline(y=100, color='gold', linestyle='--', linewidth=2,
                   label='100% de succès', alpha=0.7)

        ax4.set_xlabel('Épisodes', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Taux de Succès (%)', fontsize=12, fontweight='bold')
        ax4.set_title(f'Taux de Succès (Score ≥ 200, Fenêtre de {window_size} ép.)',
                     fontsize=14, fontweight='bold', pad=15)
        ax4.set_ylim(-5, 105)
        ax4.set_xlim(window_size, len(episode_rewards))
        ax4.legend(loc='lower right', fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

        # Annotations du taux final
        if success_rates:
            final_rate = success_rates[-1]
            ax4.annotate(f'Taux final: {final_rate:.1f}%',
                        xy=(episodes_success[-1], final_rate),
                        xytext=(episodes_success[-1]-100, final_rate+10),
                        fontsize=10, color='darkgreen', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
    else:
        ax4.text(0.5, 0.5, f'Pas assez d\'épisodes (min: {window_size})',
                ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.set_title(f'Taux de Succès (Score ≥ 200, Fenêtre de {window_size} ép.)',
                     fontsize=14, fontweight='bold', pad=15)

    # ============================================================
    # Titre global et sauvegarde
    # ============================================================
    fig.suptitle('Analyse des Performances - A2C avec GAE sur Lunar Lander',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Graphique sauvegardé : {save_path}")
    plt.show()

    return fig


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
def select_action(policy: PolicyNet, obs: np.ndarray, device: torch.device = torch.device('cpu')) -> Tuple[int, float, float]:
    """
    Returns:
      action (int)
      log_prob (float)
      entropy (float)
    """
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # [1, obs_dim]
    logits = policy(obs_t)  # [1, act_dim]
    dist = Categorical(logits=logits)
    action_t = dist.sample()
    action = action_t.item()
    log_prob = dist.log_prob(action_t).item()
    entropy = dist.entropy().item()

    return action, log_prob, entropy


def run_episode(env: gym.Env, policy: PolicyNet, device: torch.device = torch.device('cpu')) -> Tuple[List[np.ndarray], List[int], List[float], List[float], List[float], float]:
    """
    Collect one full episode (for REINFORCE).
    Returns lists: states, actions, rewards, log_probs, entropies, episode_return
    """
    states, actions, rewards, log_probs, entropies = [], [], [], [], []
    obs, _ = env.reset()
    done = False
    ep_return = 0.0

    while not done:
        action, logp, ent = select_action(policy, obs, device)
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


def collect_rollout(env: gym.Env, policy: PolicyNet, value: ValueNet,
                   rollout_steps: int, device: torch.device,
                   current_obs: np.ndarray, current_done: bool,
                   obs_normalizer: RunningMeanStd = None,
                   reward_clip: float = None,
                   obs_clip: float = 10.0) -> Tuple[Dict, np.ndarray, bool, List[float], List[np.ndarray]]:
    """
    Collect a fixed-size rollout of K steps (handles episode boundaries).

    Args:
        env: Gymnasium environment
        policy: Policy network
        value: Value network
        rollout_steps: Number of steps to collect
        device: Device for torch tensors
        current_obs: Current observation (for continuing from previous rollout)
        current_done: Whether current state is terminal
        obs_normalizer: Optional RunningMeanStd for observation normalization
        reward_clip: Optional reward clipping threshold
        obs_clip: Clip value for normalized observations

    Returns:
        rollout_data: Dict with keys ['states', 'actions', 'rewards', 'dones', 'terminateds']
        next_obs: Observation after rollout
        next_done: Done flag after rollout
        episode_returns: List of completed episode returns during rollout
        raw_observations: List of raw (unnormalized) observations for updating stats
    """
    states, actions, rewards, dones, terminateds, values = [], [], [], [], [], []
    episode_returns = []
    current_episode_return = 0.0
    raw_observations = []  # Store raw obs for updating normalizer

    # Reset if starting from terminal state
    if current_done:
        current_obs, _ = env.reset()
        current_done = False

    for _ in range(rollout_steps):
        # Store raw observation for normalizer update
        raw_observations.append(current_obs.copy())

        # Normalize observation if normalizer is provided
        if obs_normalizer is not None:
            obs_normalized = obs_normalizer.normalize(current_obs, clip=obs_clip)
        else:
            obs_normalized = current_obs

        # Select action
        obs_t = torch.tensor(obs_normalized, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            # Get action from policy
            logits = policy(obs_t)
            dist = Categorical(logits=logits)
            action_t = dist.sample()
            action = action_t.item()

            # Get value estimate
            value_pred = value(obs_t).item()

        # Take action in environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Clip reward if specified (improves stability)
        if reward_clip is not None:
            reward = np.clip(reward, -reward_clip, reward_clip)

        # Store transition
        # CRITICAL: Store 'terminated' separately from 'done' for correct GAE bootstrapping
        # - terminated = True terminal state (crashed, succeeded) -> don't bootstrap
        # - truncated = Time limit reached -> DO bootstrap (not a real terminal)
        states.append(obs_normalized)  # Store normalized observation
        actions.append(action)
        rewards.append(float(reward))
        dones.append(1.0 if done else 0.0)  # for episode tracking
        terminateds.append(1.0 if terminated else 0.0)  # for GAE (only TRUE terminals)
        values.append(value_pred)

        current_episode_return += float(reward)

        # Handle episode boundary
        if done:
            episode_returns.append(current_episode_return)
            current_episode_return = 0.0
            next_obs, _ = env.reset()
            done = False

        current_obs = next_obs
        current_done = done

    rollout_data = {
        'states': np.array(states),
        'actions': np.array(actions, dtype=np.int64),
        'rewards': np.array(rewards, dtype=np.float32),
        'dones': np.array(dones, dtype=np.float32),
        'terminateds': np.array(terminateds, dtype=np.float32),  # TRUE terminals only
        'values': np.array(values, dtype=np.float32)
    }

    return rollout_data, current_obs, current_done, episode_returns, raw_observations


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
def evaluate(cfg: Config, policy: PolicyNet, device: torch.device = torch.device('cpu'),
             update_idx: int = 0, obs_normalizer: RunningMeanStd = None) -> float:
    env = make_eval_env(cfg)
    returns = []
    for ep_idx in range(cfg.eval_episodes):
        # Use different seeds for each eval episode, or no seed for variety
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        while not done:
            # Normalize observation if normalizer is provided
            if obs_normalizer is not None:
                obs_normalized = obs_normalizer.normalize(obs, clip=cfg.obs_clip)
            else:
                obs_normalized = obs

            obs_t = torch.tensor(obs_normalized, dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy(obs_t)
            dist = Categorical(logits=logits)
            action = torch.argmax(dist.probs, dim=-1).item()  # deterministic
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)
            steps += 1

        returns.append(ep_return)

        # DEBUG: Log extreme returns (potential bugs)
        if ep_return < -500:
            print(f"    [EVAL WARNING] Episode {ep_idx+1}/{cfg.eval_episodes}: extreme return={ep_return:.1f} in {steps} steps")

    env.close()

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    min_ret = float(np.min(returns))
    max_ret = float(np.max(returns))

    # DEBUG: Log detailed stats every 100 updates
    if update_idx % 100 == 0 and update_idx > 0:
        print(f"    [EVAL STATS] mean={mean_ret:.1f} std={std_ret:.1f} min={min_ret:.1f} max={max_ret:.1f}")

    return mean_ret


# -----------------------------
# Training
# -----------------------------
def train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log configuration
    log_config(cfg)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Training mode: A2C with GAE (rollout_steps={cfg.rollout_steps})")
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] Gymnasium version: {gym.__version__}")
    print()

    set_seed(cfg.seed)

    # Training env (no rendering for speed)
    env = gym.make(cfg.env_id)
    current_obs, _ = env.reset(seed=cfg.seed)
    current_done = False

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim, cfg.hidden_size).to(device)
    value = ValueNet(obs_dim, cfg.hidden_size).to(device)

    # IMPROVED: Use AdamW with weight decay for better regularization
    opt_policy = optim.AdamW(policy.parameters(), lr=cfg.lr_policy,
                             betas=(0.9, 0.999), eps=1e-5, weight_decay=cfg.weight_decay)
    opt_value = optim.AdamW(value.parameters(), lr=cfg.lr_value,
                            betas=(0.9, 0.999), eps=1e-5, weight_decay=cfg.weight_decay)

    # IMPROVED: Initialize observation normalizer if enabled
    obs_normalizer = None
    if cfg.normalize_obs:
        obs_normalizer = RunningMeanStd(shape=(obs_dim,))
        print(f"[INFO] Observation normalization: ENABLED (clip={cfg.obs_clip})")
    else:
        print(f"[INFO] Observation normalization: DISABLED")

    if cfg.reward_clip is not None:
        print(f"[INFO] Reward clipping: ENABLED (clip=±{cfg.reward_clip})")
    else:
        print(f"[INFO] Reward clipping: DISABLED")
    print()

    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, cfg.save_name)

    # Tracking
    all_episode_returns: List[float] = []
    entropy_history: List[float] = []
    policy_loss_history: List[float] = []
    value_loss_history: List[float] = []
    best_eval = -1e9

    t0 = time.time()

    for update_idx in range(1, cfg.max_updates + 1):
        # Collect rollout (with optional normalization and reward clipping)
        rollout_data, current_obs, current_done, episode_returns, raw_obs = collect_rollout(
            env, policy, value, cfg.rollout_steps, device, current_obs, current_done,
            obs_normalizer=obs_normalizer,
            reward_clip=cfg.reward_clip,
            obs_clip=cfg.obs_clip
        )

        # Update observation normalizer with raw observations
        if obs_normalizer is not None and len(raw_obs) > 0:
            obs_normalizer.update(np.array(raw_obs))

        # Track completed episodes
        all_episode_returns.extend(episode_returns)

        # Convert to tensors
        states_t = torch.tensor(rollout_data['states'], dtype=torch.float32, device=device)
        actions_t = torch.tensor(rollout_data['actions'], dtype=torch.int64, device=device)
        rewards_t = torch.tensor(rollout_data['rewards'], dtype=torch.float32, device=device)
        terminateds_t = torch.tensor(rollout_data['terminateds'], dtype=torch.float32, device=device)
        old_values_t = torch.tensor(rollout_data['values'], dtype=torch.float32, device=device)

        # Bootstrap next value for GAE
        # CRITICAL: Only set next_value=0 if TRULY terminal (not just time limit)
        with torch.no_grad():
            # Check if last state was a true terminal (not truncated)
            last_terminated = rollout_data['terminateds'][-1]
            if last_terminated == 1.0:
                next_value = 0.0  # True terminal, don't bootstrap
            else:
                # Either ongoing or truncated, bootstrap with value
                # Normalize current_obs if normalizer is enabled
                if obs_normalizer is not None:
                    current_obs_normalized = obs_normalizer.normalize(current_obs, clip=cfg.obs_clip)
                else:
                    current_obs_normalized = current_obs
                next_obs_t = torch.tensor(current_obs_normalized, dtype=torch.float32, device=device).unsqueeze(0)
                next_value = value(next_obs_t).item()

        # Compute advantages and returns using GAE
        # Now uses 'terminateds' for correct bootstrapping (truncated episodes bootstrap)
        advantages_t, returns_t = compute_gae(
            rewards_t, old_values_t, terminateds_t, next_value, cfg.gamma, cfg.gae_lambda
        )
        advantages_t = advantages_t.to(device)
        returns_t = returns_t.to(device)

        # Normalize advantages (stability)
        adv_mean_before = advantages_t.mean()
        adv_std_before = advantages_t.std(unbiased=False) + 1e-8
        advantages_t = (advantages_t - adv_mean_before) / adv_std_before

        # Compute stats after normalization for logging
        adv_mean_after = advantages_t.mean()
        adv_std_after = advantages_t.std(unbiased=False)

        # Recompute policy distribution with current parameters
        logits = policy(states_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        # Entropy annealing (linear decay)
        progress = update_idx / cfg.max_updates
        entropy_coef = max(
            cfg.entropy_coef_final,
            cfg.entropy_coef_start * (1.0 - progress)
        )

        # Actor loss (detach advantages!)
        policy_loss = -(log_probs * advantages_t.detach()).mean() - entropy_coef * entropy

        # Critic loss (Huber/SmoothL1Loss, detach returns!)
        values_pred = value(states_t).squeeze(-1)
        value_loss = nn.SmoothL1Loss()(values_pred, returns_t.detach())

        # Total loss
        loss = policy_loss + cfg.value_coef * value_loss

        # Optimize
        opt_policy.zero_grad()
        opt_value.zero_grad()
        loss.backward()

        # Gradient clipping (both networks)
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip)
        nn.utils.clip_grad_norm_(value.parameters(), max_norm=cfg.grad_clip)

        opt_policy.step()
        opt_value.step()

        # Track metrics
        entropy_history.append(entropy.item())
        policy_loss_history.append(policy_loss.item())
        value_loss_history.append(value_loss.item())

        # Logging
        if update_idx % 10 == 0:
            recent_returns = all_episode_returns[-100:] if all_episode_returns else [0.0]
            mean_return = float(np.mean(recent_returns))

            print(
                f"Update {update_idx:4d} | "
                f"return={mean_return:7.1f} (n={len(all_episode_returns)}) | "
                f"loss={loss.item():.3f} | "
                f"policy={policy_loss.item():.3f} | "
                f"value={value_loss.item():.3f} | "
                f"entropy={entropy.item():.3f} (coef={entropy_coef:.4f}) | "
                f"adv: μ={adv_mean_after.item():.3f} σ={adv_std_after.item():.3f}"
            )

        # Periodic evaluation + checkpoint
        if update_idx % cfg.eval_every == 0:
            policy.eval()
            avg_eval = evaluate(cfg, policy, device, update_idx, obs_normalizer)
            policy.train()
            print(f"[EVAL] Update {update_idx:4d} | avg_return over {cfg.eval_episodes} eps = {avg_eval:.1f}")

            if avg_eval > best_eval:
                best_eval = avg_eval
                checkpoint = {
                    "policy_state_dict": policy.state_dict(),
                    "value_state_dict": value.state_dict(),
                    "cfg": cfg.__dict__,
                    "best_eval": best_eval,
                    "update": update_idx,
                }
                # Save obs_normalizer stats if enabled
                if obs_normalizer is not None:
                    checkpoint["obs_normalizer"] = {
                        "mean": obs_normalizer.mean,
                        "var": obs_normalizer.var,
                        "count": obs_normalizer.count
                    }
                torch.save(checkpoint, save_path)
                print(f"[SAVE] New best model saved to: {save_path}")

        # Early stop if solved
        if len(all_episode_returns) >= cfg.solved_window:
            rolling_mean = float(np.mean(all_episode_returns[-cfg.solved_window:]))
            if rolling_mean >= cfg.solved_mean_reward:
                print(f"[DONE] Solved! rolling_mean={rolling_mean:.1f} >= {cfg.solved_mean_reward}")
                break

    env.close()

    # Calculate final statistics
    training_time = time.time() - t0
    final_mean = float(np.mean(all_episode_returns[-100:])) if len(all_episode_returns) >= 100 else float(np.mean(all_episode_returns))
    final_std = float(np.std(all_episode_returns[-100:])) if len(all_episode_returns) >= 100 else float(np.std(all_episode_returns))

    # Log final results
    print("\n" + "=" * 80)
    print("📊 TRAINING SUMMARY")
    print("=" * 80)
    print(f"Training time:        {training_time:.1f}s ({training_time/60:.1f} minutes)")
    print(f"Total updates:        {update_idx}")
    print(f"Total episodes:       {len(all_episode_returns)}")
    print(f"Best eval reward:     {best_eval:.1f}")
    print(f"Final mean (100 ep):  {final_mean:.1f} ± {final_std:.1f}")
    print(f"Solved:               {'✅ YES' if best_eval >= cfg.solved_mean_reward else '❌ NO'}")
    print(f"Checkpoint:           {save_path}")
    print("=" * 80)
    print()

    # Visualize training performance
    history = {
        'episode_rewards': all_episode_returns,
        'episode_entropies': entropy_history
    }
    plot_performance(history, save_path=cfg.plot_name)

    # Log completion
    print(f"\n✅ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return history


def load_policy(cfg: Config) -> Tuple[PolicyNet, RunningMeanStd]:
    """
    Load trained policy from checkpoint.

    Returns:
        policy: Trained policy network
        obs_normalizer: Observation normalizer (None if not used during training)
    """
    ckpt_path = os.path.join(cfg.save_dir, cfg.save_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Train first!")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Get env dimensions
    env = gym.make(cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    # Use hidden_size from checkpoint's config (for backward compatibility)
    if "cfg" in ckpt and "hidden_size" in ckpt["cfg"]:
        hidden_size = ckpt["cfg"]["hidden_size"]
        print(f"[LOAD] Using hidden_size={hidden_size} from checkpoint")
    else:
        hidden_size = cfg.hidden_size
        print(f"[LOAD] Using hidden_size={hidden_size} from current config (old checkpoint)")

    # Build policy with correct architecture
    policy = PolicyNet(obs_dim, act_dim, hidden_size)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    # Load obs_normalizer if it exists in checkpoint
    obs_normalizer = None
    if "obs_normalizer" in ckpt:
        obs_normalizer = RunningMeanStd(shape=(obs_dim,))
        obs_normalizer.mean = ckpt["obs_normalizer"]["mean"]
        obs_normalizer.var = ckpt["obs_normalizer"]["var"]
        obs_normalizer.count = ckpt["obs_normalizer"]["count"]
        print(f"[LOAD] Observation normalizer loaded from checkpoint")

    return policy, obs_normalizer


def test(cfg: Config, num_episodes: int = 10, render: bool = False, device: torch.device = torch.device('cpu')):
    """
    Test the trained policy by running it for multiple episodes.

    Args:
        cfg: Configuration object
        num_episodes: Number of episodes to run (default: 10)
        render: Whether to render the environment visually (default: False)
        device: Device to run the model on

    Returns:
        Dictionary with statistics (mean, std, min, max, all_returns)
    """
    print(f"\n{'='*60}")
    print(f"🧪 Testing trained policy for {num_episodes} episodes")
    print(f"{'='*60}\n")

    # Load the trained policy and normalizer
    try:
        policy, obs_normalizer = load_policy(cfg)
        policy = policy.to(device)
        print(f"✅ Policy loaded from: {os.path.join(cfg.save_dir, cfg.save_name)}\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None
    except RuntimeError as e:
        print(f"❌ ERROR: {e}")
        return None

    # Create environment with optional rendering
    render_mode = "human" if render else None
    env = gym.make(cfg.env_id, render_mode=render_mode)

    # Run episodes
    returns = []
    policy.eval()

    with torch.no_grad():
        for ep in range(1, num_episodes + 1):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            steps = 0

            while not done:
                # Normalize observation if normalizer was used during training
                if obs_normalizer is not None:
                    obs_normalized = obs_normalizer.normalize(obs, clip=cfg.obs_clip)
                else:
                    obs_normalized = obs

                # Select action deterministically (argmax)
                obs_t = torch.tensor(obs_normalized, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(obs_t)
                dist = Categorical(logits=logits)
                action = torch.argmax(dist.probs, dim=-1).item()

                # Take action
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_return += reward
                steps += 1

            returns.append(ep_return)

            # Print episode results
            status = "✅ SUCCESS" if ep_return >= 200 else "⚠️  NEEDS WORK" if ep_return >= 0 else "❌ CRASHED"
            print(f"Episode {ep:2d}/{num_episodes} | Return: {ep_return:7.2f} | Steps: {steps:3d} | {status}")

    env.close()

    # Calculate statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)

    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 Test Results Summary:")
    print(f"{'='*60}")
    print(f"Mean Return:   {mean_return:7.2f} ± {std_return:.2f}")
    print(f"Min Return:    {min_return:7.2f}")
    print(f"Max Return:    {max_return:7.2f}")
    print(f"Success Rate:  {sum(1 for r in returns if r >= 200)}/{num_episodes} episodes (≥200 reward)")
    print(f"{'='*60}\n")

    # Return statistics
    return {
        "mean": mean_return,
        "std": std_return,
        "min": min_return,
        "max": max_return,
        "all_returns": returns
    }


def play(cfg: Config, device: torch.device = torch.device('cpu')):
    """Play 5 episodes with the trained policy (visual rendering)."""
    # Human rendering for playing
    cfg.render_eval_human = True
    cfg.record_video = False

    policy, obs_normalizer = load_policy(cfg)
    policy = policy.to(device)

    env = gym.make(cfg.env_id, render_mode="human")
    obs, _ = env.reset(seed=cfg.seed)

    # Play 5 episodes
    for i in range(5):
        # Normalize observation if normalizer exists
        if obs_normalizer is not None:
            obs_normalized = obs_normalizer.normalize(obs, clip=cfg.obs_clip)
        else:
            obs_normalized = obs

        obs_t = torch.tensor(obs_normalized, dtype=torch.float32, device=device).unsqueeze(0)
        logits = policy(obs_t)
        dist = Categorical(logits=logits)
        action = torch.argmax(dist.probs, dim=-1).item()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    # env.close()  # unreachable


if __name__ == "__main__":
    # ========================================
    # Setup Logging (logs everything to file with timestamp)
    # ========================================
    log_filepath, tee_logger = setup_logging(log_dir="logs", experiment_name="a2c_gae")
    original_stdout = sys.stdout
    sys.stdout = tee_logger

    try:
        cfg = Config()

        # ========================================
        # Configuration optionnelle
        # ========================================
        # cfg.rollout_steps = 4096        # Augmenter pour plus de stabilité
        # cfg.max_updates = 3000          # Réduire pour entraînement plus court
        # cfg.eval_every = 25             # Évaluer plus fréquemment
        # cfg.grad_clip = 1.0             # Augmenter si gradients explosent
        # cfg.lr_value = 1e-3             # Augmenter si critique apprend trop lentement
        # cfg.record_video = True
        # cfg.video_dir = "videos_a2c_gae"

        print("="*80)
        print("🚀 A2C with GAE - Lunar Lander Training")
        print("="*80)
        print(f"Rollout steps:      {cfg.rollout_steps}")
        print(f"Max updates:        {cfg.max_updates}")
        print(f"GAE lambda:         {cfg.gae_lambda}")
        print(f"Entropy annealing:  {cfg.entropy_coef_start} → {cfg.entropy_coef_final}")
        print(f"Gradient clipping:  {cfg.grad_clip}")
        print("="*80 + "\n")

        # === MODE 1: Train the model (génère automatiquement training_performance_a2c.png) ===
        history = train(cfg)

        # === MODE 2: Test the trained model (10 episodes, no rendering) ===
        # test(cfg, num_episodes=10, render=False)

        # === MODE 3: Test with visual rendering ===
        # test(cfg, num_episodes=5, render=True)

        # === MODE 4: Play interactively (5 episodes loop) ===
        # play(cfg)

        # === MODE 5: Visualiser un historique existant ===
        # Si vous avez un historique sauvegardé, vous pouvez le charger et le visualiser :
        # import pickle
        # with open('training_history.pkl', 'rb') as f:
        #     history = pickle.load(f)
        # plot_performance(history, save_path="custom_plot.png")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("Checkpoint may have been saved during training.")

    except Exception as e:
        print(f"\n\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        tee_logger.close()
        print(f"\n✅ Log saved to: {log_filepath}")
