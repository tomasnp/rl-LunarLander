import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

    max_episodes: int = 5000
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


def plot_performance(history: Dict, save_path: str = "training_performance.png"):
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
        ax2.plot(episodes, episode_entropies, color='purple', linewidth=2,
                marker='o', markersize=2, alpha=0.7)
        ax2.set_xlabel('Épisodes', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Entropy Moyenne', fontsize=12, fontweight='bold')
        ax2.set_title('Évolution de l\'Entropy (Exploration)',
                     fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax2.set_xlim(0, len(episode_rewards))

        # Annotations
        if len(episode_entropies) > 0:
            start_ent = episode_entropies[0]
            end_ent = episode_entropies[-1]
            ax2.annotate(f'Départ: {start_ent:.3f}',
                        xy=(1, start_ent), xytext=(10, start_ent),
                        fontsize=9, color='purple', alpha=0.8)
            ax2.annotate(f'Fin: {end_ent:.3f}',
                        xy=(len(episode_entropies), end_ent),
                        xytext=(len(episode_entropies)-50, end_ent),
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
    fig.suptitle('Analyse des Performances - REINFORCE sur Lunar Lander',
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
def evaluate(cfg: Config, policy: PolicyNet, device: torch.device = torch.device('cpu')) -> float:
    env = make_eval_env(cfg)
    returns = []
    for _ in range(cfg.eval_episodes):
        # Use different seeds for each eval episode, or no seed for variety
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
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
    entropy_history: List[float] = []
    best_eval = -1e9

    t0 = time.time()

    for ep in range(1, cfg.max_episodes + 1):
        states, actions, rewards, log_probs, entropies, ep_return = run_episode(env, policy, device)
        reward_history.append(ep_return)

        # Track average entropy for exploration analysis
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0
        entropy_history.append(avg_entropy)

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
            avg_eval = evaluate(cfg, policy, device)
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

    # Visualize training performance
    history = {
        'episode_rewards': reward_history,
        'episode_entropies': entropy_history
    }
    plot_performance(history, save_path="training_performance.png")

    return history


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

    # Load the trained policy
    try:
        policy = load_policy(cfg).to(device)
        print(f"✅ Policy loaded from: {os.path.join(cfg.save_dir, cfg.save_name)}\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
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
                # Select action deterministically (argmax)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
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
    # Human rendering for playing
    cfg.render_eval_human = True
    cfg.record_video = False

    policy = load_policy(cfg).to(device)

    env = gym.make(cfg.env_id, render_mode="human")
    obs, _ = env.reset(seed=cfg.seed)

    # while True:
    for i in range(5):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
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

    # === MODE 1: Train the model (génère automatiquement training_performance.png) ===
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
