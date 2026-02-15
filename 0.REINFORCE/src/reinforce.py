"""
REINFORCE avec Baseline (Critic) pour l'environnement LunarLander-v3.

Algorithme de policy gradient Monte-Carlo avec :
- Réseau de politique (acteur) et réseau de valeur (critique)
- Normalisation des avantages
- Accumulation de gradients sur plusieurs épisodes
- Décroissance du coefficient d'entropie
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# =====================================================================
#  Configuration
# =====================================================================

@dataclass
class Config:
    """Paramètres d'entraînement et d'environnement."""

    # Environnement
    env_id: str = "LunarLander-v3"
    seed: int = 42

    # Hyperparamètres d'apprentissage
    gamma: float = 0.99
    lr_policy: float = 1e-4
    lr_value: float = 5e-4

    # Entropie (exploration)
    entropy_coef: float = 0.01
    entropy_coef_decay: float = 0.995
    entropy_coef_min: float = 0.001

    # Critique
    value_coef: float = 0.5

    # Entraînement
    max_episodes: int = 10000
    eval_every: int = 50
    eval_episodes: int = 10
    hidden_size: int = 256
    grad_clip: float = 0.5
    batch_episodes: int = 4

    # Sauvegarde
    save_dir: str = "checkpoints"
    save_name: str = f"reinforce_maxEpisodes_{max_episodes}_hiddenSize_{hidden_size}.pt"
    plot_name: str = f"reinforce_maxEpisodes_{max_episodes}__hiddenSize_{hidden_size}.png"

    # Rendu / vidéo
    render_eval_human: bool = False
    record_video: bool = False
    video_dir: str = "videos_record"

    # Arrêt anticipé
    solved_mean_reward: float = 200.0
    solved_window: int = 100


# =====================================================================
#  Journalisation (Logging)
# =====================================================================

class TeeLogger:
    """Duplique stdout vers le terminal et un fichier log simultanément."""

    def __init__(self, filepath: str, mode: str = "a"):
        self.terminal = sys.stdout
        self.log = open(filepath, mode, encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup_logging(log_dir: str = "logs", experiment_name: str = "reinforce") -> Tuple[str, TeeLogger]:
    """
    Initialise la journalisation dans un fichier horodaté.

    Args:
        log_dir: Dossier de destination des logs.
        experiment_name: Préfixe du nom de fichier.

    Returns:
        Tuple (chemin du fichier log, instance TeeLogger).
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    tee = TeeLogger(log_filepath)

    print("=" * 80)
    print(f"Logging : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fichier : {log_filepath}")
    print("=" * 80 + "\n")

    return log_filepath, tee


def log_config(cfg: Config):
    """Affiche tous les paramètres de configuration."""
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    for name in cfg.__dataclass_fields__:
        print(f"  {name:25s} = {getattr(cfg, name)}")
    print("=" * 80 + "\n")


# =====================================================================
#  Utilitaires
# =====================================================================

def set_seed(seed: int):
    """Fixe les graines aléatoires pour la reproductibilité (numpy, torch)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    """
    Calcule les retours cumulés actualisés G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}.

    Args:
        rewards: Liste des récompenses d'un épisode.
        gamma: Facteur d'actualisation.

    Returns:
        Tenseur des retours pour chaque pas de temps.
    """
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


# =====================================================================
#  Réseaux de neurones
# =====================================================================

class PolicyNet(nn.Module):
    """
    Réseau de politique (acteur).

    Prend un état en entrée et produit les logits sur les actions.
    Architecture : 2 couches cachées avec LayerNorm + ReLU.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les logits (scores non-normalisés) pour chaque action."""
        return self.net(x)


class ValueNet(nn.Module):
    """
    Réseau de valeur (critique / baseline).

    Prend un état en entrée et estime la valeur V(s).
    Architecture : 2 couches cachées avec LayerNorm + ReLU.
    """

    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne la valeur scalaire V(s) pour chaque état du batch."""
        return self.net(x).squeeze(-1)


# =====================================================================
#  Collecte d'un épisode (rollout)
# =====================================================================

@torch.no_grad()
def select_action(policy: PolicyNet, obs: np.ndarray, device: torch.device) -> Tuple[int, float, float]:
    """
    Échantillonne une action selon la politique stochastique.

    Args:
        policy: Réseau de politique.
        obs: Observation courante de l'environnement.
        device: Device torch (cpu ou cuda).

    Returns:
        Tuple (action, log_prob, entropie).
    """
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits = policy(obs_t)
    dist = Categorical(logits=logits)
    action_t = dist.sample()
    return action_t.item(), dist.log_prob(action_t).item(), dist.entropy().item()


def run_episode(env: gym.Env, policy: PolicyNet, device: torch.device) -> Tuple[List[np.ndarray], List[int], List[float], List[float], List[float], float]:
    """
    Exécute un épisode complet et collecte les trajectoires.

    Args:
        env: Environnement Gymnasium.
        policy: Réseau de politique.
        device: Device torch.

    Returns:
        Tuple (états, actions, récompenses, log_probs, entropies, retour total).
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
        log_probs.append(logp)
        entropies.append(ent)

        ep_return += float(reward)
        obs = next_obs

    return states, actions, rewards, log_probs, entropies, ep_return


# =====================================================================
#  Évaluation
# =====================================================================

def make_eval_env(cfg: Config) -> gym.Env:
    """
    Crée un environnement d'évaluation (avec rendu ou enregistrement vidéo si configuré).

    Args:
        cfg: Configuration.

    Returns:
        Environnement Gymnasium prêt pour l'évaluation.
    """
    render_mode = "human" if cfg.render_eval_human else "rgb_array"
    env = gym.make(cfg.env_id, render_mode=render_mode)

    if cfg.record_video:
        from gymnasium.wrappers import RecordVideo
        os.makedirs(cfg.video_dir, exist_ok=True)
        env = RecordVideo(env, video_folder=cfg.video_dir, episode_trigger=lambda ep: True)

    return env


@torch.no_grad()
def evaluate(cfg: Config, policy: PolicyNet, device: torch.device) -> float:
    """
    Évalue la politique de manière déterministe (argmax) sur plusieurs épisodes.

    Args:
        cfg: Configuration.
        policy: Réseau de politique.
        device: Device torch.

    Returns:
        Retour moyen sur les épisodes d'évaluation.
    """
    env = make_eval_env(cfg)
    returns = []

    for _ in range(cfg.eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy(obs_t)
            action = torch.argmax(Categorical(logits=logits).probs, dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)

        returns.append(ep_return)

    env.close()
    return float(np.mean(returns))


# =====================================================================
#  Entraînement
# =====================================================================

def train(cfg: Config) -> Dict:
    """
    Boucle principale d'entraînement REINFORCE avec baseline.

    Accumule les gradients sur `cfg.batch_episodes` épisodes avant chaque mise à jour.
    Décroît le coefficient d'entropie au fil du temps pour passer de l'exploration
    à l'exploitation.

    Args:
        cfg: Configuration d'entraînement.

    Returns:
        Dictionnaire contenant l'historique (récompenses et entropies par épisode).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")

    set_seed(cfg.seed)

    # Environnement d'entraînement (sans rendu pour la vitesse)
    env = gym.make(cfg.env_id)
    env.reset(seed=cfg.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Réseaux
    policy = PolicyNet(obs_dim, act_dim, cfg.hidden_size).to(device)
    value = ValueNet(obs_dim, cfg.hidden_size).to(device)

    # Optimiseurs
    opt_policy = optim.Adam(policy.parameters(), lr=cfg.lr_policy)
    opt_value = optim.Adam(value.parameters(), lr=cfg.lr_value)

    # Sauvegarde
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, cfg.save_name)

    # Historique
    reward_history: List[float] = []
    entropy_history: List[float] = []
    best_eval = -float("inf")

    # Coefficient d'entropie (décroît au fil du temps)
    current_entropy_coef = cfg.entropy_coef

    # Buffer pour l'accumulation de gradients par batch
    batch_policy_losses: List[torch.Tensor] = []
    batch_value_losses: List[torch.Tensor] = []

    t0 = time.time()

    for ep in range(1, cfg.max_episodes + 1):
        # Collecte d'un épisode
        states, actions, rewards, _, entropies, ep_return = run_episode(env, policy, device)
        reward_history.append(ep_return)
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0
        entropy_history.append(avg_entropy)

        # Conversion en tenseurs
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
        returns_t = compute_returns(rewards, cfg.gamma).to(device)

        # Avantages = retours - baseline (critique)
        values_t = value(states_t)
        advantages = returns_t - values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # Recalcul des log-probabilités avec la politique courante
        logits = policy(states_t)
        dist = Categorical(logits=logits)
        logp_t = dist.log_prob(actions_t)
        entropy_t = dist.entropy().mean()

        # Pertes
        policy_loss = -(logp_t * advantages).mean() - current_entropy_coef * entropy_t
        value_loss = 0.5 * (returns_t - values_t).pow(2).mean()

        batch_policy_losses.append(policy_loss)
        batch_value_losses.append(value_loss)

        # Mise à jour tous les batch_episodes épisodes
        if ep % cfg.batch_episodes == 0 or ep == cfg.max_episodes:
            total_loss = torch.stack(batch_policy_losses).mean() + cfg.value_coef * torch.stack(batch_value_losses).mean()

            opt_policy.zero_grad()
            opt_value.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value.parameters()),
                max_norm=cfg.grad_clip,
            )
            opt_policy.step()
            opt_value.step()

            last_loss = total_loss.item()
            batch_policy_losses.clear()
            batch_value_losses.clear()

        # Décroissance du coefficient d'entropie
        current_entropy_coef = max(cfg.entropy_coef_min, current_entropy_coef * cfg.entropy_coef_decay)

        # Moyenne glissante des récompenses
        window = reward_history[-cfg.solved_window:]
        rolling_mean = float(np.mean(window))

        # Affichage tous les 10 épisodes
        if ep % 10 == 0:
            print(
                f"Ep {ep:4d} | retour={ep_return:8.1f} | "
                f"moy({cfg.solved_window})={rolling_mean:7.1f} | "
                f"ent_coef={current_entropy_coef:.4f} | ent={avg_entropy:.3f}"
            )

        # Évaluation périodique + sauvegarde du meilleur modèle
        if ep % cfg.eval_every == 0:
            policy.eval()
            avg_eval = evaluate(cfg, policy, device)
            policy.train()
            print(f"[EVAL] Ep {ep:4d} | retour moyen ({cfg.eval_episodes} ep) = {avg_eval:.1f}")

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
                print(f"[SAVE] Meilleur modèle sauvegardé : {save_path}")

        # Arrêt anticipé si l'environnement est résolu
        if rolling_mean >= cfg.solved_mean_reward and ep >= cfg.solved_window:
            print(f"[DONE] Résolu ! moy_glissante={rolling_mean:.1f} >= {cfg.solved_mean_reward}")
            break

    env.close()

    # Résumé final
    training_time = time.time() - t0
    final_mean = float(np.mean(reward_history[-cfg.solved_window:]))
    final_std = float(np.std(reward_history[-cfg.solved_window:]))
    solved = final_mean >= cfg.solved_mean_reward

    print("\n" + "=" * 80)
    print("RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("=" * 80)
    print(f"Durée :               {training_time:.1f}s ({training_time / 60:.1f} min)")
    print(f"Épisodes :            {len(reward_history)}")
    print(f"Meilleur eval :       {best_eval:.1f}")
    print(f"Moyenne finale :      {final_mean:.1f} +/- {final_std:.1f}")
    print(f"Résolu :              {'OUI' if solved else 'NON'}")
    print(f"Checkpoint :          {save_path}")
    print("=" * 80 + "\n")

    # Visualisation
    history = {"episode_rewards": reward_history, "episode_entropies": entropy_history}
    plot_performance(history, save_path=cfg.plot_name)

    return history


# =====================================================================
#  Chargement et test d'une politique entraînée
# =====================================================================

def load_policy(cfg: Config) -> PolicyNet:
    """
    Charge un réseau de politique depuis un checkpoint sauvegardé.

    Args:
        cfg: Configuration (pour le chemin et les dimensions).

    Returns:
        Réseau de politique en mode évaluation.

    Raises:
        FileNotFoundError: Si aucun checkpoint n'est trouvé.
    """
    ckpt_path = os.path.join(cfg.save_dir, cfg.save_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Aucun checkpoint trouvé : {ckpt_path}")

    env = gym.make(cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    policy = PolicyNet(obs_dim, act_dim, cfg.hidden_size)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


def test(cfg: Config, num_episodes: int = 10, render: bool = False, device: torch.device = torch.device("cpu")) -> Dict | None:
    """
    Teste la politique entraînée sur plusieurs épisodes (action déterministe).

    Args:
        cfg: Configuration.
        num_episodes: Nombre d'épisodes de test.
        render: Afficher le rendu visuel ou non.
        device: Device torch.

    Returns:
        Dictionnaire de statistiques (moyenne, écart-type, min, max, retours),
        ou None si le checkpoint est introuvable.
    """
    print(f"\n{'=' * 60}")
    print(f"Test de la politique sur {num_episodes} épisodes")
    print(f"{'=' * 60}\n")

    try:
        policy = load_policy(cfg).to(device)
        print(f"Politique chargée : {os.path.join(cfg.save_dir, cfg.save_name)}\n")
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        return None

    render_mode = "human" if render else None
    env = gym.make(cfg.env_id, render_mode=render_mode)
    returns = []

    with torch.no_grad():
        for ep in range(1, num_episodes + 1):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            steps = 0

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = torch.argmax(Categorical(logits=policy(obs_t)).probs, dim=-1).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_return += reward
                steps += 1

            returns.append(ep_return)
            print(f"Épisode {ep:2d}/{num_episodes} | Retour : {ep_return:7.2f} | Pas : {steps:3d}")

    env.close()

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))

    print(f"\n{'=' * 60}")
    print(f"Résultats :")
    print(f"  Moyenne : {mean_ret:7.2f} +/- {std_ret:.2f}")
    print(f"  Min :     {np.min(returns):7.2f}")
    print(f"  Max :     {np.max(returns):7.2f}")
    print(f"  Succès :  {sum(1 for r in returns if r >= 200)}/{num_episodes} (>= 200)")
    print(f"{'=' * 60}\n")

    return {"mean": mean_ret, "std": std_ret, "min": float(np.min(returns)), "max": float(np.max(returns)), "all_returns": returns}


def play(cfg: Config, num_episodes: int = 5, device: torch.device = torch.device("cpu")):
    """
    Lance la politique entraînée avec rendu visuel (mode interactif).

    Args:
        cfg: Configuration.
        num_episodes: Nombre d'épisodes à jouer.
        device: Device torch.
    """
    policy = load_policy(cfg).to(device)
    env = gym.make(cfg.env_id, render_mode="human")

    with torch.no_grad():
        for ep in range(1, num_episodes + 1):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = torch.argmax(Categorical(logits=policy(obs_t)).probs, dim=-1).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_return += reward

            print(f"Épisode {ep}/{num_episodes} | Retour : {ep_return:.1f}")

    env.close()


# =====================================================================
#  Visualisation
# =====================================================================

def plot_performance(history: Dict, save_path: str = "training_performance.png"):
    """
    Génère une figure à 4 sous-graphiques pour analyser l'entraînement :
      1. Évolution des récompenses (score brut + moyenne mobile)
      2. Évolution de l'entropie (exploration)
      3. Distribution des scores (histogramme)
      4. Taux de succès (fenêtre glissante)

    Args:
        history: Dictionnaire avec les clés 'episode_rewards' et 'episode_entropies'.
        save_path: Chemin de sauvegarde de l'image.

    Returns:
        Figure matplotlib.
    """
    episode_rewards = history["episode_rewards"]
    episode_entropies = history.get("episode_entropies", [])
    episodes = np.arange(1, len(episode_rewards) + 1)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # --- Haut-gauche : Récompenses ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episodes, episode_rewards, color="lightblue", alpha=0.6, linewidth=0.8, label="Score brut")
    if len(episode_rewards) >= 100:
        moving_avg = np.convolve(episode_rewards, np.ones(100) / 100, mode="valid")
        ax1.plot(np.arange(100, len(episode_rewards) + 1), moving_avg, color="orange", linewidth=2.5, label="Moyenne mobile (100 ep.)")
    ax1.axhline(y=200, color="green", linestyle="--", linewidth=2, label="Objectif (200)", alpha=0.8)
    ax1.set_xlabel("Épisodes", fontweight="bold")
    ax1.set_ylabel("Score", fontweight="bold")
    ax1.set_title("Évolution des Récompenses", fontsize=14, fontweight="bold", pad=15)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # --- Haut-droite : Entropie ---
    ax2 = fig.add_subplot(gs[0, 1])
    if episode_entropies:
        ax2.plot(episodes, episode_entropies, color="purple", linewidth=2, marker="o", markersize=2, alpha=0.7)
        ax2.annotate(f"Début: {episode_entropies[0]:.3f}", xy=(1, episode_entropies[0]), fontsize=9, color="purple")
        ax2.annotate(f"Fin: {episode_entropies[-1]:.3f}", xy=(len(episode_entropies), episode_entropies[-1]), fontsize=9, color="purple")
    else:
        ax2.text(0.5, 0.5, "Données non disponibles", ha="center", va="center", fontsize=12, transform=ax2.transAxes)
    ax2.set_xlabel("Épisodes", fontweight="bold")
    ax2.set_ylabel("Entropie moyenne", fontweight="bold")
    ax2.set_title("Évolution de l'Entropie (Exploration)", fontsize=14, fontweight="bold", pad=15)
    ax2.grid(True, alpha=0.3)

    # --- Bas-gauche : Distribution des scores ---
    ax3 = fig.add_subplot(gs[1, 0])
    n, bins, patches = ax3.hist(episode_rewards, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    mean_reward = np.mean(episode_rewards)
    ax3.axvline(x=mean_reward, color="red", linestyle="--", linewidth=2.5, label=f"Moyenne: {mean_reward:.1f}")
    for i, patch in enumerate(patches):
        if bins[i] >= 200:
            patch.set_facecolor("lightgreen")
            patch.set_alpha(0.8)
    ax3.set_xlabel("Score", fontweight="bold")
    ax3.set_ylabel("Fréquence", fontweight="bold")
    ax3.set_title("Distribution des Scores", fontsize=14, fontweight="bold", pad=15)
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3, axis="y")

    # --- Bas-droite : Taux de succès ---
    ax4 = fig.add_subplot(gs[1, 1])
    window_size = 50
    if len(episode_rewards) >= window_size:
        success_rates, episodes_success = [], []
        for i in range(window_size, len(episode_rewards) + 1):
            rate = sum(1 for r in episode_rewards[i - window_size : i] if r >= 200) / window_size * 100
            success_rates.append(rate)
            episodes_success.append(i)
        ax4.plot(episodes_success, success_rates, color="green", linewidth=2.5, alpha=0.8)
        ax4.fill_between(episodes_success, success_rates, alpha=0.3, color="green")
        ax4.axhline(y=100, color="gold", linestyle="--", linewidth=2, label="100%", alpha=0.7)
        ax4.set_ylim(-5, 105)
        ax4.legend(loc="lower right")
        final_rate = success_rates[-1]
        ax4.annotate(
            f"Final: {final_rate:.1f}%",
            xy=(episodes_success[-1], final_rate),
            xytext=(episodes_success[-1] - 100, final_rate + 10),
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
        )
    else:
        ax4.text(0.5, 0.5, f"Pas assez d'épisodes (min: {window_size})", ha="center", va="center", fontsize=12, transform=ax4.transAxes)
    ax4.set_xlabel("Épisodes", fontweight="bold")
    ax4.set_ylabel("Taux de succès (%)", fontweight="bold")
    ax4.set_title(f"Taux de Succès (Score >= 200, fenêtre {window_size} ep.)", fontsize=14, fontweight="bold", pad=15)
    ax4.grid(True, alpha=0.3)

    # Titre global
    fig.suptitle("Analyse des Performances - REINFORCE sur Lunar Lander", fontsize=18, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nGraphique sauvegardé : {save_path}")
    plt.show()

    return fig


# =====================================================================
#  Point d'entrée
# =====================================================================

if __name__ == "__main__":
    log_filepath, tee_logger = setup_logging(log_dir="logs", experiment_name="reinforce")
    sys.stdout = tee_logger

    try:
        cfg = Config()
        log_config(cfg)

        # Décommenter le mode souhaité :
        # train(cfg)
        # test(cfg, num_episodes=10, render=False)
        test(cfg, num_episodes=5, render=True)
        # play(cfg, num_episodes=5)

    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur (Ctrl+C).")
    except Exception as e:
        print(f"\nErreur : {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = tee_logger.terminal
        tee_logger.close()
        print(f"Log sauvegardé : {log_filepath}")
