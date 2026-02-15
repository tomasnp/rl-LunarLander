"""
A2C (Advantage Actor-Critic) avec GAE pour l'environnement LunarLander-v3.

Algorithme actor-critic synchrone avec :
- Generalized Advantage Estimation (GAE) pour réduire la variance
- Normalisation des observations (Welford en ligne)
- Clipping des récompenses et des gradients
- Décroissance linéaire du coefficient d'entropie
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
    env_id: str = "LunarLander-v3" # ID de l'environnement Gymnasium à utiliser
    seed: int = 42 # Graine aléatoire pour la reproductibilité (numpy, torch, env.reset)

    # Hyperparamètres d'apprentissage
    gamma: float = 0.99 # Facteur d'actualisation : pondère les récompenses futures vs immédiates (proche de 1 = vision long terme)
    gae_lambda: float = 0.95 # Paramètre lambda du GAE : compromis biais/variance dans l'estimation de l'avantage (1 = Monte-Carlo, 0 = TD(0))
    lr_policy: float = 5e-4 # Learning rate de l'acteur (réseau de politique)
    lr_value: float = 1e-3 # Learning rate du critique (réseau de valeur)

    # Entropie (exploration)
    entropy_coef_start: float = 0.05 # Coefficient d'entropie initial : encourage l'exploration en début d'entraînement
    entropy_coef_final: float = 0.005 # Coefficient d'entropie final après décroissance linéaire
    value_coef: float = 0.5 # Poids de la loss du critique dans la loss totale

    # Entraînement
    rollout_steps: int = 2048 # Nombre de pas collectés par rollout avant chaque update
    max_updates: int = 10000 # Nombre maximum d'itérations (rollout + update)
    eval_every: int = 50 # Fréquence d'évaluation (tous les N updates)
    eval_episodes: int = 30 # Nombre d'épisodes par évaluation (pour moyenner)
    hidden_size: int = 512 # Taille des couches cachées des réseaux acteur et critique
    grad_clip: float = 0.5 # Seuil de gradient clipping (norme max) pour stabiliser l'entraînement

    # Stabilité
    normalize_obs: bool = False # Normalisation des observations via Welford en ligne
    reward_clip: float = None # Seuil de clipping des récompenses (None = désactivé)
    weight_decay: float = 0 # Régularisation L2 dans AdamW (0 = désactivé)
    obs_clip: float = 10.0 # Clipping des obs normalisées dans [-clip, +clip] (si normalize_obs=True)

    # Sauvegarde
    save_dir: str = "checkpoints" # Dossier de sauvegarde des modèles
    save_name: str = f"a2c_r_{rollout_steps}_u_{max_updates}_h_{hidden_size}.pt" # Nom du fichier checkpoint
    plot_name: str = f"a2c_r_{rollout_steps}_u_{max_updates}_h_{hidden_size}.png" # Nom du fichier image des courbes

    # Rendu / vidéo
    render_eval_human: bool = False # Affiche le rendu visuel pendant l'évaluation
    record_video: bool = False # Enregistre les épisodes d'évaluation en vidéo
    video_dir: str = "videos_record" # Dossier de destination des vidéos

    # Arrêt anticipé
    solved_mean_reward: float = 200.0 # Seuil de score moyen pour considérer l'env comme résolu
    solved_window: int = 100 # Taille de la fenêtre glissante pour le calcul de la moyenne


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


def setup_logging(log_dir: str = "logs", experiment_name: str = "a2c") -> Tuple[str, TeeLogger]:
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
#  Normalisation des observations
# =====================================================================

class RunningMeanStd:
    """
    Calcule la moyenne et l'écart-type glissants pour normaliser les observations.

    Utilise l'algorithme de Welford en ligne pour la stabilité numérique.
    """

    def __init__(self, shape: tuple, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        """
        Met à jour les statistiques avec un nouveau batch d'observations.

        Args:
            x: Batch d'observations de forme (batch_size, *shape).
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        """Met à jour les statistiques à partir des moments du batch (Welford)."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """
        Normalise une observation et la clippe dans [-clip, +clip].

        Args:
            x: Observation brute.
            clip: Valeur de clipping.

        Returns:
            Observation normalisée et clippée.
        """
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)


# =====================================================================
#  Utilitaires
# =====================================================================

def set_seed(seed: int):
    """Fixe les graines aléatoires pour la reproductibilité (numpy, torch)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminateds: torch.Tensor,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calcule le Generalized Advantage Estimation (GAE).

    Distinction importante entre terminated et truncated :
    - terminated=1 : état véritablement terminal (crash/succès) -> pas de bootstrap
    - truncated (terminated=0) : limite de temps -> on bootstrap avec V(s_next)

    Args:
        rewards: Tenseur [T] des récompenses.
        values: Tenseur [T] des estimations V(s_t).
        terminateds: Tenseur [T] de flags terminaux (1 si terminal, 0 sinon).
        next_value: V(s_{T+1}) pour le bootstrapping.
        gamma: Facteur d'actualisation.
        gae_lambda: Paramètre lambda du GAE.

    Returns:
        Tuple (avantages [T], retours cibles [T] pour le critique).
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    values_ext = torch.cat([values, torch.tensor([next_value], dtype=torch.float32)])
    gae = 0.0

    for t in reversed(range(T)):
        not_terminal = 1.0 - terminateds[t]
        delta = rewards[t] + gamma * not_terminal * values_ext[t + 1] - values[t]
        gae = delta + gamma * gae_lambda * not_terminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# =====================================================================
#  Réseaux de neurones
# =====================================================================

class PolicyNet(nn.Module):
    """
    Réseau de politique (acteur).

    Prend un état en entrée et produit les logits sur les actions.
    Architecture : 2 couches cachées avec Tanh.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les logits (scores non-normalisés) pour chaque action."""
        return self.net(x)


class ValueNet(nn.Module):
    """
    Réseau de valeur (critique).

    Prend un état en entrée et estime la valeur V(s).
    Architecture : 2 couches cachées avec Tanh.
    """

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
        """Retourne la valeur scalaire V(s) pour chaque état du batch."""
        return self.net(x).squeeze(-1)


# =====================================================================
#  Collecte d'un rollout (K pas)
# =====================================================================

def collect_rollout(
    env: gym.Env,
    policy: PolicyNet,
    value: ValueNet,
    rollout_steps: int,
    device: torch.device,
    current_obs: np.ndarray,
    current_done: bool,
    obs_normalizer: RunningMeanStd = None,
    reward_clip: float = None,
    obs_clip: float = 10.0,
) -> Tuple[Dict, np.ndarray, bool, List[float], List[np.ndarray]]:
    """
    Collecte un rollout de taille fixe (gère les frontières d'épisodes).

    Args:
        env: Environnement Gymnasium.
        policy: Réseau de politique.
        value: Réseau de valeur.
        rollout_steps: Nombre de pas à collecter.
        device: Device torch.
        current_obs: Observation courante (continuité avec le rollout précédent).
        current_done: L'état courant est-il terminal ?
        obs_normalizer: Normalisateur d'observations (optionnel).
        reward_clip: Seuil de clipping des récompenses (optionnel).
        obs_clip: Seuil de clipping des observations normalisées.

    Returns:
        Tuple (données du rollout, obs suivante, done suivant,
               retours des épisodes terminés, observations brutes).
    """
    states, actions, rewards, terminateds, values_list = [], [], [], [], []
    episode_returns: List[float] = []
    current_ep_return = 0.0
    raw_observations: List[np.ndarray] = []

    if current_done:
        current_obs, _ = env.reset()
        current_done = False

    for _ in range(rollout_steps):
        raw_observations.append(current_obs.copy())

        # Normalisation de l'observation si activée
        obs_norm = obs_normalizer.normalize(current_obs, clip=obs_clip) if obs_normalizer else current_obs
        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            logits = policy(obs_t)
            action = Categorical(logits=logits).sample().item()
            value_pred = value(obs_t).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if reward_clip is not None:
            reward = np.clip(reward, -reward_clip, reward_clip)

        states.append(obs_norm)
        actions.append(action)
        rewards.append(float(reward))
        terminateds.append(1.0 if terminated else 0.0)
        values_list.append(value_pred)

        current_ep_return += float(reward)

        if done:
            episode_returns.append(current_ep_return)
            current_ep_return = 0.0
            next_obs, _ = env.reset()
            done = False

        current_obs = next_obs
        current_done = done

    rollout_data = {
        "states": np.array(states),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "terminateds": np.array(terminateds, dtype=np.float32),
        "values": np.array(values_list, dtype=np.float32),
    }

    return rollout_data, current_obs, current_done, episode_returns, raw_observations


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
def evaluate(cfg: Config, policy: PolicyNet, device: torch.device, obs_normalizer: RunningMeanStd = None) -> float:
    """
    Évalue la politique de manière déterministe (argmax) sur plusieurs épisodes.

    Args:
        cfg: Configuration.
        policy: Réseau de politique.
        device: Device torch.
        obs_normalizer: Normalisateur d'observations (optionnel).

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
            obs_norm = obs_normalizer.normalize(obs, clip=cfg.obs_clip) if obs_normalizer else obs
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
            action = torch.argmax(Categorical(logits=policy(obs_t)).probs, dim=-1).item()
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
    Boucle principale d'entraînement A2C avec GAE.

    À chaque itération :
    1. Collecte un rollout de `cfg.rollout_steps` pas
    2. Calcule les avantages via GAE
    3. Met à jour l'acteur et le critique
    4. Évalue périodiquement et sauvegarde le meilleur modèle

    Args:
        cfg: Configuration d'entraînement.

    Returns:
        Dictionnaire contenant l'historique (récompenses et entropies).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_config(cfg)
    print(f"[INFO] Device : {device}\n")

    set_seed(cfg.seed)

    # Environnement d'entraînement (sans rendu)
    env = gym.make(cfg.env_id)
    current_obs, _ = env.reset(seed=cfg.seed)
    current_done = False

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Réseaux
    policy = PolicyNet(obs_dim, act_dim, cfg.hidden_size).to(device)
    value = ValueNet(obs_dim, cfg.hidden_size).to(device)

    # Optimiseurs (AdamW avec régularisation L2)
    opt_policy = optim.AdamW(policy.parameters(), lr=cfg.lr_policy, eps=1e-5, weight_decay=cfg.weight_decay)
    opt_value = optim.AdamW(value.parameters(), lr=cfg.lr_value, eps=1e-5, weight_decay=cfg.weight_decay)

    # Normalisateur d'observations
    obs_normalizer = RunningMeanStd(shape=(obs_dim,)) if cfg.normalize_obs else None

    # Sauvegarde
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, cfg.save_name)

    # Historique
    all_episode_returns: List[float] = []
    entropy_history: List[float] = []
    best_eval = -float("inf")

    t0 = time.time()

    for update_idx in range(1, cfg.max_updates + 1):
        # 1. Collecte du rollout
        rollout_data, current_obs, current_done, episode_returns, raw_obs = collect_rollout(
            env, policy, value, cfg.rollout_steps, device, current_obs, current_done,
            obs_normalizer=obs_normalizer, reward_clip=cfg.reward_clip, obs_clip=cfg.obs_clip,
        )

        # Mise à jour du normalisateur
        if obs_normalizer is not None and len(raw_obs) > 0:
            obs_normalizer.update(np.array(raw_obs))

        all_episode_returns.extend(episode_returns)

        # Conversion en tenseurs
        states_t = torch.tensor(rollout_data["states"], dtype=torch.float32, device=device)
        actions_t = torch.tensor(rollout_data["actions"], dtype=torch.int64, device=device)
        rewards_t = torch.tensor(rollout_data["rewards"], dtype=torch.float32, device=device)
        terminateds_t = torch.tensor(rollout_data["terminateds"], dtype=torch.float32, device=device)
        old_values_t = torch.tensor(rollout_data["values"], dtype=torch.float32, device=device)

        # 2. Bootstrap de la valeur suivante pour le GAE
        with torch.no_grad():
            if rollout_data["terminateds"][-1] == 1.0:
                next_value = 0.0
            else:
                obs_norm = obs_normalizer.normalize(current_obs, clip=cfg.obs_clip) if obs_normalizer else current_obs
                next_obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
                next_value = value(next_obs_t).item()

        # 3. Calcul des avantages (GAE) et normalisation
        advantages_t, returns_t = compute_gae(rewards_t, old_values_t, terminateds_t, next_value, cfg.gamma, cfg.gae_lambda)
        advantages_t = advantages_t.to(device)
        returns_t = returns_t.to(device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

        # 4. Calcul des pertes
        logits = policy(states_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        # Décroissance linéaire du coefficient d'entropie
        progress = update_idx / cfg.max_updates
        entropy_coef = max(cfg.entropy_coef_final, cfg.entropy_coef_start * (1.0 - progress))

        policy_loss = -(log_probs * advantages_t.detach()).mean() - entropy_coef * entropy
        value_loss = nn.SmoothL1Loss()(value(states_t), returns_t.detach())
        loss = policy_loss + cfg.value_coef * value_loss

        # 5. Optimisation
        opt_policy.zero_grad()
        opt_value.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip)
        nn.utils.clip_grad_norm_(value.parameters(), max_norm=cfg.grad_clip)
        opt_policy.step()
        opt_value.step()

        entropy_history.append(entropy.item())

        # Affichage tous les 10 updates
        if update_idx % 10 == 0:
            recent = all_episode_returns[-100:] if all_episode_returns else [0.0]
            mean_ret = float(np.mean(recent))
            print(
                f"Update {update_idx:4d} | "
                f"retour={mean_ret:7.1f} (n={len(all_episode_returns)}) | "
                f"loss={loss.item():.3f} | "
                f"entropie={entropy.item():.3f} (coef={entropy_coef:.4f})"
            )

        # Évaluation périodique + sauvegarde
        if update_idx % cfg.eval_every == 0:
            policy.eval()
            avg_eval = evaluate(cfg, policy, device, obs_normalizer)
            policy.train()
            print(f"[EVAL] Update {update_idx:4d} | retour moyen ({cfg.eval_episodes} ep) = {avg_eval:.1f}")

            if avg_eval > best_eval:
                best_eval = avg_eval
                checkpoint = {
                    "policy_state_dict": policy.state_dict(),
                    "value_state_dict": value.state_dict(),
                    "cfg": cfg.__dict__,
                    "best_eval": best_eval,
                    "update": update_idx,
                }
                if obs_normalizer is not None:
                    checkpoint["obs_normalizer"] = {
                        "mean": obs_normalizer.mean,
                        "var": obs_normalizer.var,
                        "count": obs_normalizer.count,
                    }
                torch.save(checkpoint, save_path)
                print(f"[SAVE] Meilleur modèle sauvegardé : {save_path}")

        # Arrêt anticipé si résolu
        if len(all_episode_returns) >= cfg.solved_window:
            rolling_mean = float(np.mean(all_episode_returns[-cfg.solved_window:]))
            if rolling_mean >= cfg.solved_mean_reward:
                print(f"[DONE] Résolu ! moy_glissante={rolling_mean:.1f} >= {cfg.solved_mean_reward}")
                break

    env.close()

    # Résumé final
    training_time = time.time() - t0
    final_mean = float(np.mean(all_episode_returns[-100:])) if len(all_episode_returns) >= 100 else float(np.mean(all_episode_returns))
    final_std = float(np.std(all_episode_returns[-100:])) if len(all_episode_returns) >= 100 else float(np.std(all_episode_returns))

    print("\n" + "=" * 80)
    print("RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("=" * 80)
    print(f"Durée :               {training_time:.1f}s ({training_time / 60:.1f} min)")
    print(f"Updates :             {update_idx}")
    print(f"Épisodes :            {len(all_episode_returns)}")
    print(f"Meilleur eval :       {best_eval:.1f}")
    print(f"Moyenne finale :      {final_mean:.1f} +/- {final_std:.1f}")
    print(f"Résolu :              {'OUI' if best_eval >= cfg.solved_mean_reward else 'NON'}")
    print(f"Checkpoint :          {save_path}")
    print("=" * 80 + "\n")

    # Visualisation
    history = {"episode_rewards": all_episode_returns, "episode_entropies": entropy_history}
    plot_performance(history, save_path=cfg.plot_name)

    return history


# =====================================================================
#  Chargement et test d'une politique entraînée
# =====================================================================

def load_policy(cfg: Config) -> Tuple[PolicyNet, RunningMeanStd | None]:
    """
    Charge un réseau de politique et son normalisateur depuis un checkpoint.

    Args:
        cfg: Configuration (pour le chemin et les dimensions).

    Returns:
        Tuple (réseau de politique en mode évaluation, normalisateur ou None).

    Raises:
        FileNotFoundError: Si aucun checkpoint n'est trouvé.
    """
    ckpt_path = os.path.join(cfg.save_dir, cfg.save_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Aucun checkpoint trouvé : {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    env = gym.make(cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    # Utiliser le hidden_size du checkpoint si disponible
    hidden_size = ckpt.get("cfg", {}).get("hidden_size", cfg.hidden_size)

    policy = PolicyNet(obs_dim, act_dim, hidden_size)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    # Charger le normalisateur s'il existe dans le checkpoint
    obs_normalizer = None
    if "obs_normalizer" in ckpt:
        obs_normalizer = RunningMeanStd(shape=(obs_dim,))
        obs_normalizer.mean = ckpt["obs_normalizer"]["mean"]
        obs_normalizer.var = ckpt["obs_normalizer"]["var"]
        obs_normalizer.count = ckpt["obs_normalizer"]["count"]

    return policy, obs_normalizer


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
        policy, obs_normalizer = load_policy(cfg)
        policy = policy.to(device)
        print(f"Politique chargée : {os.path.join(cfg.save_dir, cfg.save_name)}\n")
    except (FileNotFoundError, RuntimeError) as e:
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
                obs_norm = obs_normalizer.normalize(obs, clip=cfg.obs_clip) if obs_normalizer else obs
                obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
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
    policy, obs_normalizer = load_policy(cfg)
    policy = policy.to(device)
    env = gym.make(cfg.env_id, render_mode="human")

    with torch.no_grad():
        for ep in range(1, num_episodes + 1):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0

            while not done:
                obs_norm = obs_normalizer.normalize(obs, clip=cfg.obs_clip) if obs_normalizer else obs
                obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
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
        # L'entropie est par update (pas par épisode) dans A2C
        x_axis = np.arange(1, len(episode_entropies) + 1)
        ax2.plot(x_axis, episode_entropies, color="purple", linewidth=2, marker="o", markersize=2, alpha=0.7)
        ax2.annotate(f"Début: {episode_entropies[0]:.3f}", xy=(1, episode_entropies[0]), fontsize=9, color="purple")
        if len(x_axis) > 50:
            ax2.annotate(f"Fin: {episode_entropies[-1]:.3f}", xy=(x_axis[-1], episode_entropies[-1]), fontsize=9, color="purple")
    else:
        ax2.text(0.5, 0.5, "Données non disponibles", ha="center", va="center", fontsize=12, transform=ax2.transAxes)
    ax2.set_xlabel("Updates", fontweight="bold")
    ax2.set_ylabel("Entropie moyenne", fontweight="bold")
    ax2.set_title("Évolution de l'Entropie par Update", fontsize=14, fontweight="bold", pad=15)
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
    fig.suptitle("Analyse des Performances - A2C avec GAE sur Lunar Lander", fontsize=18, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nGraphique sauvegardé : {save_path}")
    plt.show()

    return fig


# =====================================================================
#  Point d'entrée
# =====================================================================

if __name__ == "__main__":
    log_filepath, tee_logger = setup_logging(log_dir="logs", experiment_name="a2c_gae")
    sys.stdout = tee_logger

    try:
        cfg = Config()

        # Décommenter le mode souhaité :
        train(cfg)
        # test(cfg, num_episodes=10, render=False)
        # test(cfg, num_episodes=5, render=True)
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
