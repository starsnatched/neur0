from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


def make_env(env_id: str = "LunarLanderContinuous-v3", seed: int | None = None):
    import gymnasium as gym  # type: ignore
    env = gym.make(env_id, render_mode="rgb_array")
    if seed is not None:
        env.reset(seed=seed)
    return env


def heuristic_action(obs: np.ndarray) -> np.ndarray:
    x, y, vx, vy, theta, vtheta, c1, c2 = obs
    target_y = 0.6
    kp_y = 0.8
    kd_y = 1.0
    u_y = kp_y * (target_y - y) - kd_y * vy
    kp_theta = 1.5
    kd_theta = 0.6
    target_theta = -0.4 * vx
    u_theta = kp_theta * (target_theta - theta) - kd_theta * vtheta
    main = np.clip(u_y, -1.0, 1.0)
    side = np.clip(u_theta, -1.0, 1.0)
    return np.array([main, side], dtype=np.float32)


@dataclass
class EpisodeData:
    X: np.ndarray
    Y: np.ndarray


def collect_episodes(episodes: int, seq_len: int, encoder, context_factory, topk: int = 5, seed: int = 0, max_steps: int = 512, env_id: str = "LunarLanderContinuous-v3") -> EpisodeData:
    rng = np.random.default_rng(seed)
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    env = make_env(env_id, seed)
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        ctx = context_factory()
        frames = []
        feats = []
        actions = []
        for t in range(max_steps):
            frame = env.render()
            if frame is None:
                break
            f = encoder.encode(frame, topk=topk)
            z = ctx.augment(f)
            a = heuristic_action(obs)
            feats.append(z)
            actions.append(a)
            obs, reward, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        if len(feats) < seq_len:
            continue
        F = np.stack(feats, axis=0).astype(np.float32)
        A = np.stack(actions, axis=0).astype(np.float32)
        for start in range(0, len(feats) - seq_len):
            X_list.append(F[start:start + seq_len])
            Y_list.append(A[start + seq_len - 1])
    env.close()
    if not X_list:
        raise RuntimeError("no data collected; increase episodes or max_steps")
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return EpisodeData(X=X, Y=Y)

