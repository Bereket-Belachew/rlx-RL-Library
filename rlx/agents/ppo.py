"""Proximal Policy Optimization agent implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from rlx.agents.base_agent import BaseAgent
from rlx.env.manager import EnvManager


def _to_tensor(observation: Any, device: torch.device) -> torch.Tensor:
    """Convert an observation into a float32 tensor on the desired device."""
    np_obs = np.asarray(observation, dtype=np.float32)
    tensor = torch.from_numpy(np_obs).to(device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ActorCritic(nn.Module):
    """Simple shared-body actor-critic network."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(x)
        logits = self.actor_head(x)
        value = self.critic_head(x)
        return logits, value.squeeze(-1)


class PPOAgent(BaseAgent):
    """First concrete implementation that satisfies :class:`BaseAgent`."""

    def __init__(
        self,
        env: EnvManager,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
    ) -> None:
        super().__init__()
        self.env = env
        obs_shape = env.observation_space.shape
        if obs_shape is None:
            raise ValueError("Observation space must define a shape.")
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac_network = ActorCritic(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac_network.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio

    def select_action(self, observation: Any) -> Tuple[int, float]:
        """Sample an action and return its log-probability."""
        obs_tensor = _to_tensor(observation, self.device)

        with torch.no_grad():
            logits, _ = self.ac_network(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return int(action.squeeze(0).cpu().item()), float(log_prob.squeeze(0).cpu().item())

    def learn(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("PPOAgent.learn is not implemented yet.")

    def save(self, path: str) -> None:
        torch.save(self.ac_network.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.ac_network.load_state_dict(state)

