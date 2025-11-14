"""Environment management utilities for RLx."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import gymnasium as gym


@dataclass
class EnvStep:
    """Container returned by :meth:`EnvManager.step`."""

    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class EnvManager:
    """Thin wrapper around Gymnasium environments.

    The manager hides direct interactions with :func:`gymnasium.make` so agents
    can depend on a simple, testable interface.
    """

    def __init__(self, env_id: str, **env_kwargs: Any) -> None:
        self.env_id = env_id
        self.env_kwargs = env_kwargs
        self._env = gym.make(env_id, **env_kwargs)

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[Any, dict]:
        """Reset the underlying environment and return the initial observation."""
        return self._env.reset(*args, **kwargs)

    def step(self, action: Any) -> EnvStep:
        """Forward the action to the wrapped environment."""
        observation, reward, terminated, truncated, info = self._env.step(action)
        return EnvStep(observation, float(reward), bool(terminated), bool(truncated), info)

    def close(self) -> None:
        """Close the underlying environment."""
        self._env.close()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

