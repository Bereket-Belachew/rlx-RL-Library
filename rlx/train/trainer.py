"""Training loop utilities for RLx."""

from __future__ import annotations

from typing import Optional

from rlx.agents.base_agent import BaseAgent
from rlx.env.manager import EnvManager


class Train:
    """Coordinates the agent-environment interaction loop."""

    def __init__(self, agent: BaseAgent, env: EnvManager) -> None:
        self.agent = agent
        self.env = env

    def run(self, steps: int) -> None:
        """Run the rollout loop for a fixed number of steps."""
        observation, _ = self.env.reset()

        for _ in range(steps):
            action, _ = self.agent.select_action(observation)
            observation, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                observation, _ = self.env.reset()

