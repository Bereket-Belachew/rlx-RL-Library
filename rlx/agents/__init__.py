"""
Agent module initialization.

Exports the PPOAgent and BaseAgent classes.
"""

from rlx.agents.ppo import PPOAgent
from rlx.agents.base_agent import BaseAgent

__all__ = ["PPOAgent", "BaseAgent"]
