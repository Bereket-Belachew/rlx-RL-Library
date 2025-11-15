"""
RLX - A Reinforcement Learning Library

Main package initialization file that exposes the core components.
"""

from rlx.agents.ppo import PPOAgent
from rlx.env.manager import EnvManager
from rlx.train.trainer import Train

__version__ = "0.1.0"

__all__ = [
    "PPOAgent",
    "EnvManager", 
    "Train",
]
