"""Abstract base class for RLx agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple


class BaseAgent(ABC):
    """Defines the minimal surface area expected from any RLx agent."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def select_action(self, observation: Any) -> Tuple[int, float]:
        """Return an action (and metadata) given the latest observation."""

    @abstractmethod
    def learn(self, *args: Any, **kwargs: Any) -> None:
        """Run one gradient update step."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the agent to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore the agent from disk."""

