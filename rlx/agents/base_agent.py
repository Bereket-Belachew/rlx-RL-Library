import abc 
import numpy as np 
from typing import Any, Tuple

Action = int | np.ndarray
Observation = np.ndarray

class BaseAgent(abc.ABC):
    """
    Abstract Base Class (ABC) for all Reinforcement Learning Agents.

    This class is our "Agent Contract". It defines the *minimal*
    set of functions that any agent (like PPOAgent) must
    implement to be used by our 'Train' class.
    
    The 'Train' class will be type-hinted to take a 'BaseAgent',
    so it knows it can safely call '.select_action()' and '.learn()'.
    """

    @abc.abstractmethod
    def select_action(self,observation: Observation)-> Tuple[Action,Any]:
        """
        Selects an action based on the current observation.
        
        ---
        [DEV_NOTE] How this function will be called (by the Trainer):
        
        # Inside the Trainer's 'run' loop:
        # 'obs' will come from the 'env.step()' or 'env.reset()'
        action, log_probs = self.agent.select_action(obs)
        ---
        
        Args:
            observation: The current state of the environment.

        Returns:
            A tuple containing:
            - The 'Action' to take (e.g., 0 for left, 1 for right).
            - Any extra data (e.g., for PPO, we'll return the
              log-probability of that action, which we need for 'learn').
        """

        pass
    @abc.abstractmethod
    def save(self,path: str)->None:
        """
        Saves the agent's model weights to a file.
        
        ---
        [DEV_NOTE] How a user will call this function:
        
        # After training:
        agent.save("my_ppo_model.pth")
        ---
        """
        pass 
    @abc.abstractmethod
    def load(self,path:str)->None:
        """
        Loads the agent's model weights from a file.
        
        ---
        [DEV_NOTE] How a user will call this function:
        
        # For evaluation or to continue training:
        agent = Agent("ppo", env=env)
        agent.load("my_ppo_model.pth")
        ---
        """
        pass