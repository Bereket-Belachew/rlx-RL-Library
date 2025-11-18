import torch 
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    A simple MLP (Multi-Layer Perceptron) Actor-Critic network.
    
    - The "Actor" (policy) decides *what action to take*.
      (e.g., "I'm 70% sure I should go left")
      
    - The "Critic" (value) *estimates the total future reward*
      from the current state.
      (e.g., "This state is worth about +35 future points")
    """

    def __init__(self,obs_shape: int, action_dim: int):
        super(ActorCritic,self).__init__()

       # --- Actor Network ---
        self.actor_net = nn.Sequential(
            nn.Linear(obs_shape,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,action_dim)
        ) 
        # --- Critic Network ---
        self.critic_net= nn.Sequential(
            nn.Linear(obs_shape,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1) #our critic only outputs 1 value, the estimated total future reward
        )

    def get_action(self, obs: torch.Tensor )->Categorical:
          """Gets the action distribution from the actor network."""
          # The actor network returns "logits".
          logits = self.actor_net(obs)
          return Categorical(logits=logits)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
          """Gets the estimated state-value from the critic network."""
          return self.critic_net(obs)
