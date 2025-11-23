import torch 
import torch.nn as nn
from torch.distributions import Categorical, Normal #[New] for continuous action spaces 

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
    

# [NEW CLASS]an MLP for a continuous action space
class ContinuousActorCritic(nn.Module):
    """
    A "Steering Wheel" brain for Continuous action spaces. Could also be for drones or other continuous env
    """
    def __init__(self, obs_shape: int,action_dim: int):
         super().__init__()


        # Critic (Value) - Same as Discreet space
         self.critic_net=nn.Sequential(
              nn.Linear(obs_shape,64),
              nn.Tanh(),
              nn.Linear(64,64),
              nn.Tanh(),
              nn.Linear(64,1)
         )

         self.actor_body=nn.Sequential(
              nn.Linear(obs_shape,64),
              nn.Tanh(),
              nn.Linear(64,64),
              nn.Tanh(),
              # we don't have nn.Linear(64,action_dim) here because it's continuous, we need mean first
         )
        #Head 1: Mean(Mu)- the target action
         self.actor_mean= nn.Linear(64,action_dim)

        # HEAD 2: Log Std (Sigma) - The "noise/exploration"
        # We learn this as a parameter
         self.actor_logstd = nn.Parameter(torch.zeros(1,action_dim))

         # We add -> torch.Tensor to tell the user exactly what comes out
    def get_value(self,x:torch.Tensor)->torch.Tensor:
            return self.critic_net(x) 
    
    def get_action(self,x:torch.Tensor)-> Normal:
         body_out = self.actor_body(x)
         mean = self.actor_mean(body_out)

         # Calculate StdDev (must be positive, so we use exp)
         log_std = self.actor_logstd.expand_as(mean)
         std = torch.exp(log_std)

         # Create a Normal (Bell Curve) distribution
         return Normal (mean,std)
              
