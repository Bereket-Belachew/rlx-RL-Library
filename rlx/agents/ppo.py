import numpy as np
from typing import Tuple, Any
from torch.distributions import Categorical
import torch 
from torch import nn
from rlx.agents.base_agent import BaseAgent, Observation, Action
from rlx.env.manager import EnvManager

# --- The Agent's "Brain" ---
# This is a separate PyTorch module. It's good practice
# to define the network separately from the agent logic.
#
# This "brain" is the internal machinery. The user of our
# library will never see or write this code. Our PPOAgent
# will build this automatically.

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
            nn.Tanh(64,64),
            nn.Linear(64,64),
            nn.Tanh(64,64),
            nn.Linear(64,action_dim)
        ) 
        # --- Critic Network ---
        self.critic_net= nn.Sequential(
            nn.Linear(obs_shape,64),
            nn.Tanh(64,64),
            nn.Linear(64,64),
            nn.Tanh(64,64),
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

# --- The PPO Agent Class ---
class PPOAgent(BaseAgent):
   
   def __init__(self,env: EnvManager, lr: float= 3e-4):
      """
        Initializes the PPO Agent.
        
        ---
        [DEV_NOTE] How a user will call this function:
        
        env = Env("CartPole-v1")
        
        # This __init__ method is called here:
        agent = Agent("ppo", env=env, lr=0.0003)
        ---
        """
      super().__init__()
      
      self.env = env
      self.lr = lr
      

      # --- KEY LINES ---
        # Here is where we "read the dashboard" of the env.
        # This is why the 'env' object is required.
      obs_shape =env.observation_space.shape[0]
      action_dim=env.action_space.n

      # Create the "brain": this is the brian of PP=
      self.ac_network=ActorCritic(obs_shape,action_dim)

      # Create the optimizer
      # This will update the network's weights during .learn(), using the learning rate lr identfied with the PPOAgent
      self.optimezer = torch.optim.Adam(self.ac_network.parameters(),lr=self.lr)
      print(f"✅ [PPOAgent] Initialized.")
      print(f"  - Obs Shape: {obs_shape}, Action Dim: {action_dim}")
      print(f"  - Learning Rate: {self.lr}")
      
      
   def select_action(self, observation:Observation)-> Tuple[Action,any]:
         """
        Selects an action based on the current observation.
        
        This implements the "contract" from BaseAgent.
        
        ---
        [DEV_NOTE] How this function will be called (by the Trainer):
        
        # Inside the Trainer's 'run' loop:
        # 'obs' will be a single numpy array (e.g., [0.1, -0.2, 0.0, 0.3])
        action, log_prob = self.agent.select_action(obs)
        ---
        """
        # 1. Convert data: numpy -> torch.Tensor
        # We also add a "batch dimension" (unsqueeze(0)) because
        # PyTorch networks expect batches, not single samples.
        # [4] -> [1, 4]
        # (We will implement this logic in the next step)
         obs_tensor = torch.tensor(observation,dtype=torch.float32).unsqueeze(0)#Add a batch dimension → shape becomes (1, obs_dim).
         
         # Put the network in "evaluation" mode (disables dropout, etc.)
        # and tell PyTorch not to calculate gradients here. 
        #We are not learning so calculating gradient takes a lot of VRam
         self.ac_network.eval()
         with torch.no_grad():
            
            # 2. Think (Act): Get the "loaded dice"
            action_dist = self.ac_network.get_action(obs_tensor)

            # 3. Think (Criticize): Get the "fortune-teller's" prediction
            state_value = self.ac_network.get_value(obs_tensor)

            # 4. Act (Sample): "Roll the dice" to get an action
            action = action_dist.sample()
            # 5. Get Log-Prob: Ask the dice "what was the log-probability
        #    of the action we just sampled?" We need this for the
        #    'learn' method.
            log_prob = action_dist.log_prob(action)
            # 6. Return action and extra data
        # We use .item() to convert the single-item PyTorch tensors
        # back into plain Python numbers for the env.
        #
        # We return a tuple: (action, (log_prob, value))
        # The 'Trainer' will be responsible for storing this "extra data".
         extras = (log_prob.item(), state_value.item())
        
         return action.item(), extras
      


        
   def learn(self, *args, **kwargs) -> dict:
        #[Dev Note]: understand args and kwargs when you work later on this
        """
        Triggers the agent to update its policy.
        
        This implements the "contract" from BaseAgent.
        """
        # (This is where the complex PPO logic will go)
        raise NotImplementedError
      
   def save(self,path:str)->None:
         """Saves the agent's model weights to a file."""
         # PyTorch models are saved using 'state_dict'
         torch.save(self.ac_network.state_dict(), path)
         print(f"✅ [PPOAgent] Model saved to {path}")

   def load(self, path:str)->None:
        """Loads the agent's model weights from a file."""
        # We load the weights into the network
        self.ac_network.load_state_dict(torch.load(path))
        print(f"✅ [PPOAgent] Model loaded from {path}")
         

