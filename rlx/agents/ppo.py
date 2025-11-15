import numpy as np
from typing import Tuple, Any
from torch.distributions import Categorical
import torch 
from torch import nn
from rlx.agents.base_agent import BaseAgent, Observation, Action
from rlx.env.manager import EnvManager
from rlx.utils.buffer import RolloutBuffer

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

# --- The PPO Agent Class ---
class PPOAgent(BaseAgent):
   
   def __init__(self,env: EnvManager, lr: float= 3e-4,
                n_steps:int=2048,
                gamma:float=0.99,
                gae_lambda:float=0.95,
                n_epochs:int =10,
                clip_coef:float=0.2,
                entropy_coef:float=0.0,
                value_coef:float=0.5):
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
      self.n_steps = n_steps
      self.gamma = gamma
      self.gae_lambda = gae_lambda
      self.n_epochs = n_epochs
      self.clip_coef = clip_coef
      self.entropy_coef = entropy_coef
      self.value_coef = value_coef
      

      # --- KEY LINES ---
        # Here is where we "read the dashboard" of the env.
        # This is why the 'env' object is required.
      #[FIX] Get the tuple (e.g., (4,)) for the self.buffer
      obs_shape_tuple = env.observation_space.shape
      #obs_shape_int for actor critic, this take int, the buffer takes tuple version of observation
      obs_shape_int =env.observation_space.shape[0]
      action_dim=env.action_space.n

      # Create the "brain": this is the brian of PP=
      
      self.ac_network=ActorCritic(obs_shape_int,action_dim)

      # Create the optimizer
      # This will update the network's weights during .learn(), using the learning rate lr identfied with the PPOAgent
      self.optimezer = torch.optim.Adam(self.ac_network.parameters(),lr=self.lr)

        # [NEW] Build the Buffer
        # The Agent will "own" its buffer, which is a cleaner
        # design than the Trainer owning it.

      #passed obs_shaped_tuple to buffer
      self.buffer = RolloutBuffer(n_steps,obs_shape_tuple,action_dim)

      print(f"✅ [PPOAgent] Initialized.")
      print(f"  - Obs Shape: {obs_shape_int}, Action Dim: {action_dim}")
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
      
   def learn(self, batch:dict[str,torch.Tensor]) -> dict:
        #[Dev Note]: understand args and kwargs when you work later on this
        """
        Triggers the agent to update its policy.
        
        This implements the "contract" from BaseAgent.
        """
       
        # 1. Get the pre-calculated advantages and returns
        obs=batch['observations']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        advantages =batch['advantages']
        returns = batch['returns']

        # 2. Normalize advantages (a standard PPO trick for stability)
        advantages = (advantages- advantages.mean())/(advantages.std()+ 1e-8)
        # 3. Put network in "training" mode
        self.ac_network.train()

        # 4. The PPO Update Loop
        # We loop over the *same* batch of data multiple times
        for _ in range (self.n_epochs):
             
             # --- Re-evaluate the batch data ---
            # Get *new* log_probs, values, and entropy from the
            # "brain" (which is being updated every loop)

            action_dist = self.ac_network.get_action(obs)
            new_values= self.ac_network.get_value(obs)
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy()

            # --- Calculate the Actor (Policy) Loss ---
            
            # The "ratio": pi_new / pi_old
            # In log-space, this is exp(log_pi_new - log_pi_old)

            log_ratio = new_log_probs- old_log_probs
            ratio = torch.exp(log_ratio)

            # The "clipped" surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio,1-self.clip_coef,1+self.clip_coef)* advantages

            # PPO loss is the *minimum* of these two, averaged
            # We take the negative because optimizers minimize
            actor_loss = -torch.min(surr1,surr2).mean()
            
            # --- Calculate the Critic (Value) Loss ---
            # This is a simple Mean Squared Error (MSE)
            # (predicted_value - actual_return)^2
            value_loss = (new_values - returns).pow(2).mean()

            # --- Calculate the Entropy Loss ---
            # We want to *maximize* entropy (encourage exploration)
            # so we take the *negative* mean and add it to the loss.
            entropy_loss = -entropy.mean()

            # --- The Final, Combined Loss ---
            loss = (
                 actor_loss
                 +(self.value_coef*value_loss)
                 +(self.entropy_coef*entropy_loss)
            )
            # --- Backpropagation (The "Hiker" Analogy) ---
            
            # 1. Clear old gradients (wipe the "slope" info)
            self.optimezer.zero_grad()

            # 2. Calculate new gradients (stomp foot, find slope)
            loss.backward()

            # 3. Take a step downhill
            self.optimezer.step()

        return{
             "total_loss":loss.item(),
             "actor_loss":actor_loss.item(),
             "value_loss":value_loss.item()
        }

  
  
         

