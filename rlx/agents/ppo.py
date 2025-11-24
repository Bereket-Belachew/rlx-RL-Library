import torch
import torch.nn as nn
import torch.optim as optim # [FIX] Added optim import
import numpy as np
from typing import Tuple, Any

from rlx.agents.base_agent import BaseAgent, Observation, Action
from rlx.env.manager import EnvManager
from rlx.utils.buffer import RolloutBuffer

# [NEW] Import the brain from our new module, and now include Continuous Actor Critic
from rlx.networks.core import ActorCritic,ContinuousActorCritic
import gymnasium as gym #[New] used for contiuous actor critic 


class PPOAgent(BaseAgent):
   
   def __init__(self, env: EnvManager, 
                policy: torch.nn.Module = None, # [NEW] Allow user to pass a brain
                lr: float = 3e-4,
                n_steps: int = 2048,
                gamma: float = 0.99,
                gae_lambda: float = 0.95,
                n_epochs: int = 10,
                clip_coef: float = 0.2,
                entropy_coef: float = 0.0,
                value_coef: float = 0.5):
      
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
      # [FIX] Get the tuple (e.g., (4,)) for the self.buffer
      obs_shape_tuple = env.observation_space.shape
      # obs_shape_int for actor critic default
      obs_shape_int = env.observation_space.shape[0]

      # [NEW LOGIC] Detect Action Space Type
      # This is the Universal Adaptor logic
      if isinstance(env.action_space,gym.spaces.Discrete):
          print(f"🤖 [PPOAgent] Detected DISCRETE action space.")
          self.is_continuous = False
          action_dim = env.action_space.n
          DefaultPolicy = ActorCritic
      elif isinstance(env.action_space, gym.spaces.Box):
          print(f"🤖 [PPOAgent] Detected CONTINUOUS action space.")
          self.is_continuous=True 
          action_dim=env.action_space.shape[0]
          DefaultPolicy = ContinuousActorCritic
      else: 
          raise NotImplementedError(f"Action space {env.action_space} not supported yet!")
           

      # [NEW LOGIC] Plug-and-Play Brain
      if policy is not None:
          print(f"🧠 [PPOAgent] Using custom policy provided by user.")
          self.ac_network = policy
      else:
          print(f"🧠 [PPOAgent] Using default policy.")
          self.ac_network = DefaultPolicy(obs_shape_int, action_dim)
      
      # Create the optimizer
      # [FIX] Spelled 'optimizer' correctly
      self.optimizer = torch.optim.Adam(self.ac_network.parameters(), lr=self.lr)

      # [NEW] Build the Buffer with the is_continuous flag
      self.buffer = RolloutBuffer(n_steps, obs_shape_tuple, action_dim,self.is_continuous)

      print(f"✅ [PPOAgent] Initialized.")
      print(f"  - Obs Shape: {obs_shape_int}, Action Dim: {action_dim}")
      print(f"  - Learning Rate: {self.lr}")
      
      
   def select_action(self, observation: Observation) -> Tuple[Action, any]:
         # [FIX] Force conversion to numpy array first. 
         # This handles cases where observation is a list like [array]
         if not isinstance(observation, np.ndarray):
             observation = np.array(observation)
         # 1. Convert data: numpy -> torch.Tensor
         obs_tensor = torch.tensor(observation, dtype=torch.float32)

         if obs_tensor.dim()==3:#for image
             obs_tensor=obs_tensor.unsqueeze(0)
         elif obs_tensor.dim()==1: #for vector
             obs_tensor=obs_tensor.unsqueeze(0)
         
         # Put network in eval mode
         self.ac_network.eval()
         with torch.no_grad():
            
            # 2. Think (Act)
            action_dist = self.ac_network.get_action(obs_tensor)

            # 3. Think (Criticize)
            state_value = self.ac_network.get_value(obs_tensor)

            # 4. Act (Sample)
            action = action_dist.sample()
            
            # 5. Get Log-Prob
            log_prob = action_dist.log_prob(action)
            
         if self.is_continuous:
             # Continuous: Sum log_probs across dimensions (standard practice)
             # If action is [0.5, -0.2], log_prob is [lp1, lp2]. Total prob is sum.
             if log_prob.dim() > 1:
                 log_prob = log_prob.sum(dim=-1)


             action_out = action.cpu().numpy()[0]
         else:
             # Discrete: Return scalar int
             action_out = action.item()
                 
         extras = (log_prob.item(), state_value.item())
         return action_out, extras
      

   def save(self, path: str) -> None:
        """Saves the agent's brain (weights) to a file."""
        print(f"💾 [PPOAgent] Saving model to {path}...")
        torch.save({
            'model_state_dict':self.ac_network.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),

        },path)
        print("✅ Save complete.")  

   def load(self, path: str) -> None:
        """Loads a saved brain from a file."""
        print(f"📂 [PPOAgent] Loading model from {path}...")
        checkpoint = torch.load(path)
        self.ac_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Load complete. Agent is ready.")
      
   def learn(self, batch: dict[str, torch.Tensor]) -> dict:
        
        # 1. Get the pre-calculated advantages and returns
        obs = batch['observations']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        advantages = batch['advantages']
        returns = batch['returns']

        # 2. Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 3. Put network in "training" mode
        self.ac_network.train()

        # [LOGGING] Initialize accumulators for average loss
        total_loss_accum = 0.0
        actor_loss_accum = 0.0
        value_loss_accum = 0.0


        # 4. The PPO Update Loop
        for _ in range(self.n_epochs):
             
            action_dist = self.ac_network.get_action(obs)
            
            # [FIX] Added .squeeze() to handle shape mismatch [Batch, 1] -> [Batch]
            new_values = self.ac_network.get_value(obs).squeeze()
            
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy()

            # [NEW] Continuous Space Fix:
            # If we have multiple action dimensions (e.g. Car Racing),
            # we sum the log_probs and entropy across the action dimension.
            if self.is_continuous:
                
                if new_log_probs.dim()>1:
                    new_log_probs = new_log_probs.sum(dim=-1)
                if entropy.dim()>1:
                    entropy= entropy.sum(dim=-1)

            # --- Calculate Actor Loss ---
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            
            # --- Calculate Critic Loss ---
            value_loss = (new_values - returns).pow(2).mean()

            # --- Calculate Entropy Loss ---
            entropy_loss = -entropy.mean()


            # --- Final Loss ---
            loss = (
                 actor_loss
                 + (self.value_coef * value_loss)
                 + (self.entropy_coef * entropy_loss)
            )
            
            # --- Backpropagation ---
            # [FIX] Spelled 'optimizer' correctly
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # [LOGGING] Add to accumulators
            total_loss_accum += loss.item()
            actor_loss_accum += actor_loss.item()
            value_loss_accum += value_loss.item()
            n = self.n_steps
        return {
             "total_loss": total_loss_accum/ n,
             "actor_loss": actor_loss_accum/n,
             "value_loss": value_loss_accum/n
        }