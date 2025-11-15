import numpy as np
import torch 


class RolloutBuffer:

    def __init__(self,n_steps:int,obs_shape:tuple,action_dim:int):
        """
        Initializes the buffer.
        
        Args:
            n_steps (int): The number of steps to collect per rollout.
                           (e.g., 2048)
            obs_shape (tuple): The shape of a single observation.
            action_dim (int): The dimension of the action space.
        """

        self.n_steps=n_steps
        self.obs_shape = obs_shape
        self.action_dim = action_dim

        # We use 'np.zeros' to pre-allocate all the memory
        # we'll need. This is much more efficient than appending.
        self.observations = np.zeros((self.n_steps,)+self.obs_shape, dtype=np.float32)
        self.actions= np.zeros((self.n_steps,),dtype=np.int64)
        self.rewards= np.zeros((self.n_steps,),dtype=np.float32)
        self.values= np.zeros((self.n_steps,),dtype=np.float32)
        self.log_probs= np.zeros((self.n_steps,),dtype=np.float32)
        self.dones = np.zeros((self.n_steps,), dtype=np.float32) # 0=False, 1=True

        # [NEW] This is where we'll store the calculated advantages
        self.advantages = np.zeros((self.n_steps,), dtype=np.float32)

        self.step = 0 # Our current position in the buffer

    def add(self,obs,action,reward, value, log_prob,done):
        """Adds a single step's experience to the buffer."""
        self.observations[self.step]=obs
        self.actions[self.step]=action
        self.rewards[self.step]=reward
        self.values[self.step]=value
        self.log_probs[self.step]=log_prob
        self.dones[self.step]=done

        self.step += 1
    
    def is_full(self)->bool:
        """Adds a single step's experience to the buffer."""
        return self.step == self.n_steps
    def clear(self)->None:
        """Resets the buffer's position."""
        self.step =0

    def get_batch(self,last_value:float, gamma:float,gae_lambda:float)->dict[str,torch.Tensor]:
        """
        Calculates GAE and returns the full batch of data
        as PyTorch tensors.
        
        Args:
            last_value (float): The Critic's value estimate for the
                                *next* observation (after the buffer finished).
            gamma (float): The discount factor.
            gae_lambda (float): The GAE-lambda factor.
        """

       # --- GAE Calculation ---
        # We start with an empty advantage

        last_gae_lam=0
        # We loop *backwards* from the last step to the first
        for t in reversed(range(self.n_steps)):

           #If the game was 'done' at this step, the value of the
            # "next state" is 0.
            if self.dones[t] == 1.0:
                next_value = 0.0

            # Otherwise, if it wasn't the last step, the next value
            # is the one we stored from the *next* step.
            elif t<self.n_steps-1:
                next_value = self.values[t+1]

            # Otherwise, if it wasn't the last step, the next value
            # is the one we stored from the *next* step.
            else:
                next_value = last_value

            # This is the "Temporal Difference" (TD) error:
            # (reward + discounted_next_value) - current_value
            delta = self.rewards[t] + gamma * next_value - self.values[t]

            # This is the GAE magic:
            # advantage = td_error + (gamma * gae_lambda * next_advantage)
            last_gae_lam= delta + gamma*gae_lambda*last_gae_lam

            # Save the calculated advantage for this step
            self.advantages[t]=last_gae_lam

        
        # For now, just convert all data to tensors

        data ={
            "observations": torch.tensor(self.observations,dtype=torch.float32),
            "actions": torch.tensor(self.actions,dtype=torch.int64),
            "rewards": torch.tensor(self.rewards,dtype=torch.float32),
            "values": torch.tensor(self.values,dtype=torch.float32),
            "log_probs": torch.tensor(self.log_probs,dtype=torch.float32),
            "dones": torch.tensor(self.dones,dtype=torch.float32),

            # [NEW] Add the advantages to our batch
            "advantages":torch.tensor(self.advantages,dtype=torch.float32),

            # [NEW] We also return "returns", which is just advantage + value
            # This is what the critic will be trained to predict.
            "returns": torch.tensor(self.advantages+self.values,dtype=torch.float32)
        }

        return data


