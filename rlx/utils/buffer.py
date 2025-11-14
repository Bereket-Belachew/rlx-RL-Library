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

    def get_batch(self)->dict[str,torch.Tensor]:
        """
        Converts the collected data into PyTorch tensors
        to be used by the 'learn' method.
        """
        # (We will implement the GAE logic here later)
        
        # For now, just convert all data to tensors

        data ={
            "observations": torch.Tensor(self.observations,dtype=torch.float32),
            "actions": torch.Tensor(self.actions,dtype=torch.int64),
            "rewards": torch.Tensor(self.rewards,dtype=torch.float32),
            "values": torch.Tensor(self.values,dtype=torch.float32),
            "log_probs": torch.Tensor(self.log_probs,dtype=torch.float32),
            "dones": torch.Tensor(self.dones,dtype=torch.float32)
        }

        return data


