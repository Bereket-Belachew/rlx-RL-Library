import torch
import torch.nn as nn 
from torch.distributions import Categorical


class TextNetwork(nn.Module):
    """
    The Brain for reading text embeddings.
    
    Input: A vector of numbers representing the meaning of a sentence.
           (Batch, Embedding_Dim) -> e.g., (1, 384)
           
    Output: A Discrete choice of tools.
            (Batch, Action_Dim) -> e.g., (1, 3)
    """
    def __init__(self,embedding_dim:int,action_dim:int):
        super().__init__()

        self.body= nn.Sequential(
            nn.Linear(embedding_dim,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU()
        )

        # 2. The Heads (Actor & Critic)
        self.critic_head = nn.Linear(64,1) # Value (How good is this situation?)
        self.actor_head = nn.Linear(64,action_dim)# Action (Which search tool to pick?)

    def get_value(self,x):
        return self.critic_head(self.body(x))
    def get_action(self,x):
        # x is the Embedding Vector
        features=self.body(x)
        logits= self.actor_head(features)

        # Create a probability distribution over the tools (Action 0, 1, 2...)
        dist = Categorical(logits=logits)
        return dist