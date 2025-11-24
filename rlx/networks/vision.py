import torch
import torch.nn as nn 
import numpy as np 
from torch.distributions import Normal 

class NatureCNN(nn.Module):
    """
    A Vision-based Brain (The "Nature" Architecture).
    Input: Batch of Images (Batch, Channels, Height, Width)
    Output: Continuous Actions (Mean, Std)
    """

    def __init__(self, input_channels: int, action_dim: int):
        super().__init__()
        # 1. The Eyes (Convolutional Layers)
        # These scan the image to find edges, shapes, and roads.
        # Architecture: 3 Conv layers, growing in depth (32->64->64).
        self.features = nn.Sequential(
            nn.Conv2d(input_channels,32,kernel_size=8,stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # We need to calculate the size of the output coming out of the CNN.
        # We do a dummy pass to figure it out automatically.
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels,96,96)
            n_flatten = self.features(dummy).shape[1]


        # 2. The Brain (Fully Connected Layers)
        self.linear = nn.Sequential(
            nn.Linear(n_flatten,512),
            nn.ReLU()
        )

        # 3. The Heads (Actor & Critic)
        self.critic_head = nn.Linear(512,1)

        # Continuous Action Heads
        self.actor_mean = nn.Linear(512,action_dim)


        # State-Independent Log Std (The "Confidence" Parameter)
        self.actor_logstd= nn.Parameter(torch.zeros(1,action_dim))

    def get_value(self, x):
        # Gym images are (Batch, Height, Width, Channels) -> (B, 96, 96, 3)
        # PyTorch expects (Batch, Channels, Height, Width) -> (B, 3, 96, 96)
        # We need to swap the axes (permute).
        if x.dim()==4 and x.shape[-1]==3:
            x= x.permute (0,3,1,2)

        # Normalize pixels (0-255 -> 0.0-1.0)
        # This is CRITICAL for neural networks to learn.
        x= x/255.0

        feat = self.features(x)
        return self.critic_head(self.linear(feat))

    def get_action(self,x):
        # Permute and Normalize
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x / 255.0 

        feat = self.features(x)
        body = self.linear(feat)

        mean = self.actor_mean(body)
        log_std = self.actor_logstd.expand_as(mean)
        std = torch.exp(log_std)

        return Normal (mean,std)


        