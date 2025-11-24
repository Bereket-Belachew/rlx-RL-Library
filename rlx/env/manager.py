import gymnasium as gym
from typing import Optional,Tuple,Any,List

# --- DEV_NOTE: Type Aliases ---
# We define these types here for clarity.
# 'Observation' is a numpy array (e.g., [x, y, vx, vy])
# 'Action' is a number (e.g., 0 for left, 1 for right)
#
# We will eventually expand these to handle more complex
# spaces (like images), but this is perfect for CartPole.
import numpy as np
Action = int | np.ndarray # action can int or a numpy array
Observation = np.ndarray #expect an array for obs

class EnvManager:
    """
    A wrapper class for Gymnasium environments.

    This class provides a unified interface for creating, stepping,
    and managing environments. It's the "Env" part of our library.
    
    It solves the problem of "how does the Trainer talk to the env?"
    """
    def __init__(self,env_id: str,seed: Optional[int]=None, num_envs: int=1,render_mode: str =None):
        """
        Initializes the environment manager.
        
        For our MVP, 'num_envs' will be 1. In the future, this class
        will be smart enough to create 'gym.vector.AsyncVectorEnv'
        if num_envs > 1.
        
        ---
        [DEV_NOTE] How a user will call this function:
        
        # This __init__ method is called here:
        env = Env("CartPole-v1", seed=42)
        ---
        """
        self.env_id = env_id
        self.num_envs = num_envs

        #For now we only support 1 env 
        if num_envs !=1:
            raise NotImplementedError(" rlx MVP only supports num_env=1, Vectorized envs are coming soons") 
        
        #Creating a raw environment
        self.env = gym.make(env_id) #env not visible just stored internally for now

        self.seed = seed

        print(f"✅ [EnvManager] Initialized environment: {env_id}")
        print(f"  - Observation Space: {self.env.observation_space.shape}")
        print(f"  - Action Space: {self.env.action_space}")

    def step(self, action: Action) -> Tuple[Observation,float,bool,bool,dict]:
        """
        Takes a step in the environment using the given action.

        Returns a standard 5-tuple:
        (observation, reward, terminated, truncated, info)
        
        ---
        [DEV_NOTE] How this function will be called (by the Trainer):
        
        # Inside the Trainer's 'run' loop:
        new_obs, reward, done, _, info = self.env.step(action)
        ---
        """
        obs,reward,terminated,truncated,info = self.env.step(action)
        self._latest_obs = obs
        return obs,reward,terminated,truncated,info

    def reset(self)->Tuple[Observation,dict]:
        """
        Resets the environment to an initial state.
        
        ---
        [DEV_NOTE] How this function will be called (by the Trainer):
        
        # Inside the Trainer's 'run' loop (at the beginning):
        obs, info = self.env.reset()
        ---
        """
        # We reset with the *same seed* if it was provided,
        # to make experiments reproducible.
        self._latest_obs,self._latest_info = self.env.reset(seed=self.seed)
        return self._latest_obs, self._latest_info #our reset also send latest obs,extra functionality
    
    def close(self)->None:
        """ Closes the Gymnasium and cleans out the resources """
        self.env.close()

    @property
    def observation_space(self) -> gym.Space:
        """ returns The Gymnasium observation stored in self.gym """
        return self.env.observation_space
    
    @property
    def action_space(self)-> gym.Space:
        """ returns Gymnasium action action space stored in self.gym """
        return self.env.action_space