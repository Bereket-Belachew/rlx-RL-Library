from rlx.env.manager import EnvManager 
from rlx.agents.base_agent import BaseAgent
import torch 
import numpy as np # [NEW] Needed for calculating mean reward

# [NEW IMPORT]
# We now import our RolloutBuffer. The Trainer
# will "own" this buffer and use it to collect data.
from rlx.utils.buffer import RolloutBuffer
import gymnasium as gym

class Train:
    def __init__(self, agent:BaseAgent, env:EnvManager):

        """
        Initializes the Trainer.
        
        Args:
            agent (BaseAgent): The agent to train (e.g., PPOAgent).
            env (EnvManager): The environment to train in.
            n_steps (int): The number of steps to collect
                           in the buffer before learning.
        """
        self.agent= agent
        self.env=env
        


        # [NEW] Create the "shopping cart" (RolloutBuffer)
        # We get the specs from the environment
        obs_shape = self.env.observation_space.shape

        if isinstance(self.env.action_space,gym.spaces.Discrete):

            action_dim = self.env.action_space.n # if it discreet how many button does it have
        else:
            raise NotImplementedError("Only Discrete action spaces are supported for now.")
        
        ## [REMOVED] The buffer is now owned by the agent: self.buffer = RolloutBuffer(self.n_steps,obs_shape,action_dim)

    def run(self,total_time_steps:int):

        """
        [Dev Note]: User use this function as:
        #Firstly: Creating trainer
            env = Env("CartPole-v1")
            agent = Agent("ppo",env=evn)
            trainer = Train(agent,env)
       #Secondly:
            trainer.run(steps=1000000)
    
        """
        print(f"🚀 [Trainer] Starting training for {total_time_steps} steps...")

        #Our EnvManager's  .reset() will return the first observation 
        obs,info = self.env.reset()
        

        current_step= 0
        # [LOGGING] Track rewards
        current_episode_reward = 0.0
        episode_reward = []
        while current_step< total_time_steps:

            # 1. Ask Agent what to do
            #    Our PPOAgent.select_action() returns (action, (log_prob, value))
            action, extras= self.agent.select_action(obs)
            log_prob,value = extras

            # 2. Take action in Environment
            next_obs,reward,terminated,truncated,info = self.env.step(action)
            done=terminated or truncated #game is either over either way

            # [LOGGING] Update reward tracker
            current_episode_reward += reward

            # 3. Add to "shopping cart" (Buffer)
            self.agent.buffer.add(obs,action,reward,value,log_prob,done)

            # 4. Check if the cart is full
            if self.agent.buffer.is_full():
                # --- This is the "Checkout" ---
                
                # [NEW] Get the "last_value" for GAE
                # We need to get the "fortune-teller's" prediction
                # for the *next* state we are about to see
                next_obs_tensor = torch.tensor(next_obs,dtype=torch.float32).unsqueeze(0)
                self.agent.ac_network.eval()
                with torch.no_grad():
                    last_value = self.agent.ac_network.get_value(next_obs_tensor).item()

                # [NEW] We pass this last_value to the buffer
                # so it can calculate GAE
                batch = self.agent.buffer.get_batch(last_value,self.agent.gamma,self.agent.gae_lambda)

                # [LOGGING] Capture the metrics returned by learn()
                metrics = self.agent.learn(batch)

                self.agent.buffer.clear()

                # [LOGGING] Print Dashboard
                if len(episode_reward) > 0:
                    # Average of last 10 episodes
                    mean_rew  = np.mean(episode_reward[-10:])
                    print(f"{current_step:<10} | {mean_rew:<12.1f} | {metrics['actor_loss']:<10.4f} | {metrics['value_loss']:<10.4f}")
                    
            obs=next_obs
            current_step+=1
            if done:
                # [LOGGING] Save episode reward
                episode_reward.append(current_episode_reward)
                current_episode_reward= 0.0
                obs,info = self.env.reset()
        print("✅ [Trainer] Training complete.")
        self.env.close()




    