from rlx.env.manager import EnvManager 
from rlx.agents.base_agent import BaseAgent

# [NEW IMPORT]
# We now import our RolloutBuffer. The Trainer
# will "own" this buffer and use it to collect data.
from rlx.utils.buffer import RolloutBuffer
import gymnasium as gym

class Train:
    def __init__(self, agent:BaseAgent, env:EnvManager,n_steps: int=2048):

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
        self.n_steps = n_steps


        # [NEW] Create the "shopping cart" (RolloutBuffer)
        # We get the specs from the environment
        obs_shape = self.env.observation_space.shape

        if isinstance(self.env.action_space,gym.spaces.Discrete):

            action_dim = self.env.action_space.n # if it discreet how many button does it have
        else:
            raise NotImplementedError("Only Discrete action spaces are supported for now.")
        
        self.buffer = RolloutBuffer(self.n_step,obs_shape,action_dim)

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
        

        for step in range(total_time_steps):

            # 1. Ask Agent what to do
            #    Our PPOAgent.select_action() returns (action, (log_prob, value))
            action, extras= self.agent.select_action(obs)
            log_prob,value = extras

            # 2. Take action in Environment
            next_obs,reward,terminated,truncated,info = self.env.step(action)
            done=terminated or truncated #game is either over either way

            # 3. Add to "shopping cart" (Buffer)
            self.buffer.add(obs,action,reward,value,log_prob,done)

            # 4. Check if the cart is full
            if self.buffer.is_full():
                # [DEV_NOTE] This is the "checkout"
                # This line will FAIL until we implement PPOAgent.learn()
                #
                # We get the batch from the buffer...

                batch = self.buffer.get_batch()

                # ...and tell the agent to learn from it
                self.agent.learn(batch)

                self.buffer.clear()
            obs=next_obs
            if done:
                obs,info = self.env.reset()
        print("✅ [Trainer] Training complete.")
        self.env.close()




    