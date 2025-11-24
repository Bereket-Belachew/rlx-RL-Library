import rlx
import torch
from rlx.env.manager import EnvManager
from rlx.agents.ppo import PPOAgent
from rlx.networks.vision import NatureCNN
import time

def main():
    print("--- 🍿 Watch Mode: CarRacing-v3 ---")
    
    # 1. Create Environment (With Render Mode ON!)
    env = EnvManager("CarRacing-v3", render_mode="human")
    
    # 2. Setup the Brain Architecture
    action_dim = env.action_space.shape[0]
    cnn_policy = NatureCNN(input_channels=3, action_dim=action_dim)
    
    # 3. Create Agent
    agent = PPOAgent(env=env, policy=cnn_policy)
    
    # 4. Load the Saved Brain
    try:
        agent.load("my_fast_car.pth") # Make sure this filename matches what you saved!
    except FileNotFoundError:
        print("❌ Error: 'my_fast_car.pth' not found. Train the agent first!")
        return

    # 5. Enjoy Loop
    # [FIX] Unpack the tuple here
    obs, info = env.reset()
    
    while True:
        # Select action
        action, _ = agent.select_action(obs)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        try:
            env.env.render() 
        except:
            pass # If render_mode="human" works, this isn't needed, but it helps debug

        # [FIX] Slow down the loop so we can see it (and prevent console spam)
        time.sleep(0.05)
        if terminated or truncated:
            print("🔄 Episode finished. Resetting...")
            # [FIX] Unpack the tuple here too
            obs, info = env.reset()

if __name__ == "__main__":
    main()