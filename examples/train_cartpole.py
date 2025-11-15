"""
Example: Training PPO on CartPole-v1

This example demonstrates how to use the rlx library to train
a PPO agent on the classic CartPole environment from Gymnasium.
"""

import rlx

def main():
    # Step 1: Create the environment
    print("Creating CartPole environment...")
    env = rlx.EnvManager("CartPole-v1")
    
    # Step 2: Create the PPO agent
    print("Initializing PPO agent...")
    agent = rlx.PPOAgent(env)
    
    # Step 3: Create the trainer
    print("Setting up trainer...")
    trainer = rlx.Train(agent, env)
    
    # Step 4: Train the agent
    print("Starting training...")
    print("=" * 50)
    trainer.run(total_time_steps=100000)
    
    print("=" * 50)
    print("Training completed!")

if __name__ == "__main__":
    main()
