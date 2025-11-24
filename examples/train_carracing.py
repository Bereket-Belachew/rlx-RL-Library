import rlx
from rlx.env.manager import EnvManager
from rlx.agents.ppo import PPOAgent
from rlx.train.trainer import Train
from rlx.networks.vision import NatureCNN # <--- Import our new Eyes

def main():
    print("--- 🏎️ Starting Self-Driving Car Demo (CarRacing-v2) ---")
    
    # 1. Create the Environment
    # CarRacing gives us a 96x96x3 Image (RGB)
    try:
        env = EnvManager(env_id="CarRacing-v3")
    except Exception as e:
        print(f"\n❌ REAL ERROR DETAILS: {e}")  # <--- Print the actual crash reason
        import traceback
        traceback.print_exc()             # <--- Print the full confusing details
        print("Run: pip install swig && pip install 'gymnasium[box2d]'")
        return

    # 2. Setup the Vision Brain
    # Input: 3 Channels (Red, Green, Blue)
    # Output: 3 Actions (Steer, Gas, Brake)
    action_dim = env.action_space.shape[0]
    cnn_policy = NatureCNN(input_channels=3, action_dim=action_dim)
    
    print("🧠 [Demo] Loaded Custom Vision Policy (NatureCNN)")
    
    # 3. Create the Agent
    # We Plug-and-Play the CNN into the Agent!
    agent = PPOAgent(
        env=env,
        policy=cnn_policy, # <--- The Magic Moment
        lr=3e-4,
        n_steps=1024,      # Shorter steps for image tasks usually good
        n_epochs=10
    )
    
    # 4. Train
    # Note: Image training is SLOW on CPU. 
    trainer = Train(agent=agent, env=env)
    
    # Run for just a few steps to prove it doesn't crash
    trainer.run(total_time_steps=10_000,save_path="my_fast_car.pth")

if __name__ == "__main__":
    main()