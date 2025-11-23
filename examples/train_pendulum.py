import rlx
from rlx.env.manager import EnvManager
from rlx.agents.ppo import PPOAgent
from rlx.train.trainer import Train

def main():
    print("--- 🧪 Starting Continuous Action Verification (Pendulum-v1) ---")
    
    # 1. Create the Continuous Environment
    # Pendulum-v1 actions are floats: [-2.0, 2.0]
    env = EnvManager(env_id="Pendulum-v1")
    
    # 2. Create the Agent
    # Because 'env' is continuous, PPOAgent should AUTOMATICALLY
    # detect this and build a ContinuousActorCritic (The "Steering Wheel" brain).
    agent = PPOAgent(
        env=env,
        lr=3e-4,           # Standard learning rate
        n_steps=2048,      # Collect data in chunks
        gamma=0.95,        # Pendulum is a short-term game, slightly lower gamma helps
        gae_lambda=0.95,
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.0   # We let the StdDev handle exploration naturally
    )
    
    # 3. Create the Trainer
    trainer = Train(agent=agent, env=env)
    
    # 4. Run!
    # In Pendulum, scores start around -1500 (Bad).
    # "Solved" is roughly -200 or higher (closer to 0 is better).
    trainer.run(total_time_steps=100_000)

if __name__ == "__main__":
    main()