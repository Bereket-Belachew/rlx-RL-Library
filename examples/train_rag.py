import rlx
from rlx.env.rag import SimpleRAGEnv
from rlx.agents.ppo import PPOAgent
from rlx.train.trainer import Train
from rlx.networks.text import TextNetwork

def main():
    print("--- 📚 Starting RAG Optimizer Training ---")
    print("Goal: Teach the agent to switch between Keyword & Vector search based on the query.")

    # 1. Create the Environment
    # This loads the dummy database and the embedding model.
    env = SimpleRAGEnv()

    # 2. Create the "Reading" Brain
    # We know our env uses 'all-MiniLM-L6-v2', which outputs 384 dimensions.
    # We have 2 tools: Keyword (0) and Vector (1).
    policy = TextNetwork(embedding_dim=384, action_dim=2)
    
    # 3. Create the Agent
    # We plug our custom TextNetwork into the PPO Agent.
    agent = PPOAgent(
        env=env,
        policy=policy,
        lr=1e-3,         # Slightly higher LR for simple tasks often helps
        n_steps=128,     # Short rollout buffer because episodes are length 1
        n_epochs=4,
        gamma=0.99
    )

    # 4. Create the Trainer
    trainer = Train(agent=agent, env=env)

    # 5. Run Training
    # Since this is a simple 2-choice problem, it should learn VERY fast.
    # 10,000 steps should be plenty.
    trainer.run(total_time_steps=10_000)
    
    # 6. Save the Smart Router
    agent.save("rag_router.pth")
    print("✅ Saved RAG Router brain to 'rag_router.pth'")

if __name__ == "__main__":
    main()