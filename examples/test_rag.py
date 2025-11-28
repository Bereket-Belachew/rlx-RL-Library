import torch
import rlx
from rlx.env.rag import SimpleRAGEnv
from rlx.agents.ppo import PPOAgent
from rlx.networks.text import TextNetwork

def main():
    print("--- 🤖 RAG Router Inference Mode ---")
    print("Loading the brain... (This takes a few seconds)")

    # 1. Setup the Environment (To get the Embedding Model)
    # We don't need the simulator logic, just the "Ears" (env.model)
    env = SimpleRAGEnv()

    # 2. Setup the Brain Architecture
    policy = TextNetwork(embedding_dim=384, action_dim=2)

    # 3. Load the Trained Weights
    try:
        # We load the weights directly into the policy network
        checkpoint = torch.load("rag_router.pth")
        
        # Handle case where we saved the whole agent vs just the network
        if 'model_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['model_state_dict'])
        else:
            policy.load_state_dict(checkpoint)
            
        print("✅ Brain loaded successfully!")
    except FileNotFoundError:
        print("❌ Error: 'rag_router.pth' not found. Train first!")
        return

    # 4. Interactive Loop
    print("\n-------------------------------------------------")
    print("Type a query to see which Search Tool the AI picks.")
    print("Type 'exit' to quit.")
    print("-------------------------------------------------\n")

    policy.eval() # Switch to evaluation mode (turns off training layers)

    while True:
        user_query = input("USER: ")
        if user_query.lower() in ['exit', 'quit']:
            break

        # Step A: Convert Text -> Numbers (Embedding)
        # We use the environment's internal model for consistency
        embedding = env.model.encode(user_query, convert_to_tensor=True)
        
        # Step B: Prepare for Pytorch (Add Batch Dimension: [384] -> [1, 384])
        obs_tensor = embedding.unsqueeze(0).cpu()

        # Step C: Ask the Brain
        with torch.no_grad():
            # Run the input through the body
            features = policy.body(obs_tensor)
            # Get the raw scores (logits) from the actor head
            logits = policy.actor_head(features)
            # Convert logits to probabilities (0% to 100%)
            probs = torch.softmax(logits, dim=-1)
            
            # Get the final decision
            action = torch.argmax(probs).item()
            confidence = probs[0][action].item() * 100

        # Step D: Display Result
        tool_name = "🔍 KEYWORD SEARCH" if action == 0 else "🧠 VECTOR SEARCH"
        color = "\033[92m" if action == 0 else "\033[96m" # Green vs Cyan text
        reset = "\033[0m"

        print(f"AI:   I recommend {color}{tool_name}{reset}")
        print(f"      Confidence: {confidence:.1f}%")
        print(f"      (Probabilities: Keyword={probs[0][0]:.2f}, Vector={probs[0][1]:.2f})\n")

if __name__ == "__main__":
    main()