
# rlx: The "LangChain for Reinforcement Learning" (Alpha v0.3.0)

Hey everyone, Bereket here.

This is a project I'm building out of a desire for better tooling. As I've been learning AI and Reinforcement Learning, I realized that existing libraries (like `stable-baselines3`) are incredible, but they often feel like "black boxes." They are great for getting results, but hard to "tinker" with or hack.

I wanted something modular—a "box of LEGOs" for RL, just like **LangChain** is for LLMs.

That's `rlx`.

-----

## 🎯 The Goal: A "Glass Box" Framework

My goal is to create an open-source framework where every part of the RL pipeline—the brain, the memory, the algorithm—is a separate, interchangeable block.

We are moving from hard-coded scripts to declarative pipelines:

```python
from rlx.agents import PPOAgent
from rlx.train import Train
from rlx.env import EnvManager
from rlx.networks.vision import NatureCNN

# 1. Create a Vision Environment
env = EnvManager("CarRacing-v3")

# 2. Plug in a Vision Brain (The "LEGO" moment)
# We swap the default brain for a CNN just by passing an object.
cnn_policy = NatureCNN(input_channels=3, action_dim=env.action_space.shape[0])
agent = PPOAgent(env=env, policy=cnn_policy)

# 3. Train
trainer = Train(agent, env)
trainer.run(total_timesteps=1_000_000)
```

-----

## 🚀 Major Update: Vision Support\! (v0.3.0)

We have successfully built the "Holy Trinity" of RL inputs. The library now supports:

1.  **✅ Discrete Control:** (e.g., `CartPole-v1`) - Pressing buttons.
2.  **✅ Continuous Control:** (e.g., `Pendulum-v1`) - Turning knobs and steering wheels.
3.  **✅ Visual Control:** (e.g., `CarRacing-v3`) - Learning from raw pixels using CNNs.

### The "Universal Adaptor" 🤖

The `PPOAgent` is now smart.

  * If you pass a **Discrete** environment, it builds a categorical (Softmax) brain.
  * If you pass a **Continuous** environment, it builds a Gaussian (Normal Distribution) brain.
  * If you pass a **Custom Policy** (like our new `NatureCNN`), it just works.

-----

## 🧱 The Architecture

Here is how the system is currently built:

  * **`rlx/agents/ppo.py`:** The "Surgeon." Manages the learning loop and handles the "Plug-and-Play" brain logic.
  * **`rlx/env/manager.py`:** Wraps Gymnasium environments and standardizes inputs.
  * **`rlx/train/trainer.py`:** The "Driver." Includes a real-time CLI dashboard for tracking rewards and losses.
  * **`rlx/networks/`**:
      * **`core.py`:** Standard MLP brains for Discrete & Continuous tasks.
      * **`vision.py`:** [NEW] The **NatureCNN** architecture for processing images.
  * **`rlx/utils/buffer.py`:** The "Memory." Handles both scalar (button) and vector (steering) storage.

-----

## 📂 Project Structure

```text
rlx/
├── rlx/
│   ├── __init__.py
│   ├── agents/
│   │   ├── base_agent.py
│   │   └── ppo.py             # Core Algorithm
│   ├── env/
│   │   └── manager.py         # Gym Wrapper
│   ├── networks/              
│   │   ├── __init__.py
│   │   ├── core.py            # MLP Brains
│   │   └── vision.py          # CNN Eyes (The new update)
│   ├── train/
│   │   └── trainer.py         # Dashboard & Loop
│   ├── utils/
│   │   └── buffer.py          # Memory
├── examples/
│   ├── train_cartpole.py      # Demo: Discrete
│   ├── train_pendulum.py      # Demo: Continuous
│   └── train_carracing.py     # Demo: Vision
├── README.md
└── pyproject.toml
```

-----

## 🔮 What's Next?

Now that the core engine works for all data types, the next phase is **Persistence & usability.**

  * **Saving/Loading Models:** We need to save the "Self-Driving Car" brain so we can watch it drive later\!
  * **Evaluation Mode:** A script to watch the agent play without training.

Stay tuned.

— Bereket

-----

