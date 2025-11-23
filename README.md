
-----

# rlx: The "LangChain for Reinforcement Learning" (Alpha v0.2.0)

Hey everyone, Bereket here. Thanks for checking in on the progress.

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

# 1. Create the Environment (Discrete OR Continuous!)
env = EnvManager("Pendulum-v1")

# 2. Create the Agent (Auto-detects the right "Brain" for the job)
# You can plug in your own Policy here, or let rlx build a default one.
agent = PPOAgent(env=env, lr=3e-4)

# 3. Train with a built-in dashboard
trainer = Train(agent, env)
trainer.run(total_timesteps=100_000)
```

-----

## 🚀 Major Update: We Are Live\! (v0.2.0)

Since the last update, we have moved from "laying the foundation" to a **fully functional, multi-purpose library.**

We hit three massive milestones this week:

### 1\. The Engine Works (CartPole Solved) 📊

We successfully implemented the **PPO `learn()` loop**.

  * We calculated GAE (Generalized Advantage Estimation).
  * We implemented the clipped surrogate loss.
  * **Result:** The agent solves `CartPole-v1` in under 50k steps, taking rewards from \~20 to \>300.

### 2\. The "Dashboard" is Live 🖥️

We got tired of staring at a blank screen while training. The `Trainer` now includes a real-time CLI dashboard that tracks:

  * Mean Reward (Last 10 episodes)
  * Actor Loss & Value Loss
  * Current Step Count

### 3\. The "Universal Adaptor" (Discrete & Continuous) 🤖

This is the biggest technical leap. Originally, `rlx` only supported "Discrete" actions (pressing buttons, like in Mario).

We just refactored the entire stack to support **Continuous** actions (turning knobs/steering wheels, like in Robotics).

  * **Discrete Mode:** The agent builds a "Categorical" Brain (Bar Chart probability).
  * **Continuous Mode:** The agent builds a "Normal" Brain (Gaussian Bell Curve).
  * **The Magic:** You don't have to do anything. `PPOAgent` automatically looks at your environment and builds the correct brain for you.

-----

## 🧱 The "LEGO Bricks" (Current Architecture)

Here is how the system is currently built:

  * **✅ `EnvManager`:** Wraps Gymnasium environments and standardizes the inputs.
  * **✅ `BaseAgent`:** The abstract contract for all agents.
  * **✅ `PPOAgent`:** The "Surgeon."
      * It manages the learning loop.
      * It now features **Plug-and-Play Architecture**: You can pass a custom neural network into `__init__`, and the agent will use it.
  * **✅ `RolloutBuffer`:** The "Memory."
      * Updated to handle both Scalar actions (buttons) and Vector actions (steering/gas/brake).
  * **✅ `networks/core.py` (NEW):** The "Brains."
      * `ActorCritic`: For Discrete spaces.
      * `ContinuousActorCritic`: For Continuous spaces (using `torch.distributions.Normal`).
  * **✅ `Trainer`:** The "Driver." Now includes logging and safety checks.

-----

## 📂 Project Structure

We've cleaned up the architecture significantly to separate "The Agent" from "The Brain."

```
rlx/
├── rlx/
│   ├── __init__.py
│   ├── agents/
│   │   ├── base_agent.py      # The Abstract Contract
│   │   └── ppo.py             # The Logic (The "Surgeon")
│   ├── env/
│   │   └── manager.py         # The Gym Wrapper
│   ├── networks/              # [NEW] Where the Neural Nets live
│   │   ├── __init__.py
│   │   └── core.py            # Discrete & Continuous Actor-Critics
│   ├── train/
│   │   └── trainer.py         # The Loop & Dashboard
│   ├── utils/
│   │   └── buffer.py          # The Memory (Vectors & Scalars)
├── examples/
│   ├── train_cartpole.py      # Test for Discrete (Buttons)
│   └── train_pendulum.py      # Test for Continuous (Steering)
├── README.md
└── pyproject.toml
```

-----

## 🔮 What's Next?

We have the engine (PPO) and the steering wheel (Continuous Support). The next step is **The Eyes.**

I am currently working on implementing **CNN (Convolutional Neural Network)** support so we can plug a "Vision Brain" into the PPO Agent.

The goal? **To train a self-driving car in `CarRacing-v2` using raw pixels.**

Stay tuned.

— Bereket