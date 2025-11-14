# RLx

RLx is a lightweight scaffolding for reinforcement-learning experiments. The
project separates responsibilities into three "contracts":

- `rlx/env/manager.py` exposes a Gymnasium-powered `EnvManager`.
- `rlx/agents/base_agent.py` defines the abstract `BaseAgent`.
- `rlx/train/trainer.py` links agents and environments through the `Train` loop.

The first concrete appliance is the `PPOAgent`, implemented in
`rlx/agents/ppo.py`. It ships with an internal `ActorCritic` network (two
hidden `Tanh` layers feeding policy logits and a scalar value head), Adam-based
optimization, and a fully wired `select_action` method that:

1. Casts environment observations to `torch.Tensor` on the active device.
2. Produces a categorical distribution over discrete actions.
3. Samples an action and returns the pair `(action, log_prob)` for later PPO
   updates.

## Technical Review

**Strengths**
- Clean separation of concerns through the Env/Agent/Train sockets keeps the
  public API narrow and testable.
- The PPOAgent already handles tensor-device placement, deterministic saving
  and loading, and exposes typed helper utilities (`ActorCritic`, `_to_tensor`
  helper) that make future algorithm work pleasant.
- The `select_action` implementation operates entirely under
  `torch.no_grad()`, ensuring rollout collection stays cheap.

**Risks / Areas for Improvement**
- `PPOAgent.learn` is still a stub; until the optimizer step is implemented the
  agent cannot improve from experience.
- `Train.run` currently ignores rewards/log probabilities, so once learning is
  implemented the loop will need to thread those values through.
- README should evolve into living docs (installation instructions, example
  configs, etc.) once the training loop matures.

## Next Steps

1. Implement trajectory storage (e.g., GAE buffer) plus the PPO update logic.
2. Extend `Train.run` to accumulate rollouts, call `agent.learn`, and log
   statistics.
3. Add smoke tests that instantiate PPOAgent against a toy Gymnasium task.
4. Automate publishing to GitHub once repository credentials are available.

