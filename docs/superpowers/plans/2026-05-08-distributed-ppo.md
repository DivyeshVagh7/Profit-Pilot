# Distributed PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental feed-forward PPO trainer that can train one shared policy across Kaggle T4 x2 using PyTorch DistributedDataParallel.

**Architecture:** Keep the existing SB3 trainer untouched and add a separate custom PyTorch trainer under `src/profit_pilot/train/`. The trainer reuses `MultiCryptoTradingEnv` and processed bundles, launches one process per GPU with `torchrun`, synchronizes policy gradients through DDP, and saves one checkpoint plus JSON summary.

**Tech Stack:** Python, PyTorch, `torch.distributed`, Gymnasium-compatible env API, pytest.

---

### Task 1: Core Policy And PPO Math

**Files:**
- Create: `src/profit_pilot/train/distributed_ppo.py`
- Test: `tests/test_distributed_ppo.py`

- [ ] **Step 1: Write failing tests**

```python
def test_actor_critic_policy_shapes_and_finite_log_prob():
    policy = ActorCriticPolicy(obs_dim=5, action_dim=3)
    obs = torch.zeros(4, 5)
    actions, log_prob, value = policy.act(obs)
    assert actions.shape == (4, 3)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(value).all()


def test_compute_gae_matches_discounted_return_without_bootstrap():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    dones = torch.tensor([[0.0], [0.0], [1.0]])
    values = torch.zeros(4, 1)
    advantages, returns = compute_gae(rewards, dones, values, gamma=0.9, gae_lambda=1.0)
    assert torch.allclose(returns[:, 0], torch.tensor([2.71, 1.9, 1.0]), atol=1e-5)
    assert torch.allclose(advantages, returns, atol=1e-5)
```

- [ ] **Step 2: Run failing tests**

Run: `D:\Python\python.exe -m pytest tests/test_distributed_ppo.py -q`
Expected: import failure for missing `profit_pilot.train.distributed_ppo`.

- [ ] **Step 3: Implement minimal policy and GAE**

Create an `ActorCriticPolicy` with tanh-squashed Gaussian actions mapped to `[0, 1]`, log-prob correction, value head, and `compute_gae`.

- [ ] **Step 4: Run tests**

Run: `D:\Python\python.exe -m pytest tests/test_distributed_ppo.py -q`
Expected: pass.

### Task 2: Rollout Collection And PPO Update

**Files:**
- Modify: `src/profit_pilot/train/distributed_ppo.py`
- Test: `tests/test_distributed_ppo.py`

- [ ] **Step 1: Write failing tests**

```python
def test_collect_rollout_returns_expected_batch_shapes():
    envs = [TinyContinuousEnv(seed=1), TinyContinuousEnv(seed=2)]
    policy = ActorCriticPolicy(obs_dim=4, action_dim=2)
    rollout = collect_rollout(policy, envs, rollout_steps=5, device=torch.device("cpu"))
    assert rollout.observations.shape == (5, 2, 4)
    assert rollout.actions.shape == (5, 2, 2)
    assert rollout.log_probs.shape == (5, 2)
    assert rollout.rewards.shape == (5, 2)
    assert rollout.dones.shape == (5, 2)
    assert rollout.values.shape == (6, 2)


def test_ppo_update_changes_policy_parameters():
    policy = ActorCriticPolicy(obs_dim=4, action_dim=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    rollout = collect_rollout(policy, [TinyContinuousEnv(seed=3)], rollout_steps=8, device=torch.device("cpu"))
    before = [p.detach().clone() for p in policy.parameters()]
    stats = ppo_update(policy, optimizer, rollout, PPOHyperparameters(batch_size=4, update_epochs=2))
    after = list(policy.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))
    assert "loss" in stats
```

- [ ] **Step 2: Run failing tests**

Run: `D:\Python\python.exe -m pytest tests/test_distributed_ppo.py -q`
Expected: missing rollout/update functions.

- [ ] **Step 3: Implement rollout and update**

Implement synchronous env stepping, GAE/returns, flattened mini-batches, clipped PPO policy loss, value loss, entropy bonus, gradient clipping, and summary stats.

- [ ] **Step 4: Run tests**

Run: `D:\Python\python.exe -m pytest tests/test_distributed_ppo.py -q`
Expected: pass.

### Task 3: CLI And Kaggle Launch

**Files:**
- Modify: `src/profit_pilot/train/distributed_ppo.py`
- Modify: `notebooks/kaggle_p100_5m_profit_pilot_training.ipynb`
- Test: `tests/test_distributed_ppo.py`

- [ ] **Step 1: Write failing CLI/notebook contract tests**

```python
def test_distributed_launch_command_uses_torchrun_two_processes():
    command = build_torchrun_command(config_path="config/project_config_5m_train.yaml", output_dir="/kaggle/working/models/ddp")
    assert command[:4] == ["python", "-m", "torch.distributed.run", "--nproc_per_node=2"]
    assert "profit_pilot.train.distributed_ppo" in command
```

- [ ] **Step 2: Run failing tests**

Run: `D:\Python\python.exe -m pytest tests/test_distributed_ppo.py tests/test_kaggle_t4x2_notebook.py -q`
Expected: missing command builder or missing notebook marker.

- [ ] **Step 3: Implement CLI and notebook launcher**

Add argparse options for config, output dir, envs per rank, rollout steps, total timesteps, update epochs, batch size, seed, and model name. Add a notebook cell that calls `torchrun --nproc_per_node=2 -m profit_pilot.train.distributed_ppo`.

- [ ] **Step 4: Run tests**

Run: `D:\Python\python.exe -m pytest tests/test_distributed_ppo.py tests/test_kaggle_t4x2_notebook.py -q`
Expected: pass.
