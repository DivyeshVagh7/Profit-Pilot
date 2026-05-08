from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from gymnasium import spaces


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from profit_pilot.train.distributed_ppo import (  # noqa: E402
    ActorCriticPolicy,
    PPOHyperparameters,
    build_torchrun_command,
    collect_rollout,
    compute_gae,
    ppo_update,
)


class TinyContinuousEnv:
    def __init__(self, seed: int = 0) -> None:
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.steps += 1
        reward = 1.0 - float(np.square(action - 0.5).mean())
        terminated = self.steps >= 3
        truncated = False
        return self._observation(), reward, terminated, truncated, {}

    def _observation(self) -> np.ndarray:
        return self.rng.normal(size=4).astype(np.float32)


def test_actor_critic_policy_shapes_and_finite_log_prob() -> None:
    policy = ActorCriticPolicy(obs_dim=5, action_dim=3)
    obs = torch.zeros(4, 5)

    actions, log_prob, value = policy.act(obs)

    assert actions.shape == (4, 3)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)
    assert torch.all(actions >= 0.0)
    assert torch.all(actions <= 1.0)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(value).all()


def test_compute_gae_matches_discounted_return_without_bootstrap() -> None:
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    dones = torch.tensor([[0.0], [0.0], [1.0]])
    values = torch.zeros(4, 1)

    advantages, returns = compute_gae(rewards, dones, values, gamma=0.9, gae_lambda=1.0)

    expected = torch.tensor([2.71, 1.9, 1.0])
    assert torch.allclose(returns[:, 0], expected, atol=1e-5)
    assert torch.allclose(advantages[:, 0], expected, atol=1e-5)


def test_collect_rollout_returns_expected_batch_shapes() -> None:
    envs = [TinyContinuousEnv(seed=1), TinyContinuousEnv(seed=2)]
    policy = ActorCriticPolicy(obs_dim=4, action_dim=2)

    rollout = collect_rollout(policy, envs, rollout_steps=5, device=torch.device("cpu"))

    assert rollout.observations.shape == (5, 2, 4)
    assert rollout.actions.shape == (5, 2, 2)
    assert rollout.log_probs.shape == (5, 2)
    assert rollout.rewards.shape == (5, 2)
    assert rollout.dones.shape == (5, 2)
    assert rollout.values.shape == (6, 2)


def test_ppo_update_changes_policy_parameters() -> None:
    policy = ActorCriticPolicy(obs_dim=4, action_dim=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    rollout = collect_rollout(policy, [TinyContinuousEnv(seed=3)], rollout_steps=8, device=torch.device("cpu"))
    before = [p.detach().clone() for p in policy.parameters()]

    stats = ppo_update(policy, optimizer, rollout, PPOHyperparameters(batch_size=4, update_epochs=2))

    after = list(policy.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))
    assert "loss" in stats
    assert torch.isfinite(torch.tensor(stats["loss"]))


def test_distributed_launch_command_uses_torchrun_two_processes() -> None:
    command = build_torchrun_command(
        config_path="config/project_config_5m_train.yaml",
        output_dir="/kaggle/working/models/ddp",
        nproc_per_node=2,
    )

    assert command[:4] == ["python", "-m", "torch.distributed.run", "--nproc_per_node=2"]
    assert "profit_pilot.train.distributed_ppo" in command
    assert "--config" in command
    assert "--output-dir" in command
