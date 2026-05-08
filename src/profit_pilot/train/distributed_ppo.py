from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import sys
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from profit_pilot.train.train_ppo import build_env


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
ACTION_EPS = 1e-6


@dataclass
class PPOHyperparameters:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.08
    ent_coef: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 1024
    update_epochs: int = 10


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor


class ActorCriticPolicy(nn.Module):
    """Feed-forward tanh-Gaussian actor with a separate value head."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        actor_layers: list[nn.Module] = []
        critic_layers: list[nn.Module] = []
        previous_actor_dim = self.obs_dim
        previous_critic_dim = self.obs_dim
        for hidden_size in hidden_sizes:
            actor_layers.extend([nn.Linear(previous_actor_dim, hidden_size), nn.Tanh()])
            critic_layers.extend([nn.Linear(previous_critic_dim, hidden_size), nn.Tanh()])
            previous_actor_dim = hidden_size
            previous_critic_dim = hidden_size

        actor_layers.append(nn.Linear(previous_actor_dim, self.action_dim))
        critic_layers.append(nn.Linear(previous_critic_dim, 1))
        self.actor = nn.Sequential(*actor_layers)
        self.critic = nn.Sequential(*critic_layers)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.5))

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.evaluate_actions(observations, actions)

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1)

    def act(self, observations: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observations)
        raw_actions = distribution.mean if deterministic else distribution.rsample()
        actions = self._squash_raw_actions(raw_actions)
        log_prob = self._squashed_log_prob(distribution, raw_actions)
        value = self.value(observations)
        return actions, log_prob, value

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observations)
        raw_actions = self._unsquash_actions(actions)
        log_prob = self._squashed_log_prob(distribution, raw_actions)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.value(observations)
        return log_prob, entropy, value

    def _distribution(self, observations: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(observations)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def _squash_raw_actions(self, raw_actions: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(raw_actions) + 1.0) * 0.5

    def _unsquash_actions(self, actions: torch.Tensor) -> torch.Tensor:
        scaled_actions = torch.clamp(actions * 2.0 - 1.0, -1.0 + ACTION_EPS, 1.0 - ACTION_EPS)
        return 0.5 * (torch.log1p(scaled_actions) - torch.log1p(-scaled_actions))

    def _squashed_log_prob(
        self,
        distribution: torch.distributions.Normal,
        raw_actions: torch.Tensor,
    ) -> torch.Tensor:
        tanh_actions = torch.tanh(raw_actions)
        log_prob = distribution.log_prob(raw_actions)
        correction = torch.log(1.0 - tanh_actions.pow(2) + ACTION_EPS) - np.log(2.0)
        return (log_prob - correction).sum(dim=-1)


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.shape[1], dtype=rewards.dtype, device=rewards.device)
    for step in reversed(range(rewards.shape[0])):
        next_non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * values[step + 1] * next_non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values[:-1]
    return advantages, returns


def collect_rollout(
    policy: ActorCriticPolicy | DistributedDataParallel,
    envs: Sequence,
    rollout_steps: int,
    device: torch.device,
) -> RolloutBatch:
    model = unwrap_policy(policy)
    observations = []
    for env_index, env in enumerate(envs):
        observation, _ = env.reset(seed=env_index)
        observations.append(np.asarray(observation, dtype=np.float32))

    obs_tensor = torch.as_tensor(np.stack(observations), dtype=torch.float32, device=device)
    rollout_observations = []
    rollout_actions = []
    rollout_log_probs = []
    rollout_rewards = []
    rollout_dones = []
    rollout_values = []

    model.eval()
    with torch.no_grad():
        for _ in range(rollout_steps):
            actions, log_probs, values = model.act(obs_tensor)
            action_array = actions.detach().cpu().numpy().astype(np.float32)
            next_observations = []
            rewards = []
            dones = []
            for env_index, env in enumerate(envs):
                next_observation, reward, terminated, truncated, _ = env.step(action_array[env_index])
                done = bool(terminated or truncated)
                if done:
                    next_observation, _ = env.reset()
                next_observations.append(np.asarray(next_observation, dtype=np.float32))
                rewards.append(float(reward))
                dones.append(float(done))

            rollout_observations.append(obs_tensor.detach().cpu())
            rollout_actions.append(actions.detach().cpu())
            rollout_log_probs.append(log_probs.detach().cpu())
            rollout_values.append(values.detach().cpu())
            rollout_rewards.append(torch.tensor(rewards, dtype=torch.float32))
            rollout_dones.append(torch.tensor(dones, dtype=torch.float32))
            obs_tensor = torch.as_tensor(np.stack(next_observations), dtype=torch.float32, device=device)

        _, _, last_values = model.act(obs_tensor, deterministic=True)
        rollout_values.append(last_values.detach().cpu())

    return RolloutBatch(
        observations=torch.stack(rollout_observations),
        actions=torch.stack(rollout_actions),
        log_probs=torch.stack(rollout_log_probs),
        rewards=torch.stack(rollout_rewards),
        dones=torch.stack(rollout_dones),
        values=torch.stack(rollout_values),
    )


def ppo_update(
    policy: ActorCriticPolicy | DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    hyperparameters: PPOHyperparameters,
) -> dict[str, float]:
    policy.train()
    device = next(unwrap_policy(policy).parameters()).device
    observations = rollout.observations.to(device)
    actions = rollout.actions.to(device)
    old_log_probs = rollout.log_probs.to(device)
    rewards = rollout.rewards.to(device)
    dones = rollout.dones.to(device)
    values = rollout.values.to(device)

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        gamma=hyperparameters.gamma,
        gae_lambda=hyperparameters.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    flat_observations = observations.reshape(-1, observations.shape[-1])
    flat_actions = actions.reshape(-1, actions.shape[-1])
    flat_old_log_probs = old_log_probs.reshape(-1)
    flat_advantages = advantages.reshape(-1)
    flat_returns = returns.reshape(-1)

    batch_size = min(int(hyperparameters.batch_size), flat_observations.shape[0])
    stats: dict[str, float] = {}
    for _ in range(hyperparameters.update_epochs):
        indices = torch.randperm(flat_observations.shape[0], device=device)
        for start in range(0, flat_observations.shape[0], batch_size):
            minibatch = indices[start : start + batch_size]
            new_log_probs, entropy, new_values = policy(
                flat_observations[minibatch],
                flat_actions[minibatch],
            )
            log_ratio = new_log_probs - flat_old_log_probs[minibatch]
            ratio = torch.exp(log_ratio)
            unclipped_policy_loss = -flat_advantages[minibatch] * ratio
            clipped_policy_loss = -flat_advantages[minibatch] * torch.clamp(
                ratio,
                1.0 - hyperparameters.clip_range,
                1.0 + hyperparameters.clip_range,
            )
            policy_loss = torch.max(unclipped_policy_loss, clipped_policy_loss).mean()
            value_loss = torch.nn.functional.mse_loss(new_values, flat_returns[minibatch])
            entropy_loss = entropy.mean()
            loss = policy_loss + hyperparameters.value_coef * value_loss - hyperparameters.ent_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unwrap_policy(policy).parameters(), hyperparameters.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > hyperparameters.clip_range).float().mean()
            stats = {
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy_loss.detach().cpu()),
                "approx_kl": float(approx_kl.detach().cpu()),
                "clip_fraction": float(clip_fraction.detach().cpu()),
            }
    return stats


def build_torchrun_command(
    config_path: str,
    output_dir: str,
    nproc_per_node: int = 2,
    envs_per_rank: int = 2,
    rollout_steps: int | None = None,
    total_timesteps: int | None = None,
    model_name: str | None = None,
) -> list[str]:
    command = [
        "python",
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        "-m",
        "profit_pilot.train.distributed_ppo",
        "--config",
        config_path,
        "--output-dir",
        output_dir,
        "--envs-per-rank",
        str(envs_per_rank),
    ]
    if rollout_steps is not None:
        command.extend(["--rollout-steps", str(rollout_steps)])
    if total_timesteps is not None:
        command.extend(["--total-timesteps", str(total_timesteps)])
    if model_name is not None:
        command.extend(["--model-name", model_name])
    return command


def unwrap_policy(policy: ActorCriticPolicy | DistributedDataParallel) -> ActorCriticPolicy:
    return policy.module if isinstance(policy, DistributedDataParallel) else policy


def make_envs(env_factory, count: int, seed_offset: int) -> list:
    envs = []
    for index in range(count):
        env = env_factory()
        env.reset(seed=seed_offset + index)
        envs.append(env)
    return envs


def initialize_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, local_rank, world_size


def reduce_stats(stats: dict[str, float], device: torch.device, world_size: int) -> dict[str, float]:
    if world_size <= 1 or not dist.is_initialized():
        return stats
    keys = sorted(stats)
    values = torch.tensor([stats[key] for key in keys], dtype=torch.float32, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.AVG)
    return {key: float(value.detach().cpu()) for key, value in zip(keys, values)}


def save_checkpoint(
    policy: ActorCriticPolicy | DistributedDataParallel,
    output_dir: Path,
    model_name: str,
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{model_name}.pt"
    torch.save(
        {
            "model_state_dict": unwrap_policy(policy).state_dict(),
            "summary": summary,
        },
        checkpoint_path,
    )
    with (output_dir / f"{model_name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def train(args: argparse.Namespace) -> None:
    rank, local_rank, world_size = initialize_distributed()
    seed = int(args.seed) + rank * 10_000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    config, _, env_factory = build_env(args.config)
    envs = make_envs(env_factory, count=args.envs_per_rank, seed_offset=seed)
    obs_dim = int(envs[0].observation_space.shape[0])
    action_dim = int(envs[0].action_space.shape[0])

    policy = ActorCriticPolicy(obs_dim=obs_dim, action_dim=action_dim).to(device)
    if world_size > 1:
        policy = DistributedDataParallel(policy, device_ids=[local_rank] if use_cuda else None)

    learning_rate = args.learning_rate if args.learning_rate is not None else config.training.learning_rate
    optimizer = torch.optim.Adam(unwrap_policy(policy).parameters(), lr=learning_rate)
    hyperparameters = PPOHyperparameters(
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        clip_range=config.training.clip_range,
        ent_coef=config.training.ent_coef,
        batch_size=args.batch_size or config.training.batch_size,
        update_epochs=args.update_epochs,
    )
    rollout_steps = args.rollout_steps or config.training.n_steps
    total_timesteps = args.total_timesteps or config.training.total_timesteps
    model_name = args.model_name or "profit_pilot_distributed_ppo_feedforward"
    output_dir = Path(args.output_dir or config.training.model_dir)

    global_steps = 0
    iteration = 0
    rollout_width = rollout_steps * args.envs_per_rank * world_size
    if rank == 0:
        print(
            f"Distributed PPO: world_size={world_size}, envs_per_rank={args.envs_per_rank}, "
            f"rollout_width={rollout_width}, device={device}"
        )
    while global_steps < total_timesteps:
        rollout = collect_rollout(policy, envs, rollout_steps=rollout_steps, device=device)
        stats = ppo_update(policy, optimizer, rollout, hyperparameters)
        global_steps += rollout_width
        iteration += 1
        stats = reduce_stats(stats, device, world_size)
        if rank == 0:
            print(
                f"iteration={iteration} total_timesteps={global_steps} "
                f"loss={stats.get('loss', 0.0):.6f} "
                f"policy_loss={stats.get('policy_loss', 0.0):.6f} "
                f"value_loss={stats.get('value_loss', 0.0):.6f} "
                f"approx_kl={stats.get('approx_kl', 0.0):.6f}"
            )

    if rank == 0:
        summary = {
            "model_name": model_name,
            "algorithm": "distributed_feedforward_ppo",
            "world_size": world_size,
            "envs_per_rank": args.envs_per_rank,
            "rollout_steps": rollout_steps,
            "rollout_width": rollout_width,
            "total_timesteps": global_steps,
            "requested_total_timesteps": total_timesteps,
            "device": str(device),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "learning_rate": learning_rate,
            "hyperparameters": asdict(hyperparameters),
            "config": args.config,
        }
        save_checkpoint(policy, output_dir, model_name, summary)
        print(f"Saved distributed PPO checkpoint to {output_dir / (model_name + '.pt')}")

    for env in envs:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train feed-forward PPO with PyTorch DDP.")
    parser.add_argument("--config", default="config/project_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--envs-per-rank", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
