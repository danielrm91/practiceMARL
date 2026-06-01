"""Minimal multi-agent PPO for the three-reactor CSTR environment.

Assumptions:
- Cooperative multi-agent task.
- Each agent has its own local observation.
- Each agent has a discrete action space.
- Actors are decentralized.
- Critic is centralized over the concatenated observations.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from cstr_env import CSTRCoolingEnv


@dataclass
class MAPPOConfig:
    n_agents: int = 3
    obs_dim: int = 8
    action_dim: int = 5
    hidden_dim: int = 128

    rollout_steps: int = 256
    total_updates: int = 500
    ppo_epochs: int = 5
    minibatch_size: int = 512

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprobs, entropy


class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.global_states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(
        self,
        obs: np.ndarray,
        global_state: np.ndarray,
        actions: np.ndarray,
        logprobs: np.ndarray,
        reward: float,
        done: bool,
        value: float,
    ):
        self.obs.append(obs.copy())
        self.global_states.append(global_state.copy())
        self.actions.append(actions.copy())
        self.logprobs.append(logprobs.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def to_tensors(self, device: str) -> Dict[str, torch.Tensor]:
        return {
            "obs": torch.tensor(np.array(self.obs), dtype=torch.float32, device=device),
            "global_states": torch.tensor(
                np.array(self.global_states), dtype=torch.float32, device=device
            ),
            "actions": torch.tensor(np.array(self.actions), dtype=torch.long, device=device),
            "logprobs": torch.tensor(
                np.array(self.logprobs), dtype=torch.float32, device=device
            ),
            "rewards": torch.tensor(np.array(self.rewards), dtype=torch.float32, device=device),
            "dones": torch.tensor(np.array(self.dones), dtype=torch.float32, device=device),
            "values": torch.tensor(np.array(self.values), dtype=torch.float32, device=device),
        }


class MAPPO:
    def __init__(self, cfg: MAPPOConfig):
        self.cfg = cfg
        self.device = cfg.device

        self.actors = nn.ModuleList(
            [
                Actor(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim)
                for _ in range(cfg.n_agents)
            ]
        ).to(self.device)

        global_state_dim = cfg.n_agents * cfg.obs_dim
        self.critic = CentralizedCritic(global_state_dim, cfg.hidden_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actors.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def select_actions(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        global_state_t = obs_t.reshape(1, -1)

        actions = []
        logprobs = []

        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                action_i, logprob_i = actor.sample_action(obs_t[i].unsqueeze(0))
                actions.append(action_i.item())
                logprobs.append(logprob_i.item())

            value = self.critic(global_state_t).item()

        return np.array(actions, dtype=np.int64), np.array(logprobs, dtype=np.float32), value

    def compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        advantages = torch.zeros_like(rewards, device=self.device)
        last_gae = torch.tensor(0.0, device=self.device)

        for t in reversed(range(rewards.shape[0])):
            if t == rewards.shape[0] - 1:
                next_nonterminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_nonterminal = 1.0 - dones[t]
                next_val = values[t + 1]

            delta = rewards[t] + cfg.gamma * next_val * next_nonterminal - values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, buffer: RolloutBuffer, next_obs: np.ndarray) -> Dict[str, float]:
        cfg = self.cfg
        data = buffer.to_tensors(self.device)

        obs = data["obs"]
        global_states = data["global_states"]
        actions = data["actions"]
        old_logprobs = data["logprobs"]
        rewards = data["rewards"]
        dones = data["dones"]
        values = data["values"]

        with torch.no_grad():
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            next_global_state = next_obs_t.reshape(1, -1)
            next_value = self.critic(next_global_state).squeeze(0)

        advantages, returns = self.compute_gae(rewards, dones, values, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        rollout_len = rewards.shape[0]
        actor_batch_size = rollout_len * cfg.n_agents

        flat_obs = obs.reshape(actor_batch_size, cfg.obs_dim)
        flat_actions = actions.reshape(actor_batch_size)
        flat_old_logprobs = old_logprobs.reshape(actor_batch_size)
        flat_advantages = advantages.repeat_interleave(cfg.n_agents)

        actor_indices = np.arange(actor_batch_size)
        critic_indices = np.arange(rollout_len)
        last_actor_loss = 0.0
        last_critic_loss = 0.0
        last_entropy = 0.0

        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(actor_indices)

            for start in range(0, actor_batch_size, cfg.minibatch_size):
                mb_idx = actor_indices[start : start + cfg.minibatch_size]
                mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=self.device)

                mb_obs = flat_obs[mb_idx_t]
                mb_actions = flat_actions[mb_idx_t]
                mb_old_logprobs = flat_old_logprobs[mb_idx_t]
                mb_advantages = flat_advantages[mb_idx_t]
                agent_ids = mb_idx_t % cfg.n_agents

                new_logprobs = torch.zeros_like(mb_old_logprobs)
                entropy_terms = torch.zeros_like(mb_old_logprobs)

                for i, actor in enumerate(self.actors):
                    mask = agent_ids == i
                    if mask.any():
                        lp, ent = actor.evaluate_actions(mb_obs[mask], mb_actions[mask])
                        new_logprobs[mask] = lp
                        entropy_terms[mask] = ent

                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                unclipped = ratio * mb_advantages
                clipped = (
                    torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
                    * mb_advantages
                )

                actor_loss = -torch.min(unclipped, clipped).mean()
                entropy = entropy_terms.mean()
                total_actor_loss = actor_loss - cfg.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors.parameters(), cfg.max_grad_norm)
                self.actor_optimizer.step()

                last_actor_loss = float(actor_loss.item())
                last_entropy = float(entropy.item())

            np.random.shuffle(critic_indices)
            for start in range(0, rollout_len, cfg.minibatch_size):
                mb_idx = critic_indices[start : start + cfg.minibatch_size]
                mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=self.device)

                pred_values = self.critic(global_states[mb_idx_t])
                critic_loss = 0.5 * (returns[mb_idx_t] - pred_values).pow(2).mean()

                self.critic_optimizer.zero_grad()
                (cfg.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_optimizer.step()

                last_critic_loss = float(critic_loss.item())

        return {
            "actor_loss": last_actor_loss,
            "critic_loss": last_critic_loss,
            "entropy": last_entropy,
        }


def train(cfg: MAPPOConfig | None = None):
    cfg = cfg or MAPPOConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = CSTRCoolingEnv()
    agent = MAPPO(cfg)
    buffer = RolloutBuffer()

    obs, _ = env.reset(seed=cfg.seed)

    for update in range(cfg.total_updates):
        buffer.reset()
        rollout_return = 0.0
        last_info = {}

        for _ in range(cfg.rollout_steps):
            global_state = obs.reshape(-1)
            actions, logprobs, value = agent.select_actions(obs)
            next_obs, reward, done, info = env.step(actions)

            buffer.add(
                obs=obs,
                global_state=global_state,
                actions=actions,
                logprobs=logprobs,
                reward=reward,
                done=done,
                value=value,
            )

            obs = next_obs
            rollout_return += reward
            last_info = info

            if done:
                obs, _ = env.reset()

        losses = agent.update(buffer, next_obs=obs)

        if update % 10 == 0:
            print(
                f"Update {update:04d} | "
                f"rollout return: {rollout_return: .2f} | "
                f"actor loss: {losses['actor_loss']: .3f} | "
                f"critic loss: {losses['critic_loss']: .3f} | "
                f"P1: {last_info.get('P1_rate', 0.0): .3f} | "
                f"P2: {last_info.get('P2_rate', 0.0): .3f}"
            )

    env.close()
    return agent


if __name__ == "__main__":
    train()
