"""
Minimal Multi-Agent PPO (MAPPO-style) template in PyTorch.

Assumptions
-----------
- Cooperative multi-agent problem.
- Each agent has its own local observation.
- Each agent has a discrete action space.
- Actors are decentralized: pi_i(a_i | o_i).
- Critic is centralized: V(s_global), where s_global is the concatenation of all agents' observations.

This is intentionally written as a clear research template, not as a fully optimized library.
You can adapt the environment wrapper to your own batch plant / scheduling environment.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ============================================================
# 1. Configuration
# ============================================================

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

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. Networks
# ============================================================

class Actor(nn.Module):
    """Decentralized policy network for one agent."""

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
        logits = self.net(obs)
        return logits

    def get_action_and_logprob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy


class CentralizedCritic(nn.Module):
    """Centralized value function V(s_global)."""

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


# ============================================================
# 3. Rollout Buffer
# ============================================================

class RolloutBuffer:
    def __init__(self, cfg: MAPPOConfig):
        self.cfg = cfg
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
        self.obs.append(obs)
        self.global_states.append(global_state)
        self.actions.append(actions)
        self.logprobs.append(logprobs)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def to_tensors(self, device: str) -> Dict[str, torch.Tensor]:
        return {
            "obs": torch.tensor(np.array(self.obs), dtype=torch.float32, device=device),
            "global_states": torch.tensor(np.array(self.global_states), dtype=torch.float32, device=device),
            "actions": torch.tensor(np.array(self.actions), dtype=torch.long, device=device),
            "logprobs": torch.tensor(np.array(self.logprobs), dtype=torch.float32, device=device),
            "rewards": torch.tensor(np.array(self.rewards), dtype=torch.float32, device=device),
            "dones": torch.tensor(np.array(self.dones), dtype=torch.float32, device=device),
            "values": torch.tensor(np.array(self.values), dtype=torch.float32, device=device),
        }


# ============================================================
# 4. MAPPO Agent
# ============================================================

class MAPPO:
    def __init__(self, cfg: MAPPOConfig):
        self.cfg = cfg
        self.device = cfg.device

        self.actors = nn.ModuleList([
            Actor(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim)
            for _ in range(cfg.n_agents)
        ]).to(self.device)

        global_state_dim = cfg.n_agents * cfg.obs_dim
        self.critic = CentralizedCritic(global_state_dim, cfg.hidden_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actors.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def select_actions(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Parameters
        ----------
        obs : np.ndarray
            Shape: (n_agents, obs_dim)

        Returns
        -------
        actions : np.ndarray
            Shape: (n_agents,)
        logprobs : np.ndarray
            Shape: (n_agents,)
        value : float
            Centralized value estimate.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        global_state_t = obs_t.reshape(1, -1)

        actions = []
        logprobs = []

        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                action_i, logprob_i = actor.get_action_and_logprob(obs_t[i].unsqueeze(0))
                actions.append(action_i.item())
                logprobs.append(logprob_i.item())

            value = self.critic(global_state_t).item()

        return np.array(actions), np.array(logprobs), value

    def compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        T = rewards.shape[0]

        advantages = torch.zeros_like(rewards, device=self.device)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
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

    def update(self, buffer: RolloutBuffer, next_obs: np.ndarray):
        cfg = self.cfg
        data = buffer.to_tensors(self.device)

        obs = data["obs"]                         # (T, n_agents, obs_dim)
        global_states = data["global_states"]     # (T, n_agents * obs_dim)
        actions = data["actions"]                 # (T, n_agents)
        old_logprobs = data["logprobs"]           # (T, n_agents)
        rewards = data["rewards"]                 # (T,)
        dones = data["dones"]                     # (T,)
        values = data["values"]                   # (T,)

        with torch.no_grad():
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            next_global_state = next_obs_t.reshape(1, -1)
            next_value = self.critic(next_global_state).squeeze(0)

        advantages, returns = self.compute_gae(rewards, dones, values, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = rewards.shape[0]
        batch_size = T * cfg.n_agents

        # Flatten agent-specific data for actor updates.
        flat_obs = obs.reshape(T * cfg.n_agents, cfg.obs_dim)
        flat_actions = actions.reshape(T * cfg.n_agents)
        flat_old_logprobs = old_logprobs.reshape(T * cfg.n_agents)

        # Repeat centralized advantage once per agent.
        flat_advantages = advantages.repeat_interleave(cfg.n_agents)

        indices = np.arange(batch_size)

        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, cfg.minibatch_size):
                mb_idx = indices[start:start + cfg.minibatch_size]
                mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=self.device)

                mb_obs = flat_obs[mb_idx_t]
                mb_actions = flat_actions[mb_idx_t]
                mb_old_logprobs = flat_old_logprobs[mb_idx_t]
                mb_advantages = flat_advantages[mb_idx_t]

                # Identify which actor each sample belongs to.
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
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_advantages

                actor_loss = -torch.min(unclipped, clipped).mean()
                entropy_loss = -entropy_terms.mean()

                self.actor_optimizer.zero_grad()
                total_actor_loss = actor_loss + cfg.entropy_coef * entropy_loss
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors.parameters(), cfg.max_grad_norm)
                self.actor_optimizer.step()

            # Critic update using centralized states.
            pred_values = self.critic(global_states)
            critic_loss = 0.5 * (returns - pred_values).pow(2).mean()

            self.critic_optimizer.zero_grad()
            (cfg.value_coef * critic_loss).backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
            self.critic_optimizer.step()


# ============================================================
# 5. Example Environment Interface
# ============================================================

class DummyMultiAgentEnv:
    """
    Replace this with your own environment.

    Required methods:
    - reset() -> obs
    - step(actions) -> next_obs, reward, done, info

    obs shape: (n_agents, obs_dim)
    actions shape: (n_agents,)
    reward: scalar cooperative reward
    done: bool
    """

    def __init__(self, n_agents: int, obs_dim: int, action_dim: int, max_steps: int = 100):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self) -> np.ndarray:
        self.step_count = 0
        return np.random.randn(self.n_agents, self.obs_dim).astype(np.float32)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.step_count += 1

        # Dummy dynamics.
        next_obs = np.random.randn(self.n_agents, self.obs_dim).astype(np.float32)

        # Dummy cooperative reward.
        # Replace this with your process objective, e.g., profit - penalties.
        reward = -np.mean(actions.astype(np.float32))

        done = self.step_count >= self.max_steps
        info = {}
        return next_obs, reward, done, info


# ============================================================
# 6. Training Loop
# ============================================================

def train():
    cfg = MAPPOConfig()
    env = DummyMultiAgentEnv(cfg.n_agents, cfg.obs_dim, cfg.action_dim)
    agent = MAPPO(cfg)
    buffer = RolloutBuffer(cfg)

    obs = env.reset()

    for update in range(cfg.total_updates):
        buffer.reset()
        episode_return = 0.0

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
            episode_return += reward

            if done:
                obs = env.reset()

        agent.update(buffer, next_obs=obs)

        if update % 10 == 0:
            print(f"Update {update:04d} | rollout return: {episode_return:.2f}")


if __name__ == "__main__":
    train()
