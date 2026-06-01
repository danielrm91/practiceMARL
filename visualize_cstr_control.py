"""Visualize three-reactor CSTR control decisions.

This script trains a small MAPPO agent by default, rolls out one evaluation
episode, and saves plots that show how the three reactor temperature controls
evolve over time.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical

from cstr_env import CSTRCoolingEnv
from MAPPO import MAPPO, MAPPOConfig, RolloutBuffer


def train_agent(cfg: MAPPOConfig, log_every: int = 10) -> MAPPO:
    env = CSTRCoolingEnv()
    agent = MAPPO(cfg)
    buffer = RolloutBuffer()
    obs, _ = env.reset(seed=cfg.seed)

    for update in range(cfg.total_updates):
        buffer.reset()

        for _ in range(cfg.rollout_steps):
            global_state = obs.reshape(-1)
            actions, logprobs, value = agent.select_actions(obs)
            next_obs, reward, done, _ = env.step(actions)

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
            if done:
                obs, _ = env.reset()

        agent.update(buffer, next_obs=obs)

        if (update + 1) % max(log_every, 1) == 0:
            print(f"Finished training update {update + 1}/{cfg.total_updates}")

    env.close()
    return agent


def policy_action_probabilities(agent: MAPPO, obs: np.ndarray) -> np.ndarray:
    obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device)
    probabilities = []

    with torch.no_grad():
        for i, actor in enumerate(agent.actors):
            logits = actor(obs_t[i].unsqueeze(0))
            probs = Categorical(logits=logits).probs.squeeze(0)
            probabilities.append(probs.cpu().numpy())

    return np.array(probabilities)


def deterministic_actions_from_policy(agent: MAPPO, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = policy_action_probabilities(agent, obs)
    actions = np.argmax(probs, axis=1).astype(np.int64)
    return actions, probs


def run_episode(agent: MAPPO | None, seed: int, random_policy: bool = False) -> dict[str, np.ndarray]:
    env = CSTRCoolingEnv()
    obs, _ = env.reset(seed=seed)

    history = {
        "state": [],
        "actions": [],
        "action_probs": [],
        "jacket_temperatures": [],
        "p1_rate": [],
        "p2_rate": [],
        "intermediate": [],
        "setpoint": [],
        "reward": [],
    }

    done = False
    while not done:
        if random_policy:
            actions = env.action_space.sample()
            action_probs = np.full((env.n_agents, env.single_action_space.n), 1.0 / env.single_action_space.n)
        else:
            actions, action_probs = deterministic_actions_from_policy(agent, obs)

        next_obs, reward, done, info = env.step(actions)

        history["state"].append(env.state.copy())
        history["actions"].append(actions.copy())
        history["action_probs"].append(action_probs.copy())
        history["jacket_temperatures"].append(info["jacket_temperatures"].copy())
        history["p1_rate"].append(info["P1_rate"])
        history["p2_rate"].append(info["P2_rate"])
        history["intermediate"].append(info["intermediate_concentration"])
        history["setpoint"].append(info["temperature_setpoint"])
        history["reward"].append(reward)

        obs = next_obs

    env.close()
    return {key: np.array(value) for key, value in history.items()}


def plot_history(history: dict[str, np.ndarray], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state = history["state"]
    actions = history["actions"]
    action_probs = history["action_probs"]
    jackets = history["jacket_temperatures"]
    p1_rate = history["p1_rate"]
    p2_rate = history["p2_rate"]
    intermediate = history["intermediate"]
    setpoint = history["setpoint"]
    reward = history["reward"]
    steps = np.arange(len(reward))

    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

    axes[0].plot(steps, state[:, 1], label="Reactor 1")
    axes[0].plot(steps, state[:, 3], label="Reactor 2")
    axes[0].plot(steps, state[:, 5], label="Reactor 3")
    axes[0].plot(steps, setpoint, "k--", linewidth=1.2, label="Setpoint")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("Reactor Temperatures")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].step(steps, jackets[:, 0], where="post", label="Agent 1 / Reactor 1")
    axes[1].step(steps, jackets[:, 1], where="post", label="Agent 2 / Reactor 2")
    axes[1].step(steps, jackets[:, 2], where="post", label="Agent 3 / Reactor 3")
    axes[1].set_ylabel("Jacket T (K)")
    axes[1].set_title("Control Decisions")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, p1_rate, label="P1")
    axes[2].plot(steps, p2_rate, label="P2")
    axes[2].plot(steps, intermediate, label="Intermediate M")
    axes[2].set_ylabel("Rate / Concentration")
    axes[2].set_title("Production")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].step(steps, actions[:, 0], where="post", label="Agent 1")
    axes[3].step(steps, actions[:, 1], where="post", label="Agent 2")
    axes[3].step(steps, actions[:, 2], where="post", label="Agent 3")
    axes[3].set_ylabel("Action")
    axes[3].set_title("Discrete Actions, 0 = Hottest and 4 = Coldest")
    axes[3].set_yticks([0, 1, 2, 3, 4])
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(steps, reward, label="Step reward")
    axes[4].plot(steps, np.cumsum(reward), label="Cumulative reward")
    axes[4].set_xlabel("Step")
    axes[4].set_ylabel("Reward")
    axes[4].set_title("Reward")
    axes[4].legend(loc="best")
    axes[4].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    prob_path = output_path.with_name(f"{output_path.stem}_action_probabilities{output_path.suffix}")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for agent_id, ax in enumerate(axes):
        for action_id in range(action_probs.shape[2]):
            ax.plot(steps, action_probs[:, agent_id, action_id], label=f"Action {action_id}")
        ax.set_ylabel(f"Agent {agent_id + 1}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=5, fontsize=8)

    axes[-1].set_xlabel("Step")
    fig.suptitle("Policy Action Probabilities")
    fig.tight_layout()
    fig.savefig(prob_path, dpi=160)
    plt.close(fig)

    print(f"Saved control plot to {output_path}")
    print(f"Saved action probability plot to {prob_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CSTR MAPPO control behavior.")
    parser.add_argument("--random-policy", action="store_true", help="Use random actions instead of MAPPO.")
    parser.add_argument("--updates", type=int, default=50, help="Training updates before visualization.")
    parser.add_argument("--rollout-steps", type=int, default=128, help="Training rollout steps per update.")
    parser.add_argument("--ppo-epochs", type=int, default=3, help="PPO epochs per update.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Actor and critic hidden dimension.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--log-every", type=int, default=10, help="Training progress print interval.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "cstr_control_rollout.png",
        help="Output image path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.random_policy:
        trained_agent = None
    else:
        cfg = MAPPOConfig(
            rollout_steps=args.rollout_steps,
            total_updates=args.updates,
            ppo_epochs=args.ppo_epochs,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
        )
        trained_agent = train_agent(cfg, log_every=args.log_every)

    rollout = run_episode(trained_agent, seed=args.seed + 100, random_policy=args.random_policy)
    plot_history(rollout, args.output)
