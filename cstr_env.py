import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.integrate import solve_ivp


class CSTRCoolingEnv(gym.Env):
    """Three-CSTR cooperative environment for multi-agent PPO.

    Reactor 1 converts feed A into an intermediate M. The outlet from reactor 1
    is split equally into reactors 2 and 3, which consume M to produce P1 and P2.
    Each agent controls the jacket temperature of one reactor.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.n_agents = 3
        self.obs_dim = 8

        # Reactor and reaction parameters
        self.V = 100.0  # Reactor volume (L)
        self.q = 20.0  # Reactor 1 feed flowrate (L/min)
        self.q_downstream = self.q / 2.0  # Split flow to reactors 2 and 3
        self.Caf = 1.0  # Fresh feed concentration of A (mol/L)
        self.Tf = 350.0  # Fresh feed temperature (K)
        self.k0 = 15.2e10  # 1/min
        self.EoverR = 10000.0  # K
        self.mdelH = 5e4  # J/mol, positive value for -delta H
        self.rho = 1000.0  # g/L
        self.Cp = 0.239  # J/g-K
        self.UA = 5e4  # J/min-K

        # Control settings. Action 0 is hottest jacket, action 4 is coldest.
        self.Tj_base = 300.0
        self.Tj_delta = 30.0
        self.jacket_temperatures = np.array(
            [
                self.Tj_base + 2.0 * self.Tj_delta,
                self.Tj_base + self.Tj_delta,
                self.Tj_base,
                self.Tj_base - self.Tj_delta,
                self.Tj_base - 2.0 * self.Tj_delta,
            ],
            dtype=np.float32,
        )

        self.dt = 1.0  # time per step (min)
        self.max_steps = 200
        self.possible_setpoints = [290, 300, 310, 320, 330, 340, 350, 360, 370]

        # Normalization ranges
        self.C_min = 0.0
        self.C_max = 1.2
        self.T_min = 250.0
        self.T_max = 500.0
        self.product_rate_max = self.q * self.Caf

        self.action_space = spaces.MultiDiscrete([5, 5, 5])
        self.single_action_space = spaces.Discrete(5)

        low_obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        high_obs = np.ones((self.n_agents, self.obs_dim), dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # State order: [C1, T1, C2, T2, C3, T3].
        self.state = None
        self.last_jacket_temperatures = np.array([self.Tj_base] * self.n_agents, dtype=np.float32)
        self.steps = 0
        self.setpoints = self.generate_sectioned_list(self.possible_setpoints)

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n_agents,):
            raise ValueError(f"Expected {self.n_agents} actions, got shape {actions.shape}.")
        if np.any(actions < 0) or np.any(actions >= len(self.jacket_temperatures)):
            raise ValueError(f"Actions must be integers in [0, {len(self.jacket_temperatures) - 1}].")

        self.last_jacket_temperatures = self.jacket_temperatures[actions]
        sp = self.setpoints[self.steps]

        sol = solve_ivp(
            self._odes,
            [0.0, self.dt],
            self.state,
            args=(self.last_jacket_temperatures,),
            method="RK45",
        )
        if not sol.success:
            raise RuntimeError(f"CSTR integration failed: {sol.message}")

        next_state = sol.y[:, -1]
        next_state[[0, 2, 4]] = np.clip(next_state[[0, 2, 4]], self.C_min, self.C_max)
        next_state[[1, 3, 5]] = np.clip(next_state[[1, 3, 5]], self.T_min, self.T_max)
        self.state = next_state.astype(np.float32)

        p1_rate, p2_rate = self.product_rates()
        temps = self.state[[1, 3, 5]]
        temp_penalty = np.mean(np.abs(temps - sp)) / 100.0
        safety_penalty = np.mean(np.maximum(temps - 430.0, 0.0)) / 25.0
        product_reward = (p1_rate + p2_rate) / self.product_rate_max
        balance_penalty = abs(p1_rate - p2_rate) / self.product_rate_max
        reward = float(product_reward - 0.2 * temp_penalty - safety_penalty - 0.1 * balance_penalty)

        self.steps += 1
        done = self.steps >= self.max_steps

        obs = self._get_obs()
        info = {
            "P1_rate": float(p1_rate),
            "P2_rate": float(p2_rate),
            "intermediate_concentration": float(self.intermediate_concentration()),
            "temperature_setpoint": float(sp),
            "jacket_temperatures": self.last_jacket_temperatures.copy(),
        }
        return obs, reward, done, info

    def _odes(self, _t, y, jacket_temperatures):
        c1, t1, c2, t2, c3, t3 = y
        tj1, tj2, tj3 = jacket_temperatures

        c1 = max(c1, 0.0)
        c2 = max(c2, 0.0)
        c3 = max(c3, 0.0)

        r1 = self.reaction_rate(c1, t1)
        intermediate_feed = max(self.Caf - c1, 0.0)

        r2 = self.reaction_rate(c2, t2)
        r3 = self.reaction_rate(c3, t3)

        dc1dt, dt1dt = self.reactor_balance(c1, t1, self.Caf, self.Tf, self.q, tj1, r1)
        dc2dt, dt2dt = self.reactor_balance(
            c2, t2, intermediate_feed, t1, self.q_downstream, tj2, r2
        )
        dc3dt, dt3dt = self.reactor_balance(
            c3, t3, intermediate_feed, t1, self.q_downstream, tj3, r3
        )

        return [dc1dt, dt1dt, dc2dt, dt2dt, dc3dt, dt3dt]

    def reactor_balance(self, c, t, c_feed, t_feed, q, tj, reaction_rate):
        dcdt = q / self.V * (c_feed - c) - reaction_rate
        dtdt = (
            q / self.V * (t_feed - t)
            + self.mdelH / (self.rho * self.Cp) * reaction_rate
            + self.UA / (self.rho * self.Cp * self.V) * (tj - t)
        )
        return dcdt, dtdt

    def reaction_rate(self, concentration, temperature):
        temperature = max(float(temperature), 1.0)
        return self.k0 * np.exp(-self.EoverR / temperature) * concentration

    def intermediate_concentration(self):
        c1 = float(self.state[0])
        return max(self.Caf - c1, 0.0)

    def product_rates(self):
        intermediate_feed = self.intermediate_concentration()
        c2 = float(self.state[2])
        c3 = float(self.state[4])
        p1_rate = self.q_downstream * max(intermediate_feed - c2, 0.0)
        p2_rate = self.q_downstream * max(intermediate_feed - c3, 0.0)
        return p1_rate, p2_rate

    def generate_sectioned_list(self, C, num_sections=10, section_size=20):
        result = []
        for _ in range(num_sections):
            value = random.choice(C)
            section = [value] * section_size
            result.extend(section)
        return result

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        c1_0 = self.Caf
        t1_0 = 350.0 + self.np_random.uniform(-5.0, 5.0)
        c2_0 = 0.0
        t2_0 = 350.0 + self.np_random.uniform(-5.0, 5.0)
        c3_0 = 0.0
        t3_0 = 350.0 + self.np_random.uniform(-5.0, 5.0)

        self.setpoints = self.generate_sectioned_list(self.possible_setpoints)
        self.last_jacket_temperatures = np.array([self.Tj_base] * self.n_agents, dtype=np.float32)
        self.state = np.array([c1_0, t1_0, c2_0, t2_0, c3_0, t3_0], dtype=np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        c1, t1, c2, t2, c3, t3 = self.state
        intermediate = self.intermediate_concentration()
        p1_rate, p2_rate = self.product_rates()
        sp = self.setpoints[min(self.steps, len(self.setpoints) - 1)]
        step_frac = self.steps / self.max_steps

        raw_obs = np.array(
            [
                [c1, t1, self.last_jacket_temperatures[0], self.Caf, p1_rate, p2_rate, sp, step_frac],
                [c2, t2, self.last_jacket_temperatures[1], intermediate, p1_rate, p2_rate, sp, step_frac],
                [c3, t3, self.last_jacket_temperatures[2], intermediate, p1_rate, p2_rate, sp, step_frac],
            ],
            dtype=np.float32,
        )
        return self.normalize_obs(raw_obs)

    def normalize_obs(self, obs):
        norm = obs.copy()
        norm[:, 0] = (norm[:, 0] - self.C_min) / (self.C_max - self.C_min)
        norm[:, 1] = (norm[:, 1] - self.T_min) / (self.T_max - self.T_min)
        norm[:, 2] = (norm[:, 2] - self.T_min) / (self.T_max - self.T_min)
        norm[:, 3] = (norm[:, 3] - self.C_min) / (self.C_max - self.C_min)
        norm[:, 4] = norm[:, 4] / self.product_rate_max
        norm[:, 5] = norm[:, 5] / self.product_rate_max
        norm[:, 6] = (norm[:, 6] - self.T_min) / (self.T_max - self.T_min)
        norm[:, 7] = np.clip(norm[:, 7], 0.0, 1.0)
        return np.clip(norm, 0.0, 1.0).astype(np.float32)

    def render(self, mode="human"):
        c1, t1, c2, t2, c3, t3 = self.state
        p1_rate, p2_rate = self.product_rates()
        print(
            f"Step: {self.steps}, "
            f"R1(C_A={c1:.3f}, T={t1:.2f}), "
            f"R2(C_M={c2:.3f}, T={t2:.2f}, P1={p1_rate:.3f}), "
            f"R3(C_M={c3:.3f}, T={t3:.2f}, P2={p2_rate:.3f})"
        )

    def close(self):
        pass