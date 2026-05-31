import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from scipy.integrate import solve_ivp

class CSTRCoolingEnv(gym.Env):
    def __init__(self):
        super(CSTRCoolingEnv, self).__init__()
        
        # Parameters
        self.V = 100             # Reactor volume (L)
        self.q = 20             # Flowrate (L/min)
        self.Caf = 1.0           # Feed concentration (mol/L)
        self.Tf = 350            # Feed temperature (K)
        self.k0 = 15.2e10         # 1/min
        self.EoverR = 10000       # K
        self.mdelH = 5e4         # J/mol
        self.rho = 1000          # g/L
        self.Cp = 0.239          # J/g-K
        self.UA = 5e4            # J/min-K
        self.Tj_base = 300       # K
        self.Tj_delta = 30        # delta per action level
        self.dt = 1.0            # time per step (min)
        self.max_steps = 200
        self.possible_setpoints = [290,300,310,320,330,340,350,360,370]

        # Normalization ranges
        self.Ca_min = 0.0
        self.Ca_max = 1.2      # expected max around feed conc.
        
        self.T_min = 300.0
        self.T_max = 400.0      # may vary depending on dynamics


        # Action space: 3 discrete cooling settings
        self.action_space = spaces.Discrete(5)

        # Observation space: [Ca, T]
        low_obs = np.array([0.0, 300.0, 300.0], dtype=np.float32)
        high_obs = np.array([2.0, 500.0, 500.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.state = None
        self.steps = 0

    def step(self, actions):
        action1, action2, action3 = actions

        Ca, T, sp = self.state

        sp = self.setpoints[self.steps]

        # Determine Tj based on action
        if action1 == 0:
            Tj1 = self.Tj_base + 2*self.Tj_delta
        elif action1 == 1:
            Tj1 = self.Tj_base + self.Tj_delta
        elif action1 == 2:
            Tj1 = self.Tj_base

        if action2 == 0:
            Tj2 = self.Tj_base + 2*self.Tj_delta
        elif action2 == 1:
            Tj2 = self.Tj_base + self.Tj_delta
        elif action2 == 2:
            Tj2 = self.Tj_base

        if action3 == 0:
            Tj3 = self.Tj_base + 2*self.Tj_delta
        elif action3 == 1:
            Tj3 = self.Tj_base + self.Tj_delta
        elif action3 == 2:
            Tj3 = self.Tj_base

        # Integrate ODEs for 1 time step
        # Reaction 1
        def cstr_odes(t, y):
            Ca, T = y
            rA = self.k0 * np.exp(-self.EoverR / T) * Ca
            dCadt = self.q / self.V * (self.Caf - Ca) - rA
            dTdt = (self.q / self.V * (self.Tf - T)
                    + (-self.mdelH) / (self.rho * self.Cp) * rA
                    + self.UA / (self.rho * self.Cp * self.V) * (Tj1 - T))
            return [dCadt, dTdt]

        sol = solve_ivp(cstr_odes, [0, self.dt], [Ca, T], method='RK45')
        Ca_next, T_next = sol.y[:, -1]

        self.state = np.array([Ca_next, T_next, sp], dtype=np.float32)
        
        self.state = np.concat(state1,state2,state3)



        # Reward: penalize deviation from setpoint
        reward = (-abs(T_next - sp))/100

        self.steps += 1
        done = self.steps >= self.max_steps

        obs = self.normalize_state(*self.state)
        return obs, reward, done, {}

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
        Ca0 = 1.0 
        T0 = 350.0 + self.np_random.uniform(-5.0, 5.0)  #5
        sp0 = 300.0
        C = self.possible_setpoints
        self.setpoints = self.generate_sectioned_list(C)
        self.state = np.array([Ca0, T0, sp0], dtype=np.float32)
        obs = self.normalize_state(*self.state)
        return obs, {}

    def render(self, mode='human'):
        Ca, T = self.state
        print(f"Step: {self.steps}, Ca: {Ca:.3f}, T: {T:.2f}")

    def normalize_state(self, Ca, T, sp):
        Ca_norm = (Ca - self.Ca_min) / (self.Ca_max - self.Ca_min)
        T_norm = (T - self.T_min) / (self.T_max - self.T_min)
        sp_norm = (sp - self.T_min) / (self.T_max - self.T_min)
        return np.array([Ca_norm, T_norm, sp_norm], dtype=np.float32)

    def close(self):
        pass
