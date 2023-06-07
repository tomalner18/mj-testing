import numpy as np
import time
from mujoco_py import const, MjViewer
import glfw
from gym.spaces import Box, MultiDiscrete, Discrete
import time


class RolloutViewer():

    def __init__(self, env):
        self.env = env
        self.elapsed = []
        self.seed = self.env.seed()
        self.n_agents = self.env.metadata['n_actors']
        self.action_types = list(self.env.action_space.spaces.keys())
        self.num_action_types = len(self.env.action_space.spaces)
        self.num_action = self.num_actions(self.env.action_space)
        self.agent_mod_index = 0
        self.action_mod_index = 0
        self.action_type_mod_index = 0
        self.horizon = 100
        self.action = self.zero_action(self.env.action_space)
        self.env_reset()

    def num_actions(self, ac_space):
        n_actions = []
        for k, tuple_space in ac_space.spaces.items():
            s = tuple_space.spaces[0]
            if isinstance(s, Box):
                n_actions.append(s.shape[0])
            elif isinstance(s, Discrete):
                n_actions.append(1)
            elif isinstance(s, MultiDiscrete):
                n_actions.append(s.nvec.shape[0])
            else:
                raise NotImplementedError(f"not NotImplementedError")

        return n_actions

    def zero_action(self, ac_space):
        ac = {}
        for k, space in ac_space.spaces.items():
            if isinstance(space.spaces[0], Box):
                ac[k] = np.zeros_like(space.sample())
            elif isinstance(space.spaces[0], Discrete):
                ac[k] = np.ones_like(space.sample()) * (space.spaces[0].n // 2)
            elif isinstance(space.spaces[0], MultiDiscrete):
                ac[k] = np.ones_like(space.sample(), dtype=int) * (space.spaces[0].nvec // 2)
            else:
                raise NotImplementedError("MultiDiscrete not NotImplementedError")
                # return action_space.nvec // 2  # assume middle element is "no action" action
        return ac

    def env_reset(self):
        start = time.time()
        # get the seed before calling env.reset(), so we display the one
        # that was used for the reset.
        self.seed = self.env.seed()
        self.env.reset()
        self.elapsed.append(time.time() - start)


    def run(self, size=1, once=False):
        
        rts = []
        st = time.time()
        for i in range(self.horizon):
            _, _, _, env_info = self.env.step(self.action)


