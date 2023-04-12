import gym
from gym import spaces
import logging
import random
import time

import numpy as np
from gym.envs.classic_control import rendering

logger = logging.getLogger(__name__)


class SafeGrideWorld_Gym(gym.Env):
    """Custom Environment that follows gym interface."""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10}

    def __init__(self, gride_size=(4, 4), frisbee_state=None, holes_state=None, init_state=12,
                 transition_prob=0.8, max_steps=60):
        """
        This is the safe gride world in standard gym format.
        :param gridSize:        size of gride world in tuple(gridWidth, gridHeight).
        :param frisbee_state:   final goal states in list.
        :param holes_state:     holes states in list
        :param init_state:      init state for agent to start. Random position if "None".
        :param transition_prob: transition probability of env
        :param max_steps:       max steps that agent travel in env. set none if no limit.

        Default Map:
            # | A |   |   |   |       | 12| 13| 14| 15|
            # |   | 0 |   | 0 |       | 8 | 9 | 10| 11|
            # |   |   |   |   |       | 4 | 5 | 6 | 7 |
            # | 0 |   |   | x |       | 0 | 1 | 2 | 3 |
        """
        # Standard Gym Requirement
        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=float('-inf'), high=float('inf'), shape=(1,), dtype=np.float32)
        self.reward_range = (-float("inf"), float("inf"))

        # Environment information
        self.max_steps = max_steps
        self.num_step = 0
        self.episode = 0

        # Load the parameters
        self.gridWidth = gride_size[0]
        self.gridHeight = gride_size[1]
        # Environment configuration
        self.gridNum = self.gridWidth * self.gridHeight
        self.states = range(self.gridNum)
        if frisbee_state is None and holes_state is None:
            frisbee_state = [3]
            holes_state = [9, 11, 0]
        elif not (frisbee_state is not None and holes_state is not None):
            raise ValueError("Set the 'frisbee_state' and 'holes_state' simultaneously.")
        self.frisbee_states = frisbee_state
        self.holes_states = holes_state
        if init_state is None:
            self.init_state = None
            while True:
                self.state = np.random.choice(self.states)
                if (self.state not in self.frisbee_states) and (self.state not in self.holes_states):
                    break
            self.init_random = True
        else:
            self.init_state = init_state
            self.init_random = False
        self.transition_prob = transition_prob

        # Bound configuration
        #   get the boundary state using grid confiuration
        self.upBound = range(self.gridNum - self.gridWidth, self.gridNum)
        self.rightBound = []
        for i in range(self.gridHeight):
            self.rightBound.append((self.gridWidth - 1) + i * self.gridWidth)
        self.downBound = range(self.gridWidth)
        self.leftBound = []
        for i in range(self.gridHeight):
            self.leftBound.append(0 + i * self.gridWidth)

        # Action space
        #   create action dictionary, save the action and the change value of the state
        self.actionsDic = {0: self.gridWidth, 1: 1, 2: -self.gridWidth, 3: -1}
        # self.action_space = 5
        # self.actions = ['up', 'right', 'down', 'left', 'idle']
        # self.actionsDic = {'up': self.gridWidth, 'right': 1, 'down': -self.gridWidth, 'left': -1, 'idle': 0}

        # Reward Function
        #   save the reward for each action in array
        self.reward_function = np.zeros(self.gridNum)
        for state in self.frisbee_states:
            self.reward_function[state] = 1
        for state in self.holes_states:
            self.reward_function[state] = -1

        # Screen Configuration
        self.screen_width = (self.gridWidth + 2) * 100
        self.screen_height = (self.gridHeight + 2) * 100

        # Agent information
        self.gamma = 0.9
        self.state = None

        # Gym viewering
        self.viewer = None
        self.is_render = False
        self.is_sleep = False

    def step(self, action):
        observation = np.resize(np.array(self.state, dtype=np.float32), new_shape=(1,))
        reward = 0
        is_terminal = False
        info = {'goal': True, 'max_step': False}

        # Check env
        #   if env is in the terminal state, return.
        #   in case star in terminal state or did not break out
        if self.state in self.frisbee_states:
            is_terminal = True
            info['goal'] = True
            return observation, reward, is_terminal, {'goal': True}
        elif self.state in self.holes_states:
            is_terminal = True
            return observation, reward, is_terminal, {'goal': False}

        # State transition
        #   transfer the state according to action;
        #   if action out of the bound, then set the state back.
        # deterministic and non-deterministic
        state = self.state
        if random.uniform(0, 1) < self.transition_prob:
            next_state = state + self.actionsDic[action]
        else:
            # equal possibility for other actions
            _action = action
            while _action == action:
                _action = self.action_space.sample()
            action = _action
            next_state = state + self.actionsDic[action]

        # consider Boundary
        if action == 0 and state in self.upBound:
            next_state = state
        elif action == 1 and state in self.rightBound:
            next_state = state
        elif action == 2 and state in self.downBound:
            next_state = state
        elif action == 3 and state in self.leftBound:
            next_state = state
        self.state = next_state

        # Reward
        reward = self.reward_function[next_state]

        # Terminal state
        is_terminal = False
        info['goal'] = False
        if next_state in self.frisbee_states:
            is_terminal = True
            info['goal'] = True
        elif next_state in self.holes_states:
            is_terminal = True

        # Render
        # if self.is_render:
        #     self.render()
        #     if self.is_sleep:
        #         time.sleep(.1)
        self.num_step += 1
        if self.num_step == self.max_steps - 1:
            is_terminal = True
            info['max_step'] = True

        observation = np.resize(np.array(self.state, dtype=np.float32), new_shape=(1,))
        return observation, reward, is_terminal, info

    def reset(self):
        self.episode += 1
        self.num_step = 0
        print("episode", self.episode)
        observation = None
        if self.init_random:
            while True:
                self.state = np.random.choice(self.states)
                if (self.state not in self.frisbee_states) or (self.state not in self.holes_states):
                    break
        else:
            self.state = self.init_state
        observation = np.resize(np.array(self.state, dtype=np.float32), new_shape=(1,))
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        # Initial render setting
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            # Create the gride
            self.lines = []
            for i in range(1, self.gridHeight + 2):
                self.lines.append(rendering.Line((100, i * 100), ((self.gridWidth + 1) * 100, i * 100)))
            for i in range(1, self.gridWidth + 2):
                self.lines.append(rendering.Line((i * 100, 100), (i * 100, (self.gridHeight + 1) * 100)))

            # Create Frisbee
            self.frisbee = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(150 + 100 * (self.frisbee_states[0] % self.gridWidth),
                                                                150 + 100 * (self.frisbee_states[0] // self.gridWidth)))
            self.frisbee.add_attr(self.circletrans)

            # Create holes
            self.holes = []
            for i in range(len(self.holes_states)):
                state = self.holes_states[i]
                self.circletrans = rendering.Transform(translation=(150 + 100 * (state % self.gridWidth),
                                                                    150 + 100 * (state // self.gridWidth)))
                self.holes.append(rendering.make_circle(35))
                self.holes[i].add_attr(self.circletrans)

            # Create robot
            self.agent = rendering.make_circle(30)
            self.robotrans = rendering.Transform(translation=(150 + 100 * (self.init_state % self.gridWidth),
                                                              150 + 100 * (self.init_state // self.gridWidth)))
            self.agent.add_attr(self.robotrans)

            # Set color and add to viewer
            # lines
            for line in self.lines:
                line.set_color(24 / 255, 24 / 255, 24 / 255)
                self.viewer.add_geom(line)

            # frisbee
            self.frisbee.set_color(230 / 255, 44 / 255, 44 / 255)
            self.viewer.add_geom(self.frisbee)

            # holes
            for hole in self.holes:
                hole.set_color(54 / 255, 54 / 255, 54 / 255)
                self.viewer.add_geom(hole)

            # agent
            self.agent.set_color(118 / 255, 238 / 255, 0 / 255)
            self.viewer.add_geom(self.agent)

        if self.state is None:
            return None

        # Move the robot
        self.robotrans.set_translation(150 + 100 * (self.state % self.gridWidth),
                                       150 + 100 * (self.state // self.gridWidth))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()


if __name__ == '__main__':
    simple_test = True
    if simple_test:
        # Environment - 4 x 4
        gridSize = (4, 4)
        frisbee_state = [3]
        holes_state = [9, 11, 0]
        init_state = 12
        transition_prob = 0.8
        # Environment - 10 x 10
        # gridSize = (10, 10)
        # frisbee_state = [9]
        # holes_state = [29, 39, 49, 59, 69, 79, 89, 99, 98, 97, 96, 95, 94, 93, 92,
        #                77, 75, 73, 57, 37,
        #                55, 53, 35, 33]
        # init_state = 90
        # transition_prob = 0.8

        env = SafeGrideWorld_Gym(gride_size=gridSize, frisbee_state=frisbee_state,
                                 holes_state=holes_state, init_state=init_state,
                                 transition_prob=transition_prob, max_steps=60)
        print("make done!")

        env.reset()
        env.render()
        for i in range(20):
            action = env.action_space.sample()
            print('action', action)
            s, r, d, info = env.step(action)
            env.render()
            time.sleep(1)
            if d:
                break

    else:
        from stable_baselines3 import A2C
        from stable_baselines3.common.env_checker import check_env

        # Environment - 4 x 4
        gridSize = (4, 4)
        frisbee_state = [3]
        holes_state = [9, 11, 0]
        init_state = 12
        transition_prob = 0.8
        # Environment - 10 x 10
        # gridSize = (10, 10)
        # frisbee_state = [9]
        # holes_state = [29, 39, 49, 59, 69, 79, 89, 99, 98, 97, 96, 95, 94, 93, 92,
        #                77, 75, 73, 57, 37,
        #                55, 53, 35, 33]
        # init_state = 90
        # transition_prob = 0.8

        env = SafeGrideWorld_Gym(gride_size=gridSize, frisbee_state=frisbee_state,
                                 holes_state=holes_state, init_state=init_state,
                                 transition_prob=transition_prob, max_steps=60)
        print("make done!")

        check_env(env)
        print("check done!")

        print("ready to train!")
        model = A2C("MlpPolicy", env, device="cuda:0").learn(total_timesteps=10000)
        print("model done!")
