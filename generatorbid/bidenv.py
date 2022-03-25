import gym
from gym import spaces
import numpy as np

class BidEnv(gym.Env):
    def __init__(self,data):
        self.done = 0
        self.reward = 0
        self.info="start"
        self._data = data

        self.observation_space_name = np.array(['marketrate', 'generatorrate'])
        self.state_lb = [0, 0]
        self.state_ub = [2, 1]
        self.observation_space = spaces.Box(np.array(self.state_lb), np.array(self.state_ub))

        self.action_space_name = ['price', 'quantity']
        self.action_lb = [0, 1]
        self.action_ub = [1, 1.2]
        self.action_space = spaces.Box(np.array(self.action_lb), np.array(self.action_ub))

        self.observation = np.array([1.29, 0.012])  # initial observation, current rates

    def step(self, action):  # take action，return observation,reward,done,info
        self.observation = self._populateObservation(action)
        self.reward = self._populateReward(action)
        self.done = self._populateDone()
        self.info = self._populateInfo()

        return self.observation, self.reward, self.done, self.info

    def _populateObservation(self,action):
        print(action)
        mr = self._data['marketrate']  # 电力市场状态
        q_Mon = self._data['q_Mon']
        Qd = self._data['Qd']
        gr = (1 - action[0]) * q_Mon / Qd  # 发电商自身状态
        return np.array([mr,gr])

    def _populateReward(self,action):
        # 发电商月度集中竞价交易收益
        TMon = self._data['TMon']
        q_Mon = self._data['q_Mon']*action[0]
        q_YD = self._data['q_YD']
        a = self._data['a']
        b = self._data['b']
        pmc = self._data['pmc']
        income = q_Mon * pmc
        cost = a * q_Mon * (2 * q_YD + q_Mon) / TMon + b * q_Mon
        self.reward = income - cost
        return self.reward

    def _populateDone(self):
        if self.reward>5:
            return 1
        else:
            return 0

    def _populateInfo(self):
        return "step"

    def setData(self,data):
        self._data = data

    def reset(self):
        self.observation = np.array([1.29, 0.012])  # initial state, current rates
        self.done = 0
        self.reward = 0
        self.info = "reset"
        return self.observation

    def render(self):
        print("render")
        return None

    def close(self):
        print("env closed")