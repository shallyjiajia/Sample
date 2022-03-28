import gym
from gym import spaces
import numpy as np
# import logging
# from utils.logger import Logger

class BidEnv(gym.Env):
    def __init__(self,env_config):
        self.done = 0
        self.reward = 0
        self.info={"msg":"start"}
        self.env_config = env_config
        # self._log = Logger('bid.log', logging.ERROR, logging.DEBUG)

        self.observation_space_name = np.array(['marketrate', 'generatorrate'])
        self.state_lb = [0, 0]
        self.state_ub = [2, 1]
        self.observation_space = spaces.Box(np.array(self.state_lb), np.array(self.state_ub))

        self.action_space_name = ['quantity','price']
        self.action_lb = [0, 1]
        self.action_ub = [1, 1.2]
        self.action_space = spaces.Box(np.array(self.action_lb), np.array(self.action_ub))

        mr = self.env_config['marketrate']  # 电力市场状态
        gr =  self.env_config['q_Mon'] / self.env_config['Qd']  # 发电商自身状态
        self.observation = np.array([mr, gr])  # initial observation, current rates

    def step(self, action):  # take action，return observation,reward,done,info
        temp = self.reward
        self.observation = self._populateObservation(action)
        self.reward = self._populateReward(action)
        self.done = self._populateDone(temp)
        self.info = self._populateInfo()
        # self._log.debug(["step : observation,reward,done,info :", self.observation, self.reward, self.done, self.info])

        return self.observation, self.reward, self.done, self.info

    def _populateObservation(self,action):
        mr = self.env_config['marketrate']  # 电力市场状态
        q_Mon = self.env_config['q_Mon']
        Qd = self.env_config['Qd']
        gr = (1 - action[0]) * q_Mon / Qd  # 发电商自身状态
        # self._log.debug(["populate observation : action,mr,q_Mon,Qd,gr : ", action, mr, q_Mon, Qd, gr])

        return np.array([mr,gr])

    def _populateReward(self,action):
        # 发电商月度集中竞价交易收益
        TMon = self.env_config['TMon']
        q_Mon = self.env_config['q_Mon']*action[0]
        q_YD = self.env_config['q_YD']
        a = self.env_config['a']
        b = self.env_config['b']
        pmc = self.env_config['pmc']
        income = q_Mon * pmc
        cost = (a * q_Mon * (2 * q_YD + q_Mon) / TMon + b * q_Mon)/1000
        self.reward = income - cost
        # self._log.debug(["populate reward : income,cost,reward : ", income, cost, self.reward])

        return self.reward

    def _populateDone(self,temp):
        if abs(self.reward - temp) < 0.000001:
            return 0
        else:
            return 1

    def _populateInfo(self):
        return {"msg":"step"}

    def setenv_config(self,env_config):
        self.env_config = env_config

    def reset(self):
        mr = self.env_config['marketrate']  # 电力市场状态
        gr = self.env_config['q_Mon'] / self.env_config['Qd']  # 发电商自身状态
        self.observation = np.array([mr, gr])  # initial state, current rates
        self.done = 0
        self.reward = 0
        self.info = {"msg":"reset"}
        return self.observation

    def render(self):
        print("render")
        return None

    def close(self):
        print("env closed")