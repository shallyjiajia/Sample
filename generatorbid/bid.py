from utils.RL.algorithm import Algorithm
from typing import Dict
# import sympy
import numpy as np

class Bid():
    def __init__(self) -> None:
        self._strategy = None
        self._params = None
        self._data =None

    @property
    def strategy(self) -> Algorithm:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Algorithm) -> None:
        self._strategy = strategy

    def setParams(self,params:Dict):
        self._params = params

    def setData(self,data:Dict):
        self._data = data

    def run(self):
        output = self._strategy.run(self._params)
        return output

    # def cost_GT(p, a, b, c):
    #     # 发电商的总发电成本
    #     return a * (p ** 2) + b * p + c
    #
    # def cost_GA(p, a, b, c):
    #     # 发电商的平均发电成本
    #     return a * p + b + c / p
    #
    # def cost_GM(p, a, b):
    #     # 发电商的边际发电成本
    #     return 2 * a * p + beta

    """
    发电商月度负荷曲线为p=power_GT(t)

    def power_GT(t):
        #发电商月度负荷曲线1
        pass
    def quan_GMon(TMon):
        #发电商月度发电量
        t = sympy.Symbol('t')
        p = power_GT(t)
        result = sympy.integrate(p,(t,0,TMon))

        return result

    def cost_GTMon(TMon,a,b,c):
        #发电商月度总发电成本
        t = sympy.Symbol('t')
        p = power_GT(t)
        co = cost_GT(p,a,b,c)
        result = sympy.integrate(p*co,(t,0,TMon))

        return result

    def cost_GAMon(TMon,a,b,c):
        #发电商月度平均发电成本
        return cost_GTMon(TMon,a,b,c)/quan_GMon(TMon)
    """

    """
    发电商月度持续负荷曲线为t=F(p)

    def F(p):
        pass

    def quan_GMon(pMax):
        #发电商月度发电量
        p = sympy.Symbol('p')
        t = F(p)
        result = sympy.integrate(t,(p,0,pMax))

        return result

    def cost_GTMon(pMax,a,b,c):
        #发电商月度总发电成本
        p = sympy.Symbol('p')
        t = F(p)
        co = cost_GT(p,a,b,c)
        result = sympy.integrate(t*co,(p,0,pMax))

        return result

    def cost_GAMon(TMon,a,b,c):
        #发电商月度平均发电成本
        return cost_GTMon(TMon,a,b,c)/quan_GMon(TMon)
    """

    # def power_GAMon(q_YD, q_Mon, TMon):
    #     # 发电商月度平均发电出力
    #     return (q_YD + q_Mon) / TMon
    #
    # def cost_GMMon(q_YD, q_Mon, TMon, a, b):
    #     # 发电商月度平均发电出力对应的边际成本
    #     return cost_GM(power_GAMon(q_YD, q_Mon, TMon), a, b)

    # def income_GMon(q_Mon, pmc):
    #     # 发电商月度集中竞价交易收入
    #     return q_Mon * pmc

    # def reward_GMon(q_YD, q_Mon, TMon, a, b, pmc):
    #     # 发电商月度集中竞价交易收益
    #     income = q_Mon * pmc
    #     cost = a * q_Mon * (2 * q_YD + q_Mon) / TMon + b * q_Mon
    #     reward = income - cost
    #
    #     return reward
    #
    # def state(act, q_Mon, marketrate, Qd):
    #     # 状态
    #     r = marketrate  # 电力市场状态
    #     sg = (1 - act[0]) * q_Mon / Qd  # 发电商自身状态
    #
    #     return np.array([r, sg])

