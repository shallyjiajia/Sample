from utils.RL.algorithm import Algorithm
#from .maze_env import Maze
#from .RL_brain import DeepQNetwork
#from .run_this import run_maze
from .dqn import DQN
import numpy as np
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class Test_DQN():
    def test(self):
        #bid = DQN()

        strategy = DQN()
        params = {"Param1": 1, "Param2": "Flop"}


        return strategy.run(params)

