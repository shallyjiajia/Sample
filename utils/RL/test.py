from utils.RL.algorithm import Algorithm
import io
import numpy as np
import matplotlib as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class A3C(Algorithm):
    def run(self, params):
        for key,value in params.items():
            print(key,value)
        return "success"



