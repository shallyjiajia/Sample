from gym import spaces
import numpy as np
"""
Two kinds of valid input:
    Box(low=-1.0, high=1.0, shape=(3,4)) # low and high are scalars, and shape is provided
    Box(low=np.array([-1.0,-2.0]), high=np.array([2.0,4.0])) # low and high are arrays of the same shape
"""
#box = spaces.Box(low=3.0, high=4,shape=(1,4))
box = spaces.Box(low=np.array([0.5,1.0,10]), high=np.array([1.0,10.0,100]))

print(box.shape)

print(box.sample())