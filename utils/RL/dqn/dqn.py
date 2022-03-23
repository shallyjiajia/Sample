from utils.RL.algorithm import Algorithm
from .maze_env import Maze
from .RL_brain import DeepQNetwork
import numpy as np
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class DQN(Algorithm):
    def __init__(self) -> None:
        self._strategy = None
        self._params = None

    @property
    def strategy(self) -> Algorithm:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Algorithm) -> None:
        self._strategy = strategy

    def setParams(self,params):
        self._params = params

    def run(self, params):
        for key,value in params.items():
            print(key,value)

        env = Maze()
        RL = DeepQNetwork(env.n_actions, env.n_features,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=100,
                          memory_size=500,
                          # output_graph=True
                          )
        env.after(100, _run_maze(env, RL))
        env.mainloop()
        costarr = RL.plot_cost()

        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        #xs = np.random.rand(100)
        #ys = np.random.rand(100)
        #axis.plot(xs, ys)
        axis.plot(np.arange(len(costarr)), costarr)
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)

        return output

def _run_maze(env, RL):
    step = 0
    for episode in range(30):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 100) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()
