import multiprocessing
import os
import shutil
import threading

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
from utils.RL.a3cconttest import ACNet,Worker

tf.disable_v2_behavior()

class A3CCont():
    def __init__(self):
        #self._GAME = 'bidenv-v0'
        self._OUTPUT_GRAPH = True
        self._LOG_DIR = './log'
        self._N_WORKERS = multiprocessing.cpu_count()
        self._MAX_EP_STEP = 200
        self._MAX_GLOBAL_EP = 2000
        self._GLOBAL_NET_SCOPE = 'Global_Net'
        self._UPDATE_GLOBAL_ITER = 10
        self._GAMMA = 0.9
        self._ENTROPY_BETA = 0.01
        self._LR_A = 0.0001  # learning rate for actor
        self._LR_C = 0.001  # learning rate for critic
        self._GLOBAL_RUNNING_R = []
        self._GLOBAL_EP = 0

    def run(self,params):
        env = params['env']
        SESS = tf.Session()

        with tf.device("/cpu:0"):
            OPT_A = tf.train.RMSPropOptimizer(self._LR_A, name='RMSPropA')
            OPT_C = tf.train.RMSPropOptimizer(self._LR_C, name='RMSPropC')
            GLOBAL_AC = ACNet(OPT_A,OPT_C,env,self._GLOBAL_NET_SCOPE)  # we only need its params
            workers = []
            # Create worker
            for i in range(self._N_WORKERS):
                i_name = 'W_%i' % i  # worker name
                workers.append(Worker(OPT_A,OPT_C,env, i_name, GLOBAL_AC))
        # TensorFlow的Session对象是可以支持多线程的，因此多个线程可以很方便地使用同一个会话（Session）并且并行地执行操作。
        # 然而，在Python程序实现这样的并行运算却并不容易。所有线程都必须能被同步终止，异常必须能被正确捕获并报告，回话终止的时候，
        # 队列必须能被正确地关闭。
        #
        # 所幸TensorFlow提供了两个类来帮助多线程的实现：tf.Coordinator和
        # tf.QueueRunner。从设计上这两个类必须被一起使用。Coordinator类可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序
        # 报告异常。QueueRunner类用来协调多个工作线程同时将多个张量推入同一个队列中。
        COORD = tf.train.Coordinator()
        SESS.run(tf.global_variables_initializer())

        if self._OUTPUT_GRAPH:
            if os.path.exists(self._LOG_DIR):
                shutil.rmtree(self._LOG_DIR)
            tf.summary.FileWriter(self._LOG_DIR, SESS.graph)

        worker_threads = []
        for worker in workers:
            job = lambda: worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        # 有了这个join的动作，就是所有的work动运行完了后，才进行下面的动作
        COORD.join(worker_threads)

        plt.plot(np.arange(len(self._GLOBAL_RUNNING_R)), self._GLOBAL_RUNNING_R)
        plt.xlabel('step')
        plt.ylabel('Total moving reward')
        plt.show()

