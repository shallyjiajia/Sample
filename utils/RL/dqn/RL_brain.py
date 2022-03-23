"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,#这也是一种优化的策略，e值是对最优值选择的概率，本身也让他递减，就会增加随机值的出现的概率
            output_graph=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                # collections：要将变量添加到其中的图形集合键的列表.默认为 [GraphKeys.LOCAL_VARIABLES].
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1
 #没看明白？ store_transition的作用
    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        #import matplotlib.pyplot as plt
        #plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        #plt.ylabel('Cost')
        #plt.xlabel('training steps')
        #plt.show()
        return self.cost_his


# ==============np.vstack(tup)使用================
# 沿着竖直方向将矩阵堆叠起来。
# Note: the arrays must have the same shape along all but the first axis. 除开第一维外，被堆叠的矩阵各维度要一致。
# 示例代码：
# import numpy as np
#
# arr1 = np.array([1, 2, 3])
# arr2 = np.array([4, 5, 6])
# res = np.vstack((arr1, arr2))
#
# 结果如下
# array([[1, 2, 3],
#        [4, 5, 6]])
#array中的逗号之间的数可以理解是竖直方向的
# np.hstack(tup)
# 沿着水平方向将数组堆叠起来。
# Note:
# tup : sequence of ndarrays
# All arrays must have the same shape along all but the second axis.
# import numpy as np
#
# arr1 = np.array([1, 2, 3])
# arr2 = np.array([4, 5, 6])
# res = np.hstack((arr1, arr2))
# print res
#  结果如下：
# [1 2 3 4 5 6]

# arr1 = np.array([[1, 2], [3, 4], [5, 6]])
# arr2 = np.array([[7, 8], [9, 0], [0, 1]])
# res = np.hstack((arr1, arr2))
# print res
#
# 结果如下：
# [[1 2 7 8]
#  [3 4 9 0]
#  [5 6 0 1]]
# #hstack在横向的堆叠中不增加维度，类似np.array([[1, 2], [3, 4], [5, 6]])本身就是两个维度，合并以后还是两个维度，是底层维度的合并，
# vstack会增加一个维度，底层的维度内部保持不变
# ========================tf.get_collection() 主要作用：=============
# 从一个集合中取出变量
#  tf.get_collection(
#     key,
#     scope=None )
# 该函数有两个参数
# 该函数可以用来获取key集合中的所有元素，返回一个列表。列表的顺序依变量放入集合中的先后而定。scope为可选参数，表示的是名称空间（名称域），如果指定，
# 就返回名称域中所有放入‘key’的变量的列表，不指定则返回所有变量。
#
# 例子
# variables = tf.get_collection(tf.GraphKeys.VARIABLES)
#   	for i in variables:
#   	    print(i)

# out：
#  <tf.Variable 'conv1/weights:0' shape=(3, 3, 3, 96) dtype=float32_ref>
#  <tf.Variable 'conv1/biases:0' shape=(96,) dtype=float32_ref>
#  <tf.Variable 'conv2/weights:0' shape=(3, 3, 96, 64) dtype=float32_ref>
#  <tf.Variable 'conv2/biases:0' shape=(64,) dtype=float32_ref>
#  <tf.Variable 'local3/weights:0' shape=(16384, 384) dtype=float32_ref>
#  <tf.Variable 'local3/biases:0' shape=(384,) dtype=float32_ref>
#  <tf.Variable 'local4/weights:0' shape=(384, 192) dtype=float32_ref>
#  <tf.Variable 'local4/biases:0' shape=(192,) dtype=float32_ref>
#  <tf.Variable 'softmax_linear/softmax_linear:0' shape=(192, 10) dtype=float32_ref>
#  <tf.Variable 'softmax_linear/biases:0' shape=(10,) dtype=float32_ref>
#
# 又如
# tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
# 							   scope="hidden[123]")
#
# 表示获取第1,2,3隐藏层的权重

# =================tf.get_variable函数====================
# 2018-10-24 16:34 更新
# 函数：tf.get_variable
# get_variable(
#     name,
#     shape=None,
#     dtype=None,
#     initializer=None,
#     regularizer=None,
#     trainable=True,
#     collections=None,
#     caching_device=None,
#     partitioner=None,
#     validate_shape=True,
#     use_resource=None,
#     custom_getter=None
# )
# 定义在：tensorflow/python/ops/variable_scope.py
#
# 获取具有这些参数的现有变量或创建一个新变量.
# 此函数将名称与当前变量范围进行前缀,并执行重用检查.有关重用如何工作的详细说明,请参见变量范围.下面是一个基本示例:
#
# with tf.variable_scope("foo"):
#     v = tf.get_variable("v", [1])  # v.name == "foo/v:0"
#     w = tf.get_variable("w", [1])  # w.name == "foo/w:0"
# with tf.variable_scope("foo", reuse=True):
#     v1 = tf.get_variable("v")  # The same as v above.
# 如果初始化器为 None(默认),则将使用在变量范围内传递的默认初始化器.如果另一个也是 None,那么一个 glorot_uniform_initializer 将被使用.初始化器也可以是张量,在这种情况下,变量被初始化为该值和形状.
# 类似地,如果正则化器是 None(默认),则将使用在变量范围内传递的默认正则符号(如果另一个也是 None,则默认情况下不执行正则化).
#
# 如果提供了分区,则返回 PartitionedVariable.作为张量访问此对象将返回沿分区轴连接的碎片.
# 一些有用的分区可用.例如：variable_axis_size_partitioner 和 min_max_variable_partitioner.
#
# 参数：
#
# name：新变量或现有变量的名称.
# shape：新变量或现有变量的形状.
# dtype：新变量或现有变量的类型(默认为 DT_FLOAT).
# initializer：创建变量的初始化器.
# regularizer：一个函数(张量 - >张量或无)；将其应用于新创建的变量的结果将被添加到集合 tf.GraphKeys.REGULARIZATION_LOSSES 中,并可用于正则化.
# trainable：如果为 True,还将变量添加到图形集合：GraphKeys.TRAINABLE_VARIABLES.
# collections：要将变量添加到其中的图形集合键的列表.默认为 [GraphKeys.LOCAL_VARIABLES].
# caching_device：可选的设备字符串或函数,描述变量应该被缓存以读取的位置.默认为变量的设备,如果不是 None,则在其他设备上进行缓存.典型的用法的在使用该变量的操作所在的设备上进行缓存,通过 Switch 和其他条件语句来复制重复数据删除.
# partitioner：(可选)可调用性,它接受要创建的变量的完全定义的 TensorShape 和 dtype,并且返回每个坐标轴的分区列表(当前只能对一个坐标轴进行分区).
# validate_shape：如果为假,则允许使用未知形状的值初始化变量.如果为真,则默认情况下,initial_value 的形状必须是已知的.
# use_resource：如果为假,则创建一个常规变量.如果为真,则创建一个实验性的 ResourceVariable,而不是具有明确定义的语义.默认为假(稍后将更改为真).
# custom_getter：可调用的,将第一个参数作为真正的 getter,并允许覆盖内部的 get_variable 方法.custom_getter 的签名应该符合这种方法,但最经得起未来考验的版本将允许更改：def custom_getter(getter, *args, **kwargs).还允许直接访问所有 get_variable 参数：def custom_getter(getter, name, *args, **kwargs).创建具有修改的名称的变量的简单标识自定义 getter 是：python def custom_getter(getter, name, *args, **kwargs): return getter(name + '_suffix', *args, **kwargs)
# 返回值：
#
# 创建或存在Variable(或者PartitionedVariable,如果使用分区器).
#
# 可能引发的异常：
#
# ValueError：当创建新的变量和形状时,在变量创建时违反重用,或当 initializer 的 dtype 和 dtype 不匹配时.在 variable_scope 中设置重用.

# ============np.newaxis的作用===========
# 就是在这一位置增加一个一维，这一位置指的是np.newaxis所在的位置，比较抽象，需要配合例子理解。
#
# x1 = np.array([1, 2, 3, 4, 5])
# # the shape of x1 is (5,)
# x1_new = x1[:, np.newaxis]
# # now, the shape of x1_new is (5, 1)
# # array([[1],
# #        [2],
# #        [3],
# #        [4],
# #        [5]])
# x1_new = x1[np.newaxis,:]
# # now, the shape of x1_new is (1, 5)
# # array([[1, 2, 3, 4, 5]])
# 再来一个例子
# In [124]: arr = np.arange(5*5).reshape(5,5)
#
# In [125]: arr.shape
# Out[125]: (5, 5)
#
# # promoting 2D array to a 5D array
# In [126]: arr_5D = arr[np.newaxis, ..., np.newaxis, np.newaxis]
#
# In [127]: arr_5D.shape
# Out[127]: (1, 5, 5, 1, 1)


# ===================GraphKeys 类================

# tf.GraphKeys包含所有graph collection中的标准集合名，有点像Python里的build-in fuction。
#
# 首先要了解graph collection是什么。
# graph collection
# 在官方教程——图和会话中，介绍什么是tf.Graph是这么说的：
# tf.Graph包含两类相关信息：
# 图结构。图的节点和边缘，指明了各个指令组合在一起的方式，但不规定它们的使用方式。图结构与汇编代码类似：检查图结构可以传达一些有用的信息，但它不包
#        含源代码传达的的所有有用上下文。
# **图集合。**TensorFlow提供了一种通用机制，以便在tf.Graph中存储元数据集合。tf.add_to_collection函数允许您将对象列表与一个键相关联(其中tf.GraphKeys定
#          义了部分标准键)，tf.get_collection则允许您查询与键关联的所有对象。TensorFlow库的许多组成部分会使用它：例如，当您创建tf.Variable时，系统会
#           默认将其添加到表示“全局变量(tf.global_variables)”和“可训练变量tf.trainable_variables)”的集合中。当您后续创建tf.train.Saver或
#           tf.train.Optimizer时，这些集合中的变量将用作默认参数。
# 也就是说，在创建图的过程中，TensorFlow的Python底层会自动用一些collection对op进行归类，方便之后的调用。这部分collection的名字被称为tf.GraphKeys，
# 可以用来获取不同类型的op。当然，我们也可以自定义collection来收集op。
# 定义在：tensorflow/python/framework/ops.py.
#
# 用于图形集合的标准名称.
# 标准库使用各种已知的名称来收集和检索与图形相关联的值.例如,如果没有指定,则 tf.Optimizer 子类默认优化收集的变量tf.GraphKeys.TRAINABLE_VARIABLES,
# 但也可以传递显式的变量列表.
#
# 定义了以下标准键：
#
# GLOBAL_VARIABLES：默认的 Variable 对象集合,在分布式环境共享(模型变量是其中的子集).参考：tf.global_variables.通常,所有TRAINABLE_VARIABLES 变量都将在
#                   MODEL_VARIABLES,所有 MODEL_VARIABLES 变量都将在 GLOBAL_VARIABLES.
# LOCAL_VARIABLES：每台计算机的局部变量对象的子集.通常用于临时变量,如计数器.注意：使用 tf.contrib.framework.local_variable 添加到此集合.
# MODEL_VARIABLES：在模型中用于推理(前馈)的变量对象的子集.注意：使用 tf.contrib.framework.model_variable 添加到此集合.
# TRAINABLE_VARIABLES：将由优化器训练的变量对象的子集.
# SUMMARIES：在关系图中创建的汇总张量对象.
# QUEUE_RUNNERS：用于为计算生成输入的 QueueRunner 对象.
# MOVING_AVERAGE_VARIABLES：变量对象的子集,它也将保持移动平均值.
# REGULARIZATION_LOSSES：在图形构造期间收集的正规化损失.
# 定义了以下标准键,但是它们的集合并没有像其他的那样自动填充：
#
# WEIGHTS
# BIASES
# ACTIVATIONS
#
# Variable
# Tensorflow使用Variable类表达、更新、存储模型参数。
#
# Variable是在可变更的，具有保持性的内存句柄，存储着Tensor
# 在整个session运行之前，图中的全部Variable必须被初始化
# Variable的值在sess.run(init)
# 之后就确定了
# Tensor的值要在sess.run(x)
# 之后才确定
# 创建的Variable被添加到默认的collection中
#
# tf.GraphKeys中包含了所有默认集合的名称，可以通过查看__dict__发现具体集合。
#
# tf.GraphKeys.GLOBAL_VARIABLES：global_variables被收集在名为tf.GraphKeys.GLOBAL_VARIABLES的colletion中，包含了模型中的通用参数
#
# tf.GraphKeys.TRAINABLE_VARIABLES：tf.Optimizer默认只优化tf.GraphKeys.TRAINABLE_VARIABLES中的变量。
#
# 函数
# 集合名
# 意义
# tf.global_variables()
# GLOBAL_VARIABLES
# 存储和读取checkpoints时，使用其中所有变量
#
# 跨设备全局变量集合
#
# tf.trainable_variables()
# TRAINABLE_VARIABLES
# 训练时，更新其中所有变量
#
# 存储需要训练的模型参数的变量集合
#
# tf.moving_average_variables()
# MOVING_AVERAGE_VARIABLES
# ExponentialMovingAverage对象会生成此类变量
#
# 实用指数移动平均的变量集合
#
# tf.local_variables()
# LOCAL_VARIABLES
# 在global_variables()
# 之外，需要用tf.init_local_variables()
# 初始化
#
# 进程内本地变量集合
#
# tf.model_variables()
# MODEL_VARIABLES
# Key
# to
# collect
# model
# variables
# defined
# by
# layers.
#
# 进程内存储的模型参数的变量集合
#
# QUEUE_RUNNERS
# 并非存储variables，存储处理输入的QueueRunner
# SUMMARIES
# 并非存储variables，存储日志生成相关张量
# 除了上表中的函数外(上表中最后两个集合并非变量集合，为了方便一并放在这里)，还可以使用tf.get_collection(集合名)
# 获取集合中的变量，不过这个函数更多与tf.get_collection(集合名)
# 搭配使用，操作自建集合。
#
# 另，slim.get_model_variables()
# 与tf.model_variables()
# 功能近似。
#
#
#
# 回到顶部
# Summary
# Summary被收集在名为tf.GraphKeys.UMMARIES的colletion中，
#
# Summary是对网络中Tensor取值进行监测的一种Operation
# 这些操作在图中是“外围”操作，不影响数据流本身
# 调用tf.scalar_summary系列函数时，就会向默认的collection中添加一个Operation
#
# 回到顶部
# 自定义集合
# 除了默认的集合，我们也可以自己创造collection组织对象。网络损失就是一类适宜对象。



