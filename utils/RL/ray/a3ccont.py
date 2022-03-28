import gym
import ray
import ray.rllib.agents.a3c as a3c
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib import agents
from generatorbid.bidenv import BidEnv

ray.init()
'''
config = {'gamma': 0.9,
          'lr': 1e-2,
          'num_workers': 2,
          'train_batch_size': 200,
          'model': {
              'fcnet_hiddens': [128, 128]
          }}
trainer = agents.ppo.PPOTrainer(env='CartPole-v0', config=config)
results = trainer.train()
'''

config = {
    "env_config" : {'Qd': 70794,
                     'marketrate': 1.29,
                     'pmc': 0.3285,
                     'a': 0.088,
                     'b': 305,
                     'TMon': 744,
                     'q_YD': 12686,
                     'q_Mon': 5170}

}
trainer = agents.a3c.A3CTrainer(env=BidEnv, config=config)
results = trainer.train()
