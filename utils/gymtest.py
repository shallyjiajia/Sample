import gym

env = gym.make('custenv-v0') #CustEnv()

for i_episode in range(3):
    observation = env.reset()
    for t in range(5):
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
