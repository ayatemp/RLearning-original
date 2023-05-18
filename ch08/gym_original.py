import gym
import numpy as np

env = gym.make('CartPole-v1',render_mode="human")
state = env.reset()
done = False


while not done:
    env.render()
    action = np.random.choice([0, 1])
    result = env.step(action)
    next_state, reward, done, info = result[:4] 
    print(next_state, reward, done, info)

env.close()
