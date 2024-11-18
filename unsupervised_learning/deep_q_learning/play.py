import gym
import numpy as np
from keras.models import load_model
from rl.policy import GreedyQPolicy

# Load the trained policy network
model = load_model('policy.h5')

# Initialize the environment
env = gym.make('CartPole-v1')
state = env.reset()
done = False
policy = GreedyQPolicy()

while not done:
    env.render()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    q_values = model.predict(state)
    action = policy.select_action(q_values=q_values)
    next_state, reward, done, _ = env.step(action)
    state = next_state

env.close()