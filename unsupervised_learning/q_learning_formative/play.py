import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py
import tensorflow as tf
import keras
from keras import layers

# Load the model
model = keras.models.load_model('policy.h5')

# Register ALE environments (this is not usually necessary if ale_py is installed properly)
gym.register_envs(ale_py)

# Create the Breakout environment
env_name = 'ALE/Breakout-v5'
env = gym.make(env_name)

max_steps_per_episode = 1000

# Play the game using the trained model
for episode in range(10):  # Play 10 episodes
    observation, _ = env.reset()
    state = np.array(observation)
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        env.render()

        # Predict action Q-values
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        action = tf.argmax(action_probs[0]).numpy()

        # Apply the sampled action in our environment
        state_next, reward, done, _, _ = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward
        state = state_next

        if done:
            print(f"Episode: {episode + 1}, Reward: {episode_reward}")
            break

env.close()