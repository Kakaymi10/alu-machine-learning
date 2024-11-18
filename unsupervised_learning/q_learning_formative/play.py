import pickle
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py

# Register ALE environments (this is not usually necessary if ale_py is installed properly)
gym.register_envs(ale_py)

# Create the Breakout environment
env = gym.make('ALE/Breakout-v5')


# Load the Q-table from the file
with open('q_table.pkl', 'rb') as f:
    q_table = pickle.load(f)

# Play using the loaded Q-table
num_episodes = 5
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(q_table[state])  # Choose the best action from Q-table
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = next_state
        total_reward += reward

        # Display the game frame
        plt.imshow(state)
        plt.axis('off')  # Optional: Hide axes for better visualization
        plt.pause(0.01)  # Pause to create a frame-by-frame effect

    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

env.close()
plt.show()  # Display the final frame