
# **Breakout DQN Model for Atari Environment**

## Overview
This project involves building and training a Deep Q-Network (DQN) to play the Atari Breakout game. The goal is to create an AI agent that can learn to play Breakout through reinforcement learning, using a deep neural network to predict actions that maximize the cumulative reward over time. This implementation uses the Stable-Baselines3 library, TensorFlow, and other related dependencies to set up and train the model, evaluate its performance, and visualize the learning process.


Key features:
- **Algorithm**: DQN
- **Framework**: Stable Baselines3
- **Environment**: OpenAI Gym
- **State Space**: discrete
- **Action Space**: 
  
---

## Training Details

### Hyperparameters:
- **Learning Rate**: 0.0001
- **Exploration Rate**: Decays over time from 0.666 to 0.657
- **Total Episodes**: 3,900+
- **Training Time**: 63 minutes
- **Loss**: Average loss over episodes, fluctuates during training
- **Updates**: 8,765 updates during training
- **FPS**: 128
- **Total Timesteps**: 36,064 timesteps

### Training Logs Summary:
The model was trained with the following performance metrics over multiple rollouts and training updates:

| Metric               | Value     |
|----------------------|-----------|
| **Episodes**         | 3,900     |
| **Total Timesteps**  | 36,064    |
| **Average Episode Length** | 289 steps |
| **Average Episode Reward** | 3.31   |
| **Exploration Rate** | 0.657     |
| **Loss** (last recorded)  | 0.021    |
| **Learning Rate**    | 0.0001    |
| **Training Time**    | 63 mins  |
| **Updates**          | 8,765     |

### Training Process:
- The agent’s exploration rate gradually decayed over time from an initial rate of 0.666 to 0.657.
- The learning rate remained constant throughout training at 0.0001.
- The loss experienced slight fluctuations, with the final recorded value at 0.021.

---

## Results

### Training Performance:
After training, the model achieved an average episode length of 289 steps and an average episode reward of 3.31. The exploration rate decreased from 0.666 to 0.657 over 3,900 episodes. The model continued to improve in efficiency, demonstrated by the loss reduction observed during training.


## Features
- **Reinforcement Learning (RL):** The agent learns to play Breakout through trial and error, maximizing rewards by destroying bricks.
- **Model Architecture:** Uses a CNN-based DQN model to process raw pixel input from the game and select actions.
- **Hyperparameter Tuning:** Configurable training parameters such as learning rate, batch size, and buffer size.
- **Visualization:** Real-time visualization of agent's actions and performance metrics during training and evaluation.
- **Model Saving and Loading:** The trained model is saved and can be loaded for evaluation or deployment.
  
## Table of Contents
1. [Installation](#installation)
2. [Environment Setup](#environment-setup)
3. [Training the Model](#training-the-model)
4. [Evaluation and Testing](#evaluation-and-testing)
5. [Metrics and Results](#metrics-and-results)
6. [Model Saving and Loading](#model-saving-and-loading)
7. [License](#license)

## Installation
To get started, install the required dependencies using pip:

```bash
pip install stable-baselines3[extra]
pip install gymnasium[atari]
pip install ale-py==0.8.1
pip install pyvirtualdisplay
```

For environments that require display, you may also need to install additional packages:

```bash
!apt-get update
!apt-get install -y xvfb x11-utils
```

## Environment Setup
The project uses the **ALE (Arcade Learning Environment)** with the **Breakout-v5** environment. The environment is set up with image-based observations, and frames are stacked to provide temporal context for the agent.

```python
env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
env = AtariWrapper(env)
env = VecFrameStack(env, n_stack=4)  # Stack 4 frames for temporal context
```

## Training the Model
The DQN model is implemented using the **Stable-Baselines3** library, with the following configuration:

```python
model = DQN(
    "CnnPolicy", 
    env, 
    learning_rate=0.0001, 
    buffer_size=100000, 
    learning_starts=1000, 
    batch_size=32, 
    target_update_interval=1000, 
    train_freq=4, 
    gradient_steps=1, 
    verbose=1
)
```

- **Learning Rate:** 0.0001
- **Buffer Size:** 100,000 experiences
- **Batch Size:** 32
- **Target Update Interval:** 1000
- **Training Frequency:** Every 4 steps

You can train the model for a specified number of timesteps:

```python
model.learn(total_timesteps=35_000)
```

## Evaluation and Testing
After training the model, it is evaluated by testing its performance in the environment. The mean reward and standard deviation of the model's actions are calculated over 10 episodes.

```python
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
```

## Metrics and Results
During training, the agent's performance is monitored through rewards and other metrics:

- **Mean Reward**: The average reward obtained by the model during evaluation.
- **Epsilon Decay**: The epsilon value used in epsilon-greedy action selection decreases over time as the model learns, balancing exploration and exploitation.
- **Rewards Plot**: The total rewards collected by the agent are plotted to visualize learning progress.

### Example Rewards Plot
```python
plt.plot(rewards)
```

## Model Saving and Loading
After training, the model is saved to Google Drive and can be loaded back for further testing or evaluation.

```python
model.save("/content/gdrive/MyDrive/Breakout_DQN_model")
```

To load the model:

```python
loaded_model = DQN.load("/content/gdrive/MyDrive/Breakout_DQN_model")
```




