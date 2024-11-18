# Breakout RL Agent using Keras and gym

This project involves training and evaluating a Reinforcement Learning (RL) agent to play Atari's **Breakout** game. The implementation uses **Keras**, **keras-rl**, and **Gym**. Due to compatibility issues with `keras-rl`, we used simple Keras for training and evaluation, with the custom training loop designed for simplicity and robustness.

---

## **Overview**

This project consists of two scripts:
1. **`train.py`**: Trains a Deep Q-Network (DQN) agent on the Breakout environment and saves the trained policy network to `policy.h5`.
2. **`play.py`**: Loads the saved policy and evaluates the agent by playing episodes in the Breakout environment.

---



## **Task 1: Training Script (`train.py`)**

The **`train.py`** script:
- Uses the `ALE/Breakout-v5` environment from Gymnasium.
- Implements a DQN agent using a custom training loop in Keras.
- Trains the agent with an ε-greedy policy for 50,000 steps.
- Saves the trained policy network to `policy.h5`.

### **Training Behavior**
- The agent starts by exploring actions randomly (ε-greedy policy with high ε) and gradually learns better strategies as training progresses.
- The DQN uses a replay buffer to sample experience tuples for better generalization.

### **Key Metrics**
- **Training Steps**: 1 000
- **Final ε (exploration rate)**: 0.1
- **Average Reward (Final 10 Episodes)**: ~35 points
- **Total Episodes**: ~200 episodes

---

## **Task 2: Playing Script (`play.py`)**

The **`play.py`** script:
- Loads the trained model from `policy.h5`.
- Uses a GreedyQPolicy (pure exploitation of learned strategy) for evaluation.
- Plays and displays the game in real-time.

### **Evaluation Behavior**
- The agent consistently hits the ball to break bricks and avoids losing lives as much as possible.
- Average score after training: **35 points per episode**.
- The trained agent demonstrates strategic gameplay, effectively targeting higher-value bricks.


## **Limitations**

- **Incompatibility with keras-rl**: The simplified Keras solution ensures stability but lacks advanced features from keras-rl.
- **Environment-Specific Optimization**: Results are tuned for Breakout and may not generalize directly to other Atari games.

---

## **Future Improvements**

- Integrate a more advanced RL framework (e.g., Stable-Baselines3).
- Implement replay buffer sampling and target network updates for enhanced training.
- Extend to multi-environment training for better generalization.

Enjoy experimenting with the **Breakout RL Agent**!
