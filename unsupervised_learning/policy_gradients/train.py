import numpy as np
from policy_gradient import policy_gradient

def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Trains a policy using the policy gradient method over a specified number of episodes.
    
    Parameters:
    - env: the environment to train in.
    - nb_episodes: the number of episodes to train for.
    - alpha: learning rate for the policy gradient update.
    - gamma: discount factor for future rewards.
    - show_result: boolean indicating whether to render the environment every 1000 episodes.
    
    Returns:
    - scores: list containing the total reward (score) for each episode.
    """
    # Initialize the weight matrix for the policy
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state = env.reset()[np.newaxis, :]
        episode_reward = 0
        grads = []
        rewards = []
        
        done = False
        while not done:
            if show_result and episode % 1000 == 0:
                env.render()
            action, gradient = policy_gradient(state, weight)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[np.newaxis, :]

            # Store the gradient and reward for this step
            grads.append(gradient)
            rewards.append(reward)
            episode_reward += reward
            state = next_state

        # Calculate discounted rewards
        for t in range(len(rewards)):
            Gt = sum([gamma**k * rewards[k] for k in range(t, len(rewards))])
            weight += alpha * grads[t] * Gt  # Update weight using policy gradient

        scores.append(episode_reward)
        print(f"Episode {episode + 1}/{nb_episodes}, Score: {episode_reward}", end="\r", flush=True)

    return scores
