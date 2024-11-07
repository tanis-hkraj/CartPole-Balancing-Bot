import gym
import numpy as np

def discretize_state(state):
    """Discretizes the continuous state of the CartPole environment.

    Args:
        state: A numpy array containing the continuous state.

    Returns:
        A tuple representing the discretized state.
    """
    
    # Define the bin edges for each state variable
    cart_position_bins = np.linspace(-2.4, 2.4, num=10)  # Cart position ranges from -2.4 to 2.4
    cart_velocity_bins = np.linspace(-3.0, 3.0, num=10)  # Cart velocity ranges from -3.0 to 3.0
    pole_angle_bins = np.linspace(-0.209, 0.209, num=6)   # Pole angle ranges from -12 degrees to 12 degrees (in radians)
    pole_angular_velocity_bins = np.linspace(-3.5, 3.5, num=6)  # Pole angular velocity ranges from -3.5 to 3.5

    # Digitize the state values into discrete bins
    cart_position = np.digitize(state[0], cart_position_bins) - 1  # -1 to make bins zero-indexed
    cart_velocity = np.digitize(state[1], cart_velocity_bins) - 1
    pole_angle = np.digitize(state[2], pole_angle_bins) - 1
    pole_angular_velocity = np.digitize(state[3], pole_angular_velocity_bins) - 1

    return (cart_position, cart_velocity, pole_angle, pole_angular_velocity)

  

def epsilon_greedy(q_values, epsilon):
  """Epsilon-greedy action selection.

  Args:
    q_values: A numpy array of Q-values for each action.
    epsilon: The exploration rate.

  Returns:
    The selected action index.
  """

  if np.random.rand() < epsilon:
    return np.random.randint(len(q_values))
  else:
    return np.argmax(q_values)

def train_q_agent(env, num_episodes, alpha, gamma, epsilon, epsilon_decay, min_epsilon):
    """Trains a Q-learning agent.

    Args:
        env: The environment to train in.
        num_episodes: The number of training episodes.
        alpha: The learning rate.
        gamma: The discount factor.
        epsilon: The initial exploration rate.
        epsilon_decay: The exploration decay rate.
        min_epsilon: The minimum exploration rate.

    Returns:
        The trained Q-table.
    """

    num_bins = [10, 10, 6, 6]
    q_table = np.zeros((num_bins[0], num_bins[1], num_bins[2], num_bins[3], env.action_space.n))

    for episode in range(num_episodes):
        state = env.reset()
        state = discretize_state(state)
        done = False

        while not done:
            # Choose action using epsilon-greedy strategy
            action = epsilon_greedy(q_table[state], epsilon)

            next_state, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state)
            q_table[tuple(state) + (action,)] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[tuple(state) + (action,)])
            state = next_state
            epsilon = max(epsilon * epsilon_decay, min_epsilon)

        if episode % 100 == 0:
            print(f"Episode: {episode}, Epsilon: {epsilon}")

    return q_table

# Example usage:
env = gym.make('CartPole-v1')
q_table = train_q_agent(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01)