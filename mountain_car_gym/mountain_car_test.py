import gymnasium as gym
import numpy as np

# Initialize environment
env = gym.make("MountainCar-v0", render_mode="human")

# Reset once to get space info
observation, info = env.reset(seed=42)

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


# Discretization
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Initialize Q-table
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Discretization helper
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

# Training loop
for episode in range(EPISODES):
    observation, info = env.reset()
    discrete_state = get_discrete_state(observation)

    if episode % SHOW_EVERY == 0:
        print(f"Episode: {episode}")
        render = True
    else:
        render = False

    done = False
    step = 0
    while not done:
        if render:
            env.render()

        # Choose action
        action = np.argmax(q_table[discrete_state])

        # Take action
        new_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        new_discrete_state = get_discrete_state(new_observation)

        # Q-learning update
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_observation[0] >= env.unwrapped.goal_position:
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
        step += 1

    # End of episode
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()
