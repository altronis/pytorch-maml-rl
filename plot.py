import os
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_moving_avg(results_dir, label):
    results_path = os.path.join(results_dir, 'avg_returns.npy')
    returns = np.load(results_path)

    # Calculate moving average
    window_size = 5
    moving_avg = moving_average(returns, window_size)
    plt.plot(np.arange(window_size - 1, len(returns)), moving_avg, label=label)


# Plotting
plt.figure(figsize=(10, 5))

plot_moving_avg('maml-halfcheetah-vel-trpo', 'TRPO')
plot_moving_avg('maml-halfcheetah-vel-ppo', 'PPO')

plt.title('Half-cheetah, Goal Velocity')
plt.xlabel('Iteration')
plt.ylabel('Average Return')
plt.legend()
plt.show()
