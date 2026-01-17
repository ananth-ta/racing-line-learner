import gym
import numpy as np

# Create the env
env = gym.make('f110_gym:f110-v0', map='maps/vegas', map_ext='.png', num_agents=1)
obs, step_reward, done, info = env.reset(np.array([[0.0, 0.0, 0.0]]))

# Render one frame
# env.render()
print("Environment created successfully!")
