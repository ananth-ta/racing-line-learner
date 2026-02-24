# F1TENTH Autonomous Racing: PPO vs. Neuroevolution

Graduate project for CS 5180 - Reinforcement Learning and Sequential Decision Making, Spring 2026 Semester @ Northeastern University.

This repository explores and compares two distinct artificial intelligence approaches for high-speed autonomous racing in a continuous control environment:
1. **Gradient-Based (PPO):** Utilizing Proximal Policy Optimization (via Stable-Baselines3) to maximize a custom, speed-incentivized reward function.
2. **Gradient-Free (Neuroevolution):** A population-based Genetic Algorithm designed to bypass local optima traps (e.g., the "safety plateau") through aggressive, randomized exploration.

**Phase 3 Stretch Goal:** An Evolutionary Reinforcement Learning (ERL) hybrid architecture. This approach extracts the optimal policy weights from the Genetic Algorithm and injects them into the PPO Actor network—training the Critic while the Actor is temporarily frozen to prevent catastrophic forgetting.

## Usage
- `baseline.py`: Hard-coded, geometric "Follow The Gap" (FTG) controller used as a safety and performance benchmark.
- `train.py`: The primary Reinforcement Learning training loop for the PPO agent.
- `wrapper.py`: Custom environment wrapper handling Lidar downsampling (1080 -> 20 rays), velocity tracking, and reward shaping.
- `render_video.py`: Headless rendering utility to evaluate agents and export laps as MP4 videos.
- `ga_model.py`: Implementation of `MutantMLP` with flat array extraction and injection utilities for the crossover/mutation pipelines.
- `ga_train.py`: Parallelized evolutionary loop utilizing `SubprocVecEnv` across multiple CPU cores to evaluate the population.