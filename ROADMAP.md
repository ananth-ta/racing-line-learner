# 🏁 Project Roadmap: F1TENTH Autonomous Racing

## Phase 1: Infrastructure & Setup ✅ (Completed Jan 2026)
- [x] Configure WSL2 with CUDA/GPU support (RTX 4070).
- [x] Install F1TENTH Gym (Legacy v0.19) and Stable-Baselines3.
- [x] Create `Wrapper` class to downsample Lidar (1080 -> 20 rays) and normalize Observation Space.
- [x] Implement `baseline.py`: "Follow the Gap" geometric controller (Benchmark: 11.64s survival).
- [x] Implement `train.py`: PPO Training loop verified (Current Avg: 35s survival).

## Phase 2: PPO Tuning & Perception ✅ (Completed Mid Feb)
*Focus: Improve lap times and bypass the "safety plateau" (Local Optima).*
- [x] **Custom Reward:** Modify `wrapper.py` to reward high speed (PPO now highly aggressive).
- [x] **Braking Logic (The "2 m/s" fix):** Implement a Speed Floor (e.g., `clip(speed, 3.5, 20.0)`) or add massive collision penalties at high velocities to force the agent to learn to brake.
- [x] **Observation Stacking:** Stack the last 3 Lidar frames so the PPO agent can perceive acceleration and wall-approach velocity. 

*(Midterms Break: Late Feb to Early March)*

## Phase 3: Neuroevolution / Genetic Algorithm (Current Step - Mid March)
*Focus: Gradient-Free Exploration vs. Gradient-Based Exploitation.*
- [x] **Parallel Environments:** Implement `SubprocVecEnv` to run $N=50$ F1TENTH physics simulations simultaneously across CPU cores.
- [x] **Population Initialization:** Initialize a population of 50 Actor networks with random weights.
- [ ] **Evolutionary Loop:** Implement Evaluation (Lap Distance), Selection (Tournament), Crossover, and Gaussian Mutation. 
- [ ] **Colab Pro Training:** Offload the 48-hour parallel GA training loop to Google Colab Pro to protect local hardware thermals.

## Phase 4: Hybrid ERL (Stretch Goal - Early April)
*Focus: Seeding gradient-based optimization with gradient-free exploration.*
- [ ] **Weight Extraction:** Extract the PyTorch state dictionary from the best-performing Genetic Algorithm agent.
- [ ] **Brain Transplant:** Load the GA weights into the PPO Actor network.
- [ ] **The "Blind Critic" Fix:** Freeze the Actor network (`requires_grad=False`), run the simulation, and train the PPO Critic to map the state-values without causing catastrophic forgetting.
- [ ] **Fine-Tuning:** Unfreeze the Actor, drop the learning rate, and let PPO optimize the GA's racing line.

## Phase 5: Synthesis & Evaluation (Mid-April)
*Focus: Data analysis and final report generation.*
- [ ] **Statistical Sweeps:** Run 5 different random seeds for the final PPO and GA models to generate reliable performance curves.
- [ ] **Visual Analysis:** Use `render_video.py` to generate side-by-side comparisons of PPO vs. GA optimal racing lines.
- [ ] **Report Formatting:** Convert findings into the mandatory 2-column, US Letter AAAI conference paper LaTeX format.
- [ ] **Submission:** Submit final PDF report and code repository (Due April 22nd).