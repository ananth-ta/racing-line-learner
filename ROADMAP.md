# 🏁 Project Roadmap: F1TENTH Autonomous Racing

## Phase 1: Infrastructure & Setup ✅ (Completed Jan 2026)
- [x] Configure WSL2 with CUDA/GPU support (RTX 4070).
- [x] Install F1TENTH Gym (Legacy v0.19) and Stable-Baselines3.
- [x] Create `Wrapper` class to downsample Lidar (1080 -> 20 rays) and normalize Observation Space.
- [x] Implement `baseline.py`: "Follow the Gap" geometric controller (Benchmark: 11.64s survival).
- [x] Implement `train.py`: PPO Training loop verified (Current Avg: 35s survival).

## Phase 2: Reward Function Engineering (Next Step)
*Focus: Improve lap times, not just survival.*
- [ ] **Custom Reward:** Modify `wrapper.py` to reward high speed, not just survival.
- [ ] **Penalty Logic:** Add penalties for frequent steering changes (jitter) to smooth out the drive.
- [ ] **Curriculum Learning:** Start slow (1 m/s), then increase speed limit as the agent gets smarter.

## Phase 3: Algorithm Comparison
*Focus: Sample Efficiency vs. Stability.*
- [ ] **SAC (Soft Actor-Critic):** Implement SAC to compare against PPO (SAC is usually better for continuous control like steering).
- [ ] **Hyperparameter Tuning:** Tune Learning Rate and Batch Size for the specific track dynamics.

## Phase 4: Perception & Sim-to-Real Prep
*Focus: Handling noise and realistic constraints.*
- [ ] **Observation Stacking:** Stack the last 3 frames so the AI can perceive acceleration/jerk.
- [ ] **Latency Simulation:** Artificially add delay to the wrapper to mimic real-world sensor lag.
- [ ] **Map Generalization:** Train on 'Vegas', validate on 'Silverstone' map to test generalization.

## Phase 5: Advanced Optimization
- [ ] **Vectorized Environments:** Run 4-8 simulations in parallel to increase FPS from 150 to 1,000+.
- [ ] **Racing Line Optimization:** Feed the optimal racing line (waypoints) as input to the agent.
