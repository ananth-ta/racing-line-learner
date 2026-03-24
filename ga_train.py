import glob
import re
import os
import torch
import numpy as np
import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from wrapper import F1TenthWrapper
from ga_model import MutantMLP

MAP_NAME = 'maps/vegas'

def make_env(rank):
    """
    Utility function to instantiate the wrapper.
    """
    def _init():
        # Instantiate the custom F1TENTH environment
        env = gym.make('f110_gym:f110-v0', map=MAP_NAME, map_ext='.png', num_agents=1)
        # Apply the wrapper
        env = F1TenthWrapper(env)
        return env
    return _init

def evaluate_population(env, population, current_gen, total_gens):
    num_envs = len(population)
    fitness_scores = np.zeros(num_envs)
    active_envs = np.ones(num_envs, dtype=bool)
    time_since_last_cp = np.zeros(num_envs)
    
    # Curriculum Learning: Scale steering penalty from 0.0 to 2.0 linearly
    # Simulated Annealing / Cyclic Penalty Curriculum
    # The penalty drops back to near 0 every 100 generations to allow aggressive exploration
    # to burst through plateaus at very sharp corners.
    cycle_length = 100
    cycle_pos = current_gen % cycle_length
    penalty_weight = 2.0 * (max(0, cycle_pos - 1) / max(1, cycle_length - 1))
    
    # Stagnation limit: If 3000 timesteps pass without hitting the next checkpoint, kill the mutant.
    STAGNATION_LIMIT = 3000
    
    # Reset the VecEnv
    obs = env.reset()

    # Add step limits to the while loop to kill any immortal mutants
    step_count = 0
    MAX_STEPS = 10000
    
    # The Stepping Loop
    while active_envs.any() and step_count < MAX_STEPS:
        actions = []
        for i in range(num_envs):
            if active_envs[i]:
                # Pass observation through the corresponding active MutantMLP
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
                with torch.no_grad():
                    action = population[i](obs_tensor).squeeze(0).numpy()
                actions.append(action)
            else:
                # Dummy action for inactive envs
                actions.append(np.zeros(2))
                
        obs, rewards, dones, infos = env.step(np.array(actions))
                
        for i in range(num_envs):
            if active_envs[i]:
                steering = actions[i][0]
                
                # Extract continuous distance-closed reward from environment info to combat U-turn farming
                progress_score = infos[i].get('progress_reward', 0.0)
                steering_delta = infos[i].get('steering_delta', 0.0) # <-- Extract Delta   

                # Base wrapper reward (contains Gaussian speed incentive, jitter penalty, collision penalty)
                env_reward = rewards[i]
                
                # Heavy quadratic penalty for sharp turning to stop donuts. 
                # Curriculum learning scales this penalty dynamically.
                steering_penalty = (abs(steering) ** 2) * penalty_weight 

                # THE SMOOTHNESS TAX: Massive penalty for jittering the wheel
                jitter_tax = steering_delta * 5.0
                
                # Waypoint Progression Check
                waypoint_cleared = infos[i].get('waypoint_cleared', False)
                waypoint_bonus = 500.0 if waypoint_cleared else 0.0
                
                fitness_scores[i] += (env_reward + progress_score + waypoint_bonus - steering_penalty - jitter_tax)
                                
                if dones[i]:
                    active_envs[i] = False
                
                if waypoint_cleared:
                    time_since_last_cp[i] = 0
                else:
                    time_since_last_cp[i] += 1
                    
                # Stagnation Pruning
                if time_since_last_cp[i] > STAGNATION_LIMIT:
                    active_envs[i] = False
                    
        step_count += 1

    return fitness_scores
    
def evolve_population(population, fitness_scores):
    num_envs = len(population)
    
    # Elitism: Sort population by fitness
    sorted_indices = np.argsort(fitness_scores)[::-1]
    
    new_population = [MutantMLP() for _ in range(num_envs)]
    
    # The absolute best agent (Top 1) copies its exact weights unharmed
    elite_idx = sorted_indices[0]
    best_weights = population[elite_idx].get_weights()
    new_population[0].set_weights(best_weights)
    
    # Tournament Selection & Breeding for the remaining 49 slots
    for i in range(1, num_envs):
        # Parent A Tournament
        tournament_A = np.random.choice(range(num_envs), size=3, replace=False)
        parent_A_idx = tournament_A[np.argmax(fitness_scores[tournament_A])]
        weights_A = population[parent_A_idx].get_weights()
        
        # Parent B Tournament
        tournament_B = np.random.choice(range(num_envs), size=3, replace=False)
        parent_B_idx = tournament_B[np.argmax(fitness_scores[tournament_B])]
        weights_B = population[parent_B_idx].get_weights()
        
        # Uniform Crossover
        mask = np.random.rand(len(weights_A)) > 0.5
        child_weights = np.where(mask, weights_A, weights_B)
        
        # Gaussian Mutation
        # Bumping mutation rate to 35% probability and 0.3 variance for faster exploration
        mutation_mask = np.random.rand(len(child_weights)) < 0.35
        mutations = np.random.normal(0, 0.3, size=len(child_weights))
        child_weights = np.where(mutation_mask, child_weights + mutations, child_weights)
        
        # Load the new arrays back into the PyTorch models
        new_population[i].set_weights(child_weights)
        
    return new_population

if __name__ == '__main__':
    num_envs = 50
    
    # Initialize a SubprocVecEnv with a population size of 50
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    # Initialize a list of 50 MutantMLP instances to serve as our starting generation
    population = [MutantMLP() for _ in range(num_envs)]
    
    num_generations = 500
    checkpoint_dir = "/content/drive/MyDrive/F1Tenth/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # --- AUTO-RESUME LOGIC ---
    start_gen = 1
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "best_mutant_gen_*.pt"))
    
    if checkpoint_files:
        # Find the file with the highest generation number using regex
        latest_cp = max(checkpoint_files, key=lambda x: int(re.search(r'gen_(\d+)', x).group(1)))
        
        # Extract the generation number to know where to resume
        last_gen = int(re.search(r'gen_(\d+)', latest_cp).group(1))
        start_gen = last_gen + 1
        
        # Load the champion's brain into the #1 slot of the new population
        population[0].load_state_dict(torch.load(latest_cp))
        
        print(f"🔄 Auto-Resuming from {latest_cp} for headless execution. Loaded champion weights. Starting at Generation {start_gen}.")
    else:
        print(f"Initialized fresh Genetic Algorithm loop with {num_envs} Parallel Environments and {len(population)} initial mutants.")
    # -------------------------
    
    for gen in range(start_gen, num_generations + 1):
        # Evaluate the current population
        fitness_scores = evaluate_population(env, population, gen, num_generations)
        
        # Logging
        max_fitness = np.max(fitness_scores)
        mean_fitness = np.mean(fitness_scores)
        min_fitness = np.min(fitness_scores)
        print(f"Generation {gen}/{num_generations} - Max Fitness: {max_fitness:.1f} | Mean Fitness: {mean_fitness:.1f} | Min Fitness: {min_fitness:.1f}")
        
        # Checkpointing
        if gen % 10 == 0:
            best_idx = np.argmax(fitness_scores)
            best_model = population[best_idx]
            checkpoint_path = os.path.join(checkpoint_dir, f"best_mutant_gen_{gen}.pt")
            torch.save(best_model.state_dict(), checkpoint_path)
            print(f"🔥 Checkpoint saved: {checkpoint_path}")
            
        # Evolve population for the next generation
        population = evolve_population(population, fitness_scores)

    env.close()