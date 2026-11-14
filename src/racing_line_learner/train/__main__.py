from .ppo import main

# Guard required: with --n_envs > 1, SubprocVecEnv starts workers that
# re-import the main module, which must not start training again.
if __name__ == "__main__":
    main()
