# RL-trainer-script
This script helps one to train and tune hyperparameters of RL agents from terminal using python, using argparse, gymnasium for environments and stable-baselines3 for algortihms

DEPENDENCIES:

[Argparse](https://docs.python.org/3/library/argparse.html)

[Gymnasium](https://gymnasium.farama.org/)

[Optuna](https://optuna.org/)

[Stable Baselines 3](https://stable-baselines.readthedocs.io/en/master/)

[Vizdoom](https://github.com/Farama-Foundation/ViZDoom)

ALGORITHMS SUPPORTED:

From Stable-Baselines3

[PPO](https://paperswithcode.com/method/ppo)

[A2C](https://paperswithcode.com/method/a2c)

[DQN](https://paperswithcode.com/method/dqn)

[SAC](https://paperswithcode.com/method/soft-actor-critic)

ENVIRONMENTS SUPPORTED:

All the environments available at [Gymnasium](https://gymnasium.farama.org/)

[Vizdoom](https://github.com/Farama-Foundation/ViZDoom)

OPTIMIZABLE MODELS:

DQN (More support to be added)

OPTIMIZABLE PARAMETERS:

learning_rate

learning_starts

tau

target_update_interval

exploration_fraction

train_freq

exploration_initial_eps

exploration_final_eps




usage: main.py [-h] [-t path_to_model] [-n NENVS] [-c RESUME] [-s path_to_model] [-o] [-buf BUFFER] [--optimize_ram] [-op OPTIMIZATION_PARAMETERS] [-nt N_TRIALS] [-ne N_EVALS]
               [-tme TIMESTEPS] [-l LEARNING_RATE] [-g] [-sde] [-e ENTROPY_COEFFICIENT]
               gymenv algo policy

Train or test model.

positional arguments:
  gymenv                Gymnasium environment i.e. Humanoid-v4
  
  algo                  StableBaseline3 RL algorithm i.e. SAC, TD3
  
  policy                Enter type of input or policy

options:
  -h, --help            show this help message and exit
  
  -t path_to_model, --train path_to_model
  
  -n NENVS, --nenvs NENVS
                        Number of parallel environments for vectorized environments
                        
  -c RESUME, --resume RESUME
                        Continue training a saved model
                        
  -s path_to_model, --test path_to_model
  
  -o, --optimize        Optimize using Optuna mode
  
  -buf BUFFER, --buffer BUFFER
                        Buffer size that affects RAM
                        
  --optimize_ram        Optimize memory usage
  
  -op OPTIMIZATION_PARAMETERS, --optimization_parameters OPTIMIZATION_PARAMETERS
                        Parameters to study using Optuna to be entered within quotes separated by space
                        
  -nt N_TRIALS, --n_trials N_TRIALS
                        Number of trials for Optuna
                        
  -ne N_EVALS, --n_evals N_EVALS
                        Evaluattion frequency for Optuna
                        
  -tme TIMESTEPS, --timesteps TIMESTEPS
                        Number of Timesteps
                        
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        Custom learning rate if needed
                        
  -g, --gpu             Enable usage of gpu[0]
  
  -sde, --state_dependent_exploration
                        Enable usage of state dependant exploration
                        
  -e ENTROPY_COEFFICIENT, --entropy_coefficient ENTROPY_COEFFICIENT
                        Specify the entropy that is the degree the exploration

  -sde, --state_dependent_exploration      Enable usage of state dependant exploration
  
  -e ENTROPY_COEFFICIENT, --entropy_coefficient ENTROPY_COEFFICIENT      Specify the entropy that is the degree the exploration
