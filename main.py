import gymnasium as gym
from stable_baselines3 import SAC, PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
import os
import argparse
import optuna
def train(env, algo, model_dir, timesteps=25000, use_sde=False, learning_rate=0.004,use_gpu=True, entropy='auto'):
    try: os.makedirs(model_dir)
    except: pass
    match algo:
        case 'SAC':
            model=SAC('MlpPolicy', env=env, verbose=1, device='cuda' if use_gpu else 'cpu', learning_rate=learning_rate, tensorboard_log=model_dir+"/logs", use_sde=use_sde, ent_coef=entropy)
        case 'PPO':
            model=PPO('MlpPolicy', env=env, verbose=1, device='cuda' if use_gpu else 'cpu', learning_rate=learning_rate, tensorboard_log=model_dir+"/logs", use_sde=use_sde, ent_coef=entropy)
        case 'DQN':
            #Dude, remember that you have changed handle_timeout_termination = True to False
            model=DQN('MlpPolicy', env=env, verbose=1, device='cuda' if use_gpu else 'cpu', learning_rate=learning_rate, tensorboard_log=model_dir+"/logs", buffer_size=75000, exploration_fraction=1, optimize_memory_usage= True, learning_starts= 500000)
        case 'A2C':
            model=A2C('MlpPolicy', env=env, verbose=1, device='cuda' if use_gpu else 'cpu', learning_rate=learning_rate, tensorboard_log=model_dir+"/logs", use_sde=use_sde, ent_coef=entropy)
        case _:
            print("Wrong Algorithm or Algorithm not imported")
            return
    iters=0
    while True:
        iters+=1
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        model.save(f"{model_dir}/{algo}/{timesteps*iters}")
def test(env, algo, path_to_model):

    match algo:
        case 'SAC':
            model = SAC.load(path_to_model, env=env)
        case 'DQN':
            model = DQN.load(path_to_model, env=env)
        case 'PPO':
            model = PPO.load(path_to_model, env=env)
        case 'A2C':
            model = A2C.load(path_to_model, env=env)
        case _:
            print('Algorithm not found')
            return

    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break
def study(env, algo, params, timesteps, n_trials, n_evals):
    def objective(trial):
        learning_rate=0.0001
        learning_starts=50000
        tau=1
        target_update_interval=10000
        exploration_fraction=0.1
        exploration_initial_eps=1
        exploration_final_eps=0.05
        if 'learning_rate' in params:
            learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1)
        if 'learning_starts' in params:
            learning_starts=trial.suggest_discrete_uniform('learning_starts', 10000, 1000000, 1000)
        if 'tau' in params:
            tau=trial.suggest_uniform('tau', 0, 1)
        if 'train_freq' in params:
            train_freq=int(trial.suggest_discrete_uniform('trial_freq', 1, 10, 1))
        if 'exploration_fraction' in params:
            exploration_fraction=trial.suggest_loguniform('exploration_fraction', 1, 10)
        if 'target_update_interval' in params:
            target_update_interval=trial.suggest_discrete_uniform('target_update_interval', 1000, 100000, 1000)
        if 'exploration_initial_eps' in params:
            exploration_initial_eps=trial.suggest_uniform('exploration_initial_eps', 0, 10)
        if 'exploration_initial_eps' in params:
            exploration_final_eps=trial.suggest_uniform('exploration_final_eps', 0, 10)
        if 'all' in params:
            learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1)
            learning_starts=trial.suggest_discrete_uniform('learning_starts', 10000, 1000000, 1000)
            tau=trial.suggest_uniform('tau', 0, 1)
            train_freq=int(trial.suggest_discrete_uniform('trial_freq', 1, 10, 1))
            exploration_fraction=trial.suggest_loguniform('exploration_fraction', 1, 10)
            target_update_interval=trial.suggest_discrete_uniform('target_update_interval', 1000, 100000, 1000)
            exploration_initial_eps=trial.suggest_uniform('exploration_initial_eps', 0, 10)
            exploration_final_eps=trial.suggest_uniform('exploration_final_eps', 0, 10)
        envm=gym.make(env)
        if algo == 'DQN':
            model=DQN('MlpPolicy', env=envm, verbose=1, buffer_size=75000, optimize_memory_usage= True, device='cuda', learning_rate=learning_rate, learning_starts=learning_starts, tau=tau, target_update_interval=target_update_interval, exploration_fraction=exploration_fraction,train_freq=train_freq, exploration_initial_eps=exploration_initial_eps, exploration_final_eps=exploration_final_eps)
        model.learn(total_timesteps=timesteps)

        eval_env=gym.make(env)
        r, _ =evaluate_policy(model, eval_env, n_eval_episodes=n_evals)
        return r
    study= optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3", study_name="Pacman-v3")
    study.optimize(objective, n_trials=n_trials)
    print("Best values:", study.best_params)

# Parse command line inputs
parser = argparse.ArgumentParser(description='Train or test model.')
parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
parser.add_argument('algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
parser.add_argument('-t', '--train', metavar='path_to_model')
parser.add_argument('-s', '--test', metavar='path_to_model')
parser.add_argument('-o', "--optimize", help="Optimize using Optuna mode", action='store_true')
parser.add_argument('-op', "--optimization_parameters", help="Parameters to study using Optuna to be entered within quotes separated by space", type=str)
parser.add_argument('-nt', '--n_trials', help='Number of trials for Optuna', type=int)
parser.add_argument('-ne', '--n_evals', help="Evaluattion frequency for Optuna", type=int)
parser.add_argument('-tme','--timesteps', help='Number of Timesteps', type=int, default=25000)
parser.add_argument('-l', '--learning_rate', help="Custom learning rate if needed", type=float, default=0.004)
parser.add_argument('-g', '--gpu', help="Enable usage of gpu[0]", action='store_true')
parser.add_argument('-sde', '--state_dependent_exploration', help='Enable usage of state dependant exploration', action='store_true')
parser.add_argument('-e', '--entropy_coefficient', help='Specify the entropy that is the degree the exploration', type=float, default=0.01)
args = parser.parse_args()


if args.train:
    gymenv = gym.make(args.gymenv, render_mode=None)
    train(env=args.gymenv, algo=args.algo, model_dir=args.train, timesteps=args.timesteps, learning_rate=args.learning_rate, use_gpu=args.gpu, use_sde=args.state_dependent_exploration, entropy=args.entropy_coefficient)
if(args.test):
    if os.path.isfile(args.test):
        gymenv = gym.make(args.gymenv, render_mode='human')
        print(gymenv.action_space)
        test(gymenv, args.algo, path_to_model=args.test)
    else:
        print(f'{args.test} not found.')
if (args.optimize and args.optimization_parameters):
    study(args.gymenv, args.algo, args.optimization_parameters, args.timesteps, args.n_trials, args.n_evals)
else:
    print("Optimization parameters not given")