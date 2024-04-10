import gymnasium as gym
from stable_baselines3 import SAC, PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
import os
import argparse
import optuna
import tensorflow as tf
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from time import sleep
from vizdoom import gymnasium_wrapper
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
import vizdoom.gymnasium_wrapper

import psutil
import nvitop as nvtop

class CustomCallback(BaseCallback):
    def __init__(self, log_dir, log_steps):
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.log_steps = log_steps
        self.step = 0
        self.n_calls = 0
        self.update_locals = 0
        self.episode_reward = 0
        self.episode_length = 0
        self.best_mean_episode_reward = -np.inf
        self.best_episode_reward = -np.inf
        self.saved_best_model_path = None
    def update_locals(self, args):
        pass

    def _on_step(self, batch, logs, **kwargs):
        self.n_calls += 1
        # Perform operations after each training step
        pass

    def _on_episode_end(self, episode_length, episode_reward):
        self.episode_length = episode_length
        self.episode_reward = episode_reward

        if np.mean(self.episode_reward) > self.best_mean_episode_reward:
            self.best_mean_episode_reward = np.mean(self.episode_reward)
            self.best_episode_reward = episode_reward
            self.saved_best_model_path = self.callback_order[0].save(self.log_dir + "best_model")

    def _on_training_end(self):
        self.writer.close()

    def _on_training_start(self):
        self.step = 0

    def _on_rollout_start(self):
        pass

    def _on_train_batch_end(self, batch, logs):
        if self.step % self.log_steps == 0:
            ram_usage = psutil.virtual_memory().percent
            gpu_nvtop = nvtop.Nvtop()
            gpu_usage, gpu_temp = gpu_nvtop.get_metrics()

            with self.writer.as_default():
                tf.summary.scalar('RAM Usage', data=ram_usage, step=self.step)
                tf.summary.scalar('GPU Usage', data=gpu_usage, step=self.step)
                tf.summary.scalar('GPU Temperature', data=gpu_temp, step=self.step)

        self.step += 1
def train(env, algo, model_dir, policy='MlpPolicy', timesteps=25000, use_sde=False, learning_rate=0.004,use_gpu=True, entropy='auto', cont=None, buffer=75000, optimize_memory=False):
    try: os.makedirs(model_dir)
    except: pass
    match algo:
        case 'SAC':
            model=SAC(policy, env=env, verbose=1, device='cuda' if use_gpu else 'cpu', learning_rate=learning_rate, tensorboard_log=model_dir+"/logs", use_sde=use_sde, ent_coef=entropy)
        case 'PPO':
            model=PPO(policy, env=env, verbose=1, device='cuda' if use_gpu else 'cpu', learning_rate=learning_rate, tensorboard_log=model_dir+"/logs", use_sde=use_sde, ent_coef=entropy)
        case 'DQN':
            #Dude, remember that you have changed handle_timeout_termination = True to False
            model=DQN(policy, env=env, verbose=1, device='cuda' if use_gpu else 'cpu', tensorboard_log=model_dir+"/logs", buffer_size=buffer, exploration_fraction=1, optimize_memory_usage=optimize_memory)
        case 'A2C':
            model=A2C(policy, env=env, verbose=1, device='cuda' if use_gpu else 'cpu', learning_rate=learning_rate, tensorboard_log=model_dir+"/logs", use_sde=use_sde, ent_coef=entropy)
        case _:
            print("Wrong Algorithm or Algorithm not imported")
            return
    #callbacks= CustomCallback(log_dir="./logs/", log_steps=100)
    iters=0
    if cont!=None:
        print(f'Resuming training from {cont}')
        model.load(path=cont, env=env, device='cuda' if use_gpu else 'cpu')
    try:
        while True:
            iters+=1
            detes=model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
            model.save(f"{model_dir}/{algo}/{timesteps*iters}")
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        model.save(f"{model_dir}/{algo}/{'paused'}")
        print("Model saved.")
def test(env, algo, nenvs,  path_to_model):
    gymenv = gym.make(env, render_mode='human')
    if args.gymenv in ["ALE/Pacman", "ALE/DonkeyKong", "ALE/Breakout"]:
        print("Atari game detected. Stacking frames")
        gymenv= make_atari_env(env, n_envs=nenvs)
        gymenv = VecFrameStack(gymenv, n_stack=4)
    match algo:
        case 'SAC':
            model = SAC.load(path_to_model, env=gymenv)
        case 'DQN':
            model = DQN.load(path_to_model, env=gymenv)
        case 'PPO':
            model = PPO.load(path_to_model, env=gymenv)
        case 'A2C':
            model = A2C.load(path_to_model, env=gymenv)
        case _:
            print('Algorithm not found')
            return
    if args.gymenv in ["ALE/Pacman", "ALE/DonkeyKong", "ALE/Breakout"]:
        gymenv.metadata['render_fps']=60
        obs = gymenv.reset()
        done = False
        e = 0
        while True:
            action, _ = model.predict(obs)
            obs, _, done, _ = gymenv.step(action)
            gymenv.render('human')
            sleep(0.05)

            if done.all():
                e+=1
                if e==2:
                    break
    else:
        obs = gymenv.reset()[0]
        done = False
        extra_steps = 500
        while True:
            action, _ = model.predict(obs)
            obs, _, done, _, _ = gymenv.step(action)

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
parser.add_argument('policy', help='Enter type of input or policy', type=str)
parser.add_argument('-t', '--train', metavar='path_to_model')
parser.add_argument('-n', '--nenvs', help='Number of parallel environments for vectorized environments', type=int)
parser.add_argument('-c', "--resume", help="Continue training a saved model", type=str)
parser.add_argument('-s', '--test', metavar='path_to_model')
parser.add_argument('-o', "--optimize", help="Optimize using Optuna mode", action='store_true')
parser.add_argument('-buf', '--buffer', help="Buffer size that affects RAM", type=int)
parser.add_argument('--optimize_ram', help="Optimize memory usage", action='store_true')
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
    if args.gymenv in ["ALE/Pacman", "ALE/DonkeyKong", "ALE/Breakout"]:
        print("Atari game detected. Stacking frames...")
        print(f"Having {args.nenvs} environments")
        gymenv= make_atari_env(args.gymenv, n_envs=args.nenvs)
        gymenv = VecFrameStack(gymenv, n_stack=4)
    if args.gymenv in ["VizdoomCorridor", "VizdoomBasic"]:
        print("Vizdoom env detected. Stacking frames...")
        gymenv= make_vec_env(args.gymenv, n_envs=args.nenvs)
        gymenv = VecFrameStack(gymenv, n_stack=4)
    train(env=gymenv, algo=args.algo, policy=args.policy, model_dir=args.train, timesteps=args.timesteps, learning_rate=args.learning_rate, use_gpu=args.gpu, use_sde=args.state_dependent_exploration, entropy=args.entropy_coefficient, cont=args.resume, buffer=args.buffer, optimize_memory=args.optimize_ram)
if(args.test):
    if os.path.isfile(args.test):
        test(args.gymenv, args.algo, args.nenvs, path_to_model=args.test)
    else:
        print(f'{args.test} not found.')
if (args.optimize and args.optimization_parameters):
    study(args.gymenv, args.algo, args.optimization_parameters, args.timesteps, args.n_trials, args.n_evals)
