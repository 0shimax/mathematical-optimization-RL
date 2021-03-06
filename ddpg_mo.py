from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import argparse
import sys

import chainer
from chainer import optimizers
import gym
gym.undo_logger_setup()
from gym import spaces
import gym.wrappers
import numpy as np

# from chainerrl.agents.ddpg import DDPG
# from chainerrl.agents.ddpg import DDPGModel
from ddpg_agent import DDPG
from ddpg_agent import DDPGModel

from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
# from chainerrl import policy
import deterministic_policy as policy
# from chainerrl import q_functions
import q_func as q_functions
from chainerrl import replay_buffer

from environment import Easy2D, Griewank


def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='out')
    parser.add_argument('--env', type=str, default='griewank')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--obs-dim', type=int, default=1,
                        help='int')
    parser.add_argument('--action-dim', type=int, default=1,
                        help='int')
    parser.add_argument('--action-low', type=int, default=-100,
                        help='int. action state low limit.')
    parser.add_argument('--action-high', type=int, default=100,
                        help='int. action state high limit.')
    parser.add_argument('--gpu', type=int, default=-1)
    # parser.add_argument('--final-exploration-steps',
    #                     type=int, default=10 ** 4)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--steps', type=int, default=10 ** 4)
    parser.add_argument('--n-hidden-channels', type=int, default=300)
    parser.add_argument('--n-hidden-layers', type=int, default=3)
    parser.add_argument('--replay-start-size', type=int, default=5000)
    parser.add_argument('--n-update-times', type=int, default=1)
    parser.add_argument('--target-update-interval',
                        type=int, default=1)
    parser.add_argument('--target-update-method',
                        type=str, default='soft', choices=['hard', 'soft'])
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=10 ** 3)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--minibatch-size', type=int, default=200)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--use-bn', action='store_true', default=False)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--sigma-scale-factor', type=float, default=5e-2)
    args = parser.parse_args()

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    # Set a random seed used in ChainerRL
    # misc.set_random_seed(args.seed, gpus=(args.gpu,))
    misc.set_random_seed(args.seed)

    def clip_action_filter(a):
        print("clip_", a)
        return np.clip(a, args.action_low, args.action_high)

    def reward_filter(r):
        return r * args.reward_scale_factor

    def make_env(test, env_name='griewank', obs_dim=1, action_dim=1,
                 action_low=-100, action_high=100):
        env = None
        if env_name == 'griewank':
            env = Griewank(test=test, obs_dim=obs_dim,
                           action_dim=action_dim,
                           action_low=args.action_low,
                           action_high=args.action_high)
        elif env_name == 'easy2d':
            env = Easy2D(test=test, obs_dim=obs_dim,
                         action_dim=action_dim,
                         action_low=args.action_low,
                         action_high=args.action_high)
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # if args.monitor:
        #     env = gym.wrappers.Monitor(env, args.outdir)
        # if isinstance(env.action_space, spaces.Box):
        #     misc.env_modifiers.make_action_filtered(env, clip_action_filter)
        if not test:
            misc.env_modifiers.make_action_filtered(env, clip_action_filter)
            misc.env_modifiers.make_reward_filtered(env, reward_filter)
        # if args.render and not test:
        #     misc.env_modifiers.make_rendered(env)
        return env

    env = make_env(test=False, env_name=args.env, obs_dim=args.obs_dim,
                   action_dim=args.action_dim,
                   action_low=args.action_low,
                   action_high=args.action_high)
    # timestep_limit = env.spec.tags.get(
    #     'wrapper_config.TimeLimit.max_episode_steps')
    timestep_limit = env.timestep_limit
    obs_size = np.asarray(env.observation_space.shape).prod()
    action_space = env.action_space

    action_size = np.asarray(action_space.shape).prod()
    print("obs_size, action_size", obs_size, action_size)
    if args.use_bn:
        q_func = q_functions.FCBNLateActionSAQFunction(
            obs_size, action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            normalize_input=True)
        pi = policy.FCBNDeterministicPolicy(
            obs_size, action_size=action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            min_action=action_space.low, max_action=action_space.high,
            bound_action=True,
            normalize_input=True)
    else:
        q_func = q_functions.FCSAQFunction(
            obs_size, action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers)
        pi = policy.FCDeterministicPolicy(
            obs_size, action_size=action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            min_action=action_space.low, max_action=action_space.high,
            bound_action=True)
    model = DDPGModel(q_func=q_func, policy=pi)
    opt_a = optimizers.Adam(alpha=args.actor_lr)
    opt_c = optimizers.Adam(alpha=args.critic_lr)
    opt_a.setup(model['policy'])
    opt_c.setup(model['q_function'])
    opt_a.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_a')
    opt_c.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_c')

    rbuf = replay_buffer.ReplayBuffer(5 * 10 ** 5)

    def phi(obs):
        return obs.astype(np.float32)

    def random_action():
        a = action_space.sample()
        if isinstance(a, np.ndarray):
            a = a.astype(np.float32)
        return a

    ou_sigma = (action_space.high - action_space.low) * args.sigma_scale_factor
    explorer = explorers.AdditiveOU(sigma=ou_sigma)
    agent = DDPG(model, opt_a, opt_c, rbuf, gamma=args.gamma,
                 explorer=explorer, replay_start_size=args.replay_start_size,
                 target_update_method=args.target_update_method,
                 target_update_interval=args.target_update_interval,
                 update_interval=args.update_interval,
                 soft_update_tau=args.soft_update_tau,
                 n_times_update=args.n_update_times,
                 phi=phi, gpu=args.gpu, minibatch_size=args.minibatch_size)

    if len(args.load) > 0:
        agent.load(args.load)

    eval_env = make_env(test=True, env_name=args.env, obs_dim=args.obs_dim,
                        action_dim=args.action_dim,
                        action_low=args.action_low,
                        action_high=args.action_high)
    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_runs=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_env=eval_env,
            eval_n_runs=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir,
            max_episode_len=timestep_limit)


if __name__ == '__main__':
    main()
