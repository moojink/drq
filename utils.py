import math
import os
import random
from collections import deque

import numpy as np
import scipy.linalg as sp_la

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.util.shape import view_as_windows
from torch import distributions as pyd


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


class ActionRepeatWrapper(gym.Wrapper):
    """Gym wrapper for repeating actions."""
    def __init__(self, env, action_repeat, discount):
        gym.Wrapper.__init__(self, env)
        self._env = env
        self._action_repeat = action_repeat
        self._discount = discount

    def reset(self):
        return self._env.reset()

    def step(self, action):
        total_reward = 0.0
        discount = 1.0
        for _ in range(self._action_repeat):
            obs, reward, done, info = self._env.step(action)
            total_reward += reward * discount
            discount *= self._discount
            if done:
                break
        return obs, total_reward, done, info

    def render(self, **kwargs):
        return self._env.render(**kwargs)


class FrameStack(gym.Wrapper):
    """Gym wrapper for stacking image observations."""
    def __init__(self, view, env, k):
        self.view = view
        gym.Wrapper.__init__(self, env)
        self._k = k

        if str(self.view) == 'both':
            self._frames1 = deque([], maxlen=k)
            self._frames3 = deque([], maxlen=k)
        else:
            self._frames = deque([], maxlen=k)
        self._ee_grip_stack = deque([], maxlen=k)
        self._ee_pos_rel_base_stack = deque([], maxlen=k)
        self._contact_flags_stack = deque([], maxlen=k)
        shp = env.observation_space['im_rgb'].shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space['im_rgb'].dtype)
        self._max_episode_steps = env.env.env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            if str(self.view) == 'both':
                self._frames1.append(obs['im_rgb1'])
                self._frames3.append(obs['im_rgb3'])
            else:
                self._frames.append(obs['im_rgb'])
            self._ee_grip_stack.append(obs['ee_grip'])
            self._ee_pos_rel_base_stack.append(obs['ee_pos_rel_base'])
            self._contact_flags_stack.append(obs['contact_flags'])
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if str(self.view) == 'both':
            self._frames1.append(obs['im_rgb1'])
            self._frames3.append(obs['im_rgb3'])
        else:
            self._frames.append(obs['im_rgb'])
        self._ee_grip_stack.append(obs['ee_grip'])
        self._ee_pos_rel_base_stack.append(obs['ee_pos_rel_base'])
        self._contact_flags_stack.append(obs['contact_flags'])
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._ee_grip_stack) == self._k
        assert len(self._ee_pos_rel_base_stack) == self._k
        assert len(self._contact_flags_stack) == self._k
        ee_grip_obs = np.concatenate(list(self._ee_grip_stack), axis=0)
        ee_pos_rel_base_obs = np.concatenate(list(self._ee_pos_rel_base_stack), axis=0)
        contact_flags_obs = np.concatenate(list(self._contact_flags_stack), axis=0)
        if str(self.view) == 'both':
            assert len(self._frames1) == self._k
            assert len(self._frames3) == self._k
            img_obs1 = np.concatenate(list(self._frames1), axis=0)
            img_obs3 = np.concatenate(list(self._frames3), axis=0)
            return img_obs1, img_obs3, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs
        else:
            assert len(self._frames) == self._k
            img_obs = np.concatenate(list(self._frames), axis=0)
            return img_obs, ee_grip_obs, ee_pos_rel_base_obs, contact_flags_obs


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu