import functools

from autograph.lib.envs.mineworldenv import MineWorldEnv
from autograph.net.curiosity.rnd_models import RND
from autograph.net.mazenet import Mazenet
from autograph.net.minenet import Minenet, minecraft_obs_rewrite


def num_channels(env: MineWorldEnv):
    """How many output channels the environment will have"""
    return len(env.config.inventory) + len(env.config.placements) + 1


def minenet_v1(env: MineWorldEnv, num_blocks, **kwargs):
    return Minenet(env.shape, num_channels(env), num_blocks, env.action_space.n, **kwargs)


def mine_mazenet_v1(env: MineWorldEnv):
    return Mazenet(env.shape, actions_n=env.action_space.n, in_channels=num_channels(env), initstride=1, initpadding=2)


def minernd_v1(env: MineWorldEnv, feature_space: int):
    return RND(num_channels(env), env.shape, feature_space, init_stride=1)


def mine_obs_rewriter_creator(env: MineWorldEnv):
    return functools.partial(minecraft_obs_rewrite, env.shape)
