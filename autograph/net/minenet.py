import functools
import torch
import torch.nn as nn
from torch import Tensor

from autograph.lib.envs.mazeenv import n_hot_grid
from autograph.lib.util import const_plane
from autograph.net.curiosity.utils import Flatten


class Residual(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input):
        output = self.inner(input)
        return input + output


class Scalarize(nn.Module):
    def forward(self, input: Tensor):
        return input.view((input.size(0),))


KERNEL_SIZE = 3
PADDING_AMOUNT = 1


class Minenet(nn.Module):
    def __init__(self, shape, num_channels, num_blocks, num_actions,
                 num_intermediate_filters=32,
                 num_policy_head_filters=2,
                 num_value_head_filters=1,
                 disable_value=False,
                 disable_policy=False,
                 separate_networks=False):
        super().__init__()

        grid_size = 1
        for dim in shape:
            grid_size *= dim

        self.disable_value = disable_value
        self.disable_policy = disable_policy
        self.num_actions = num_actions
        self.separate_networks = separate_networks

        # Basically the architecture from AlphaGo
        def generate_common():
            init_conv = nn.Sequential(
                nn.Conv2d(num_channels, num_intermediate_filters, kernel_size=KERNEL_SIZE, padding=PADDING_AMOUNT),
                nn.BatchNorm2d(num_intermediate_filters),
                nn.LeakyReLU()
            )

            blocks = [nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(in_channels=num_intermediate_filters,
                                  out_channels=num_intermediate_filters,
                                  kernel_size=KERNEL_SIZE,
                                  padding=PADDING_AMOUNT),
                        nn.BatchNorm2d(num_intermediate_filters),
                        nn.LeakyReLU(),
                        nn.Conv2d(in_channels=num_intermediate_filters,
                                  out_channels=num_intermediate_filters,
                                  kernel_size=KERNEL_SIZE,
                                  padding=PADDING_AMOUNT),
                        nn.BatchNorm2d(num_intermediate_filters))
                ),
                nn.LeakyReLU()
            ) for _ in range(num_blocks)]

            return nn.Sequential(
                init_conv, *blocks
            )

        self.policy_trunk = generate_common()
        if separate_networks:
            self.value_trunk = generate_common()

        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=num_intermediate_filters,
                      out_channels=num_policy_head_filters,
                      kernel_size=1),
            nn.BatchNorm2d(num_policy_head_filters),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(grid_size * num_policy_head_filters, num_actions)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=num_intermediate_filters,
                      out_channels=num_value_head_filters,
                      kernel_size=1),
            nn.BatchNorm2d(num_value_head_filters),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(grid_size * num_value_head_filters, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            # Scalarize(),
            nn.Tanh()
        )

    def forward(self, x: Tensor):
        policy_trunk_out = self.policy_trunk(x)

        if self.separate_networks:
            value_trunk_out = self.value_trunk(x)
        else:
            value_trunk_out = policy_trunk_out

        batch = x.size(0)
        if self.disable_policy:
            pol = torch.full(size=(batch, self.num_actions), fill_value=1.0 / self.num_actions, device=x.device,
                             requires_grad=True)
        else:
            pol = self.policy_head(policy_trunk_out)

        if self.disable_value:
            val = torch.zeros(size=(batch, 1), device=x.device, requires_grad=True)
        else:
            val = self.value_head(value_trunk_out)

        return pol, val


@functools.lru_cache(16384)
def minecraft_obs_rewrite(shape, obs):
    position, tile_locs, inventories = obs
    # Convert to float?
    position_tile_layers = tuple(torch.from_numpy(n_hot_grid(shape, layer)).float() for layer in (position, *tile_locs))
    inventory_layers = tuple(const_plane(shape, layer).float() for layer in inventories)
    return torch.stack((*position_tile_layers, *inventory_layers), dim=0)
