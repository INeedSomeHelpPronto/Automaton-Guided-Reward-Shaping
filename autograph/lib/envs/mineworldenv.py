from collections import Counter
from random import Random
import random
from typing import Tuple, TypeVar, Union, List, Dict, Collection

import numpy as np
import math
from gym import spaces

from autograph.lib.envs.gridenv import GridEnv
from autograph.lib.envs.saveloadenv import SaveLoadEnv
from autograph.lib.util import element_add


class MineWorldTileType:
    """A single special tile in the mine world"""

    def __init__(self, consumable: bool, inventory_modifier: Counter, ap_name: str, grid_letter: str,
                 wall: bool = False):
        """
        :param consumable: Does this tile disappear after being activated
        :param inventory_modifier: How does this modify the inventory (e.g. wood -2, desk +1)
        :param ap_name: What atomic proposition should be true the round that this tile is activated
        :param grid_letter: What letter should be displayed on the grid
        """
        self.consumable = consumable
        self.inventory = inventory_modifier
        self.ap_name = ap_name
        self.grid_letter = grid_letter
        self.wall = wall

    def apply_inventory(self, prev_inventory: Counter):
        """
        Get the new inventory of the player after interacting with this tile, or errors if the player is unable to
        interact with the tile
        :param prev_inventory: The current inventory of the player
        """

        # Apply all the inventory changes and make sure that no item is negative
        new_inv = prev_inventory.copy()
        new_inv.update(self.inventory)
        if any([(new_inv[i] < 0) for i in new_inv]):
            raise ValueError()
        else:
            return new_inv

    @staticmethod
    def from_dict(dict):
        wall = dict.get("wall", False)
        return MineWorldTileType(consumable=dict["consumable"], inventory_modifier=Counter(dict["inventory_modifier"]),
                                 ap_name=dict["ap_name"], grid_letter=dict["grid_letter"], wall=wall)


T = TypeVar("T")
MaybeRand = Union[T, str]


class TilePlacement:
    def __init__(self, tile: MineWorldTileType, fixed_placements: Collection[Tuple[int, int]] = tuple(),
                 random_placements: int = 0, parlay_placement: int = 0):
        self.tile = tile
        self.fixed_placements = fixed_placements
        self.random_placements = random_placements
        self.parlay_placement = parlay_placement

    @staticmethod
    def from_dict(dict):
        tile = MineWorldTileType.from_dict(dict["tile"])
        fixed_raw = dict.get("fixed_placements", [])
        fixed_placements = [tuple(coord) for coord in fixed_raw]
        random_placements = dict.get("random_placements", 0)
        parlay_placement = dict.get("parlay_placement", 0)
        return TilePlacement(tile=tile,
                             fixed_placements=fixed_placements,
                             random_placements=random_placements,
                             parlay_placement=parlay_placement)


class InventoryItemConfig:
    def __init__(self, name: str, default_quantity: int, capacity: int):
        """
        :param name: Name of the item, like wood or iron
        :param default_quantity: How many of these items to start with
        :param capacity: Maximum amount of this item the agent can hold. Also used for scaling of NN inputs.
        """
        self.name = name
        self.default_quantity = default_quantity
        self.capacity = capacity

    @staticmethod
    def from_dict(dict):
        return InventoryItemConfig(**dict)


class MineWorldConfig:
    def __init__(self, shape: Tuple[int, int], initial_position: Union[Tuple[int, int], None],
                 placements: List[TilePlacement], inventory: List[InventoryItemConfig], fixed: bool = False):
        self.placements = placements
        self.shape = shape
        self.initial_position = initial_position
        self.inventory = inventory
        self.fixed = fixed

    @staticmethod
    def from_dict(dict):
        shape = tuple(dict["shape"])
        ip = dict["initial_position"]
        fixed = dict.get("fixed_across_episodes", False)
        initial_position = ip if ip is None else tuple(ip)
        placement = [TilePlacement.from_dict(i) for i in dict["placements"]]
        inventory = list(map(InventoryItemConfig.from_dict, dict["inventory"]))

        return MineWorldConfig(shape=shape, initial_position=initial_position, placements=placement,
                               inventory=inventory, fixed=fixed)


class MineWorldEnv(GridEnv, SaveLoadEnv):

    @staticmethod
    def from_dict(dict):
        return MineWorldEnv(MineWorldConfig.from_dict(dict))

    def __init__(self, config: MineWorldConfig, *args, **kwargs):
        super().__init__(shape=config.shape, *args, **kwargs)

        self.action_space = spaces.Discrete(6)
        self.config = config
        self.default_inventory = Counter(
            {inv_type.name: inv_type.default_quantity for inv_type in self.config.inventory})
        self.rand = Random()
        self.new_pqr = False
        """
        Up: 0,
        Right:1,
        Down: 2,
        Left: 3,
        No-op: 4,
        Tile action: 5"""

        self.done = True
        self.position: Tuple[int, int] = (0, 0)
        self.special_tiles: Dict[Tuple[int, int], MineWorldTileType] = dict()
        self.inventory = Counter()
        self.starting_state = None

    def step(self, action: int):
        assert self.action_space.contains(action)
        assert not self.done

        atomic_propositions = set()

        if action < 5:
            # Movement or no-op
            action_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
            new_place = element_add(self.position, action_offsets[action])

            can_move = self._in_bounds(new_place)

            # takes into account that the pit is a one way gate in and can't get out.
            can_move = can_move and self.can_move_with_pit(self.position, new_place)

            if new_place in self.special_tiles:
                if self.special_tiles[new_place].wall:
                    can_move = False
                if self.new_pqr:
                    this_tile: MineWorldTileType = self.special_tiles[self.position]
                    try:
                        new_inv = this_tile.apply_inventory(self.inventory)
                        for inv_config in self.config.inventory:
                            if new_inv[inv_config.name] > inv_config.capacity:
                                new_inv[inv_config.name] = inv_config.capacity
                        self.inventory = new_inv
                        atomic_propositions.add(this_tile.ap_name)
                    except ValueError:  # Couldn't apply inventory
                        pass
            if can_move:
                self.position = new_place
        elif not self.new_pqr:
            if self.position in self.special_tiles:
                this_tile: MineWorldTileType = self.special_tiles[self.position]

                try:
                    new_inv = this_tile.apply_inventory(self.inventory)
                    for inv_config in self.config.inventory:
                        if new_inv[inv_config.name] > inv_config.capacity:
                            new_inv[inv_config.name] = inv_config.capacity
                    self.inventory = new_inv

                    atomic_propositions.add(this_tile.ap_name)
                    if this_tile.consumable:
                        del self.special_tiles[self.position]
                    if this_tile.ap_name == "itemB":
                        # dirty hard-coding 3 in here for the abcpit game. 3 is the index of pit tile type in config
                        self.special_tiles[self.position] = self.config.placements[3].tile
                        atomic_propositions.add("pit")

                except ValueError:  # Couldn't apply inventory
                    pass

        info = {
            'atomic_propositions': atomic_propositions,
            'inventory': self.inventory.copy()
        }
        #self.render()
        # Reward is always 0 because it's minecraft. Exploration is reward in itself, and the only limit is imagination
        return self._get_observation(), 0, self.done, info

    def seed(self, seed=None):
        self.rand.seed(seed)

    def reset(self):
        self.done = False

        self.position = self.config.initial_position
        if not self.position:
            self.position = self.rand.randrange(0, self.shape[0]), self.rand.randrange(0, self.shape[1])
        self.inventory = self.default_inventory.copy()
        self.special_tiles = self._get_tile_positioning()

        # this should help us maintain the fixed randomly generated state across the 30k steps
        if self.config.fixed:
            if not self.starting_state:
                self.starting_state = self.save_state()
            else:
                self.load_state(self.starting_state)

        return self._get_observation()

    def generate_pqr(self, size):
        n = size
        if(n<3):
            print("YOU WILL DESTROY THE UNIVERSE IF YOU RUN IT WITH N < 3, DON'T DO IT")
            return
        rows, cols = (size,size)
        board = [['#']*cols for _ in range(rows)]
        sr = random.randint(0,size-1)
        sc = random.randint(0,size-1)
        while(sr == 0 or sr == 1):
            sr = random.randint(0,size-1)
        while(sc == n-2 or sc == n-1):
            sc = random.randint(0,size-1)

        er, ec = (0,0)
        while(True):
            er = random.randint(0,size-1)
            ec = random.randint(0,size-1)
            if er >= sr-1 or ec <= sc+1:
                continue
            if er == sr and ec == sc:
                continue
            break

        board[sr][sc] = 'S'
        board[er][ec] = 'H'

        for i in range (er, sr):
            board[i][sc] = 'p'
            if i > er:
                board[i][ec] = 'q'
        for i in range (sc+1,ec+1):
            board[sr][i] = 'q'
            if i < ec:
                board[er][i] = 'p'

        for i in range(sr-2, er, -2):
            col = ec - 1
            while (col - 2 >= sc and board[i][col-2] == '#' and board[i][col-1] == '#'):
                board[i][col] = 'p'
                col-=1

            if col <= sc+1 or col>=ec-1:
                continue
            for j in range (er+1, i+1):
                board[j][col] = 'q'

        #if you want to see the board!:
        '''
        for row in board:
            print(row)
        '''
        return board

    def can_move_with_pit(self, cur_space, new_space):
        if cur_space in self.special_tiles and \
                (self.special_tiles[cur_space].ap_name == "pit" or self.special_tiles[cur_space].ap_name == 'b'):
            if new_space in self.special_tiles and \
                    (self.special_tiles[new_space].ap_name == "pit" or self.special_tiles[new_space].ap_name == 'b'):
                return True
            else:
                return False
        return True

    # modified to randomly place the pit and "a", "b", "c" tiles so that the positions to the agent's initial position
    # Agent->a   <    agent->b   <    agent->c
    def _get_tile_positioning(self) -> Dict[Tuple[int, int], MineWorldTileType]:

        tiles = {}
        board = {}
        pit = False
        new_pqr = False
        size = 1
        for tile_type in self.config.placements:
            if tile_type.tile.ap_name == 'start':
                # Used in step function for pqr and new_pqr games, because these 2 have the home tile type in json
                self.new_pqr = True
                self.pqr = True
                #(IMPORTANT) the board size for new pqr is based on parlay_placement value for "start" in json
                size = tile_type.parlay_placement
                if size != 10 and size != 25:
                    self.new_pqr = False
                else:
                    self.pqr = False
            if tile_type.tile.ap_name == 'pit':
                pit_tile_type = tile_type.tile
                pit = True
            for fixed in tile_type.fixed_placements:
                tiles[fixed] = tile_type.tile

        all_spaces = set(np.ndindex(self.config.shape))
        pit_possibles = set(np.ndindex(tuple([self.config.shape[0] - 2, self.config.shape[1] - 2])))
        open_spaces = all_spaces.difference(tiles.keys())
        if (0, 0) in open_spaces:
            open_spaces.remove((0, 0))

        # if its a pit we must randomly generate a pit
        if pit:
            def get_pit_set(pit_center, max_apothem):
                pit_spaces = set()
                for i in range(pit_center[0]-max_apothem, pit_center[0] + max_apothem):
                    for j in range(pit_center[1]-max_apothem, pit_center[1] + max_apothem):
                        pit_spaces.add((i, j))
                return pit_spaces
            pit_spaces = set()
            pit_size_gotten = False
            while not pit_size_gotten:
                max_apothem = math.floor((self.config.shape[0] / 2) - 1)
                # Here, we pick the random apothem
                apothem = self.rand.randint(1, max_apothem)
                pit_center = self.rand.sample(pit_possibles, 1)[0]

                pit_check_up = tuple([pit_center[0], pit_center[1]+apothem])
                pit_check_down = tuple([pit_center[0], pit_center[1]-apothem])
                pit_check_left = tuple([pit_center[0] + apothem, pit_center[1]])
                pit_check_right = tuple([pit_center[0] - apothem, pit_center[1]])

                if pit_check_up in open_spaces and pit_check_down in open_spaces and \
                    pit_check_left in open_spaces and pit_check_right in open_spaces:

                    pit_spaces = get_pit_set(pit_center, apothem)
                    # THIS IS WHERE PIT ASSIGNMENT HAPPENS. can't start in pit
                    if self.position in pit_spaces:
                        pass
                    else:
                        for space in pit_spaces:
                            tiles[space] = pit_tile_type
                        pit_size_gotten = True

            o_spaces = open_spaces.difference(pit_spaces)

            def check_parlay(a, b, c, agent):
                agent_to_a = math.hypot(a[0]-agent[0], a[1]-agent[1])
                agent_to_b = math.hypot(b[0]-agent[0], b[1]-agent[1])
                agent_to_c = math.hypot(c[0]-agent[0], c[1]-agent[1])

                return agent_to_a < agent_to_b < agent_to_c

            parlay_correct = False
            while not parlay_correct:
                open_sp = o_spaces.copy()
                a_space, b_space, c_space = (0, 0), (0, 0), (0, 0)
                for tile_type in self.config.placements:
                    if tile_type.parlay_placement != 0:
                        if tile_type.tile.ap_name == 'itemA':
                            a_space = self.rand.sample(open_sp, 1)[0]
                            a_tile = tile_type.tile
                            open_sp.difference_update(a_space)
                        if tile_type.tile.ap_name == 'itemB':
                            b_space = self.rand.sample(pit_spaces, 1)[0]
                            b_tile = tile_type.tile
                        if tile_type.tile.ap_name == 'itemC':
                            c_space = self.rand.sample(open_sp, 1)[0]
                            c_tile = tile_type.tile
                            open_sp.difference_update(c_space)
                parlay_correct = check_parlay(a_space, b_space, c_space, self.position)

            o_spaces = open_sp.copy()
            tiles[a_space] = a_tile
            tiles[b_space] = b_tile
            tiles[c_space] = c_tile
        elif self.new_pqr:
            #print(size)
            board = self.generate_pqr(size)

            for i in range(size):
                for j in range(size):
                    if board[i][j] == 'S':
                        self.position = (i,j)
                        self.config.initial_position = (i,j)
                    for tile_type in self.config.placements:
                        if tile_type.tile.grid_letter == board[i][j]:
                            tiles[(i,j)] = tile_type.tile

        #for i in range(size):
            #for j in range(size):
                #print(tiles[(i,j)])

        for tile_type in self.config.placements:
            tile, num_placements = tile_type.tile, tile_type.random_placements
            spaces = self.rand.sample(open_spaces, num_placements)
            open_spaces.difference_update(spaces)


            for space in spaces:
                tiles[space] = tile

        return tiles

    def _get_observation(self):

        tiles = tuple(
            frozenset(space for space, content in self.special_tiles.items() if content is placement.tile) for
            placement in self.config.placements)

        inv_ratios = tuple(
            self.inventory[inv_config.name] / inv_config.capacity for inv_config in self.config.inventory)

        return (
            self.position,
            tiles,
            inv_ratios
        )

    def render(self, mode='human'):
        def render_func(x, y):
            agent_str = "@" if self.position == (x, y) else " "
            tile_str = self.special_tiles[(x, y)].grid_letter if (x, y) in self.special_tiles else " "
            return agent_str + tile_str, False, False

        print(self._render(render_func, 2), end="")
        print(dict(self.inventory))
        print(self.position)

    def save_state(self):
        return self.position, self.done, self.special_tiles.copy(), self.inventory.copy()

    def load_state(self, state):
        self.position, self.done, spec_tile, inv = state
        self.special_tiles = spec_tile.copy()
        self.inventory = inv.copy()