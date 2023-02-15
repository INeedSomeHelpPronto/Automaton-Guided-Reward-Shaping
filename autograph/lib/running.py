import functools
import multiprocessing.context
import os
import random
from _signal import SIGKILL
from collections import deque
from copy import deepcopy
from typing import Callable, Any, Tuple, List, Union

import numpy
import psutil
import ptan
import torch
from decorator import contextmanager
from gym import Env
from tensorboardX import SummaryWriter
from torch import multiprocessing as multiprocessing
from torch.nn import functional as F
from torch.optim import Optimizer

from autograph.lib.envs.saveloadenv import SaveLoadEnv
from autograph.lib.loss_functions import LossFunction
from autograph.lib.mcts import MCTS
from autograph.lib.util.trace_info_processor import TraceInfoPreprocessor
from autograph.lib.util.trace_return_step import TraceStep, TraceReturnStep
from autograph.net.curiosity.curiosity_optimizer import ModuleCuriosityOptimizer
from autograph.net.mazenet import Mazenet


# play steps happen here.
def run_episode_generic(env: Env, action_value_generator: Callable[[Any, int], Tuple[List[float], float]],
                        max_length: int,
                        max_len_reward: Union[int, None],
                        action_selector: ptan.actions.ActionSelector = ptan.actions.ArgmaxActionSelector(),
                        state_observer: Callable[[Any], None] = None, render: bool = False):
    """
    Run an episode in an environment
    :param env: The environment to run the episode in
    :param action_value_generator: Given a state and step number, produce some action probabilities and a value estimate
    :param max_length: The step number after which to cut off the episode
    :param action_selector: How to decide the right actions to select given NN "probabilities"
    :param state_observer:A callback function to notify of all states encountered
    :return: A list of steps taken, and the value estimate of the final state
    """
    done = False
    next_state = env.reset()
    next_action_raw, next_value = action_value_generator(next_state, 0)
    length = 0

    trace = []

    while not done:
        length += 1

        # Take an action and get results of the action
        state, action_raw, value = next_state, next_action_raw, next_value
        # action_probs = F.softmax(action_raw)
        # action_selected = action_selector(action_probs.unsqueeze(0).cpu().numpy())[0]
        # TODO
        action_selected = action_selector(numpy.array([action_raw]))[0]
        next_state, reward, done, info = env.step(action_selected)

        if render:
            env.render()

        if state_observer:
            state_observer(next_state)

        done_from_env = 1 if done else 0

        # Get the next action/value pair (now instead of beginning of loop so that we easily have access to next value)
        if done:
            next_value = 0
        else:
            next_action_raw, next_value = action_value_generator(next_state, length)
            if length >= max_length:
                done = True
                if max_len_reward is not None:
                    next_value = max_len_reward

        trace.append(
            TraceStep(state, value, action_raw, action_selected, next_state, next_value, reward, info, done_from_env))

    return trace, next_value


def run_episode(net: Mazenet, env: Env, max_length: int, device,
                action_selector: ptan.actions.ActionSelector = ptan.actions.ArgmaxActionSelector(),
                state_observer: Callable[[Any], None] = None) -> Tuple[List[TraceStep], float]:
    """
    Run an episode using the policy from the given network
    :param net: The network whose policy to follow- should have a forward_obs method that returns the action outputs and
    value estimate given a single observation
    :param env: The environment to run the simulation in
    :param max_length: How long the episode should be allowed to run before cutting it off
    :param action_selector: A callable that accepts a list of action probabilities and returns the action to choose
    :param state_observer: A callable that will be called with each state
    :param device: What device to run the network on
    :return: A list of TraceStep tuples and a value estimate of the last state
    """

    def obs_helper(state, step):
        action, value = net.forward_obs(state, device)
        return action.detach(), value.detach().squeeze(-1)

    return run_episode_generic(env, obs_helper, max_length, action_selector, state_observer)


def run_mcts_episode(net: Mazenet, env: SaveLoadEnv, max_length: int, curiosity: ModuleCuriosityOptimizer, device,
                     c_puct,
                     num_batches, batch_size,
                     state_observer: Callable[[Any], None] = None) -> Tuple[List[TraceStep], float]:
    """
    Run an episode using MCTS with curiosity as the action selection
    :param net: The policy/value network
    :param env: The environment to run the simulation in
    :param max_length: When to cut off the simulation
    :param curiosity: Something to calculate the relative "newness" of a state
    :param device: The device to run the simulation on
    :param c_puct: Puct constant of MCTS
    :param num_batches: How many groups of MCTS sims to run
    :param batch_size: How many MCTS sims per group
    :param state_observer: Function to call for every state seen
    :return: A trace and final value estimate
    """

    def curiosity_evaluator(sars):
        states, actions, rewards, next_states = zip(*sars)
        rewards = curiosity.get_curiosity(states, actions, next_states)
        return rewards.tolist()

    def curiosity_trainer(sars):
        states, actions, rewards, next_states = zip(*sars)
        curiosity.train(states, actions, next_states, train_rounds=1)

    def state_evaluator(states):
        states_transformed = torch.stack(tuple(net.rewrite_obs(s) for s in states))
        pols, vals = net(states_transformed.to(device))
        pollist = F.softmax(pols, dim=-1).tolist()
        vallist = vals.squeeze(-1).tolist()

        return list(zip(pollist, vallist))

    mcts = MCTS(env.action_space.n, curiosity_evaluator, state_evaluator, curiosity_trainer, c_puct=c_puct)

    def action_value_generator(state, step):
        mcts.mcts_batch(env, state, num_batches, batch_size)
        probs, values = mcts.get_policy_value(state, 1)
        return probs, max(values)

    return run_episode_generic(env, action_value_generator, max_length, ptan.actions.ProbabilityActionSelector(),
                               state_observer)


def calculate_returns_adv(trace: List[TraceStep], last_value: float, discount: float) \
        -> List[Tuple[float, float, float]]:
    """
    Given a trace, get the discounted returns, advantage, and return advantage value. Advantage is calculated based on the actual
    discounted return versus the value estimate of the state.
    :param trace: The information from a single run
    :param last_value: A value estimate of the final state
    :param discount: The discount factor
    :return: A list of (discounted return, advantage, return_advantage) tuples
    """
    discounted_return = last_value
    next_value = last_value
    returns_adv = []

    for step in reversed(trace):
        discounted_return *= discount
        discounted_return += step.reward
        advantage = step.reward + (discount * next_value) - step.value
        return_advantage = discounted_return - float(step.value)
        returns_adv.append((discounted_return, advantage, return_advantage))

        next_value = step.value

    returns_adv.reverse()

    return returns_adv


def parallel_queue_worker(queue: multiprocessing.SimpleQueue, function_to_run: Callable) -> None:
    """
    Worker that repeatedly adds the result of a function to a queue, but eventually quits when the parent process dies.
    :param queue: The queue to add results to
    :param function_to_run: A function that takes no arguments and produces no results
    :return: When the parent process dies
    """
    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    while True:
        result = function_to_run()

        # If the parent process isn't python, that means that it got terminated
        # (and this orphan process got reassigned to init or to some display manager)
        parent = psutil.Process(os.getppid())
        if "python" not in parent.name():
            return

        queue.put(result)
        # pr.dump_stats("performance.cprof")


class RenderEnvAndReturnWrapper:
    """
    Wraps a given callable such that if the callable accepts an env as a parameter, the env is printed before returning.

    Not a nested function for pickling reasons.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, env: Env, **kwargs):
        ret = self.func(env=env, **kwargs)
        env.render()
        return ret


@contextmanager
def get_parallel_queue(num_processes, episode_runner, env, **kwargs):
    """
    Create a queue that has a bunch of parallel feeder processes
    :param num_processes: How many feeder processes
    :param episode_runner: A function that produces a trace to be added to the queue
    :param env: The environment to run the simulations in
    :param kwargs: Additional arguments to the episode runner
    :return: The queue that the processes are feeding into, and a list of the created processes
    """
    multiprocessing.set_start_method("spawn", True)

    sim_round_queue = multiprocessing.SimpleQueue()
    processes: List[multiprocessing.context.Process] = []
    for i in range(num_processes):
        newenv = deepcopy(env)
        render = (i == 0)

        if render:
            this_episode_runner = RenderEnvAndReturnWrapper(episode_runner)
        else:
            this_episode_runner = episode_runner

        mcts_with_args = functools.partial(this_episode_runner, env=newenv, **kwargs)

        p = multiprocessing.Process(target=parallel_queue_worker,
                                    args=(sim_round_queue, mcts_with_args))
        p.daemon = True
        p.name = "Worker thread " + str(i)
        p.start()
        processes.append(p)

    try:
        yield sim_round_queue
    finally:
        for p in processes:
            os.kill(p.pid, SIGKILL)


class RandomReplayTrainingLoop:
    """
    A training loop that reads random replays from a replay buffer and trains a loss function based on that
    """

    def __init__(self, discount: float, replay_buffer_len: int, min_trace_to_train: int, train_rounds: int,
                 obs_processor: Callable, writer: SummaryWriter, device):
        self.device = device
        self.obs_processor = obs_processor
        self.train_rounds = train_rounds
        self.min_trace_to_train = min_trace_to_train
        self.discount = discount
        self.writer = writer

        self.trace_hooks: List[Callable[[List[TraceReturnStep], float], None]] = []
        self.round_hooks: List[Callable[[int], None]] = []

        self.recent_traces: deque[TraceReturnStep] = deque(maxlen=replay_buffer_len)
        self.global_step = 0
        self.num_rounds = 0

    def add_trace_hook(self, hook: Callable[[List[TraceReturnStep], float], None]):
        self.trace_hooks.append(hook)

    def add_round_hook(self, hook: Callable[[int], None]):
        self.round_hooks.append(hook)

    def process_trace(self, trace: List[TraceStep], last_val: float):
        """
        Calculate the returns on a trace and add it to the replay buffer
        :param trace: The actions actually taken
        :param last_val: The value estimate of the final state
        """
        ret_adv = calculate_returns_adv(trace, last_val, discount=self.discount)

        trace_adv = [TraceReturnStep(*twi, *ra) for twi, ra in zip(trace, ret_adv)]

        for hook in self.trace_hooks:
            hook(trace_adv, last_val)

        self.recent_traces.extend(trace_adv)

        self.writer.add_scalar("run/total_reward", sum([step.reward for step in trace]), self.global_step)
        self.writer.add_scalar("run/length", len(trace), self.global_step)

        self.global_step += len(trace)

    def train_on_traces(self, traces: List[TraceReturnStep], loss_function: LossFunction, optimizer: Optimizer):
        """
        Minimize a loss function based on a given set of replay steps.
        :param traces:
        :return:
        """
        trinfo = TraceInfoPreprocessor(traces, self.obs_processor, self.device)

        loss, logs = loss_function(trinfo)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(None)

        logs["total_loss"] = loss

        return logs

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "recent_traces": self.recent_traces,
            "num_rounds": self.num_rounds
        }

    def load_state_dict(self, state):
        renames = {
            "globalstep": "global_step",
            "numrounds": "num_rounds"
        }

        for key, value in renames.items():
            if key in state and value not in state:
                state[value] = state[key]

        self.global_step = state["global_step"]
        self.num_rounds = state["num_rounds"]
        self.recent_traces = state["recent_traces"]

    def __call__(self, sim_round_queue, loss_function, optimizer):
        self.num_rounds += 1

        trace, last_val = sim_round_queue.get()
        self.process_trace(trace, last_val)

        if len(self.recent_traces) < self.min_trace_to_train:
            return

        for i in range(self.train_rounds):
            rand_traces = random.sample(self.recent_traces, self.min_trace_to_train)
            logs = self.train_on_traces(rand_traces, loss_function, optimizer)

            if i == self.train_rounds - 1:
                for key, value in logs.items():
                    self.writer.add_scalar(key, value, self.global_step)

        for hook in self.round_hooks:
            hook(self.num_rounds)
