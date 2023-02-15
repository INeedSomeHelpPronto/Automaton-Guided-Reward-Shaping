from typing import List, Callable, Any, Dict

from gym import Wrapper, Env

from autograph.lib.automata import AutomatonSet
# Action, Observation, Reward, Done, Info
from autograph.lib.envs.saveloadenv import SaveLoadEnv

AtomProp = Callable[[Any, Any, float, bool, Dict], bool]


class AutShapingWrapper(Wrapper, SaveLoadEnv):
    """
    An environment wrapper for gym that allows reward shaping using an automaton.
    """

    def __init__(self, env: Env, aps: List[AtomProp],
                 init_automaton: AutomatonSet,
                 potential_reward: float = 0,
                 discount: float = 0,
                 terminate_on_fail: bool = False, terminate_on_accept: bool = False,
                 termination_fail_reward: float = 0, termination_accept_reward: float = 1,
                 use_potential: bool = True):
        """
        Construct a new instance of the shaping wrapper
        :param env: The environment to wrap
        :param aps: A list of atomic propositions, as functions of the observation space
        :param init_automaton: The starting state of the automaton
        :param potential_reward: A reward amount to apply to
        :param discount: The discount factor to apply to future observations
        """
        super(AutShapingWrapper, self).__init__(env)
        self.termination_accept_reward = termination_accept_reward
        self.terminate_on_accept = terminate_on_accept
        self.termination_fail_reward = termination_fail_reward
        self.use_potential = use_potential
        self.terminate_on_fail = terminate_on_fail
        self.aps = aps
        self.init_automaton: AutomatonSet = init_automaton
        self.current_automaton: AutomatonSet = init_automaton
        self.reward = potential_reward
        self.discount = discount

    def reset(self, *args, **kwargs):
        """
        Reset the environment to the initial state
        """
        self.current_automaton = self.init_automaton
        return self.env.reset(*args, **kwargs)

    def _evaluate_state(self, obs: AutomatonSet) -> float:
        return len(obs.acceptance()) * self.reward

    def _evaluate_aut_transition(self, prev_aut: AutomatonSet, cur_aut: AutomatonSet) -> float:
        return (self.discount * self._evaluate_state(cur_aut)) - self._evaluate_state(prev_aut)

    def _transition(self, aut: AutomatonSet, action, obs, rew, done, info):
        ap_results = [ap(action, obs, rew, done, info) for ap in self.aps]
        return aut.transition(ap_results)

    def step(self, action):
        """
        Take an action in the environment, also transitioning in the automaton and shaping the reward
        """
        obs, rew, done, info = self.env.step(action)
        transitioned_aut = self._transition(self.current_automaton, action, obs, rew, done, info)
        if self.use_potential:
            rew_shape = self._evaluate_aut_transition(self.current_automaton, transitioned_aut)
        else:
            rew_shape = self._evaluate_state(transitioned_aut)

        self.current_automaton = transitioned_aut

        if self.terminate_on_accept:
            if len(self.current_automaton.acceptance()) > 0:
                rew_shape += self.termination_accept_reward
                done = True

        if self.terminate_on_fail:
            if not self.current_automaton.reaches_acceptance():
                rew_shape += self.termination_fail_reward
                done = True

        outer_info = {
            "inner": info,
            "automaton_states": self.current_automaton.states
        }
        #print(outer_info)
        return obs, rew + rew_shape, done, outer_info

    def save_state(self):
        if isinstance(self.env, SaveLoadEnv):
            return self.current_automaton, self.env.save_state()
        else:
            raise NotImplemented

    def load_state(self, state):
        if isinstance(self.env, SaveLoadEnv):
            new_aut, new_env_state = state
            self.env.load_state(new_env_state)
            self.current_automaton = new_aut
        else:
            raise NotImplemented
