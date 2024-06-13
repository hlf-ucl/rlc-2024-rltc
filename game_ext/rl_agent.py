"""
Code for reinforcement learning.

Assumes states and actions are encoded as integers.
E.g. if there are N actions in total,
each action is assigned a unique number from 0 to N-1.
"""

from typing import Optional

import numpy as np


def _argmax_random_tiebreak(array: np.ndarray) -> np.ndarray:
    """Greedy policy with uniform random tie-breaking.

    Args:
        array: array of values on which argmax is performed.
    Returns:
        probs: a probability distribution over argmaxes,
            same dimension as the input array.

    Example:
        array = [3, 2, 3, 1, 1]
        probs = [0.5, 0, 0.5, 0, 0]
    """

    # Example:
    # array = [3, 2, 3, 1, 1]; a_max = 3
    # mask = [1, 0, 1, 0, 0]; probs = [0.5, 0, 0.5, 0, 0]
    a_max = np.max(array)
    mask = (array == a_max).astype(float)
    probs = mask / mask.sum()
    return probs


def _epsilon_greedy_sample(action_prefs: np.ndarray, epsilon, seed=None) -> int:
    """Sample action from epsilon-greedy policy."""
    random = np.random.default_rng(seed)
    num_actions = action_prefs.shape[0]
    rand = random.random()
    if rand < epsilon:
        # P(uniform random) = eps
        policy = np.ones(num_actions) / num_actions
    else:
        # P(greedy) = 1 - eps
        policy = _argmax_random_tiebreak(action_prefs)

    return random.choice(num_actions, p=policy)


def _greedy_sample(action_prefs: np.ndarray, random_tiebreak=True, seed=None) -> int:
    """Sample action from greedy policy with optional random tiebreaking."""
    if random_tiebreak:
        random = np.random.default_rng(seed)
        num_actions = action_prefs.shape[0]
        probs = _argmax_random_tiebreak(action_prefs)
        return random.choice(num_actions, p=probs)
    else:
        return int(np.argmax(action_prefs))


class QLearningAgent:
    """Tabular Q Learning with epsilon-greedy behaviour."""

    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 step_size: float,
                 discount: float,
                 epsilon: float,
                 is_train=True,
                 seed=None):
        """
        Args:
            num_states: number of unique states
            num_actions: number of unique actions
            step_size: learning step size
            discount: reward discount factor
            epsilon: epsilon-greedy exploration parameter
            is_train: whether the agent should start learning or not.
                This can be changed later via the is_train attribute.
            seed: random seed
        """
        self.rng = np.random.default_rng(seed)
        self._num_actions = num_actions
        self._num_states = num_states

        self._last_state = None
        self._last_action = None

        self._q_values = np.zeros((num_states, num_actions))

        self._step_size = step_size
        self._discount = discount
        self._epsilon = epsilon

        # Train/eval mode.
        self._is_train = is_train

    @property
    def q(self):
        """Reference to Q(s, a) table."""
        return self._q_values

    @property
    def is_train(self):
        """Is the agent learning or evaluating now?"""
        return self._is_train

    @is_train.setter
    def is_train(self, is_train: bool):
        self._is_train = is_train

    def step(self, reward: Optional[float], next_state: int, done: bool,
             epsilon: float = None) -> Optional[int]:
        """Having performed A_t in state S_t,
        Now given R_{t+1} and S_{t+1},
        choose the next action A_{t+1}.

        Args:
            reward: R_{t+1}. If the reward is None, then assume that we are
                at the start of an episode (hence no feedback is provided)
            next_state: S_{t+1}
            done: whether the episode has terminated.
                If True, then `reward` is final and
                `next_state` is a terminal state with future returns = 0.
            epsilon: use a custom value for epsilon (epsilon-greedy parameter)

        Returns:
            Next action A_{t+1}, None if done=True
        """
        if epsilon is None:
            epsilon = self._epsilon

        # Assume that this is the start of an episode,
        # i.e. next_state is S_0, choose A_0.
        if reward is None:
            assert self._last_state is None
            assert self._last_action is None
            assert not done

            # Sample initial action A_0
            if self.is_train:
                next_action = _epsilon_greedy_sample(self.q[next_state],
                                                     epsilon,
                                                     seed=self.rng)
            else:
                next_action = _greedy_sample(self.q[next_state],
                                             random_tiebreak=True,
                                             seed=self.rng)

            # Save info for next step() call
            self._last_state = next_state
            self._last_action = next_action
            return next_action

        # Typical t > 0 case
        assert self._last_state is not None
        assert self._last_action is not None

        # Extract (S_t, A_t)
        state = self._last_state
        action = self._last_action

        # Perform Q-learning update during training
        if self.is_train:
            if done:
                # Terminal state and non-existent "future" states have reward 0
                q_next = 0.0
            else:
                # Get the next target action
                next_action_tgt = _greedy_sample(self.q[next_state],
                                                 random_tiebreak=True,
                                                 seed=self.rng)
                q_next = self.q[next_state, next_action_tgt]

            # Apply Q-value update equation
            q_curr = self.q[state, action]
            error = reward + (self._discount * q_next) - q_curr
            self.q[state, action] += self._step_size * error

        # Pick next action
        if done:
            next_action = None
        else:
            if self.is_train:
                next_action = _epsilon_greedy_sample(self.q[next_state],
                                                     epsilon,
                                                     seed=self.rng)
            else:
                next_action = _greedy_sample(self.q[next_state],
                                             random_tiebreak=True,
                                             seed=self.rng)

        # Save info for next step() call
        self._last_state = next_state
        self._last_action = next_action

        return next_action

    def reset(self, reset_q=False):
        """ Reset agent variables at the end of an episode,
        but keep the Q-value table by default. """
        self._last_state = None
        self._last_action = None

        if reset_q:
            self._q_values.fill(0.0)
