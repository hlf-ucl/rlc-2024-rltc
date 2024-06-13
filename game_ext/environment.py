"""
Environment for communication consensus.
"""
import networkx as nx
import numpy as np


def square_lattice_graph(num_nodes):
    """Create a NetworkX square lattice graph with the given number of nodes.

    Args:
         num_nodes: square number
    Returns:
        NetworkX graph object with nodes labeled 0, 1, ..., num_nodes - 1.
    """
    m = n = int(np.sqrt(num_nodes))
    assert m * n == num_nodes

    graph = nx.generators.lattice.grid_2d_graph(m, n, periodic=False)

    # Need to relabel nodes from 2D grid coordinates to ints
    mapping = {n: i for i, n in enumerate(graph.nodes)}
    graph = nx.relabel_nodes(graph, mapping)
    return graph


class Process:
    """
    Represents a process/node in the communication network.

    Note on terminology: a "process" refers to a generic autonomous entity/
    computing element. This usage is often found in distributed systems literature.
    """
    def __init__(self, pid, seed=None):
        self.id = pid

        # For temporarily storing incoming messages/values from neighbours
        self.inbox = []

        # References to neighbouring Processes
        self.neighbours = []

        # Trust score dictionary mapping neighbour IDs to {0, 1}
        self.trusts = dict()

        # Process' own value. None if uninitialised, {0, 1} otherwise.
        self.vote = None

        # RNG (could use environment's one)
        self.random = np.random.default_rng(seed)

        # Reliable by default
        self.is_reliable = True

    def reset(self):
        """Reset the process' state, then reset all trust scores to 1,
        assuming we trust all neighbours at the start of each episode."""
        self.inbox.clear()
        self.trusts.clear()
        self.vote = None

        # Trust by default initially
        for nbr in self.neighbours:
            self.trusts[nbr.id] = 1

    def set_neighbours(self, neighbours):
        """Set the process' neighbours.

        Should only be set once, since the network structure doesn't vary
        between users for now.
        """
        self.neighbours.clear()
        self.neighbours.extend(neighbours)

    def set_vote(self, value):
        """Set the process' internal value, which must be 0 or 1.

        This should only be used at the start of each episode as initialisation,
        or debug purposes."""
        assert value in {0, 1}
        self.vote = value

    def gather(self):
        """Collect data from our trusted neighbours."""
        for nbr in self.neighbours:
            # Only get values from trusted neighbours
            if self.trusts[nbr.id] == 1:
                self.inbox.append((nbr.id, nbr.vote))

    def aggregate(self):
        """Update current state using all info currently in the inbox.

        The agent's inbox is cleared after this procedure.
        """
        # Assume inbox values came from trusted neighbours
        data = [self.vote]
        while len(self.inbox) > 0:
            sender_id, value = self.inbox.pop()
            assert self.trusts[sender_id]
            data.append(value)

        # Pick a random neighbour's value
        self.vote = self.random.choice(data)

    def flip(self, neighbour_id):
        """Toggle the trust score of a neighbour.

        This corresponds to an action in RL/MDP.
        """
        assert neighbour_id in self.trusts

        # Relies on trust scores being either 0 or 1
        self.trusts[neighbour_id] = 1 - self.trusts[neighbour_id]

    def state_to_int(self):
        """Convert the process state to a unique integer representation
        (e.g. for tabular learning)"""
        # The agent's MDP state only consists of the trust dict, for now.
        # (basically converting binary to int below)
        res = 0
        mult = 1
        for nbr in self.neighbours:
            value = int(self.trusts[nbr.id])
            res += value * mult
            mult *= 2
        return res


class UnreliableProcess(Process):
    """Unreliable process that performs some kind of predetermined behaviour.
    Does not have an accompanying trained RL agent.

    Currently, an unreliable process is either/or:
    - fixed: sticks to its initial value and ignores the neighbours' values.
    - random: sets its local value to either 0 or 1 with probability 1/2
      upon aggregation. 
    """

    def __init__(self, pid, behaviour="fixed", seed=None):
        super(UnreliableProcess, self).__init__(pid, seed=seed)
        self.is_reliable = False

        assert behaviour in {"fixed", "random"}
        self.behaviour = behaviour

    def gather(self):
        pass

    def aggregate(self):
        if self.behaviour == "random":
            # Set local value randomly during the aggregation step
            self.vote = int(self.random.random() < 0.5)

    def flip(self, i):
        pass


class Environment:
    """
    A communication network scenario with unreliable processes.

    We assume that reliable (or normal) processes are trainable,
    e.g. make decisions according to its corresponding Q-Learning agent,
    while unreliable processes follow a predetermined behaviour according to
    some failure model.
    """

    def __init__(self,
                 num_processes,
                 frac_reliable,
                 noise,
                 episode_timesteps,
                 unreliable_behaviour="fixed",
                 seed=None):

        assert 0.0 <= frac_reliable <= 1.0

        self.num_processes = num_processes
        self.frac_reliable = frac_reliable
        self.num_reliable = int(frac_reliable * num_processes)

        # True global value (1)
        self.true_value = 1

        # Observation noise (probability of a process observing not true_value)
        assert 0 <= noise < 1
        self.noise = noise

        # Length (number of timesteps) per episode
        self.episode_timesteps = episode_timesteps
        self.current_timestep = 0

        # Environment rng
        self.random = np.random.default_rng(seed)

        # NetworkX Graph object
        # Assuming that nodes are labelled 0, 1, ... num_processes - 1
        # and correspond to process integer IDs.
        self.graph = self._gen_graph()

        reliable_ids = self.random.choice(num_processes, size=self.num_reliable,
                                          replace=False)
        reliable_ids = set(reliable_ids)

        # Unreliable behaviour: fixed or random
        self.unreliable_behaviour = unreliable_behaviour

        # Initialise processes
        # Each process shares the environment's RNG object
        # (fine for sequential execution)
        # Additional lists to speed up loops
        # (at the cost of additional memory, but not a big issue)
        # e.g. each time we want to loop through reliable processes,
        # avoid iterating over ALL processes and check each one's identity.
        self.processes = []
        self.reliable_processes = []
        self.unreliable_processes = []
        for pid in range(num_processes):
            if pid in reliable_ids:
                process = Process(pid, seed=self.random)
                self.reliable_processes.append(process)
            else:
                process = UnreliableProcess(pid, 
                    behaviour=self.unreliable_behaviour,
                    seed=self.random)
                self.unreliable_processes.append(process)
            self.processes.append(process)

        assert len(self.processes) == num_processes
        assert len(self.reliable_processes) == self.num_reliable
        assert len(self.unreliable_processes) == num_processes - self.num_reliable

        # Assign neighbours according to communication graph (only done once)
        for process in self.processes:
            nbr_ids = self.graph.neighbors(process.id)
            nbrs = [self.processes[i] for i in nbr_ids]
            assert len(nbrs) > 0
            process.set_neighbours(nbrs)

    def reset(self):
        """Reset the environment for a new episode.

        Also randomly resets each agent's initial vote
        """
        self.current_timestep = 0

        for process in self.processes:
            process.reset()

        for process in self.reliable_processes:
            # Assign random value
            value = int(self.random.random() > self.noise)
            process.set_vote(value)

        # For now, unreliable processes is defined as sending the wrong value
        for process in self.unreliable_processes:
            value = int(not self.true_value)
            # value = int(self.random.random() > self.noise)
            process.set_vote(value)

    def step(self, actions=None):
        """Update the environment state according to agents' actions.

        Returns:
            next_states (list of int): per-agent local state
            reward (list of float): per-agent reward
            done (bool): is it the end of the episode?
            info (dict): a dictionary containing additional information
                (e.g. for debugging purposes)

        Actions correspond to those of reliable (trainable) processes only.
        If actions=None, assume all agents choose to do nothing, i.e. NO-OP
        """
        assert self.current_timestep < self.episode_timesteps
        assert actions is None or len(actions) == len(self.reliable_processes)

        self.current_timestep += 1

        if actions is not None:
            for action, process in zip(actions, self.reliable_processes):
                # Currently each action refers to a particular neighbour.
                # The action flips its binary trust score,
                # The exception is when action = 0, which is NO-OP.
                assert action >= 0
                if action > 0:
                    target_idx = action - 1
                    neighbour_id = process.neighbours[target_idx].id
                    process.flip(neighbour_id)

        # Follow the typical "protocol" of gathering and aggregating
        # neighbours' information
        for process in self.processes:
            process.gather()

        for process in self.processes:
            process.aggregate()

        next_states = [p.state_to_int() for p in self.reliable_processes]
        rewards = [self._compute_reward(p) for p in self.reliable_processes]
        done = self.current_timestep == self.episode_timesteps
        info = {}

        return next_states, rewards, done, info

    def _compute_reward(self, process):
        """Compute the reward in the current state of the process.
        (full transition not needed for now). """
        # Local agreement: the process has the same vote as its trusted neighbours.
        # Structure:
        # -1  : no local agreement
        # -1 : local agreement, wrong value
        # +1 : local agreement, correct value

        local_agree = all(process.vote == n.vote for n in process.neighbours
                          if process.is_reliable)
        if not local_agree:
            return -1.

        if process.vote == self.true_value:
            return 1.
        else:
            return -1.

    def get_observations(self):
        return [p.state_to_int() for p in self.reliable_processes]

    def _gen_graph(self) -> nx.Graph:
        """Generate the NetworkX graph structure.

        Currently a square lattice.
        """
        return square_lattice_graph(self.num_processes)

    def compute_success_rate(self):
        """Compute percentage of agents that get the correct value."""
        success_count = sum(
            p.vote == self.true_value for p in self.reliable_processes)
        return success_count / len(self.reliable_processes)

    def compute_average_trust_rate(self):
        """Compute trust rate among reliable processes.

        For each reliable process, compute the fraction of trusted neighbours.
        Sum all of the above then take an average over reliable processes.
        """
        trust_sum = 0.0
        for process in self.reliable_processes:
            trusted_nbrs = sum(process.trusts.values())
            all_nbrs = len(process.trusts)
            trust_sum += (trusted_nbrs / all_nbrs)
        return trust_sum / len(self.reliable_processes)

    def compute_average_trust_accuracy(self):
        """Compute average trust accuracy among reliable processes.

        For each reliable process and its neighbours,
        check if its trust scores are "correct",
        i.e. A trusts B if B is actually reliable, distrust otherwise,
        as a fraction of the total number of neighbours.
        Then, average over all reliable processes.
        """
        res = 0.0
        for process in self.reliable_processes:
            count = 0
            for nbr_id, trust_nbr in process.trusts.items():
                nbr = self.processes[nbr_id]
                count += (bool(trust_nbr) == nbr.is_reliable)
            res += (count / len(process.trusts))
        return res / len(self.reliable_processes)

    def compute_mutual_trust_rate(self):
        """Count the pairs of reliable processes that have mutual trust
        (i.e. A trusts B and B trusts A) as a fraction of the
        total number of edges between reliable processes
        (function returns 0 if there aren't any)
        """
        count = 0
        num_edges = 0

        num_reliable = len(self.reliable_processes)
        for i in range(num_reliable):
            for j in range(i + 1, num_reliable):
                fst = self.reliable_processes[i]
                snd = self.reliable_processes[j]
                if fst in snd.neighbours:
                    assert snd in fst.neighbours
                    num_edges += 1
                    count += int(fst.trusts[snd.id] and snd.trusts[fst.id])

        if num_edges == 0:
            return 0.0
        else:
            return count / num_edges
