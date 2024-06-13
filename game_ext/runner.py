"""
Code for running RL training and evaluation.
"""
import itertools
import multiprocessing

from game_ext.environment import Environment
from game_ext.rl_agent import QLearningAgent


class Runner:

    def __init__(self,
                 num_processes,
                 frac_reliable,
                 noise,
                 episode_length,
                 run_mode,
                 step_size,
                 discount,
                 epsilon,
                 epsilon_decay_factor,
                 unreliable_behaviour,
                 seed=None
                 ):
        self.env_seed = seed
        self.num_processes = num_processes
        self.frac_reliable = frac_reliable
        self.noise = noise
        self.episode_length = episode_length

        assert run_mode in {"trained", "oracle", "trust_all"}
        self.run_mode = run_mode

        self.step_size = step_size
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay_factor = epsilon_decay_factor
        self.unreliable_behaviour = unreliable_behaviour

        # Initialise a RL agent per process
        self.agents = []

        self.env = Environment(self.num_processes,
                               self.frac_reliable,
                               self.noise,
                               self.episode_length,
                               unreliable_behaviour=self.unreliable_behaviour,
                               seed=self.env_seed)
        self.env.reset()

        # Reliable processes are trainable, i.e. have accompanying Q-Learners
        # Assume that the unreliable ones follow some predetermined behaviour
        # (e.g. random, fixed) rather than also learning via RL

        agent_seed = self.env_seed + 1
        for i, process in enumerate(self.env.reliable_processes):
            num_neighbours = len(process.neighbours)

            # All possible trust score array configurations,
            # which is a 2^n binary array
            num_states = 2 ** num_neighbours

            # Action space is the choice of neighbour trust score to flip,
            # and also a null/NO-OP action
            num_actions = num_neighbours + 1
            agent = QLearningAgent(num_states, num_actions,
                                   self.step_size, self.discount, self.epsilon,
                                   True,
                                   seed=agent_seed + i)
            self.agents.append(agent)

    def run(self, is_train, num_episodes):

        for agent in self.agents:
            agent.is_train = is_train

        all_episodes_average_reward = []
        all_episodes_success_rate = []
        all_episodes_average_trust_rate = []
        all_episodes_mutual_trust_rate = []
        all_episodes_average_trust_accuracy = []
        all_episodes_epsilon_decay = []

        # Might decay
        epsilon = self.epsilon

        for episode in range(num_episodes):

            self.env.reset()
            for agent in self.agents:
                agent.reset()
            
            if self.run_mode != "trained":
                # Fixed processes/agents
                assert self.run_mode in {"oracle", "trust_all"}
                for process in self.env.reliable_processes:
                    # Disable flip function
                    process.flip = lambda nid: None
                    for nbr in process.neighbours:
                        if self.run_mode == "trust_all":
                            # Don't use trust mechanism: "trust" all neighbours
                            process.trusts[nbr.id] = 1
                        elif self.run_mode == "oracle":
                            # "Oracle" solution: only trust neighbours that are actually reliable
                            process.trusts[nbr.id] = int(nbr.is_reliable)

            actions = [None for _ in self.agents]
            states = self.env.get_observations()
            rewards = [None for _ in self.agents]
            done = False

            episode_average_reward = 0.0
            episode_success_rate = 0.0
            episode_average_trust_rate = 0.0
            episode_mutual_trust_rate = 0.0
            episode_average_trust_accuracy = 0.0

            for _ in itertools.count(0, 1):

                for i, agent in enumerate(self.agents):
                    actions[i] = agent.step(rewards[i], states[i], done,
                                            epsilon=epsilon)

                if done:
                    break

                states, rewards, done, info = self.env.step(actions)

                episode_average_reward += (sum(rewards) / len(rewards))
                episode_success_rate += self.env.compute_success_rate()
                episode_average_trust_rate += self.env.compute_average_trust_rate()
                episode_mutual_trust_rate += self.env.compute_mutual_trust_rate()
                episode_average_trust_accuracy += self.env.compute_average_trust_accuracy()

            # Decay epsilon after each episode
            all_episodes_epsilon_decay.append(epsilon)
            epsilon *= self.epsilon_decay_factor

            # Compute and save statistics
            all_episodes_average_reward.append(episode_average_reward)
            all_episodes_success_rate.append(episode_success_rate)
            all_episodes_average_trust_rate.append(episode_average_trust_rate)
            all_episodes_mutual_trust_rate.append(episode_mutual_trust_rate)
            all_episodes_average_trust_accuracy.append(episode_average_trust_accuracy)

        stats = {
            "average_reward": all_episodes_average_reward,
            "success_rate": all_episodes_success_rate,
            "average_trust_rate": all_episodes_average_trust_rate,
            "mutual_trust_rate": all_episodes_mutual_trust_rate,
            "average_trust_accuracy": all_episodes_average_trust_accuracy
        }

        return stats


def run_experiment(config):
    """Run a single experiment.

    Returns a tuple of results for training and evaluation respectively.
    """
    num_train_episodes = config["num_train_episodes"]
    num_eval_episodes = config["num_eval_episodes"]

    del config["num_train_episodes"]
    del config["num_eval_episodes"]

    runner = Runner(**config)

    # Run modes 'oracle' and 'trust_all' don't require training
    if config["run_mode"] != "trained":
        num_train_episodes = 0

    train_results = runner.run(True, num_train_episodes)
    eval_results = runner.run(False, num_eval_episodes)

    return train_results, eval_results


def run_all(base_config,
            seeds,
            num_workers=1):
    """Perform separate runs of the same experiment same configuration
    over a list of random seeds."""
    # Set up config dicts per run
    all_configs = []
    for seed in seeds:
        config = base_config.copy()
        config["seed"] = seed
        all_configs.append(config)

    # Run N separate experiments
    results = []
    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as p:
            results = p.map(run_experiment, all_configs)
    else:
        for config in all_configs:
            res = run_experiment(config)
            results.append(res)

    return results
