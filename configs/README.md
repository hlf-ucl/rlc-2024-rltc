# Configuration files

This folder contains JSON files that specify the combinations of parameters
that were used to run experiments.
Each entry contains a key-value pair, where the key refers to the parameter name and the value  is a list of different values that the parameter will take on during the experiments.
Supply one of these files to `game_ext.main` (see root README on how to run experiments), which will perform training/evaluation on all the possible combinations of parameters within it (taking `itertools.product` over the lists).

## Descriptions
 - `config.json`: the main results presented in the paper, using 16 agents and fixed failure model. For the 9-agent results in the appendix, change `num_processes` to `[9]`
 instead of `[16]`.
 - `config-random.json`: the same as `config.json` but it uses the random failure model instead (results in appendix).
 - `scalability.json`: tests increasing communication grid sizes `sqrt(N)` as presented in the main paper.
 - `trustaccuracy.json`: investigates a pattern in the trust accuracy for `0.75 <= f <= 1` (results in appendix).

## Other notes

- The parameter `num_processes` refers to the number of agents/nodes in the environment (rather than the number of parallel processes, for instance). We take the term _process_ from distributed systems literature where it refers to a generic autonomous entity or computing element, in our case an agent/node in the communication network. `num_processes` is assumed to be a square number, e.g. if we want to test a 4-by-4 communication grid then put `16`.
- `run_mode` refers to the type of agent: `trained` is RLTC (learnt trust mechanisms using IQL), `oracle` and `trust_all` (both fixed and not trained, see the main paper for descriptions).
- `unreliable_behaviour` is either `fixed` (unreliable agents output the wrong value) or `random` (output wrong value 1/2 the time at random).