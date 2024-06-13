"""
Command line program for running experiments and saving results.
"""
import argparse
import datetime
import json
import os
import pathlib
from datetime import timedelta
from pprint import pprint
from timeit import default_timer

from game_ext.hyperparam_utils import gen_hparam_dicts
from game_ext.runner import run_all


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run consensus RL experiments.")
    parser.add_argument("config_path",
                        help="Path to configuration parameter ranges.")
    parser.add_argument("seed_path",
                        help="Path to text file containing random seeds.")
    parser.add_argument("--results", dest="results_path",
                        default=os.getcwd(),
                        help="Where to save results to (a new folder called `results` will be created here)")
    parser.add_argument("--num-workers", type=int, dest="num_workers", default=1)

    args = parser.parse_args()

    # Load configuration parameter ranges from json file
    configfile_path = pathlib.Path(args.config_path)
    with open(configfile_path, "r") as f:
        config_ranges = json.load(f)

    print("Configuration parameter ranges:")
    pprint(config_ranges)

    # For each config combination, do multiple runs with different seeds
    # Load random seeds from file
    seedfile_name = pathlib.Path(args.seed_path)
    with open(seedfile_name, "r") as f:
        seeds = [int(line.strip()) for line in f.readlines()]

    assert len(seeds) > 0
    print(f"Using {len(seeds)} random seeds: {seeds}")

    num_workers = args.num_workers
    print(f"Running with {num_workers} worker(s).")

    # Write results to JSON files
    result_save_dir_root = pathlib.Path(args.results_path, "results")
    if not os.path.exists(result_save_dir_root):
        os.mkdir(result_save_dir_root)

    # Run experiments for each configuration
    start_time = default_timer()
    for config_idx, config in enumerate(gen_hparam_dicts(config_ranges)):
        print("Running config:")
        pprint(config)

        results = run_all(config, seeds, num_workers=num_workers)

        end_time = default_timer()
        seconds_elapsed = end_time - start_time
        print("Time elapsed: {}".format(timedelta(seconds=seconds_elapsed)))

        # Specific save directory for this run
        # Format: "res_{timestamp}_{index}"
        # The index, unique for each config, is just in case the consecutive
        # for-loop iterations happen in too quick succession,
        # resulting in clashing timestamps (although unlikely)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
        result_name = "res_" + timestamp + f"_{config_idx}"
        result_save_dir = result_save_dir_root.joinpath(result_name)
        assert not result_save_dir.exists()
        result_save_dir.mkdir()

        print(f"Saving results to {result_save_dir}")
        with open(result_save_dir.joinpath("config.json"), "w") as f:
            json.dump(config, f)

        with open(result_save_dir.joinpath("results.json"), "w") as f:
            json.dump(results, f)
