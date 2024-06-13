import itertools


def gen_hparam_dicts(hparam_list_dict):
    """Given a dictionary of hyperparameter ranges,
    yield a separate dictionary per combination.
    """

    keys = hparam_list_dict.keys()
    vals = hparam_list_dict.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
