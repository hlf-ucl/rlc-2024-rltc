from game_ext.hyperparam_utils import gen_hparam_dicts


def test_gen_hparam_dicts_one():
    data = {"a": [1], "b": [2]}
    res = list(gen_hparam_dicts(data))
    assert len(res) == 1
    output = res[0]
    assert output.keys() == data.keys()
    assert output["a"] == 1
    assert output["b"] == 2


def test_gen_hparam_dicts_multiple():
    data = {"a": [1, 2], "b": [3, 4]}
    res = list(gen_hparam_dicts(data))
    assert len(res) == 4
