from bayesian_scooters.strategies import Namer


def test_get_word_list():
    names = Namer().get_word_list()
    assert type(names) == list
    for n in names:
        assert type(n) == str


def test_get_name():
    name = Namer().get_random_name()
    assert type(name) == str
