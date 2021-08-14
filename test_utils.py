from utils import parse_time


def test_parse_time():
    assert parse_time("0:0:30") == 30
