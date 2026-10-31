import pytest

from persona_neurons.shards import parse_shard, shard


def test_shards_cover_every_item_once():
    items = [f"role{i}" for i in range(11)]
    parts = [shard(reversed(items), f"{i}/4") for i in range(4)]
    assert sorted(x for p in parts for x in p) == sorted(items)
    assert max(map(len, parts)) - min(map(len, parts)) <= 1
    assert shard(items, None) == sorted(items)


@pytest.mark.parametrize("bad", ["4/4", "-1/3", "1"])
def test_bad_shard(bad):
    with pytest.raises(ValueError):
        parse_shard(bad)
