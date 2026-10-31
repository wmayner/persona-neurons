"""Splitting a step's work across machines: `--shard i/n` takes the i-th of n near-equal parts (i from 0)."""


def parse_shard(text: str) -> tuple[int, int]:
    """"i/n" to (i, n), with 0 <= i < n."""
    i, _, n = text.partition("/")
    i, n = int(i), int(n)
    if not 0 <= i < n:
        raise ValueError(f"shard {text!r}: need 0 <= i < n")
    return i, n


def shard(items, part: str | None) -> list:
    """The items of shard `part` ("i/n"), taking every n-th of the sorted items from the i-th; all items without one."""
    items = sorted(items)
    if part is None:
        return items
    i, n = parse_shard(part)
    return items[i::n]
