"""One reply from the public Qwen3-8B neuron persona library (neuronpersonas.com), fetched at run time. The
library has no licence, so none of its data is stored in this repository or in the data bundle.

Each layer file holds every reply for that layer, in rows ordered neuron 0 steered up, neuron 0 steered down, neuron 1
up, and so on, with the reply as the last field of its row.
"""
import gzip
import json
import re
import urllib.error
import urllib.request

SITE = "https://neuronpersonas.com/data/qwen/layers"


def reply(layer: int, neuron: int, sign: int) -> str | None:
    """The library's reply when the neuron is steered alone (sign +1 up, -1 down), or None if it cannot be fetched."""
    try:
        with urllib.request.urlopen(f"{SITE}/L{layer:02d}.gz", timeout=120) as r:
            raw = r.read()
        rows = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)["rows"]
        return rows[2 * neuron + (0 if sign > 0 else 1)][-1]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError) as e:
        print(f"could not fetch the library's reply: {e}")
        return None


def first_sentence(layer: int, neuron: int, sign: int) -> str | None:
    """The reply's first sentence, or None if it cannot be fetched."""
    text = reply(layer, neuron, sign)
    return re.match(r"\s*(.*?[.!?])(?=\s|$)", text, re.S).group(1) if text and re.search(r"[.!?]", text) else text
