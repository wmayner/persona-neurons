"""Fetching one reply from the public neuron library, without the network."""
import gzip
import json
import urllib.error

from persona_neurons import neuron_library


class Response:
    def __init__(self, raw):
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.raw


def serve(monkeypatch, raw):
    monkeypatch.setattr(neuron_library.urllib.request, "urlopen", lambda url, timeout: Response(raw))


def test_rows_alternate_up_and_down(monkeypatch):
    rows = [[0, "up 0"], [0, "down 0"], [0, "up 1"], [0, "I am a wine. It is good."]]
    serve(monkeypatch, gzip.compress(json.dumps({"rows": rows}).encode()))
    assert neuron_library.reply(9, 0, 1) == "up 0"
    assert neuron_library.reply(9, 1, 1) == "up 1"
    assert neuron_library.first_sentence(9, 1, -1) == "I am a wine."


def test_plain_json_is_read_too(monkeypatch):
    serve(monkeypatch, json.dumps({"rows": [[0, "up"], [0, "no full stop here"]]}).encode())
    assert neuron_library.first_sentence(9, 0, -1) == "no full stop here"


def test_no_network_gives_none(monkeypatch):
    def fail(url, timeout):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(neuron_library.urllib.request, "urlopen", fail)
    assert neuron_library.first_sentence(9, 0, -1) is None


def test_a_neuron_past_the_end_gives_none(monkeypatch):
    serve(monkeypatch, gzip.compress(json.dumps({"rows": []}).encode()))
    assert neuron_library.reply(9, 5, 1) is None
