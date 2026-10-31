"""Shared fixtures. No test loads Qwen3-8B's weights: tests that need a model use a tiny, randomly initialised Qwen3
built from Qwen3-8B's own config and saved with its real tokenizer. Its folder is named tiny-qwen3, so that the
authors' Qwen check ("qwen" in name.lower()) sees it as Qwen."""
from pathlib import Path

import pytest

from persona_neurons.config import MODEL_ID, MODEL_REVISION
from persona_neurons.upstream import fetch_upstream, model_snapshot

# The files a tokenizer and config need; the 16 GB of weights are never downloaded here
CONFIG_AND_TOKENIZER = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                        "vocab.json", "merges.txt"]
TINY_SEED = 20261027


@pytest.fixture(scope="session")
def upstream_checkout(tmp_path_factory) -> Path:
    return fetch_upstream(tmp_path_factory.mktemp("upstream") / "assistant-axis")


@pytest.fixture(scope="session")
def qwen3_8b_config_snapshot() -> Path:
    """Qwen3-8B's snapshot at the fixed revision, holding only its config and tokenizer files."""
    return model_snapshot(MODEL_ID, MODEL_REVISION, allow_patterns=CONFIG_AND_TOKENIZER)


@pytest.fixture(scope="session")
def tiny_qwen3_dir(tmp_path_factory, qwen3_8b_config_snapshot) -> Path:
    """Qwen3-8B's config with 2 layers, hidden size 64, intermediate size 128, 2 attention heads and 1 key-value head,
    randomly initialised with seed TINY_SEED, saved with the real Qwen3-8B tokenizer to a folder named tiny-qwen3."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(qwen3_8b_config_snapshot)
    n_layers = 2
    config.num_hidden_layers = n_layers
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 2
    config.num_key_value_heads = 1
    config.head_dim = 32
    config.max_window_layers = n_layers
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = config.layer_types[:n_layers]
    config.torch_dtype = torch.float32
    # fork_rng keeps the seed from touching the global generator other tests use
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(TINY_SEED)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    out = tmp_path_factory.mktemp("models") / "tiny-qwen3"
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(qwen3_8b_config_snapshot).save_pretrained(out)
    (out / "seed.txt").write_text(f"{TINY_SEED}\n")
    return out
