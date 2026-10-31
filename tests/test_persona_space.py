"""Persona space (`persona_neurons.persona_space`): PCA on the role vectors with the authors' code."""
import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def upstream(upstream_checkout):
    from persona_neurons.upstream import use_upstream

    use_upstream(upstream_checkout)


def test_loadings_are_cosines_and_default_loads_positively(upstream):
    from persona_neurons.persona_space import pca_roles

    rng = np.random.default_rng(0)
    vectors = {f"role{i}": torch.tensor(rng.normal(size=(3, 16)), dtype=torch.float32) for i in range(20)}
    vectors["default"] = torch.tensor(rng.normal(size=(3, 16)), dtype=torch.float32)
    pcs = pca_roles(vectors, layer=1)
    x = torch.stack([vectors[r][1] for r in pcs.roles]).numpy()
    centred = x - x.mean(0)
    expected = centred @ pcs.components[0] / np.linalg.norm(centred, axis=1)
    assert np.allclose(pcs.loadings[:, 0], expected, atol=1e-5)
    d = vectors["default"][1].numpy() - x.mean(0)
    assert (d @ pcs.components.T > 0).all()
    # the first component is the direction of largest variance of the centred vectors
    top = np.linalg.svd(centred, full_matrices=False)[2][0]
    assert abs(abs(top @ pcs.components[0]) - 1) < 1e-4
