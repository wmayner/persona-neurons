"""The controls: one violin per condition over the 275 roles, the share of replies with the persona's traits at 4×
strength, for the persona vector, the top 100 neurons, 100 randomly rotated directions, the top 100 principal components and 100
random neurons. The randomly rotated directions and the random neurons are means over five draws.

Reads role_steering/ and role_steering_draws/ and writes figures/fig3_controls/ in the data directory (figure.png,
figure.pdf, data.parquet).

    uv run figures/fig3_controls.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _violins
from persona_neurons import figures
from persona_neurons.thread_tables import five_draw_per_role

d = five_draw_per_role()
fk = figures.kit("post")
fig = _violins.draw(fk, d, columns=5, width=0.8, figsize=(8.5, 5.0),
                    title="The neuron basis is privileged")
# 8.5 in wide so every name fits on two lines; the check's width and text-size warnings are expected
figures.save(fk, fig, "fig3_controls", d)
