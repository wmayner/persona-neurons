"""What every figure script shares: the figure kit in figures/_kit with this repository's colour roles, and one
output folder per figure, figures/<name>/ in the data directory, holding the PNG, the PDF and the plotted data. With
PN_FIG_THEME=dark in the environment, figures are drawn in a dark theme, in figures/<name>_dark/."""
import json
import os
import sys
from pathlib import Path

from persona_neurons.config import data_dir

REPO = Path(__file__).resolve().parents[1]
ROLES_FILE = REPO / "figures" / "figure-roles.json"
DARK = os.environ.get("PN_FIG_THEME") == "dark"
THREAD = os.environ.get("PN_FIG_WORDING") != "webpage"
CONTROL = "#B0B0B0" if DARK else "#5E5E5E"   # every control and its text
# dark theme: surface, ink, secondary ink, muted, rule, then the vector and neuron colours
SURFACE, INK, INK_2, MUTED, RULE, VECTOR, NEURONS = "#161A1F", "#E9ECF0", "#C6CDD6", "#959FAD", "#2C323A", "#56B4E9", "#FF8A3D"


def wording(webpage: str, thread: str) -> str:
    """The text for the current wording: the thread says "persona" throughout; the webpage says "role" for a prompted item of
    the set and "persona" for what the model takes on."""
    return thread if THREAD else webpage


def kit(size: str = "screen"):
    """figkit, set up for `size` with the repository's colour roles."""
    path = str(REPO / "figures" / "_kit")
    if path not in sys.path:
        sys.path.insert(0, path)
    import figkit as fk

    fk.ROLE_COLORS["control"] = CONTROL
    if DARK:
        fk.ROLE_COLORS.update(intervention=NEURONS, second=VECTOR)
        fk.ACCENT, fk.BLUE, fk.INK, fk.MUTED, fk.GREY, fk.LIGHT = NEURONS, VECTOR, INK, MUTED, "#7D8794", RULE
    fk.use(size=size, roles=json.loads(ROLES_FILE.read_text()))
    if DARK:
        import matplotlib.pyplot as plt

        plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK,
                             "axes.labelcolor": INK, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                             "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "grid.color": RULE})
    return fk


def save(fk, fig, name: str, data) -> Path:
    """Save figures/<name>/figure.png, figure.pdf and data.parquet (the plotted data) in the data directory."""
    out = data_dir() / "figures" / (name + ("" if THREAD else "_webpage") + ("_dark" if DARK else ""))
    out.mkdir(parents=True, exist_ok=True)
    fk.save(fig, out / "figure.png")
    fig.savefig(out / "figure.pdf")
    data.to_parquet(out / "data.parquet", index=False)
    return out
