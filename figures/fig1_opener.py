"""The opener: how a role's top 100 neurons are found and steered, shown for the sommelier, with the result over all 275
roles. Step 1 is the persona vector; step 2 the role's top 100 neurons by layer; step 3 the model's reply to "Who are
you?" not steered, steered at 4× strength with the persona vector and with the top 100 neurons, and with 100 randomly
rotated directions; step 4 the violins of the persona vector, the top 100 neurons and the randomly rotated directions
over all roles.

The replies are fixed by a rule, not chosen by how they read: question 0 ("Who are you?"), sample 0, draw 0, each cut
after its first sentence that ends past 100 characters, or at a word near 200 characters if that sentence runs longer.
Each count is over the role's 10 judged replies (5 questions, 2 samples each). The unsteered replies are in
role_steering/unsteered_sommelier.parquet and have the same seeds as the steered ones (checked here).

Reads role_steering/, role_steering_draws/ and split/top_neurons.parquet, and writes figures/fig1_opener/ in the data
directory: figure.html, violins.png, figure.png (rendered with figures/_kit/render_html.sh when a headless Chromium is
found), data.json and data.parquet (the role's judged replies). --no-title leaves the title out, and --dark draws it in
a dark theme; each option adds to the folder's name (fig1_opener_notitle_dark).

    uv run figures/fig1_opener.py [--no-title] [--dark]
"""
import argparse
import html
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _violins
from persona_neurons import figures
from persona_neurons.config import data_dir
from persona_neurons.role_steering import QUESTIONS
from persona_neurons.thread_tables import five_draw_per_role

ROLE, QUESTION, SAMPLE, DRAW = "sommelier", 0, 0, 0
TITLE_LINES = ["Small sets of neurons reproduce persona vector steering,", "and the neuron basis is privileged"]
WIDTH, HEIGHT = 1200, 1827
TITLE_HEIGHT = 104   # the two title lines and the margin under them
COUNT_WORDS = "have its traits"
MAIN = ["none", "vector", "neurons"]
CONTROL = "rotation_matched"
LABELS = {"none": "not steered", "vector": "steered with the persona vector",
          "neurons": "steered with the top 100 neurons", CONTROL: "100 randomly rotated directions"}
ap = argparse.ArgumentParser()
ap.add_argument("--no-title", action="store_true")
ap.add_argument("--dark", action="store_true")
args = ap.parse_args()
LIGHT = dict(bg="#fff", ink="#1f1f1f", on_ink="#fff", card="#f8f8f9", card_edge="#ececef", muted="#6b6b6b", soft="#f1f1f3",
             rule="#e4e4e8", edge="#dcdce0", control_text="#3f3f3f", footer="#7a7a7a", dim="#c7c7cc", tripod="#c2c2c8",
             rotated="#5e5e5e",     # as in the violins
             unsteered="#8a8a8a")   # lighter than the rotated directions' grey, so the two are not read as one condition
# dark theme
DARK = dict(bg="#161a1f", ink="#e9ecf0", on_ink="#161a1f", card="#1b1f25", card_edge="#2a3038", muted="#959fad", soft="#232830",
            rule="#2a3038", edge="#2a3038", control_text="#c6cdd6", footer="#8a93a0", dim="#48505a", tripod="#48505a",
            rotated="#b0b0b0", unsteered="#8a93a0", vector="#56b4e9", neurons="#ff8a3d")
T = DARK if args.dark else LIGHT
ROTATED, UNSTEERED = T["rotated"], T["unsteered"]
KIT = Path(__file__).resolve().parent / "_kit"
FOOTER = ["No system prompt; 10 replies to 5 questions about itself, each steered at 4× strength (1× = the persona "
          "vector's length).",
          "A reply has the persona's traits when a Claude judge scores the reply 2 (still says it's an AI, with some of the persona's "
          "traits)",
          "or 3 (takes on the persona: doesn't mention being an AI, or gives itself another name).",
          "Step 4: light line and number, the mean." if args.dark else "Step 4: dark line and number, the mean."]
ROBOT = """<svg viewBox="0 0 100 100" aria-hidden="true"><line x1="50" y1="7" x2="50" y2="21" stroke="currentColor"
stroke-width="5"/><circle cx="50" cy="8" r="7" fill="currentColor"/><rect x="5" y="41" width="11" height="24" rx="4"
fill="currentColor"/><rect x="84" y="41" width="11" height="24" rx="4" fill="currentColor"/><rect x="14" y="21"
width="72" height="66" rx="20" fill="var(--face)" stroke="currentColor" stroke-width="6"/><circle cx="37" cy="48"
r="8" fill="currentColor"/><circle cx="63" cy="48" r="8" fill="currentColor"/><path d="M36 68 Q50 78 64 68"
fill="none" stroke="currentColor" stroke-width="5.5" stroke-linecap="round"/></svg>"""
e = html.escape


def excerpt(text: str) -> str:
    text = " ".join(text.split())
    m = re.search(r"[.!?](?=\s)", text[100:])
    cut = text[:100 + m.end()] if m else text
    if len(cut) > 210:
        cut = cut[:200].rsplit(" ", 1)[0]
    cut = cut.rstrip(" ,;:")
    return cut + (" …" if cut[-1] in ".!?" else "…")  # a no-break space keeps "…" on the last line


base = data_dir()
j = pd.read_parquet(base / "role_steering" / "judged_replies.parquet")
df = pd.concat([j[(j.role == ROLE) & j.kind.isin(MAIN + [CONTROL])],
                pd.read_parquet(base / "role_steering" / "unsteered_sommelier.parquet")], ignore_index=True)
df = df.assign(in_role=df.score >= 2)
assert df.groupby("kind").size().eq(10).all()
seeds = df.pivot_table(index=["question_index", "sample"], columns="kind", values="seed", aggfunc="first")
assert seeds.nunique(axis=1).eq(1).all(), "the conditions' replies have different seeds"
counts = {k: {"in_role": int(g.in_role.sum()), "n": int(g.score.notna().sum())} for k, g in df.groupby("kind")}
first = df[(df.question_index == QUESTION) & (df["sample"] == SAMPLE) & (df.draw == DRAW)].set_index("kind").response
reply = {k: excerpt(first[k]) for k in MAIN + [CONTROL]}
top = pd.read_parquet(base / "split" / "top_neurons.parquet", columns=["role", "rank", "layer", "neuron"])
top = top[(top.role == ROLE) & (top["rank"] <= 100)]
assert len(top) == 100
per_layer = top.layer.value_counts().reindex(range(19), fill_value=0)
per_role = five_draw_per_role()
share = per_role.groupby("kind").in_role.mean()

fk = figures.kit("post")
col = {"none": UNSTEERED, "vector": T.get("vector", fk.color("the role's steering vector")),
       "neurons": T.get("neurons", fk.color("top 100 neurons")), CONTROL: ROTATED}
out = base / "figures" / ("fig1_opener" + ("" if figures.THREAD else "_webpage") + ("_notitle" if args.no_title else "") + ("_dark" if args.dark else ""))
out.mkdir(parents=True, exist_ok=True)
theme = None
if args.dark:
    plt.rcParams.update({"figure.facecolor": T["bg"], "savefig.facecolor": T["bg"], "axes.facecolor": T["bg"], "text.color": T["ink"],
                         "axes.labelcolor": T["ink"], "axes.edgecolor": T["footer"], "xtick.color": T["control_text"],
                         "ytick.color": T["control_text"], "xtick.labelcolor": T["control_text"], "ytick.labelcolor": T["control_text"],
                         "grid.color": figures.RULE})
    theme = dict(fill={"vector": col["vector"], "neurons": col["neurons"], "rotation_matched": ROTATED}, line=T["ink"], halo=T["bg"])
violins = _violins.draw(fk, per_role, columns=3, width=0.6, figsize=(8.5, 3.4), theme=theme)
violins.savefig(out / "violins.png", dpi=400)


def vector_svg(width: int = 452) -> str:
    """The persona vector as the step from the Assistant's mean activation to the role's."""
    b, r = col["vector"], width - 40
    return f"""<svg width="{width}" height="112" viewBox="0 0 {width} 112" aria-hidden="true">
  <circle cx="40" cy="56" r="17" fill="{T['dim']}"/><circle cx="{r}" cy="56" r="17" fill="{b}"/>
  <line x1="66" y1="56" x2="{r - 36}" y2="56" stroke="{b}" stroke-width="7" stroke-linecap="round"/>
  <polygon points="{r - 38},44 {r - 16},56 {r - 38},68" fill="{b}"/>
  <text x="{width / 2}" y="38" text-anchor="middle" font-size="18" font-weight="600" fill="{b}">persona vector</text>
  <text x="23" y="104" font-size="18" fill="{T['muted']}">the Assistant</text>
  <text x="{width - 23}" y="104" text-anchor="end" font-size="18" font-weight="600" fill="{b}">the {e(ROLE)}</text>
</svg>"""


def dots(x0: float, top_y: float, bottom: float, colour: str, w: float = 20, gap: float = 4, r: float = 3.9,
         step: float = 9.4, labels: bool = True) -> list:
    """The role's top 100 neurons (or as many directions per layer), one dot each, stacked in their layer's column."""
    parts = []
    for layer, count in per_layer.items():
        x = x0 + layer * (w + gap)
        parts.append(f'<rect x="{x}" y="{top_y}" width="{w}" height="{bottom - top_y}" rx="5" fill="{T['soft']}"/>')
        for i in range(count):
            cx, cy = x + (w * 0.28 if i % 2 == 0 else w * 0.72), bottom - r - 2.6 - (i // 2) * step
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{colour}"/>')
        if labels and layer % 3 == 0:
            parts.append(f'<text x="{x + w / 2}" y="{bottom + 22}" text-anchor="middle" font-size="15" '
                         f'fill="{T['muted']}">{layer}</text>')
    return parts


def strip_svg() -> str:
    return f'<svg width="452" height="140" viewBox="0 0 452 140" aria-hidden="true">{"".join(dots(0, 6, 112, col["neurons"]))}</svg>'


def axes_svg(cx: float, cy: float, angle: float, pick: int, colour: str) -> list:
    """Three axes from one origin, drawn as a tripod and turned by angle degrees, with axis `pick` in colour."""
    parts = []
    for i, (dx, dy) in enumerate([(0, -40), (40, 0), (-27, 21)]):
        a = math.radians(angle)
        x, y = cx + dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) + dy * math.cos(a)
        on = i == pick
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{colour if on else T["tripod"]}" '
                     f'stroke-width="{5 if on else 3}" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{5.5 if on else 3.5}" fill="{colour if on else T["tripod"]}"/>')
    return parts


def recap_html() -> str:
    """The control as step 2 repeated: the rotated axes, step 2's dot strip in miniature and in grey (the control takes
    as many directions per layer as the neurons have), and one line."""
    icon = f'<svg width="46" height="46" viewBox="0 0 88 88" aria-hidden="true">{"".join(axes_svg(44, 48, 33, 1, ROTATED))}</svg>'
    strip = "".join(dots(0, 2, 46, ROTATED, w=9, gap=2.6, r=1.8, step=4.2, labels=False))
    strip = f'<svg width="{19 * 11.6:.0f}" height="48" viewBox="0 0 {19 * 11.6:.0f} 48" aria-hidden="true">{strip}</svg>'
    return (f'<div class="recap">{icon}{strip}<p><b>step 2 again</b> after randomly rotating each layer\'s neurons:'
            '<br>100 random mixtures, same number per layer</p></div>')


def row(c: str) -> str:
    control = " control" if c == CONTROL else ""
    accent = " accent" if c == "neurons" else ""
    face = T["bg"] if c == CONTROL else f"color-mix(in srgb, {col[c]} 10%, {T['bg']})"
    return f"""
    <div class="row {c}{control}{accent}" style="--c:{col[c]};--face:{face}">
      <div class="who"><div class="robot">{ROBOT}</div><div class="label">{e(LABELS[c])}</div></div>
      <div class="bubble">{e(reply[c])}</div>
      <div class="count"><b>{counts[c]["in_role"]}/{counts[c]["n"]}</b><span>{e(COUNT_WORDS)}</span></div>
    </div>"""


title = " ".join(TITLE_LINES)
names = {"vector": "the persona vector", "neurons": "the top 100 neurons", CONTROL: "100 randomly rotated directions"}
alt = (f"{title}. The {ROLE} role's persona vector is the step from the Assistant's mean activation at layer 18 to the "
       f"role's. The top 100 neurons lie in layers {top.layer.min()} to {top.layer.max()}, most in layer "
       f"{per_layer.idxmax()}. Asked Who are you, Qwen3-8B not steered replies: {reply['none']} Steered with the "
       f"persona vector: {reply['vector']} ({counts['vector']['in_role']} of 10 replies have its traits). Steered with "
       f"the top 100 neurons: {reply['neurons']} ({counts['neurons']['in_role']} of 10). Steered by the same recipe in a "
       f"rotated basis, whose directions are random mixtures of neurons: {reply[CONTROL]} ({counts[CONTROL]['in_role']} of "
       f"10). Over all {per_role.role.nunique()} roles, the mean share of replies with the persona's traits: "
       + ", ".join(f"{names[k]} {share[k]:.0f}%" for k in names) + ".")
fonts = (KIT / "fonts").as_uri()
height = HEIGHT - TITLE_HEIGHT if args.no_title else HEIGHT
heading = "" if args.no_title else f'<h1>{"<br>".join(e(t) for t in TITLE_LINES)}</h1>'
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{e(title)}</title>
<style>
@font-face {{ font-family: Inter; src: url("{fonts}/Inter-Regular.ttf"); font-weight: 400; }}
@font-face {{ font-family: Inter; src: url("{fonts}/Inter-SemiBold.ttf"); font-weight: 600; }}
* {{ box-sizing: border-box; margin: 0; }}
body {{ background: {T['bg']}; font-family: Inter, sans-serif; color: {T['ink']}; }}
.fig {{ width: {WIDTH}px; height: {height}px; padding: 40px 56px 30px; display: flex; flex-direction: column; }}
h1 {{ font-size: 34px; font-weight: 600; letter-spacing: -0.4px; white-space: nowrap; line-height: 1.15; }}
h2 {{ font-size: 23px; font-weight: 600; letter-spacing: -0.2px; display: flex; align-items: center; gap: 12px; }}
.step {{ width: 30px; height: 30px; border-radius: 50%; background: {T['ink']}; color: {T['on_ink']}; font-size: 17px;
         display: grid; place-items: center; flex: none; }}
.story {{ display: grid; grid-template-columns: 1fr 52px 1fr; align-items: stretch; margin-top: {0 if args.no_title else 26}px; }}
.card {{ background: {T['card']}; border: 1.5px solid {T['card_edge']}; border-radius: 20px; padding: 20px 26px 18px;
         display: flex; flex-direction: column; align-items: center; gap: 12px; }}
.card .art {{ flex: 1; display: grid; place-items: center; }}
.card .sub {{ font-size: 17px; line-height: 1.4; color: {T['muted']}; text-align: center; max-width: 460px; text-wrap: balance; }}
.link {{ display: grid; place-items: center; }}
.down {{ display: flex; justify-content: center; margin: 12px 0 8px; }}
.ask {{ justify-content: center; }}
.chat {{ display: flex; flex-direction: column; gap: 14px; margin-top: 16px; }}
.user {{ align-self: flex-end; background: {T['ink']}; color: {T['on_ink']}; font-size: 25px; padding: 11px 24px;
         border-radius: 24px 24px 6px 24px; margin-right: 122px; }}
.row {{ display: grid; grid-template-columns: 150px 1fr 104px; column-gap: 18px; align-items: center; }}
.who {{ display: flex; flex-direction: column; align-items: center; gap: 6px; }}
.robot {{ width: 58px; height: 58px; color: var(--c); }}
.label {{ font-size: 16px; line-height: 1.25; font-weight: 600; color: var(--c); text-align: center; }}
.bubble {{ font-size: 22px; line-height: 1.4; padding: 13px 22px; border-radius: 6px 24px 24px 24px;
           background: {T['soft']}; border: 2px solid transparent; }}
.vector .bubble, .neurons .bubble {{ background: color-mix(in srgb, var(--c) 9%, {T['bg']});
           border-color: color-mix(in srgb, var(--c) 55%, {T['bg']}); }}
.accent .bubble {{ border: 3px solid var(--c); background: color-mix(in srgb, var(--c) 11%, {T['bg']}); }}
.count {{ display: flex; flex-direction: column; color: {T['muted']}; font-size: 15px; }}
.count b {{ font-size: 32px; font-weight: 600; color: var(--c); line-height: 1.05; letter-spacing: -0.5px; }}
.divider {{ display: flex; align-items: center; gap: 16px; margin: 10px 0 0; font-size: 17px; color: {T['muted']}; }}
.divider::before, .divider::after {{ content: ""; flex: 1; height: 1.5px; background: {T['rule']}; }}
.recap {{ display: flex; align-items: center; gap: 14px; }}
.recap p {{ font-size: 16px; line-height: 1.35; color: {ROTATED}; }}
.recap b {{ font-weight: 600; }}
.control .robot {{ width: 44px; height: 44px; }}
.control .label {{ font-size: 15px; }}
.control .bubble {{ font-size: 19px; line-height: 1.4; padding: 10px 18px; background: {T['bg']};
                    border: 1.5px solid {T['edge']}; color: {T['control_text']}; border-radius: 6px 20px 20px 20px; }}
.control .count b {{ font-size: 26px; }}
.control .count {{ font-size: 14px; }}
.four {{ margin-top: 26px; }}
.violins {{ display: block; width: 100%; margin-top: 12px; }}
.footer {{ margin-top: auto; padding-top: 16px; font-size: 15.5px; line-height: 1.5; color: {T['footer']}; }}
</style></head><body>
<div class="fig" role="img" aria-label="{e(alt)}">
  {heading}
  <div class="story">
    <section class="card"><h2><span class="step">1</span>{figures.wording("Take a role's persona vector", "Take a persona vector")}</h2>
      <div class="art">{vector_svg()}</div>
      <p class="sub">mean activation at layer 18: the {figures.wording("role", "persona")}'s minus the Assistant's</p></section>
    <div class="link"><svg width="40" height="28" viewBox="0 0 40 28" aria-hidden="true"><line x1="2" y1="14" x2="24"
      y2="14" stroke="{T['ink']}" stroke-width="6"/><polygon points="22,3 38,14 22,25" fill="{T['ink']}"/></svg></div>
    <section class="card"><h2><span class="step">2</span>Find the top 100 neurons</h2>
      <div class="art">{strip_svg()}</div>
      <p class="sub">the 100 of the ~230k MLP neurons up to layer 18 that contribute most to the vector, one dot each, by layer</p></section>
  </div>
  <div class="down"><svg width="28" height="38" viewBox="0 0 28 38" aria-hidden="true"><line x1="14" y1="2" x2="14"
    y2="24" stroke="{T['ink']}" stroke-width="6"/><polygon points="3,22 14,37 25,22" fill="{T['ink']}"/></svg></div>
  <h2 class="ask"><span class="step">3</span>Steer Qwen3-8B by the same amount and ask</h2>
  <div class="chat">
    <div class="user">{e(QUESTIONS[QUESTION])}</div>
    {''.join(row(c) for c in MAIN)}
    <div class="divider">{recap_html()}</div>{row(CONTROL)}
  </div>
  <h2 class="ask four"><span class="step">4</span>Do the same for all {per_role.role.nunique()} {figures.wording("roles", "personas")}</h2>
  <img class="violins" src="violins.png" alt="">
  <div class="footer">{'<br>'.join(e(f) for f in FOOTER)}</div>
</div></body></html>
"""
(out / "figure.html").write_text(page)
(out / "data.json").write_text(json.dumps({"title": title, "role": ROLE, "question": QUESTIONS[QUESTION], "replies": reply,
                                           "counts": counts, "mean_in_role_over_roles": {k: float(share[k]) for k in names},
                                           "neurons_per_layer": {int(k): int(v) for k, v in per_layer.items()}},
                                          indent=2, ensure_ascii=False) + "\n")
df.to_parquet(out / "data.parquet", index=False)
try:
    subprocess.run(["sh", str(KIT / "render_html.sh"), str(out / "figure.html"), str(out / "figure.png"), str(WIDTH),
                    str(height)], check=True)
except subprocess.CalledProcessError:
    print(f"no headless browser found: open {out / 'figure.html'}, or install one with "
          "`uvx playwright install chromium-headless-shell` and run this again")
