# Persona neurons in Qwen3-8B

**Explore the results: [https://willmayner.com/persona-neurons](https://willmayner.com/persona-neurons)**

Code and data for every figure and number in the thread on persona steering with MLP neurons.

- Data: https://huggingface.co/datasets/wmayner/persona-neurons

A persona vector is the difference between a role's mean activations and the default Assistant's at layer 18. It splits exactly into contributions from the 233,472 MLP neurons of layers 0 to 18 (442,368 in all), the embedding and the attention layers. We steer Qwen3-8B at 4× strength with each role's top neurons by contribution, ask it 5 questions about itself, and have Claude Haiku 5.5 score the replies with the authors' judge prompt. All numbers below are means over 275 roles.

- The top 50 neurons give 61% replies with the persona's traits and the top 100 give 64%, against 69% for the persona vector.
- 100 randomly rotated directions give 11% and the top 100 principal components give 8%, though those components contribute about twice as much to the vector as the top 100 neurons.
- Personas farther from the Assistant share more of their top neurons (Spearman 0.91). Split by role-specificity (a neuron's contribution for the role as a share of its summed contribution over all 275), the 50 most persona-specific of a persona's top 100 give 65%, and the 50 least persona-specific give 33%. The output of the 50 least lines up more with the Assistant Axis (cosine 0.181, against 0.083).
- Under the role's own system prompt, where 87% of replies have the traits, removing the top 100 neurons' contribution lowers that by 0.2 points, and removing the whole persona vector lowers it by 9.8. The prompt does use them, about as much as other neurons with a similar summed contribution: the log-probability of the role's reply minus the Assistant's falls 11% as far as under the whole vector's removal, against 9% to 10% for lower-ranked neurons and 0% for 100 random neurons.
- Steered far past its natural range, a single neuron gives a persona about as often as a random mixture of a layer's neurons (15.4% and 15.6%) and less often than a principal component (20.2%).

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/). `uv sync` installs the rest.
- `PN_DATA_DIR`: the data directory every step reads and writes (default `./data`).
- To regenerate data: an NVIDIA GPU with at least 40 GB of memory, about 100 GB of disk, and `ANTHROPIC_API_KEY` for the judge.

These are fetched at run time and are not stored here or in the data bundle:
- the authors' code, [safety-research/assistant-axis](https://github.com/safety-research/assistant-axis) at commit `a98961956072224eaf244eb289d6c01700b63795`, which has no licence;
- `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`;
- Lu et al.'s published Qwen3-32B vectors, [`lu-christina/assistant-axis-vectors`](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors) at revision `3b3b788432ad33e3a28d9ff08e88a530c0740814`, for the PC1 check;
- one quoted reply in `figures/fig6b_sommelier_neurons.py`, from the [neuron persona library](https://neuronpersonas.com). Without network access the figure is drawn without the quote.

## Figures and numbers from the data bundle

The bundle is about 1 GB, and every figure draws from it in seconds. The download command takes the dataset's address.

```
uv sync
export PN_DATA_DIR=data
uv run download_data.py https://huggingface.co/datasets/wmayner/persona-neurons/resolve/main
uv run steps/5_split.py finish
uv run steps/7_single_neurons.py tables
uv run figures/print_numbers.py         # every number in the thread, labelled by tweet
```

Each figure script writes `figures/<name>/` in the data directory, with the PNG, the PDF and the plotted data. `PN_FIG_THEME=dark` draws a dark version, and the opener takes `--dark` and `--no-title`. `PN_FIG_WORDING=webpage` draws the webpage's wording, which says "role" for a prompted item of the set, in `figures/<name>_webpage/`. The opener's PNG needs a headless Chromium (`uvx playwright install chromium-headless-shell`).

The last column is the cost of regenerating the figure's data, on top of steps 1 to 4 (33 GPU hours and 330,000 judge requests).

| Figure | Command | Bundle folders | Regeneration |
|---|---|---|---|
| Opener | `uv run figures/fig1_opener.py` | `role_steering*`, `split` | steps 5, 6, 6b: 42 GPU hours, 143,000 judge requests |
| Set size | `uv run figures/fig2_set_size.py` | `role_steering*` | steps 5, 6, 6b: as above |
| Controls | `uv run figures/fig3_controls.py` | `role_steering*` | steps 5, 6, 6b: as above |
| Contribution | `uv run figures/fig4_contribution.py` | `role_steering*` | steps 5, 6, 6b: as above |
| Single neurons | `uv run figures/fig5_single_neurons.py` | `single_neurons` | step 7: 2 GPU hours, 25,000 judge requests |
| Roles sharing neurons | `uv run figures/fig6a_shared_roles.py` | `shared_neurons` | steps 5, 5b: CPU only |
| Sommelier's neurons | `uv run figures/fig6b_sommelier_neurons.py` | `shared_neurons` | steps 5, 5b: CPU only |
| Halves | `uv run figures/fig6c_halves.py` | `role_steering`, `role_steering_sizes` | steps 5, 6, 6b: as for the opener |
| Necessity | `uv run figures/fig7_necessity.py` | `necessity` | steps 5, 6, 8: about 7 GPU hours and 16,500 judge requests beyond step 6 |
| PC1 check | `uv run steps/check_published.py`, then `uv run figures/check_pc1_loadings.py` | `vectors` | steps 1 to 4 only; downloads the published vectors |

The PC1 check compares each role's loading on the first principal component of persona space in Qwen3-8B with its loading in Lu et al.'s published Qwen3-32B vectors, computed with the authors' code.

The numbers also read two tables with no figure: `necessity_same_size` (step 8b: 5 GPU hours, 8,250 judge requests) and `shared_neurons/on_axis*` (step 5b's `axis`, CPU only). The scaled run in `necessity_scaled` (step 8 with `--scaled`: about 6 GPU hours, 11,000 judge requests) backs numbers on the webpage only.

To recompute tables from the bundle without a model, add `--to` with a new folder: `uv run steps/8_necessity.py analyze --to necessity/again`, or the same for `steps/8b_necessity_same_size.py`.

## Regenerating the data

Run the steps in order from this folder, on any machine with a suitable GPU. Lines marked GPU take `--shard i/n` (i from 0) to do the i-th of n parts, for machines that share the data directory or whose outputs are copied into one. A stopped step skips the work it has finished. Each step's `--help` lists its parts and outputs, and the method is in the docstrings of `persona_neurons/`. GPU hours are for an A40 or L40S.

The model runs in bfloat16, so a sampled reply can change with what else is in its batch, and a rerun's numbers can differ slightly from ours.

```
steps/1_generate.sh                                # GPU, 19 hours
uv run steps/2_judge.py submit                     # 330,000 requests; `collect` once the batches have ended
uv run steps/3_activations.py                      # GPU, 14 hours
uv run steps/4_vectors.py

uv run steps/5_split.py split                      # 16 GB
uv run steps/5_split.py pca --layers 0-4           # and 5-9, 10-14, 15-18; 20 GB, up to an hour on 8 cores each
uv run steps/5_split.py finish
uv run steps/5b_shared_neurons.py inputs           # 16 GB
uv run steps/5b_shared_neurons.py tables
uv run steps/5b_shared_neurons.py axis

uv run steps/6_role_steering.py prepare
uv run steps/6_role_steering.py components --layers 0-4    # and 5-9, 10-14, 15-18; 24 GB each
uv run steps/6_role_steering.py rotate --layers 0-4        # and 5-9, 10-14, 15-18
uv run steps/6_role_steering.py finish
uv run steps/6_role_steering.py generate           # GPU, 8 hours
uv run steps/6_role_steering.py judge submit       # 13,750 requests; `judge collect` once they have ended
uv run steps/6_role_steering.py build

uv run steps/6b_more_conditions.py prepare
uv run steps/6b_more_conditions.py components --layers 0-4           # and 5-9, 10-14, 15-18
uv run steps/6b_more_conditions.py rotate --draw 0 --layers 0-4      # and 5-9, 10-14, 15-18, for draws 0 to 4
uv run steps/6b_more_conditions.py finish          # 40 GB of disk
uv run steps/6b_more_conditions.py generate        # GPU, 34 hours
uv run steps/6b_more_conditions.py judge submit    # 129,250 requests; `judge collect` once they have ended
uv run steps/6b_more_conditions.py build

uv run steps/7_single_neurons.py prepare
uv run steps/7_single_neurons.py natural           # GPU
uv run steps/7_single_neurons.py push              # GPU; 2 hours with `natural`
uv run steps/7_single_neurons.py collect
uv run steps/7_single_neurons.py label submit      # 25,000 requests; `label collect` once they have ended
uv run steps/7_single_neurons.py tables

uv run steps/8_necessity.py prepare
uv run steps/8_necessity.py generate --changes     # GPU, 7 hours
uv run steps/8_necessity.py judge submit           # 16,500 requests; `judge collect` once they have ended
uv run steps/8_necessity.py build
uv run steps/8_necessity.py analyze

uv run steps/8b_necessity_same_size.py prepare
uv run steps/8b_necessity_same_size.py score       # GPU; 5 hours with `generate`
uv run steps/8b_necessity_same_size.py generate    # GPU
uv run steps/8b_necessity_same_size.py judge submit    # 8,250 requests; `judge collect` once they have ended
uv run steps/8b_necessity_same_size.py build
uv run steps/8b_necessity_same_size.py analyze

uv run steps/8_necessity.py prepare-scaled         # the scaled run, for the webpage's numbers
uv run steps/8_necessity.py generate --scaled --changes    # GPU, 6 hours
uv run steps/8_necessity.py judge submit --scaled  # 11,000 requests; `judge collect --scaled` once they have ended
uv run steps/8_necessity.py build --scaled
uv run steps/8_necessity.py analyze --scaled
uv run steps/8_necessity.py usual-range
```

The GPU hours for steps 8 and 8b are estimates, scaled from the original runs.

## Tests

`uv run pytest` runs the tests on a tiny, randomly initialised model with Qwen3-8B's configuration; no test loads the real weights.

## Credits

The setup follows Chen et al., *Persona vectors*, and Lu et al. (2026), *The Assistant Axis*, with the authors' code used unchanged. The single-neuron steering protocol is borrowed from Josh's public [Qwen3-8B neuron persona library](https://neuronpersonas.com/#method); the replies and labels here are our own.

## Licence

The code is under the MIT licence (see `LICENSE`). The data bundle is under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The figure kit in `figures/_kit/` is adapted from [better-figures](https://github.com/wmayner/better-figures) (MIT). The Inter font in `figures/_kit/fonts/` is under the SIL Open Font License (`Inter-LICENSE.txt`).
