# qfield

Code, data and machine-checked proofs for

> **Amortized quadrature for posterior expectations in inverse problems.**
> Ali Siahkoohi. Preprint, 2026.

## Overview

Posterior expectations average an integrand over posterior samples, at the `O(M^-1/2)` error
of Monte-Carlo estimation. Designed quadratures improve on that rate, but solve an optimization
problem for every new observation. The quadrature field instead maps an observation and its
samples to a signed-weight quadrature in one forward pass, reading the posterior only through
its kernel mean.

The kernel mean also scores a quadrature, so the field returns the best of three candidates and
the samples are one of them. What comes back is never worse than the samples it was handed,
exactly when the kernel mean is exact and with high probability when it is estimated. The
theory is machine-checked in Lean 4, see [Formal verification](#formal-verification).

## References

| reference | how the posterior is supplied |
|---|---|
| Gaussian, `d = 2` | closed-form kernel mean |
| Gaussian mixture, `d = 10` | closed-form kernel mean |
| banana, `d = 2` | a bank of samples |
| limited-angle tomography, `16 x 16` | exact conditional posterior in a rank-7 informed subspace |
| limited-angle tomography, `64 x 64` | trained conditional flow in a rank-32 subspace |
| groundwater flow, 100-d field, 33 sensors | trained HINT flow |
| Lotka-Volterra family | trained conditional flow |

## Installation

```bash
git clone https://github.com/luqigroup/qfield
cd qfield
pip install -e .
```

Python 3.10 or newer with PyTorch. Everything except the conditional flows runs on a CPU.

## The method in ten lines

The field takes samples of the reference, the observation, and the node count. Nothing else.

```python
import torch
from qfield import QuadratureField

field = QuadratureField(d=2)
field.load_state_dict(torch.load("checkpoint.pth"))   # trained by the scripts below

out = field.emit(samples, m=64)                       # nodes and weights, one forward pass
expectation = (out["w"] * integrand(out["z"])).sum()  # what the samples were wanted for
```

`samples` are samples of the reference at this observation. The first `m` seed the nodes and
the rest estimate the kernel mean the weight solve needs, which is the only thing the method
reads of the posterior beyond the samples themselves. Pass `mu_fn=` instead where the reference
gives the kernel mean in closed form.

`emit` runs the forward pass, solves the weights at the samples and at the displaced nodes,
scores the three candidates by the same criterion and returns the best. `out["name"]` says
which one won, `out["z"]` and `out["w"]` are its nodes and weights, and the weights sum to one.

The bandwidth is not passed either. When it is not given it is taken from the seeds by the
median heuristic, the rule every experiment in the paper uses, and reported as `out["sigma"]`.

For a family of posteriors the observation comes too:

```python
from qfield import ConditionalQuadratureField

field = ConditionalQuadratureField(d=2, cond_kind="vector", cond_vec_dim=3)
field.load_state_dict(torch.load("checkpoint.pth"))

out = field.emit(samples, y, m=64)
```

Two toy references build the samples for either case without training anything:
`gaussian_reference` and `conditional_gaussian_reference`.

## Layout

The method is four modules and nothing else:

| module | what it holds |
|---|---|
| `qfield/designed_quadrature/net.py` | the equivariant body: set features, feature-wise modulation, the blocks, and the unconditional field |
| `qfield/designed_quadrature/cond_net.py` | the observation pathway: the encoders and the conditional field |
| `qfield/designed_quadrature/weights.py` | the closed-form unit-sum weight solve, usable without a network |
| `qfield/designed_quadrature/mmd.py` | the discrepancy the field is trained and scored by |

Around them:

| module | what it holds |
|---|---|
| `qfield/api.py` | the public surface, re-exported from `qfield` |
| `qfield/designed_quadrature/safeguard.py` | the three-candidate comparison |
| `qfield/designed_quadrature/certsel.py` | the certification split and its slack |
| `qfield/designed_quadrature/move.py` | per-observation descent, the construction the field amortizes |
| `qfield/designed_quadrature/target.py` | the reference behind one interface: kernel mean, self-affinity, sampler |
| `qfield/designed_quadrature/{herding,sbq,thinning,recombination}.py` | the compression peers |
| `qfield/dataset/` | one problem per module |
| `qfield/models/`, `qfield/samplers/` | the reference flows and the Markov chain they are audited against |
| `qfield/subspace.py` | the likelihood-informed subspace the tomography problems run in |

## Data

Trained networks are not distributed. What is hosted is the raw data each problem trains from,
so a fresh clone retrains rather than restoring someone else's run. Every script calls
`qfield.download.ensure` on the files it opens, which is a no-op once the file is on disk.
Paths are resolved by `projorg`, so nothing is built by hand.

```python
from qfield.download import ensure_tier
ensure_tier("groundwater")       # the 300,000 simulated permeability and pressure pairs
ensure_tier("lotka_volterra")    # the family, its conditioners and the exact chains
```

The files are hosted and the links are in `qfield/download.py`, so there is nothing to fetch by
hand. Limited-angle tomography needs no download: its exact posterior is closed form and the
samples its flow trains on are simulated by the script.

The records the figures read are small and ship with the repository under `data/records`, so
three of the five figures render with no network at all.

## Reproducing the paper's figures

```bash
python scripts/render_depth_ratio.py          # error against the node count, six references
python scripts/render_certification_rate.py   # certified fraction against split size
python scripts/render_lv_benchmark.py         # the Lotka-Volterra benchmark
python scripts/render_banana_field.py         # one forward pass on the banana reference
python scripts/render_field_groundwater.py    # one forward pass on a groundwater observation
```

The first three read only `data/records` and finish in seconds. The last two load a trained
field, so they need one to have been trained first.

## Training

Each problem has a script that owns its run identity through `projorg`: every configuration
field becomes a command-line override, and the resulting experiment name is the directory the
checkpoint and the figures are written to.

```bash
python scripts/designed_quadrature_gaussian_amortized.py --phase train
python scripts/designed_quadrature_banana2_amortized.py --phase train
python scripts/dq_tomography.py --phase train
python scripts/dq_darcy_stuart.py --phase train
```

Any value of `--phase` other than `train` renders from the saved checkpoint.

## Tests

```bash
pytest
```

The suite checks that the package imports without pulling in anything it does not need, that
the weight solve satisfies its own optimality conditions, that the three-candidate comparison
never returns something worse than the samples it was handed, that every download entry is
reachable from a script, and that the Lean development is complete and free of `sorry`.

## Formal verification

The paper's theoretical results are machine-checked in Lean 4 against Mathlib.

```bash
cd formal
lake exe cache get
lake build
lake env lean Axioms.lean
```

The audit prints the axioms each theorem rests on, which are only `propext`,
`Classical.choice` and `Quot.sound`. `formal/README.md` maps each statement of the paper to the
theorem that checks it, and states what is imported rather than proved. The lake manifest is
committed, which pins the dependencies the prebuilt cache is keyed on.

## License

MIT, see [LICENSE](LICENSE).
