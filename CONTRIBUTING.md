# Contributing

Corrections are as welcome as additions — arguably more so. If a scale figure, licence or
benchmark number in this list is wrong, that is a bug.

## The one rule

**Edit the YAML, not the README.** Everything between the `<!-- BEGIN:x -->` and
`<!-- END:x -->` markers in `README.md`, and the whole of `docs/index.html`, is generated
from [`data/`](data/). A PR that edits the README tables directly will be overwritten by the
next build.

```bash
pip install -r requirements.txt
$EDITOR data/models.yaml        # or datasets.yaml / reading.yaml
python scripts/build.py         # regenerates README.md and docs/index.html
```

CI runs `python scripts/build.py --check` and fails if the generated files are stale, plus a
weekly link check.

## What belongs here

A model, dataset or benchmark qualifies if it **jointly learns from, aligns, or translates
between two or more chemically meaningful modalities**: SMILES/SELFIES/IUPAC, 2D graphs,
3D conformers or crystals, spectra (IR, NMR, MS), images, natural language, reaction roles
and conditions, or knowledge graphs.

Also in scope, deliberately:

- **Single-modality specialists that multimodal models are measured against** — MolScribe,
  DECIMER, MSNovelist. Without them the comparisons are meaningless.
- **Tool-augmented agents** — ChemCrow, Coscientist. They are how multimodal chemistry
  currently reaches the bench.
- **Critical evaluations** — papers arguing that a benchmark is broken belong in
  `reading.yaml` under `critiques`.

Out of scope: general-purpose LLMs with no chemistry-specific training or evaluation; pure
2D-only property predictors with no second modality; blog posts and press releases.

## Entry format

### `data/models.yaml`

```yaml
- name: ModelName                    # short, as the authors write it
  title: "Exact paper title"
  year: 2025                         # first PUBLIC release (preprint if earlier than the venue)
  venue: "NeurIPS 2025"
  paper: https://arxiv.org/abs/...   # arXiv abs preferred, else the DOI URL
  code: https://github.com/...       # or null
  license: Apache-2.0                # as the repo states, or "unspecified"
  category: mllm                     # see below
  modalities: [graph, 3d, text]      # see below
  tasks: [captioning, property prediction]
  note: One sentence on why it is worth reading. Be specific; avoid "novel framework".
```

`category` — one of `molecule-text`, `multi-view`, `mllm`, `image`, `reaction`, `spectra`,
`materials`, `agent`.

`modalities` — drawn from `text`, `smiles`, `graph`, `3d`, `image`, `ir`, `nmr`, `ms`,
`reaction`, `kg`, `crystal`. Add a new tag only if none of these fits, and update
`MODALITY_LABEL` in `scripts/build.py`.

### `data/datasets.yaml`

Same shape, plus `kind` (`corpus`, `pretraining`, `instruction`, `benchmark`) and `scale`.

**`scale` must be quoted from the source**, not estimated. Write
`"231k spectra over 29k unique structures"`, not `"large MS/MS dataset"`.

### `data/reading.yaml`

Three sections: `surveys`, `critiques`, `governance`.

## Standards

1. **Verify before you add.** Open the paper. Open the repo. If the repo 404s, set
   `code: null`, do not link a fork.
2. **`unspecified` is a valid answer** and is more useful than a plausible guess. Missing
   licences and undisclosed parameter counts are signal.
3. **Notes are one sentence** and should say what is *distinctive*, ideally with a number.
   "Reduces motif hallucination by 40%" beats "improves molecular understanding".
4. **Year means first public release.** A 2024 arXiv preprint published at ICML 2025 has
   `year: 2024` and `venue: "ICML 2025"`. This keeps the timeline honest.
5. **No self-promotion caveats needed** — adding your own paper is fine, as long as it meets
   the bar and the note is not marketing copy.

## Opening a PR

1. Fork, branch, edit the YAML.
2. Run `python scripts/build.py` and commit the regenerated `README.md` and
   `docs/index.html` alongside your YAML change.
3. In the PR description, link the paper and say in one line why it belongs.

Thanks for helping keep this accurate.
