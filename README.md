<div align="center">

<img src="assets/banner.svg" alt="Awesome Multi-Modal Chemistry" width="100%">

<br>

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
[![License: CC0-1.0](https://img.shields.io/badge/licence-CC0--1.0-5eead4?style=flat-square)](LICENSE)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-a78bfa?style=flat-square)](CONTRIBUTING.md)
[![Site](https://img.shields.io/badge/browse-searchable%20site-7dd3fc?style=flat-square)](https://axelrolov.github.io/awesome-multimodal-chemistry/)
[![Link check](https://img.shields.io/github/actions/workflow/status/AxelRolov/awesome-multimodal-chemistry/links.yml?style=flat-square&label=links)](../../actions/workflows/links.yml)
[![Stars](https://img.shields.io/github/stars/AxelRolov/awesome-multimodal-chemistry?style=flat-square&color=f0abfc)](../../stargazers)

**Models, datasets and benchmarks that learn across chemically meaningful modalities —
graphs, 3D structures, spectra, images, reactions and text.**

<!-- BEGIN:stats -->
**57** models and systems · **31** datasets and benchmarks · **23** surveys, critiques and governance references · **53/57** models with public code

<!-- END:stats -->

[**Browse the searchable site →**](https://axelrolov.github.io/awesome-multimodal-chemistry/) · [**Read the full review →**](REVIEW.md) · [**Contribute →**](CONTRIBUTING.md)

</div>

---

## Why this list

Most chemistry ML lists are organised by *task*. This one is organised by *what the model
can see*. That turns out to be the more useful axis: a model that reads only SMILES and a
model that reads SMILES, a conformer and an IR spectrum are not competing on the same
problem, even when they report the same benchmark number.

Every entry here was checked against its primary source. Scale figures are quoted from the
paper, not estimated. Where a paper reports no code, no licence or no parameter count, this
list says `unspecified` rather than guessing, because in this field the gaps are often the
most informative part.

The tables below are generated from [`data/*.yaml`](data/), so the machine-readable version
is the source of truth, not a byproduct.

## Contents

- [Scope and definitions](#scope-and-definitions)
- [The field at a glance](#the-field-at-a-glance)
- [Models and systems](#models-and-systems)
- [Datasets and benchmarks](#datasets-and-benchmarks)
- [Timeline](#timeline)
- [What actually works](#what-actually-works)
- [Open problems](#open-problems)
- [Reading list](#reading-list)
- [Contributing](#contributing)
- [Licence and citation](#licence-and-citation)

## Scope and definitions

A **multi-modal model in chemistry** is one that jointly learns from, aligns, or translates
between two or more chemically meaningful modalities:

| Modality | What it encodes | Typical encoder |
|---|---|---|
| `SMILES` `SELFIES` `IUPAC` | composition and connectivity, as a sequence | seq2seq or causal transformer |
| `graph` | explicit atom–bond topology | GNN, graph transformer |
| `3D` `crystal` | sterics, geometry, symmetry | E(3)/SE(3)-equivariant network |
| `IR` `NMR` `MS` | instrument signal, tightly coupled to structure | 1D CNN, spectrum transformer |
| `image` | how chemists actually communicate structures | ViT, CNN |
| `text` | descriptive, procedural and conceptual knowledge | scientific or chemical LM |
| `reaction` | roles, conditions, yields — symbolic *and* continuous | reaction-aware LM |
| `KG` | curated relational knowledge | graph attention over triples |

Two families are worth keeping apart:

- **Multi-view molecular modelling** combines alternative encodings of the *same* molecule
  (2D graph + 3D geometry). It buys physical fidelity and transfer on property tasks.
  → *GraphMVP, Uni-Mol, MolLM, 3D-MolT5, Mol-LLaMA*
- **Cross-domain multimodality** bridges structured chemical input with human- or
  instrument-generated signals (graph↔text, image↔structure, spectra↔structure). It buys
  new *tasks*: retrieval, captioning, document intelligence and structure elucidation.
  → *MoleculeSTM, ChemVLM, RxnIM, DiffSpectra, ChemDFM-X*

The boundary between representation and generation is porous, so both are included here. A
contrastive encoder usually becomes the conditioning front-end of a generator; generative
pretraining usually yields reusable representations.

## The field at a glance

```mermaid
flowchart LR
    A["SMILES / SELFIES / IUPAC"] --> A1["Transformer seq2seq or causal LM"]
    B["2D molecular graphs"] --> B1["GNNs, graph transformers"]
    C["3D conformers, crystals"] --> C1["E(3)/SE(3)-equivariant networks"]
    D["Spectra: IR, NMR, MS"] --> D1["1D CNNs, spectrum transformers"]
    E["Images: depictions, schemes"] --> E1["ViT / CNN encoders"]
    F["Text: captions, procedures, QA"] --> F1["Scientific / chemical LMs"]

    A1 --> G(("Fusion"))
    B1 --> G
    C1 --> G
    D1 --> G
    E1 --> G
    F1 --> G

    G --> H["Contrastive alignment"]
    G --> I["Cross-attention / Q-Former projector"]
    G --> J["Unified tokenisation"]
    G --> K["Instruction tuning"]
    G --> L["Conditional diffusion"]

    H --> M["Retrieval, zero-shot transfer"]
    I --> N["Assistants, QA, captioning"]
    J --> N
    K --> N
    L --> O["Molecules, crystals, spectra, procedures"]
```

Five architectural families cover almost everything in this list:

| Family | Objective | Best at | Examples |
|---|---|---|---|
| **Dual-encoder contrastive** | InfoNCE-style alignment | retrieval, zero-shot transfer | MoMu, MoleculeSTM, MolFM, CReSS |
| **Encoder–decoder translator** | reconstruction / denoising | captioning, sequence generation | MolT5, BioT5, MSNovelist, Spec2Mol |
| **Projector-based MLLM** | instruction tuning on a frozen LM | broad assistant tasks | InstructMol, MolCA, ChemVLM, Mol-LLaMA |
| **Unified token model** | next-token prediction over mixed tokens | one stream, no modality boundary | UniMoT, 3D-MolT5, HIGHT |
| **Conditional diffusion** | denoising in latent, graph or 3D space | structured outputs with global coherence | 3M-Diffusion, LDMol, DiffMS, DiffSpectra, MatterGen |

## Models and systems

> Sorted by first public release within each group. `unspecified` means the source does not
> state it.

<!-- BEGIN:models -->
### Molecule ↔ Text

Contrastive alignment, translation and text-guided generation between molecular structure and natural language.

| Model | Year | Modalities | Paper | Code | Licence | Why it matters |
|---|:--:|---|---|---|---|---|
| **Text2Mol** | 2021 | `text` `SMILES` `graph` | [EMNLP 2021](https://aclanthology.org/2021.emnlp-main.47/) | [repo](https://github.com/cnedwards/text2mol) | unspecified | The paper that made molecule–language retrieval a task, and the origin of the ChEBI-20 benchmark. |
| **KV-PLM** | 2022 | `text` `SMILES` | [Nature Communications](https://doi.org/10.1038/s41467-022-28494-3) | [repo](https://github.com/thunlp/KV-PLM) | unspecified | Earliest unified SMILES + biomedical-text language model; sequence-only, no explicit structure. |
| **MolT5** | 2022 | `text` `SMILES` | [EMNLP 2022](https://arxiv.org/abs/2204.11817) | [repo](https://github.com/blender-nlp/MolT5) | BSD-3-Clause | Established molecule ↔ language translation as a benchmark family; still the default seq2seq baseline. |
| **MoMu** | 2022 | `text` `graph` | [arXiv](https://arxiv.org/abs/2209.05481) | [repo](https://github.com/BingSu12/MoMu) | non-commercial academic use | First contrastive graph–text foundation model; supervision harvested from paper text, so pairs are weak. |
| **BioT5** | 2023 | `text` `SMILES` | [EMNLP 2023](https://arxiv.org/abs/2310.07276) | [repo](https://github.com/QizhiPei/BioT5) | MIT | SELFIES-based T5 that fixes MolT5's invalid-output problem. |
| **GIMLET** | 2023 | `graph` `text` | [NeurIPS 2023](https://arxiv.org/abs/2306.13089) | [repo](https://github.com/zhao-ht/GIMLET) | MIT | Encodes graph and instruction text in one transformer via generalised position embeddings, with no separate graph encoder. |
| **MolCA** | 2023 | `graph` `text` `SMILES` | [EMNLP 2023](https://arxiv.org/abs/2310.12798) | [repo](https://github.com/acharkq/MolCA) | unspecified | The Q-Former-style projector that most later graph–LLM assistants reuse; ships PubChem324k. |
| **MoleculeSTM** | 2023 | `text` `graph` `SMILES` | [Nature Machine Intelligence](https://arxiv.org/abs/2212.10789) | [repo](https://github.com/chao1224/MoleculeSTM) | NVIDIA Source Code License (non-commercial) | Introduced open-vocabulary text-based molecule editing; dataset release constrained by text licensing. |
| **MolFM** | 2023 | `graph` `text` `KG` | [arXiv](https://arxiv.org/abs/2307.09484) | [repo](https://github.com/PharMolix/OpenBioMed) | MIT | Adds a knowledge graph as a third modality; +12.13 pp zero-shot cross-modal retrieval. |
| **MolReGPT** | 2023 | `text` `SMILES` | [IEEE TKDE 2024](https://arxiv.org/abs/2306.06615) | [repo](https://github.com/phenixace/MolReGPT) | unspecified | Retrieval-augmented in-context prompting that matches fine-tuned MolT5 with no training at all; the baseline any new model should beat. |
| **MolXPT** | 2023 | `text` `SMILES` | [ACL 2023](https://arxiv.org/abs/2305.10688) | — | unspecified | Replaces molecule names in PubMed text with their SMILES inline, so structure and surrounding prose condition each other in one stream. |
| **nach0** | 2023 | `text` `SMILES` `reaction` | [Chemical Science 2024](https://arxiv.org/abs/2311.12410) | [repo](https://github.com/insilicomedicine/nach0) | unspecified | Instruction-tuned encoder-decoder pretrained jointly on papers, patents and SMILES, spanning biomedical NLP and chemical generation. |
| **Text+Chem T5** | 2023 | `text` `SMILES` `reaction` | [ICML 2023](https://arxiv.org/abs/2301.12586) | [repo](https://github.com/GT4SD/multitask_text_and_chemistry_t5) | MIT | One multi-task T5 covering captioning, text-conditional generation and reaction prediction without per-task fine-tuning. |
| **3M-Diffusion** | 2024 | `text` `graph` | [COLM 2024](https://arxiv.org/abs/2403.07179) | [repo](https://github.com/huaishengzhu/3MDiffusion) | unspecified | Aligns a graph latent space to text, then diffuses in it — the first strong text→graph diffusion result. |
| **Atomas** | 2024 | `text` `SMILES` | [ICLR 2025](https://arxiv.org/abs/2404.16880) | [repo](https://github.com/yikunpku/Atomas) | unspecified | Aligns SMILES fragments to text at three semantic levels rather than only molecule-to-sentence, which is where most contrastive models stop. |
| **BioT5+** | 2024 | `text` `SMILES` | [Findings of ACL 2024](https://arxiv.org/abs/2402.17810) | [repo](https://github.com/QizhiPei/BioT5) | MIT | 15 task types over 21 datasets; first place in the ACL 2024 text-based molecule generation track. |
| **LDMol** | 2024 | `text` `SMILES` | [ICML 2025](https://arxiv.org/abs/2405.17829) | [repo](https://github.com/jinhojsk515/ldmol) | Apache-2.0 | Direct evidence that latent diffusion beats autoregressive decoding when the latent space is chemistry-aware. |
| **MolBind** | 2024 | `text` `graph` `3D` | [arXiv](https://arxiv.org/abs/2403.08167) | — | unspecified | Binds four modalities to a shared space through pairwise contrastive training, in the manner of ImageBind; ships the MolBind-M4 dataset. |
| **UTGDiff** | 2024 | `text` `graph` | [arXiv](https://arxiv.org/abs/2408.09896) | [repo](https://github.com/ran1812/UTGDiff) | unspecified | One denoising network over a joint text–graph token space rather than two aligned encoders. |

### Multi-View: 2D + 3D

Alternative encodings of the *same* molecule — graph topology fused with geometry.

| Model | Year | Modalities | Paper | Code | Licence | Why it matters |
|---|:--:|---|---|---|---|---|
| **GraphMVP** | 2021 | `graph` `3D` | [ICLR 2022](https://arxiv.org/abs/2110.07728) | [repo](https://github.com/chao1224/GraphMVP) | MIT | The canonical 2D↔3D multi-view pretraining recipe; 3D as a teacher for a 2D encoder. |
| **Uni-Mol** | 2023 | `3D` `graph` | [ICLR 2023](https://openreview.net/forum?id=6K2RM6wVqKu) | [repo](https://github.com/deepmodeling/Uni-Mol) | MIT | 209M conformations + 3M pockets; the 3D backbone that later multimodal assistants plug in. |
| **3D-MolT5** | 2024 | `3D` `text` `SMILES` | [ICLR 2025](https://arxiv.org/abs/2406.05797) | [repo](https://github.com/QizhiPei/3D-MolT5) | Apache-2.0 | Discretises 3D geometry into T5 tokens, so structure and language share one autoregressive stream. |
| **MolLM** | 2024 | `text` `graph` `3D` | [Bioinformatics (ISMB 2024)](https://doi.org/10.1093/bioinformatics/btae260) | [repo](https://github.com/gersteinlab/MolLM) | unspecified | Clean ablation showing what explicit 3D adds on top of graph–text contrastive pretraining. |
| **MV-Mol** | 2024 | `graph` `text` `KG` | [KDD 2024](https://arxiv.org/abs/2406.09841) | [repo](https://github.com/PharMolix/OpenBioMed) | MIT | Text prompts select which "view" of a molecule to encode, fusing structure with biomedical text and knowledge-graph triples. |
| **Mol-LLaMA** | 2025 | `graph` `3D` `text` | [NeurIPS 2025](https://arxiv.org/abs/2502.13449) | [repo](https://github.com/DongkiKim95/Mol-LLaMA) | unspecified | Blends 2D and 3D encoders through cross-attention before the Q-Former; 7B and 8B checkpoints released. |

### Multimodal LLMs & Assistants

Instruction-tuned models that inject structural embeddings into a language backbone.

| Model | Year | Modalities | Paper | Code | Licence | Why it matters |
|---|:--:|---|---|---|---|---|
| **BioMedGPT** | 2023 | `graph` `text` | [arXiv](https://arxiv.org/abs/2308.09442) | [repo](https://github.com/PharMolix/OpenBioMed) | MIT | Aligns molecule-graph and protein encoders into a 7B backbone so both can be queried in free text. |
| **GIT-Mol** | 2023 | `graph` `image` `text` `SMILES` | [Computers in Biology and Medicine](https://arxiv.org/abs/2308.06911) | [repo](https://github.com/AI-HPC-Research-Team/GIT-Mol) | MIT | First graph+image+text chemistry model; GIT-Former with cross-modal matching and contrastive losses. |
| **InstructMol** | 2023 | `graph` `text` `SMILES` | [COLING 2025](https://arxiv.org/abs/2311.16208) | [repo](https://github.com/IDEA-XL/InstructMol) | Apache-2.0 (code); CC BY-NC 4.0 (data) | Two-stage alignment + instruction tuning on a frozen LLM; the practical assistant template. |
| **3D-MoLM** | 2024 | `3D` `text` | [ICLR 2024](https://arxiv.org/abs/2401.13923) | [repo](https://github.com/lsh0520/3D-MoLM) | unspecified | Projects a 3D encoder into a language model and instruction-tunes on 3D-MoIT for properties that 2D topology cannot answer. |
| **ChemDFM** | 2024 | `text` `SMILES` | [Cell Reports Physical Science](https://arxiv.org/abs/2401.14818) | [repo](https://github.com/OpenDFM/ChemDFM) | Apache-2.0 | The text-only dialogue backbone that ChemDFM-X extends with modality encoders. |
| **ChemDFM-X** | 2024 | `graph` `3D` `image` `MS` `IR` `text` `SMILES` | [Science China Information Sciences](https://arxiv.org/abs/2409.13194) | [repo](https://github.com/OpenDFM/ChemDFM-X) | Apache-2.0 | The broadest modality coverage of any open chemistry model; 7.6M synthesised instruction pairs. |
| **HIGHT** | 2024 | `graph` `text` | [ICML 2025](https://arxiv.org/abs/2406.14021) | [repo](https://github.com/LFhase/HIGHT) | Apache-2.0 | Node-level tokenisation causes motif hallucination; hierarchical tokens cut it by 40% across 14 benchmarks. |
| **InstructBioMol** | 2024 | `text` `graph` `3D` | [Nature Machine Intelligence 2025](https://arxiv.org/abs/2410.07919) | [repo](https://github.com/HICAI-ZJU/InstructBioMol) | MIT | Motif-guided extraction encodes 2D graph and 3D structure together; any-to-any instruction following across molecules and proteins. |
| **UniMoT** | 2024 | `graph` `text` `SMILES` | [IJCAI 2025](https://arxiv.org/abs/2408.00863) | — | unspecified | VQ-tokenises molecules into the LLM's own vocabulary; elegant unification, sparse public code. |
| **ChemMLLM** | 2025 | `image` `text` `SMILES` | [arXiv / Cell Reports Physical Science](https://arxiv.org/abs/2505.16326) | [repo](https://github.com/bbsbz/ChemMLLM) | unspecified | Understands *and* generates molecule images; 34B hits 0.87 avg sim on img2SMILES, 4.27 vs 1.97 over GPT-4o on image optimisation. |

### Images, OCR & Document Intelligence

Molecular depictions and reaction schemes as they actually appear in papers and patents.

| Model | Year | Modalities | Paper | Code | Licence | Why it matters |
|---|:--:|---|---|---|---|---|
| **DECIMER** | 2021 | `image` `SMILES` | [Journal of Cheminformatics](https://doi.org/10.1186/s13321-021-00538-8) | [repo](https://github.com/Kohulan/DECIMER-Image_Transformer) | MIT | Fully open OCSR pipeline with released training data; the other specialist baseline to beat. |
| **Img2Mol** | 2021 | `image` `SMILES` | [Chemical Science](https://doi.org/10.1039/D1SC01839F) | [repo](https://github.com/bayer-science-for-a-better-life/Img2Mol) | Apache-2.0 (code); CC BY-NC 4.0 (weights) | CNN encoder feeding a pretrained SMILES decoder; recovers up to 88% of depictions and predates the transformer OCSR wave. |
| **MolScribe** | 2022 | `image` `graph` | [J. Chem. Inf. Model.](https://arxiv.org/abs/2205.14311) | [repo](https://github.com/thomas0809/MolScribe) | MIT | The image-to-graph OCSR reference; explicit atoms and bonds with confidence estimates. |
| **RxnScribe** | 2023 | `image` `reaction` | [J. Chem. Inf. Model.](https://arxiv.org/abs/2305.11845) | [repo](https://github.com/thomas0809/RxnScribe) | MIT | Casts diagram parsing as sequence generation; 80.0% soft-match F1 over 1,378 annotated diagrams, and the direct predecessor to RxnIM. |
| **ChemVLM** | 2024 | `image` `text` | [AAAI 2025](https://arxiv.org/abs/2408.07246) | [repo](https://github.com/lijunxian111/ChemVlm) | unspecified | InternViT-6B + ChemLLM-20B (26B total, 16×A100); SOTA on 5 of 6 tasks, 31.6% on CMMU vs 24.2% for GPT-4V. |
| **MolNexTR** | 2024 | `image` `graph` | [Journal of Cheminformatics](https://arxiv.org/abs/2403.03691) | [repo](https://github.com/CYF2000127/MolNexTR) | Apache-2.0 | ConvNext+ViT dual stream; one of the specialists that generalist MLLMs still have not caught. |
| **RxnIM** | 2025 | `image` `reaction` `text` | [Chemical Science](https://arxiv.org/abs/2503.08156) | [repo](https://github.com/CYF2000127/RxnIM) | MIT | Turns published reaction schemes into machine-readable records; 88% average F1, ~5 pts over prior baselines. |

### Reactions, Conditions & Procedures

Reaction context, condition recommendation and experimental procedure prediction.

| Model | Year | Modalities | Paper | Code | Licence | Why it matters |
|---|:--:|---|---|---|---|---|
| **MM-RCR** | 2024 | `reaction` `graph` `text` `SMILES` | [arXiv (also released as Chemma-RC)](https://arxiv.org/abs/2407.15141) | — | unspecified | 1.2M pairwise QA instructions; the clearest treatment of reaction conditions as a first-class modality. |
| **PRESTO** | 2024 | `graph` `text` `reaction` | [Findings of EMNLP 2024](https://arxiv.org/abs/2406.13193) | [repo](https://github.com/IDEA-XL/PRESTO) | Apache-2.0 | Models interaction between multiple molecule graphs at once, which single-molecule projectors cannot represent for reactions. |
| **ReactXT** | 2024 | `reaction` `graph` `text` `SMILES` | [Findings of ACL 2024](https://arxiv.org/abs/2405.14225) | [repo](https://github.com/syr-cn/ReactXT) | MIT | Role-aware forward/backward reaction context as a pretraining signal; ships the OpenExp dataset. |

### Spectra & Structure Elucidation

IR, NMR and MS as model inputs — retrieval, interpretation and de novo generation.

| Model | Year | Modalities | Paper | Code | Licence | Why it matters |
|---|:--:|---|---|---|---|---|
| **CReSS** | 2021 | `NMR` `SMILES` | [Analytical Chemistry](https://doi.org/10.1021/acs.analchem.1c04307) | [repo](https://github.com/Qihoo360/CReSS) | unspecified | CLIP-style contrastive alignment applied to 13C NMR — the spectroscopy analogue of MoleculeSTM. |
| **MSNovelist** | 2022 | `MS` `SMILES` | [Nature Methods](https://doi.org/10.1038/s41592-022-01486-3) | [repo](https://github.com/zamboni-lab/MSNovelist) | AGPL-3.0 | Fingerprint prediction then RNN decoding; the first credible non-retrieval answer to unknown MS/MS. |
| **Spec2Mol** | 2023 | `MS` `SMILES` | [Communications Chemistry](https://doi.org/10.1038/s42004-023-00932-3) | [repo](https://github.com/KavrakiLab/Spec2Mol) | BSD-3-Clause | Straight spectra→SMILES encoder–decoder; the simplest baseline the diffusion models are measured against. |
| **DiffMS** | 2025 | `MS` `graph` | [ICML 2025](https://arxiv.org/abs/2502.09571) | [repo](https://github.com/coleygroup/DiffMS) | MIT | Formula-constrained graph diffusion conditioned on MS/MS; the strongest MS-only generative baseline. |
| **DiffSpectra** | 2025 | `IR` `NMR` `MS` `graph` `3D` | [arXiv](https://arxiv.org/abs/2507.06853) | [repo](https://github.com/AzureLeon1/DiffSpectra) | Apache-2.0 | SpecFormer conditioning an SE(3)-equivariant diffusion transformer; 40.76% top-1 and 99.49% top-10. |
| **MolSpectra / SpecFormer** | 2025 | `IR` `NMR` `3D` | [ICLR 2025](https://arxiv.org/abs/2502.16284) | [repo](https://github.com/AzureLeon1/MolSpectra) | unspecified | Uses energy spectra as a physical supervision signal for 3D encoders, not just as a retrieval query. |
| **MultimodalAnalytical** | 2025 | `IR` `NMR` `MS` `SMILES` | [ChemRxiv / AI4Mat @ NeurIPS 2025](https://doi.org/10.26434/chemrxiv-2025-q80r9) | [repo](https://github.com/rxn4chemistry/MultimodalAnalytical) | MIT | Analytical chemistry consolidating into reusable foundation-model tooling rather than one-off models. |

### Materials & Crystals

Property-conditioned generation and simulation of inorganic solids.

| Model | Year | Modalities | Paper | Code | Licence | Why it matters |
|---|:--:|---|---|---|---|---|
| **CDVAE** | 2021 | `crystal` `3D` `graph` | [ICLR 2022](https://arxiv.org/abs/2110.06197) | [repo](https://github.com/txie-93/cdvae) | MIT | The SE(3)-equivariant diffusion decoder that MatterGen and most later crystal generators build on. |
| **CrystaLLM** | 2023 | `text` `crystal` | [Nature Communications](https://arxiv.org/abs/2307.04340) | [repo](https://github.com/lantunes/CrystaLLM) | MIT | Treats the CIF file itself as the language to model, making crystals a text modality rather than a geometric one. |
| **GNoME** | 2023 | `crystal` `graph` `3D` | [Nature](https://doi.org/10.1038/s41586-023-06735-9) | [repo](https://github.com/google-deepmind/materials_discovery) | Apache-2.0 (code); CC BY-NC 4.0 (database) | 2.2M structures and 381,000 predicted stable materials; the filter that gives generative crystal models something to be checked against. |
| **MatterSim** | 2024 | `crystal` `3D` | [arXiv](https://arxiv.org/abs/2405.04967) | [repo](https://github.com/microsoft/mattersim) | MIT | The forward simulator that pairs with MatterGen to close the generate-then-verify loop. |
| **MatterGen** | 2025 | `crystal` `3D` | [Nature](https://doi.org/10.1038/s41586-025-08628-5) | [repo](https://github.com/microsoft/mattergen) | MIT | Joint diffusion over atom types, coordinates and lattice; fine-tunable on property constraints. |

<!-- END:models -->

## Datasets and benchmarks

Grouped by the modality pair each one actually aligns, because that is what decides which
models can be trained or evaluated on it. A dataset with 3M samples and no text is not a
substitute for one with 300k molecule–description pairs.

### Where the data comes from

Most of these are not independent. Four upstream sources feed nearly everything below, and a
model can meet the same molecule several times over under different names:

```mermaid
flowchart LR
    PC["PubChem<br/>118.6M compounds"] --> PCSTM["PubChemSTM"]
    PC --> PC324["PubChem324k"]
    PC --> HIPC["HiPubChem"]
    PC --> LM24["L+M-24"]
    PC --> SMI["SMolInstruct"]
    PC --> MI["Mol-Instructions"]

    PC -- compounds --> CH20["ChEBI-20"]
    CHEBI["ChEBI"] -- descriptions --> CH20
    CH20 --> MQA["MoleculeQA"]
    CH20 --> MOTIF["MotifHallu"]
    CH20 --> SMI

    ZINC["ZINC-250K"] --> TOMG["TOMG-Bench"]
    PC --> TOMG

    USPTO["USPTO patents"] --> U50["USPTO-50k"]
    USPTO --> UFULL["USPTO-full"]
    USPTO --> ORD["ORD"]
    ORD --> ORDLY["ORDerly"]
    ORD --> OEXP["OpenExp"]
    USPTO --> OEXP
    USPTO --> MSD["Multimodal<br/>Spectroscopic"]
    USPTO --> IRNMR["IR–NMR dataset"]
    UFULL --> SMI
    USPTO --> MI
```

ChEBI-20 is the most commonly mis-cited node here: its *compounds* come from PubChem and its
*descriptions* from ChEBI, so crediting it to either source alone is wrong in both directions.

Three consequences worth keeping in mind:

- **Leakage is structural, not accidental.** ChEBI-20 supplies the description corpus for
  both MoleculeQA and MotifHallu, and is itself folded into SMolInstruct. Instruction-tune on
  SMolInstruct, evaluate on MoleculeQA, and the compounds have already been seen. The
  questions are new; the molecules are not.
- **The spectra are simulated.** Both large spectroscopic corpora compute IR and NMR from
  patent-extracted structures rather than measuring them, so a model can learn the
  simulator's systematic error as if it were chemistry.
- **Licence terms are inherited, not chosen.** PubChem itself is permissive, but its textual
  annotations carry mixed provenance, which is why PubChemSTM could not be released in full.

MaCBench is the deliberate exception: its images were created for the benchmark, by building
and photographing lab setups rather than harvesting existing figures.

<!-- BEGIN:datasets -->
### Structure ↔ Text

Molecule–description, QA and instruction pairs. The scarcest resource in the field, and the one most constrained by text licensing.

| Dataset | Year | Type | Modalities | Scale | Licence | Paper | Data |
|---|:--:|---|---|---|---|---|---|
| **ChEBI-20** | 2021 | Benchmark | `SMILES` `text` | 33,010 molecule–description pairs, 80/10/10 split | unspecified | [paper](https://doi.org/10.18653/v1/2021.emnlp-main.47) | [download](https://github.com/cnedwards/text2mol) |
| **ChEMBL** | 2023 | Corpus | `SMILES` `text` | >20.3M bioactivity measurements on 2.4M compounds from >1.6M assays | CC BY-SA 3.0 | [paper](https://doi.org/10.1093/nar/gkad1004) | [download](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/) |
| **ChemLLMBench** | 2023 | Benchmark | `text` `SMILES` `reaction` | 8 tasks: naming, property, yield, reaction and retrosynthesis prediction, text-based design, captioning, reagent selection | unspecified | [paper](https://arxiv.org/abs/2305.18365) | [download](https://github.com/ChemFoundationModels/ChemLLMBench) |
| **PubChem324k** | 2023 | Pretraining set | `graph` `SMILES` `text` | 324k molecule–text pairs; 15k high-quality subset (>19 words) | unspecified | [paper](https://aclanthology.org/2023.emnlp-main.966/) | [download](https://huggingface.co/datasets/acharkq/PubChem324kV2) |
| **PubChemSTM** | 2023 | Pretraining set | `graph` `SMILES` `text` | >280,000 structure–text pairs | unspecified | [paper](https://arxiv.org/abs/2212.10789) | [download](https://huggingface.co/datasets/chao1224/MoleculeSTM) |
| **L+M-24** | 2024 | Corpus | `SMILES` `text` | 160,492 training pairs; 21,839 evaluation pairs per track; ~1.5M property–molecule pairs | unspecified | [paper](https://arxiv.org/abs/2403.00791) | [download](https://github.com/language-plus-molecules/LPM-24-Dataset) |
| **Mol-Instructions** | 2024 | Instruction set | `SMILES` `text` | 148.4K molecule, 505K protein and 53K biotext instructions | CC BY 4.0 | [paper](https://arxiv.org/abs/2306.08018) | [download](https://huggingface.co/datasets/zjunlp/Mol-Instructions) |
| **MoleculeQA** | 2024 | Benchmark | `SMILES` `text` | 62K QA pairs over 23K molecules, one correct and three distractor options each | MIT | [paper](https://arxiv.org/abs/2403.08192) | [download](https://huggingface.co/datasets/hcaoaf/MoleculeQA) |
| **MolLM dataset** | 2024 | Pretraining set | `graph` `3D` `text` | 160K molecule–text pairings with 2D and 3D information | unspecified | [paper](https://doi.org/10.1093/bioinformatics/btae260) | [download](https://github.com/gersteinlab/MolLM) |
| **SMolInstruct** | 2024 | Instruction set | `SMILES` `text` | 14 tasks, >3M samples | CC BY 4.0 | [paper](https://arxiv.org/abs/2402.09391) | [download](https://huggingface.co/datasets/osunlp/SMolInstruct) |
| **TOMG-Bench** | 2024 | Benchmark | `SMILES` `text` | 3 tasks (MolEdit, MolOpt, MolCustom), 9 subtasks, 45,000 test samples | unspecified | [paper](https://arxiv.org/abs/2412.14642) | [download](https://huggingface.co/datasets/Duke-de-Artois/TOMG-Bench) |
| **HiPubChem** | 2025 | Pretraining set | `graph` `text` | 295k molecule–text pairs with motif annotations (Jan 2024 cutoff) | CC BY-NC 4.0 | [paper](https://arxiv.org/abs/2406.14021) | [download](https://huggingface.co/datasets/lfhase/HIGHT) |
| **MotifHallu** | 2025 | Benchmark | `graph` `text` | 23,924 questions over 3,300 molecules and 38 RDKit functional groups | CC BY-NC 4.0 | [paper](https://arxiv.org/abs/2406.14021) | [download](https://huggingface.co/datasets/lfhase/HIGHT) |
| **PubChem** | 2025 | Corpus | `SMILES` `graph` `3D` `text` | 118.6M compounds, 322.4M substances, 295.4M bioactivity data points (Sep 2024) | unspecified | [paper](https://doi.org/10.1093/nar/gkae1059) | [download](https://pubchem.ncbi.nlm.nih.gov/docs/downloads) |

### Structure ↔ Spectra

IR, NMR and MS paired with the structure that produced them. Mostly simulated, mostly patent-derived.

| Dataset | Year | Type | Modalities | Scale | Licence | Paper | Data |
|---|:--:|---|---|---|---|---|---|
| **MassSpecGym** | 2024 | Benchmark | `MS` `SMILES` | 231k high-quality spectra over 29k unique structures | MIT | [paper](https://arxiv.org/abs/2410.23326) | [download](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym) |
| **MolPuzzle** | 2024 | Benchmark | `IR` `MS` `NMR` `image` `text` | 217 instances, 868 spectrum images, 23,678 QA samples | MIT | [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f2b9e8e7a36d43ddfd3d55113d56b1e0-Abstract-Datasets_and_Benchmarks_Track.html) | [download](https://huggingface.co/datasets/kguo2/MolPuzzle_data) |
| **Multimodal Spectroscopic Dataset** | 2024 | Corpus | `NMR` `IR` `MS` `SMILES` | 1H/13C/HSQC NMR, IR and ±MS for 790,000 patent-derived molecules | CDLA-Sharing-1.0 | [paper](https://arxiv.org/abs/2407.17492) | [download](https://doi.org/10.5281/zenodo.14770232) |
| **IR–NMR multimodal dataset** | 2025 | Corpus | `IR` `NMR` `3D` `SMILES` | 177,461 molecules with IR; 1,255 with DFT 1H/13C shifts | CDLA-Permissive-2.0 | [paper](https://doi.org/10.1038/s41597-025-05729-8) | [download](https://doi.org/10.5281/zenodo.15669241) |

### Structure ↔ Image

Depictions and reaction schemes paired with machine-readable structure. Largely synthetic renderings rather than scraped figures.

| Dataset | Year | Type | Modalities | Scale | Licence | Paper | Data |
|---|:--:|---|---|---|---|---|---|
| **MMChemOCR / MMCR-Bench / MMChemBench** | 2024 | Benchmark | `image` `text` `SMILES` | 1,000 OCR pairs; 1,000 exam questions; 700 captioning/property samples | MIT | [paper](https://arxiv.org/abs/2408.07246) | [download](https://huggingface.co/datasets/Duke-de-Artois/ChemVLM_test_data) |
| **MaCBench** | 2025 | Benchmark | `image` `text` `crystal` | 1,153 expert-curated questions across 35 configurations | MIT | [paper](https://doi.org/10.1038/s43588-025-00836-3) | [download](https://huggingface.co/datasets/jablonkagroup/MaCBench) |

### Reaction, Conditions & Procedures

Reaction records with roles, conditions, yields and free-text procedures.

| Dataset | Year | Type | Modalities | Scale | Licence | Paper | Data |
|---|:--:|---|---|---|---|---|---|
| **USPTO-50k** | 2016 | Benchmark | `reaction` | 50k reactions of 10 types; 40,008 / 5,001 / 5,007 split | unspecified | [paper](https://doi.org/10.1021/acs.jcim.6b00564) | [download](https://github.com/Hanjun-Dai/GLN) |
| **USPTO-full** | 2019 | Benchmark | `reaction` | ~1M unique reactions, 800k / 100k / 100k split | unspecified | [paper](https://arxiv.org/abs/2001.01408) | [download](https://doi.org/10.6084/m9.figshare.5104873) |
| **Open Reaction Database (ORD)** | 2021 | Corpus | `reaction` `text` | structured reaction records with conditions, yields and free-text procedures | CC BY-SA 4.0 | [paper](https://doi.org/10.1021/jacs.1c09820) | [download](https://huggingface.co/datasets/open-reaction-database/ord-data) |
| **OpenExp** | 2024 | Benchmark | `reaction` `text` | 274k reaction–procedure pairs, 8:1:1 split | CC BY-SA | [paper](https://arxiv.org/abs/2405.14225) | [download](https://github.com/syr-cn/ReactXT) |
| **ORDerly** | 2024 | Benchmark | `reaction` | 356,906 reactions in the condition-prediction benchmark, from 1.7M ORD reactions | MIT | [paper](https://doi.org/10.1021/acs.jcim.4c00292) | [download](https://figshare.com/articles/dataset/ORDerly_chemical_reactions_condition_benchmarks/23298467) |

### Structure ↔ 3D Geometry

Conformers and quantum labels. These carry no text, but they are what every 3D branch of a multi-modal model is pretrained on.

| Dataset | Year | Type | Modalities | Scale | Licence | Paper | Data |
|---|:--:|---|---|---|---|---|---|
| **QM9** | 2014 | Benchmark | `3D` | 133,885 species with up to nine heavy atoms, 15 DFT properties | CC BY-NC-SA 4.0 | [paper](https://doi.org/10.1038/sdata.2014.22) | [download](https://doi.org/10.6084/m9.figshare.978904) |
| **MoleculeNet** | 2018 | Benchmark | `SMILES` `graph` `3D` | 17 datasets, >800 tasks over ~700,000 compounds | CC BY-NC 3.0 | [paper](https://doi.org/10.1039/C7SC02664A) | [download](https://github.com/deepchem/deepchem) |
| **Alchemy** | 2019 | Benchmark | `3D` | 119,487 molecules up to 14 heavy atoms, 12 quantum properties | MIT | [paper](https://arxiv.org/abs/1906.09427) | [download](https://github.com/tencent-alchemy/Alchemy) |
| **PCQM4Mv2 / OGB-LSC** | 2021 | Benchmark | `graph` `3D` | 3,746,619 molecules with DFT HOMO–LUMO gaps | CC BY 4.0 | [paper](https://arxiv.org/abs/2103.09430) | [download](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) |
| **GEOM** | 2022 | Corpus | `3D` | 37M conformations for >450,000 molecules | CC0 | [paper](https://doi.org/10.1038/s41597-022-01288-4) | [download](https://doi.org/10.7910/DVN/JNGTDF) |

### Crystals & Materials

Periodic structures with computed properties.

| Dataset | Year | Type | Modalities | Scale | Licence | Paper | Data |
|---|:--:|---|---|---|---|---|---|
| **Materials Project** | 2013 | Corpus | `crystal` `3D` | 86,680 inorganic compounds and 530,243 nanoporous materials (live) | CC BY 4.0 | [paper](https://doi.org/10.1063/1.4812323) | [download](https://materialsproject.org/) |

<!-- END:datasets -->

## Timeline

<!-- BEGIN:timeline -->
```mermaid
timeline
    title Multi-modal chemistry, by first public release
    2021 : CDVAE
         : CReSS
         : DECIMER
         : GraphMVP
         : Img2Mol
         : Text2Mol
    2022 : KV-PLM
         : MolScribe
         : MolT5
         : MoMu
         : MSNovelist
    2023 : BioMedGPT
         : BioT5
         : CrystaLLM
         : GIMLET
         : GIT-Mol
         : GNoME
         : InstructMol
         : MolCA
         : MoleculeSTM
         : MolFM
         : MolReGPT
         : MolXPT
         : nach0
         : RxnScribe
         : Spec2Mol
         : Text+Chem T5
         : Uni-Mol
    2024 : 3D-MoLM
         : 3D-MolT5
         : 3M-Diffusion
         : Atomas
         : BioT5+
         : ChemDFM
         : ChemDFM-X
         : ChemVLM
         : HIGHT
         : InstructBioMol
         : LDMol
         : MatterSim
         : MM-RCR
         : MolBind
         : MolLM
         : MolNexTR
         : MV-Mol
         : PRESTO
         : ReactXT
         : UniMoT
         : UTGDiff
    2025 : ChemMLLM
         : DiffMS
         : DiffSpectra
         : MatterGen
         : Mol-LLaMA
         : MolSpectra / SpecFormer
         : MultimodalAnalytical
         : RxnIM
```

<!-- END:timeline -->

## What actually works

Findings below are drawn from the papers indexed above; the [full review](REVIEW.md) carries
the argument and the numbers.

<details open>
<summary><b>Specialists still beat generalists on tightly-scoped tasks</b></summary>

<br>

On strict image-to-structure conversion, MolScribe, MolNexTR and DECIMER remain ahead of
generalist multimodal LLMs. ChemMLLM and ChemVLM narrow the gap and cover far more tasks,
but if you need one number to be right, use the specialist. MaCBench makes the same point
from the evaluation side: vision-language models perceive chemistry images competently and
then fail at spatial reasoning and cross-modal synthesis.

</details>

<details>
<summary><b>Modalities help when they are complementary, not redundant</b></summary>

<br>

3D geometry helps when the target is physically grounded (quantum properties, binding).
Text helps when the target depends on assay context, semantics or biochemical description.
Stacking two encodings of the same information mostly buys parameters. MolLM's ablations
and the GraphMVP line of work are the cleanest evidence here.

</details>

<details>
<summary><b>Tokenisation design drives chemical hallucination</b></summary>

<br>

HIGHT showed that node-centric graph tokenisation induces *motif* hallucination, where models
confidently assert functional groups that are not present, and that a chemically meaningful
token hierarchy cuts it by 40% while improving 14 benchmarks. This is a design bug, not a
scale problem.

</details>

<details>
<summary><b>Spectroscopy is where multimodality is chemically necessary, not merely convenient</b></summary>

<br>

Human experts elucidate structures by combining orthogonal evidence streams, and the models
now do the same. DiffSpectra reports 40.76% top-1 and 99.49% top-10 structure elucidation
accuracy by conditioning an SE(3)-equivariant diffusion transformer on multiple spectra at
once, a regime that single-modality models such as MSNovelist and Spec2Mol could not reach.

</details>

<details>
<summary><b>Reaction work is consolidating into a pipeline, not competing models</b></summary>

<br>

RxnIM parses published reaction schemes into machine-readable records; ReactXT uses reaction
context to predict procedures and retrosynthetic routes; MM-RCR recommends conditions. These
are stages of one stack, and they are increasingly being built as such.

</details>

<details>
<summary><b>Diffusion is winning wherever the output is a structured object</b></summary>

<br>

For molecules, crystals and spectra alike, denoising in a chemically meaningful latent or
geometric space now matches or beats autoregressive decoding. LDMol, 3M-Diffusion, DiffMS,
DiffSpectra and MatterGen all point the same way. Autoregression still owns text output.

</details>

## Open problems

| # | Problem | Why it is hard | Where to look |
|:--:|---|---|---|
| 1 | **Paired-data scarcity** | There is no chemistry equivalent of a web-scale image–text corpus. Molecule–text sets are in the 10⁵ range; aligned all-modal corpora barely exist. | ChemDFM-X synthesises 7.6M instructions; most others use two-stage training on frozen backbones |
| 2 | **Data rights and reproducibility** | PubChem is permissive but its *textual* annotations inherit heterogeneous licences, which is why PubChemSTM could not be fully released. InstructMol's code is Apache-2.0 while its data is research-only. | [MoleculeSTM](https://arxiv.org/abs/2212.10789), [OECD data governance](https://www.oecd.org/en/publications/ai-data-governance-and-privacy_2476b1a4-en.html) |
| 3 | **Benchmark fragility** | Cleaning choices silently inflate reaction-condition scores; USPTO random splits are in-distribution; the same molecule recurs across literature, patents, images and derived synthetic data, so leakage is compounded. | [ORDerly](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00292), [syntheseus](https://pubs.rsc.org/en/content/articlelanding/2025/fd/d4fd00093e), [OOD splits](https://pubs.acs.org/doi/10.1021/acscentsci.5c00055) |
| 4 | **Compute concentration** | ChemVLM: 26B params on 16×A100. Uni-Mol2: 1.1B params, 800M conformers. Frontier work increasingly means *adapting* someone else's large backbone. | Parameter-efficient projectors are the field's answer |
| 5 | **Cross-modal consistency** | Many models align modalities; few can guarantee that a generated molecule, its IR spectrum, its NMR features, its depiction and its verbal description all cohere physically. | The clearest unsolved problem in the field |
| 6 | **Dual use** | Generative molecular models can be inverted toward harmful design. This is demonstrated, not hypothetical. | [Urbina et al. 2022](https://www.nature.com/articles/s42256-022-00465-9), [EU AI Act Ch. V](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng), [NIST AI RMF](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10) |

## Reading list

<!-- BEGIN:reading -->
### Surveys & reviews

- **[A Perspective on Foundation Models in Chemistry](https://pubs.acs.org/doi/10.1021/jacsau.4c01160)** — Junyoung Choi et al., *JACS Au* (2025).  
  Defines what actually qualifies as a chemistry foundation model, and where multimodal pretraining does not transfer.
- **[A Survey of Large Language Models for Text-Guided Molecular Discovery](https://arxiv.org/abs/2505.16094)** — Ziqing Wang et al., *arXiv* (2025).  
  Focused on language-conditioned molecule generation and optimisation.
- **[A review of large language models and autonomous agents in chemistry](https://pubs.rsc.org/en/content/articlelanding/2025/sc/d4sc03921a)** — Mayk Caldas Ramos et al., *Chemical Science* (2025).  
  The reference review for chemistry LLM agents and tool use, with a maintained companion repository.
- **[Advancements in Molecular Property Prediction: A Survey of Single and Multimodal Approaches](https://arxiv.org/abs/2408.09461)** — Tanya Liyaqat et al., *Archives of Computational Methods in Engineering* (2025).  
  Directly contrasts single-modality against multi-view encoders for property prediction.
- **[Artificial Intelligence in Spectroscopy: Advancing Chemistry from Prediction to Generation and Beyond](https://arxiv.org/abs/2502.09897)** — Kehan Guo et al., *IJCAI 2025 Survey Track* (2025).  
  Best current survey for AI-driven structure elucidation across MS, NMR and IR.
- **[From Generalist to Specialist: A Survey of Large Language Models for Chemistry](https://arxiv.org/abs/2412.19994)** — Yang Han et al., *COLING 2025* (2025).  
  Explicitly frames how graphs, 3D structures and spectra get injected into chemistry LLMs, plus a benchmark audit.
- **[Molecular representation learning: cross-domain foundations and future frontiers](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00170f)** — Rahul Sheshanarayana, Fengqi You, *Digital Discovery* (2025).  
  Flags hybrid multimodal fusion and 3D-aware representations as the main open frontiers.
- **[Scientific Large Language Models: A Survey on Biological & Chemical Domains](https://arxiv.org/abs/2401.14656)** — Qiang Zhang et al., *ACM Computing Surveys* (2025).  
  Spans textual, molecular, protein and genomic modalities, so it shows where chemistry sits inside scientific LLMs generally.
- **[A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery](https://arxiv.org/abs/2406.10833)** — Yu Zhang et al., *arXiv* (2024).  
  Situates molecule, protein and material LLMs relative to text-only scientific models.
- **[Bridging Text and Molecule: A Survey on Multimodal Frameworks for Molecule](https://arxiv.org/abs/2403.13830)** — Yi Xiao et al., *arXiv* (2024).  
  Organises molecule–text models by pretraining objective, alignment strategy and downstream task.
- **[From Words to Molecules: A Survey of Large Language Models in Chemistry](https://arxiv.org/abs/2402.01439)** — Chang Liao et al., *arXiv* (2024).  
  Early systematic taxonomy of molecular input representations for LLMs.
- **[Leveraging Biomolecule and Natural Language through Multi-Modal Learning: A Survey](https://arxiv.org/abs/2403.01528)** — Qizhi Pei et al., *arXiv* (2024).  
  The most complete map of biomolecule–text cross-modal modelling, covering molecules, proteins and the datasets that join them to language.
- **[Materials science in the era of large language models: a perspective](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00074a)** — Ge Lei, Ronan Docherty, Samuel J. Cooper, *Digital Discovery* (2024).  
  Argues LLMs are best used as tireless task automators rather than idea generators.
- **[A Systematic Survey of Chemical Pre-trained Models](https://www.ijcai.org/proceedings/2023/760)** — Jun Xia et al., *IJCAI 2023 Survey Track* (2023).  
  Covers the pretraining strategies that multimodal encoders inherited, and the pre-LLM baselines they are measured against.

### Critical evaluations & benchmark hygiene

- **[A framework for evaluating the chemical knowledge and reasoning abilities of large language models against the expertise of chemists](https://www.nature.com/articles/s41557-025-01815-x)** — Adrian Mirza et al., *Nature Chemistry* (2025).  
  ChemBench — models beat expert chemists on average yet fail on safety-relevant items and are badly calibrated.
- **[Challenging Reaction Prediction Models to Generalize to Novel Chemistry](https://pubs.acs.org/doi/10.1021/acscentsci.5c00055)** — John Bradshaw et al., *ACS Central Science* (2025).  
  Random splits of USPTO-style data are in-distribution; proposes document, author, time and reaction-type OOD splits.
- **[Probing the limitations of multimodal language models for chemistry and materials research](https://www.nature.com/articles/s43588-025-00836-3)** — Nawaf Alampara et al., *Nature Computational Science* (2025).  
  MaCBench — perception is fine, spatial reasoning and cross-modal synthesis are not.
- **[Re-evaluating retrosynthesis algorithms with syntheseus](https://pubs.rsc.org/en/content/articlelanding/2025/fd/d4fd00093e)** — Krzysztof Maziarz et al., *Faraday Discussions* (2025).  
  Published retrosynthesis rankings largely dissolve once search and evaluation are standardised.
- **[ORDerly: Data Sets and Benchmarks for Chemical Reaction Data](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00292)** — Daniel S. Wigh et al., *J. Chem. Inf. Model.* (2024).  
  Undisclosed cleaning choices silently inflate reported reaction-condition performance.

### Safety, dual use & governance

- **[AI, data governance and privacy: Synergies and areas of international co-operation](https://www.oecd.org/en/publications/ai-data-governance-and-privacy_2476b1a4-en.html)** — OECD, *OECD Artificial Intelligence Papers* (2024).  
  Reference statement on data governance, relevant to provenance and licensing of chemical corpora.
- **[Regulation (EU) 2024/1689 — Artificial Intelligence Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng)** — European Parliament and Council, *EUR-Lex* (2024).  
  Binding source text for general-purpose AI provider obligations (Chapter V, Arts. 51–56).
- **[Artificial Intelligence Risk Management Framework (AI RMF 1.0)](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10)** — NIST, *NIST AI 100-1* (2023).  
  The Govern/Map/Measure/Manage framework most chemistry-AI risk policies are written against.
- **[Dual use of artificial-intelligence-powered drug discovery](https://www.nature.com/articles/s42256-022-00465-9)** — Fabio Urbina, Filippa Lentzos, Cédric Invernizzi, Sean Ekins, *Nature Machine Intelligence* (2022).  
  Inverting a toxicity model turns a generative drug-design pipeline into a toxin-design pipeline.

<!-- END:reading -->

## Contributing

Contributions are very welcome, especially corrections. See [CONTRIBUTING.md](CONTRIBUTING.md).

The short version: **edit the YAML, not the README.** The tables are generated.

```bash
pip install pyyaml
$EDITOR data/models.yaml          # or datasets.yaml / reading.yaml
python scripts/build.py           # regenerates README.md and docs/index.html
```

CI checks that the generated files are in sync and that every URL still resolves.

## Licence and citation

This list is released under [CC0 1.0](LICENSE), public domain. The linked papers, code and
datasets remain under their own licences, which are recorded in the tables above; check them
before use, particularly the non-commercial ones.

```bibtex
@misc{awesome_multimodal_chemistry,
  title        = {Awesome Multi-Modal Chemistry},
  author       = {Orlov, Axel and contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/AxelRolov/awesome-multimodal-chemistry}}
}
```

<div align="center">
<br>
<sub>If a paper is missing, mis-tagged, or its number is wrong, <a href="../../issues/new/choose">open an issue</a>. Corrections are the point.</sub>
</div>
