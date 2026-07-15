# Spotlight-Guard

Reproducibility package for the paper **"Spotlight-Guard: A Layered Defense
Against Indirect Prompt Injection in Tool-Augmented LLM Agents."**

Spotlight-Guard is a local, training-free, layered defense for tool-augmented
LLM agents. It combines three layers into a single pipeline:

1. **Spotlighting** — input isolation that marks untrusted tool content so the
   model can separate it from trusted instructions.
2. **LLM detect-and-quarantine** — a guard model classifies tool content and
   quarantines suspicious payloads before they reach the agent.
3. **HMAC instruction integrity** — cryptographic signing that verifies trusted
   instructions have not been tampered with.

This repository contains the full evaluation pipeline, the defense
implementations, the test partitions derived from the InjecAgent benchmark, and
the aggregated result files needed to reproduce every metric reported in the
paper.

## Repository layout

```
src/
  data/         dataset preparation and adaptive-attack generation
  defenses/     baseline, spotlighting-only, detector-only, Spotlight-Guard (full), react
  evaluation/   evaluation harness, judge, partitioning, metric recomputation
  llm/          Ollama client
  metrics/      bootstrap confidence intervals and classification metrics
  utils/        spotlighting, HMAC signing, detection, I/O, logging
tests/          unit tests (pytest)
data/
  processed/    test_cases.json, partitions_25.json (stratified case definitions)
  results_v4/   main 250-sample run used in the paper
  analysis_250.json   aggregated metrics consumed by the figure scripts
_analyze_paper.py     builds analysis_250.json from a results.json run
_make_figures.py      generates the paper figures (EPS/PNG)
_make_diagrams.py     generates schematic diagrams
_summarize_run.py     prints a per-variant summary table for a results.json
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) running locally (default endpoint
  `http://localhost:11434`)
- The three open-weight models used in the paper:
  - `qwen2.5:7b` (guard)
  - `mistral:7b` (quarantine)
  - `deepseek-coder:6.7b` (fallback)
- A GPU is recommended for the full sweep (the paper used an NVIDIA A100 80GB),
  but the pipeline runs on any machine where the three models fit.

## Setup

```bash
# 1. Start Ollama (if not already running)
ollama serve &

# 2. Pull the three models (first run only; downloads several GB)
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull deepseek-coder:6.7b

# 3. Create the Python environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the unit tests (should all pass)
python -m pytest -q
```

## Running the evaluation

Quick smoke test (6 samples, core defenses only):

```bash
python -m src.evaluation.optimized_evaluation2 --sample-size 6 --output-tag smoke
```

Full publication run (250 adversarial + 250 benign cases per configuration:
core defenses, extra baselines, ablations, and adaptive attacks):

```bash
python -m src.evaluation.optimized_evaluation2 \
    --sample-size 250 \
    --guard-model qwen2.5:7b \
    --quarantine-model mistral:7b \
    --fallback-model deepseek-coder:6.7b \
    --extra-baselines \
    --ablations \
    --adaptive \
    --output-tag full
```

Results are written to `data/results_v4/<date>__<tag>/results.json`.

> The default `num_predict`, `repeat_penalty`, and `repeat_last_n` values in
> `src/llm/ollama_client.py` must be kept; lowering them reintroduces model
> degeneration (runaway repetition).

## Reproducing the paper metrics and figures

```bash
# Print a per-variant summary table for a run
python _summarize_run.py data/results_v4/12-06-2026-13-55__full_a100/results.json

# Aggregate the run into analysis_250.json
python _analyze_paper.py

# Generate the figures used in the manuscript
python _make_figures.py
```

## Headline results

On a stratified, fixed-seed benchmark of 250 adversarial and 250 benign cases
per configuration, the full Spotlight-Guard system:

- reduces attack success rate (ASR) from **36.0% to 17.2%**,
- maintains a **97.2%** benign-task success rate,
- raises detection **F1 from 0.749 to 0.892**, and
- holds a **6.7%** ASR under adaptive attacks crafted to evade the pipeline.

All metrics are reported with bootstrap 95% confidence intervals (1000
iterations).

## Naming note

The proposed system appears in the source code and result files under the
internal variant key `dual_signed_full` (and `dual_signed` for the
spotlighting + HMAC variant). These are the internal identifiers for
**Spotlight-Guard (Full)** and **Spotlight-Guard** as named in the paper.

## Data source

The adversarial and benign cases are derived from the publicly available
[InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) benchmark.
