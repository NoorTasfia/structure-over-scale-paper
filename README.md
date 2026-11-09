# Structure over Scale

**Can code-graph navigation tools make up for smaller and quantized Gemma 4 coding agents?**

Code, all 900 raw trajectories, and analysis for our entry to the
[Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)
(Kaggle, 2026). The paper is in [`paper/paper.md`](paper/paper.md) /
[`paper/paper.pdf`](paper/paper.pdf).

## Study design

| Factor | Levels |
|---|---|
| Tools | **A** list_dir/read_file/grep · **B** A + outline/find_definition/find_references/callees (jedi + ast) · **C** A + issue-ranked repo map in the prompt |
| Precision axis | gemma-4-**E2B**-it at BF16 / Q8_0 / Q4_0 |
| Size axis | **E2B / E4B / 12B**-it, all at Q4_0 |
| Tasks | 60 random SWE-bench Lite instances (seed 0), localization only |

5 model labels × 3 toolsets × 60 instances = **900 episodes**, all run on one
RTX 4070 (12 GB) in 21.7 GPU-hours. Models are the official
`ggml-org/gemma-4-*-it-GGUF` builds, served by llama.cpp (`--jinja`).

## Repository contents

- `gemma_nav/` — agent loop, tools (search / graph / repo map), lenient tool-call parser, scoring
- `run.py`, `summarize.py` — episode driver (resumable) and aggregation
- `scripts/` — `analyze.py` (paired bootstrap: CIs, gains, interactions), `taxonomy.py` (failure classification + hand-label sample), `figures.py`, PowerShell runners
- `results/` — `runs_final.zip` (all 900 trajectories), `analysis.json`, `taxonomy.json`, `failures.csv` (with manual labels)
- `figures/` — the paper's two figures
- `tests/` — parser and tool tests

## Reproduce

```powershell
# serve one model (example: E4B Q4_0)
llama-server.exe -m gemma-4-E4B-it-Q4_0.gguf --n-gpu-layers 99 -c 16384 --port 8000 --jinja
# run one cell, then aggregate
python run.py --label e4b-q4 --toolset B --n 60
python summarize.py
python scripts/analyze.py
python scripts/taxonomy.py
python scripts/figures.py
```

Greedy decoding (temperature 0) makes runs deterministic up to llama.cpp kernel
nondeterminism. The full grid reruns in under a day on a 12 GB GPU.

## License

Apache-2.0 (see [LICENSE](LICENSE)). Our main-competition agent is a separate,
still-private artifact and will be released after the main competition ends
(Dec 2, 2026).
