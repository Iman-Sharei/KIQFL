# KIQFL

**KAN-Enhanced Indirect Quantum Federated Learning**

Research code for hybrid quantum–classical federated learning under a **central-server** topology. Clients train Hybrid CNN–QNN models on private data; the server aggregates parameters. Circuit-state fidelity (optionally mixed by a ring DTQW and softened by a KAN) guides aggregation weights. All quantum dynamics use **classical state-vector simulation** (NISQ-style noise models optional)—not a claim of real-QPU wall-clock speedup.

| | |
|---|---|
| **Language** | Python ≥ 3.10 |
| **Stack** | PyTorch · Flask · pykan · Opacus (optional DP) |
| **UI** | Local web app at `http://127.0.0.1:5000` |
| **CLI** | `python -m core …` or `kiqfl …` after install |

---

## Suggested repository names

If you publish under a new GitHub repo, pick one of these (recommended first):

1. **`kiqfl`** — short, matches the method acronym and package name  
2. **`kiqfl-federated`** — clearer for search (“federated learning”)  
3. **`hybrid-qnn-federated`** — descriptive if you want less acronym-first branding  

Clone example (replace with your user/org):

```bash
git clone https://github.com/<you>/kiqfl.git
cd kiqfl
```

---

## What this is (and is not)

**Is**

- Central-server federated learning (broadcast → local train → upload → aggregate)
- Hybrid CNN → amplitude encoding → variational circuit → classical readout
- Fidelity-guided aggregation vs the **previous** global circuit state
- Optional ring **DTQW** mixing of guidance states + **KAN** mixing coefficient \(\alpha\)
- Baselines: FedAvg, FQNGD, centralized
- Experiment CLI + interactive Flask UI
- Datasets via torchvision: MNIST, Fashion-MNIST, CIFAR-10/100 (downloaded on first use)

**Is not**

- Peer-to-peer / server-free federated learning (the ring is for circuit-state guidance only)
- Execution on real quantum hardware
- A privacy proof by itself (use optional DP / Opacus when you need formal privacy)

---

## Repository layout

```text
.
├── core/                 # Library + CLI
│   ├── quantum/          # Circuit, fidelity, DTQW, KAN helpers
│   ├── models/           # Hybrid CNN–QNN and classical models
│   ├── fl/               # Data, aggregation, algorithms, metrics
│   └── experiments/      # Runner, analyze, pipeline, resume
├── webapp/               # Flask UI (templates + static)
├── test/                 # Smoke / unit tests
├── run_pipeline.sh       # Helper for the full multi-suite pipeline
├── requirements.txt
├── pyproject.toml
└── README.md
```

Runtime folders (gitignored; created automatically):

- `data/` — torchvision downloads  
- `result/` — experiment CSVs, summaries, checkpoints  

---

## Requirements

- Python **3.10+**
- A virtual environment recommended
- GPU optional but strongly preferred for full suites; CPU is fine for `--quick` and small UI runs

---

## Installation

```bash
git clone https://github.com/<you>/kiqfl.git
cd kiqfl

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# PyTorch (pick the index for your platform / CUDA from pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -e .
# alternatively:
# pip install -r requirements.txt
```

After `pip install -e .`, the `kiqfl` console script is available (same as `python -m core`).

---

## Quick start

### 1) Smoke tests

```bash
python -m unittest discover -s test -v
```

### 2) Web UI

```bash
python -m webapp
# or:  python -m core web
# or:  kiqfl web
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Use **Quick Test** presets for short runs (few rounds / qubits). Heavy presets (many clients, CIFAR, 6–10 qubits) can take a long time on CPU.

### 3) Short CLI experiment

```bash
python -m core experiments --quick --suite main
python -m core analyze
```

Results land under `result/` (e.g. `experiments_log.csv`, summaries).

### 4) Full pipeline (long)

```bash
bash run_pipeline.sh
# same as: python -m core pipeline
```

On a gaming GPU this can take on the order of **~18–25 hours**. If interrupted:

```bash
python -m core resume
```

---

## CLI reference

```bash
python -m core --help
python -m core experiments --help
```

| Command | Purpose |
|---------|---------|
| `experiments` | Run suites (`main`, `ablation`, `noise`, …); supports `--quick` |
| `analyze` | Summarize `result/experiments_log.csv` |
| `pipeline` | Full multi-suite experiment pipeline |
| `resume` | Continue unfinished suites from checkpoints |
| `web` | Start the Flask UI on `127.0.0.1:5000` |

Examples:

```bash
python -m core experiments --quick --suite main
python -m core experiments --suite ablation --dataset mnist
kiqfl analyze
```

Useful experiment flags (see `--help` for the full list): `--dataset`, `--data-fraction`, `--num-clients`, `--epochs`, `--num-qubits`, `--num-layers`, `--nisq-error`.

---

## Method (short)

1. Server broadcasts \(\omega_{\mathrm{global}}\).  
2. Selected clients train a Hybrid CNN–QNN for \(K\) local epochs and upload parameters plus circuit states.  
3. Optional DTQW on a ring mixes guidance states; a KAN sets \(\alpha\) for local–walked blending.  
4. Server forms weights from pure-state fidelity  
   \(F(\lvert\psi_n\rangle,\lvert\psi_{\mathrm{global}}\rangle)=|\langle\psi_n\mid\psi_{\mathrm{global}}\rangle|^2\)  
   against the **previous** global circuit state, normalizes them, and aggregates **classical** parameters:  
   \(\omega_{\mathrm{global}} \leftarrow \sum_n w_n\,\omega_n\).

Details and figures: open **About** in the web UI.

---

## Reproducibility tips

- Prefer `--quick` while validating install and UI wiring.  
- Fix seeds via experiment config / runner defaults where applicable.  
- Keep `result/` and `data/` out of git (already in `.gitignore`).  
- Stop in the UI waits for the **current** global round/epoch to finish, then saves a config checkpoint you can resume from setup.

---

## Citation

If you use this code in academic work, please cite your thesis / paper and link this repository. Example BibTeX stub (fill in your details):

```bibtex
@mastersthesis{sharei_kiqfl,
  title   = {KAN-Enhanced Indirect Quantum Federated Learning},
  author  = {Sharei, Iman},
  school  = {<Your University>},
  year    = {2026},
  note    = {Code: https://github.com/<you>/kiqfl}
}
```

---

## License

[MIT](LICENSE) — Copyright (c) 2026 KIQFL contributors.

---

## Acknowledgements

Built for research on hybrid quantum–classical models in federated settings. Quantum layers are simulated classically; federated aggregation remains a standard central-server parameter average with fidelity-informed weights.
