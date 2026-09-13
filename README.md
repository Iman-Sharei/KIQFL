<div align="center">

# ⚛️ KIQFL

### KAN-Enhanced Indirect Quantum Federated Learning

**Hybrid quantum–classical federated learning** with fidelity-guided aggregation  

🖥️ Central-server FL · 🧠 Hybrid CNN–QNN · 🌀 Ring DTQW · 🧩 KAN · 🔥 PyTorch + Flask

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-UI-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-5F8F73?style=flat-square)](LICENSE)

```text
  📱 Clients (private data)  ──upload ωₙ, |ψₙ⟩──►  🖥️ Central Server
           ▲                                            │
           └──────── broadcast ω_global ────────────────┘
                           ▲
         🌀 DTQW + 🧩 KAN α  →  📐 fidelity weights  →  📊 aggregate
```

</div>

---

## 📖 Overview

KIQFL is research code for studying **federated learning** with hybrid quantum–classical client models.

| | Piece | Role |
|:---:|:------|:-----|
| 🖥️ | **Central server** | Broadcasts the global model, collects updates, aggregates parameters |
| 🧠 | **Hybrid CNN–QNN** | CNN features → amplitude encoding → variational circuit → classical head |
| 📐 | **Fidelity weights** | \(F=\lvert\langle\psi_n\mid\psi_{\mathrm{global}}\rangle\rvert^2\) vs the *previous* global circuit state |
| 🌀 | **DTQW + KAN** | Optional ring walk mixes *guidance states*; KAN sets mixing strength \(\alpha\) |

> ⚠️ **Scope.** Classical **state-vector simulation** (optional NISQ-style noise).  
> ❌ Not peer-to-peer parameter exchange. Not a claim of real-QPU wall-clock speedup.  
> 🔒 Formal privacy still needs DP (optional Opacus)—not the uncertainty principle alone.

---

## ✨ Features

- 🖥️ Central-server FL with Hybrid CNN–QNN clients  
- 🎯 Fidelity-guided aggregation (KIQFL) plus baselines: **FedAvg**, **FQNGD**, **centralized**  
- ⌨️ Experiment CLI (`experiments`, `analyze`, `pipeline`, `resume`)  
- 🌐 Local **Flask** web UI for configure / simulate / compare  
- 📦 Datasets via torchvision: MNIST, Fashion-MNIST, CIFAR-10/100 (auto-download)  
- 🛡️ Optional differential privacy hooks (Opacus)

---

## 📁 Repository layout

```text
KIQFL/
├── core/                  # 📚 Library + CLI
│   ├── quantum/           # ⚛️ Circuit · fidelity · DTQW · KAN
│   ├── models/            # 🧠 Hybrid CNN–QNN · classical models
│   ├── fl/                # 🔗 Data · aggregation · algorithms · metrics
│   └── experiments/       # 🧪 Runner · analyze · pipeline · resume
├── webapp/                # 🌐 Flask UI (templates + static)
├── test/                  # ✅ Smoke / unit tests
├── run_pipeline.sh        # 🚀 Full multi-suite helper
├── requirements.txt
├── pyproject.toml
└── README.md
```

| Path | Created at runtime (gitignored) |
|:-----|:--------------------------------|
| 📥 `data/` | Torchvision downloads |
| 📊 `result/` | Experiment CSVs, summaries, checkpoints |

---

## 📋 Requirements

- 🐍 **Python** ≥ 3.10  
- 📦 Virtual environment recommended  
- 🎮 **GPU** optional; CPU is fine for `--quick` and small UI runs  

---

## ⚙️ Installation

```bash
git clone https://github.com/Iman-Sharei/KIQFL.git
cd KIQFL

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install PyTorch for your platform/CUDA: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -e .
# or:  pip install -r requirements.txt
```

After editable install, the `kiqfl` command mirrors `python -m core`.

---

## 🚀 Quick start

<table>
<tr>
<td width="50%">

### ✅ Tests

```bash
python -m unittest discover -s test -v
```

### 🌐 Web UI

```bash
python -m webapp
# kiqfl web
```

Open **http://127.0.0.1:5000**  
Use **Quick Test** presets for short runs.

</td>
<td width="50%">

### ⚡ Short CLI run

```bash
python -m core experiments --quick --suite main
python -m core analyze
```

### 🏁 Full pipeline *(long)*

```bash
bash run_pipeline.sh
# ~18–25h on a gaming GPU

python -m core resume   # if interrupted
```

</td>
</tr>
</table>

---

## ⌨️ CLI

```bash
python -m core --help
python -m core experiments --help
```

| Command | Purpose |
|:--------|:--------|
| 🧪 `experiments` | Run suites (`main`, `ablation`, `noise`, …); supports `--quick` |
| 📈 `analyze` | Summarize `result/experiments_log.csv` |
| 🏁 `pipeline` | Full multi-suite experiment pipeline |
| ▶️ `resume` | Continue unfinished suites from checkpoints |
| 🌐 `web` | Start Flask UI on `127.0.0.1:5000` |

```bash
# examples
python -m core experiments --quick --suite main
python -m core experiments --suite ablation --dataset mnist
kiqfl analyze
kiqfl web
```

Useful flags: `--dataset` · `--data-fraction` · `--num-clients` · `--epochs` · `--num-qubits` · `--num-layers` · `--nisq-error`

---

## 🔬 Method (one round)

1. 📡 Server broadcasts \(\omega_{\mathrm{global}}\)  
2. 📱 Clients train Hybrid CNN–QNN for \(K\) local epochs; upload \(\omega_n\) and \(\lvert\psi_n\rangle\)  
3. 🌀 Optional DTQW on a ring + 🧩 KAN \(\alpha\) blend local and walked guidance states  
4. 📐 Weights \(w_n \propto F(\lvert\psi_n\rangle,\lvert\psi_{\mathrm{global}}\rangle)\), then  
   \(\displaystyle \omega_{\mathrm{global}} \leftarrow \sum_n w_n\,\omega_n\)  
   (classical parameter average; fidelity uses the **previous** global circuit state)

More detail: **About** page in the web UI 📖

---

## 🔁 Reproducibility tips

- ✅ Validate install with `--quick` before long runs  
- 🙈 Keep `data/` and `result/` out of git (already ignored)  
- ⏹ UI **Stop** finishes the *current* global round/epoch, then saves a resumeable config  

---

## 📚 Citation

If you use this code, please cite your thesis/paper and link this repository:

```bibtex
@mastersthesis{sharei_kiqfl,
  title   = {KAN-Enhanced Indirect Quantum Federated Learning},
  author  = {Sharei, Iman},
  school  = {Allameh Tabataba'i University},
  year    = {2026},
  note    = {Code: https://github.com/Iman-Sharei/KIQFL}
}
```

---

## 📄 License

Released under the [MIT License](LICENSE).

---

<div align="center">

⚛️ **KIQFL** · Quantum-inspired federated learning · Classical simulation for research

</div>
