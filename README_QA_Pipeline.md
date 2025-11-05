
# Questionnaires QA Pipeline

End-to-end QA over donor CSV data with optional vector search and semantic normalization.

## Core Components
- **Core orchestrator**: `qapipeline.Orchestrator`
- **Data contracts**: `qapipeline.SubQuestion`, `qapipeline.SplitPlan`
- **Vector ingest**: `ingest_vectors.py`
- **Demo/entrypoints**: `main_demo.py`, `main.py`
- **Server**: `server.py`
- **Sample dataset path**: `CRM_Donor_Simulation_Dataset.csv`
- **Python deps**: `requirements.txt`

---

## Prerequisites

- Python 3.10+
- Optional: local embeddings via `langchain_huggingface`
- Optional: local vector store via `langchain_chroma` (used in-process, no server required)
- Optional: Ollama running locally if you enable LLM-based splitting

---

## Setup

Create virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure Python can import the src package:
```bash
export PYTHONPATH="$(pwd)/questionnaires/src"
```

---

## Environment Variables

### Data and Vectors:
```bash
export DATA_CSV="questionnaires/data/CRM_Donor_Simulation_Dataset.csv"
export VECTORS_DIR="questionnaires/vectors"
```

### LLM Splitting (optional):
```bash
export SPLITTER_TRY_LLM="1"              # enable LLM-backed splitting if wired
export LLM_PROVIDER="ollama"             # or: chatgpt, deepseek (if implemented)
export LLM_MODEL="llama3.1"              # model name for provider
export OLLAMA_HOST="http://localhost:11434"
```

### Embeddings (optional):
```bash
export HF_EMBEDDINGS_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

---

## Prepare Vectors

Ingest the CSV into a local Chroma directory store:
```bash
python ingest_vectors.py
```

This will populate vectors which the orchestrator uses by default.

---

## Run the Demo

```bash
python main_demo.py
```

Or:
```bash
python main.py
```

---

## Start the Server

```bash
python server.py
```

See `server.py` for the exact routes and port.

---

## Programmatic Use

Minimal scaffold:
```python
import os
os.environ.setdefault("PYTHONPATH", "<repo>/questionnaires/src")
from qapipeline.orchestrator import Orchestrator

orch = Orchestrator(
    vectordb_path=os.getenv("VECTORS_DIR", "questionnaires/vectors"),
    donations_csv=os.getenv("DATA_CSV", "questionnaires/data/CRM_Donor_Simulation_Dataset.csv"),
    debug=True,
)
```

Your app should build a `SplitPlan` and feed it to the orchestrator.
See contracts for shapes:
- `qapipeline.SplitPlan`
- `qapipeline.SubQuestion`

And mirror the usage in `main_demo.py`.

---

## Example Questions and Output

**Questions you can try:**
- compute total_donation_amount trend & check decline
- top donors in OH in the last 90 days
- count donors by state with median donation

**Typical Output (JSON-like):**
```json
{
  "answer": "Summary for: compute total_donation_amount trend & check decline ...",
  "used_llm_in_splitter": false,
  "plan_steps": ["Q1: compute total_donation_amount trend & check decline"],
  "validation_score": 1
}
```

> If `used_llm_in_splitter=false`, the splitter used rule-based heuristics.

To try Ollama:
```bash
export SPLITTER_TRY_LLM=1
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3.1
```

---

## Notes

- Vector results depend on vectors being present and aligned with `CRM_Donor_Simulation_Dataset.csv`.

---

## Files Reference

- **Orchestrator**: `qapipeline.Orchestrator`
- **Contracts**: `qapipeline.SubQuestion`, `qapipeline.SplitPlan`
- **Demo**: `main_demo.py`
- **Server**: `server.py`
- **Ingest**: `ingest_vectors.py`
- **Requirements**: `requirements.txt`


Open it with ⌘⇧P (Mac).