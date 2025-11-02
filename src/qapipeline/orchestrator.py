# python
# filepath: /Users/gauravsingh/study/AI/DeependraBhaiyaproject/questionier/questionnaires/src/qapipeline/orchestrator.py
from __future__ import annotations
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re

import pandas as pd

from .contracts import OrchestratorInput, OrchestratorOutput, QAResult

# Resilient imports for Chroma and HuggingFaceEmbeddings
Chroma = None
HuggingFaceEmbeddings = None
try:
    from langchain_chroma import Chroma
except Exception:
    try:
        from langchain_community.vectorstores import Chroma
    except Exception:
        try:
            from langchain.vectorstores import Chroma
        except Exception:
            Chroma = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
        except Exception:
            HuggingFaceEmbeddings = None


class Orchestrator:
    """Answers sub-questions via DB (CSV) and/or Chroma, in parallel per step, with parent→child context injection."""
    def __init__(self, vectordb_path: str = "vectors", donations_csv: Optional[str] = "data/CRM_Donor_Simulation_Dataset.csv"):
        self.vectordb_path = vectordb_path
        self.donations_csv = donations_csv
        self.vectordb = None
        self._donations_mode: Optional[str] = None  # "tx" or "summary"

        # Lazy init vector DB if available and directory exists
        if Chroma is not None and HuggingFaceEmbeddings is not None and Path(vectordb_path).exists():
            try:
                self.vectordb = Chroma(
                    persist_directory=vectordb_path,
                    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                )
            except Exception:
                self.vectordb = None

    # ---------- CSV loading and normalization ----------
    @lru_cache(maxsize=1)
    def _load_donations(self) -> Optional[pd.DataFrame]:
        if not self.donations_csv:
            return None
        path = Path(self.donations_csv)
        if not path.exists():
            return None
        df = pd.read_csv(path)

        # Fuzzy map headers to donor_id / donation_date / amount
        cols = {c.lower().strip(): c for c in df.columns}

        def pick_col(cands: set, contains_any: Optional[set] = None) -> Optional[str]:
            for key in cols:
                if key in cands:
                    return cols[key]
            if contains_any:
                for key in cols:
                    if any(tok in key for tok in contains_any):
                        return cols[key]
            return None

        donor_col = pick_col({"donor_id", "donorid", "donor id", "donor", "id"}, contains_any={"donor"})
        date_col = pick_col({"donation_date", "date", "donated_at", "timestamp"}, contains_any={"last_donation_date", "last_gift_date", "gift_date"})
        amount_col = pick_col({"amount", "donation_amount", "total"}, contains_any={"total_donations_amount", "lifetime_amount", "total_giving", "totaldonationsamount"})

        if not donor_col or not date_col or not amount_col:
            return None

        df = df.rename(columns={donor_col: "donor_id", date_col: "donation_date", amount_col: "amount"})
        df["donor_id"] = df["donor_id"].astype(str)
        df["donation_date"] = pd.to_datetime(df["donation_date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df = df.dropna(subset=["donation_date", "donor_id"])

        # Detect mode: transaction rows vs per-donor summary
        max_rows_per_donor = df.groupby("donor_id").size().max() if not df.empty else 0
        self._donations_mode = "tx" if max_rows_per_donor and max_rows_per_donor > 1 else "summary"
        return df

    # ---------- Helpers ----------
    def _parse_days(self, text: str, default_days: int = 90) -> int:
        m = re.search(r"last\s+(\d+)\s*day", text.lower())
        return int(m.group(1)) if m else default_days

    # ---------- DB logic ----------
    def _answer_from_db(self, text: str) -> Optional[str]:
        df = self._load_donations()
        if df is None:
            return None

        t = text.lower()
        days = self._parse_days(t, 90)
        cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=days))

        # Q1: how many donors gave in the last N days
        if ("donor" in t and "last" in t and ("how many" in t or "count" in t)):
            if self._donations_mode == "tx":
                recent = df[df["donation_date"] >= cutoff]
                donors = int(recent["donor_id"].nunique())
                total_amt = float(recent["amount"].sum())
                return f"{donors} donors gave in the last {days} days (total amount {total_amt:.2f})."
            else:
                # summary per donor: filter by last donation date
                recent_donors = df[df["donation_date"] >= cutoff]
                donors = int(recent_donors["donor_id"].nunique())
                total_amt = float(recent_donors["amount"].sum())
                return f"{donors} donors have a last donation within {days} days (sum of their totals {total_amt:.2f})."

        # Q2: median donation total in last N days
        if "median" in t and "donation" in t and "total" in t:
            if self._donations_mode == "tx":
                recent = df[df["donation_date"] >= cutoff]
                if recent.empty:
                    return f"No donations in the last {days} days."
                per_donor_totals = recent.groupby("donor_id")["amount"].sum()
                median_total = float(per_donor_totals.median())
                donors = int(per_donor_totals.index.nunique())
                return f"Median per-donor total in the last {days} days is {median_total:.2f} across {donors} donors."
            else:
                # summary mode: median of total column for donors with recent last donation
                recent_donors = df[df["donation_date"] >= cutoff]
                if recent_donors.empty:
                    return f"No donors with a recent last donation in the last {days} days."
                median_total = float(recent_donors["amount"].median())
                donors = int(recent_donors["donor_id"].nunique())
                return f"Median of per-donor totals for donors with a last donation in {days} days is {median_total:.2f} across {donors} donors."

        return None

    # ---------- Vector logic ----------
    def _answer_from_vectors(self, text: str) -> Optional[str]:
        if self.vectordb is None:
            return None
        try:
            docs = self.vectordb.similarity_search(text, k=3)
            if not docs:
                return None
            joined = "\n".join(getattr(d, "page_content", str(d)) for d in docs)
            return joined[:1500]
        except Exception:
            return None

    def _search_parallel(self, query: str) -> Tuple[str, str]:
        """Run DB and Vector retrieval in parallel, return (answer, source)."""
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_db = ex.submit(self._answer_from_db, query)
            fut_vs = ex.submit(self._answer_from_vectors, query)
            db_ans = vs_ans = None
            for fut in as_completed([fut_db, fut_vs]):
                try:
                    val = fut.result()
                except Exception:
                    val = None
                if fut is fut_db:
                    db_ans = val
                else:
                    vs_ans = val

        # Prefer vector if present (as requested), fall back to DB, else default note
        if vs_ans:
            if db_ans:
                return f"{vs_ans}\n\n(DB check: {db_ans})", "both"
            return vs_ans, "chroma"
        if db_ans:
            return db_ans, "db"
        return "No data found.", "none"

    # ---------- Main run ----------
    def run(self, plan: OrchestratorInput) -> OrchestratorOutput:
        results: List[QAResult] = []
        errors: Dict[str, str] = {}
        parent_ctx: Dict[str, str] = {}

        for step in plan.ordered_steps:
            try:
                # Inject parent answers into the query for dependent steps
                query = step.text
                if getattr(step, "requires", None):
                    ctx = {pid: parent_ctx.get(pid, "") for pid in step.requires}
                    if any(ctx.values()):
                        query = f"{step.text}\n\nContext:\n{json.dumps(ctx)[:2000]}"

                answer, source = self._search_parallel(query)
                results.append(QAResult(
                    question_id=step.id,
                    question_text=step.text,
                    answer=answer,
                    meta={"source": source},
                ))
                parent_ctx[step.id] = answer
            except Exception as e:
                errors[step.id] = str(e)

        return OrchestratorOutput(results=results, errors=errors)