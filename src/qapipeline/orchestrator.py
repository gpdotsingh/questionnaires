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
from .semantic import (
    load_semantics, normalize_text_with_semantics, parse_time_window_days,
    extract_state_filter, wants_group_by_state, requested_metrics, SemanticModel
)

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
    """Semantic-aware Orchestrator: normalize queries via a semantic model, then query CSV and/or Chroma."""
    def __init__(
        self,
        vectordb_path: str = "vectors",
        donations_csv: Optional[str] = "data/CRM_Donor_Simulation_Dataset.csv",
        semantic_path: Optional[str] = "semantics.yaml"
    ):
        self.vectordb_path = vectordb_path
        self.donations_csv = donations_csv
        self.semantic: SemanticModel = load_semantics(semantic_path)
        self.vectordb = None
        self._donations_mode: Optional[str] = None  # "tx" or "summary"

        if Chroma is not None and HuggingFaceEmbeddings is not None and Path(vectordb_path).exists():
            try:
                self.vectordb = Chroma(
                    persist_directory=vectordb_path,
                    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                )
            except Exception:
                self.vectordb = None

    # ---------- Domain guard ----------
    def _is_donor_entity_query(self, text: str) -> bool:
        t = text.lower()
        # After normalization, require 'donor' keyword present
        return "donor" in t or "donors" in t

    # ---------- CSV loading and normalization ----------
    @lru_cache(maxsize=1)
    def _load_donations(self) -> Optional[pd.DataFrame]:
        if not self.donations_csv:
            return None
        path = Path(self.donations_csv)
        if not path.exists():
            return None
        df = pd.read_csv(path)

        # Prefer semantic column mapping
        def col_or_default(dim_name: str, fallback: Optional[str]) -> Optional[str]:
            dim = self.semantic.dimensions.get(dim_name)
            return dim.column if dim else fallback

        donor_col = col_or_default("donor_id", None)
        date_col = col_or_default("donation_date", None)
        amount_col = col_or_default("amount", None)
        gifts_col = col_or_default("total_gifts", None)
        state_col = col_or_default("state", None)
        engagement_col = col_or_default("engagement_score", None)
        event_col = col_or_default("event_participation", None)

        # Validate presence; if missing, try fuzzy fallback
        cols = set(df.columns)
        def pick_one(cands: List[str]) -> Optional[str]:
            for c in cands:
                if c in cols:
                    return c
            return None

        donor_col = donor_col or pick_one(["DonorID","donor_id","Donor Id","ID"])
        date_col = date_col or pick_one(["LastDonationDate","donation_date","Date","GiftDate"])
        amount_col = amount_col or pick_one(["TotalAmountDonated","Amount","DonationAmount","Total"])
        gifts_col = gifts_col or pick_one(["TotalGifts","Gifts","GiftCount"])
        state_col = state_col or pick_one(["State"])
        engagement_col = engagement_col or pick_one(["EngagementScore","Engagement","Score"])
        event_col = event_col or pick_one(["EventParticipation","Participated","Event"])

        if not donor_col or not date_col or not amount_col:
            return None

        rename_map = {
            donor_col: "donor_id",
            date_col: "donation_date",
            amount_col: "amount",
        }
        if gifts_col: rename_map[gifts_col] = "total_gifts"
        if state_col: rename_map[state_col] = "state"
        if engagement_col: rename_map[engagement_col] = "engagement_score"
        if event_col: rename_map[event_col] = "event_participation"

        df = df.rename(columns=rename_map)
        df["donor_id"] = df["donor_id"].astype(str)
        df["donation_date"] = pd.to_datetime(df["donation_date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        if "state" in df: df["state"] = df["state"].astype(str).str.upper()
        if "engagement_score" in df: df["engagement_score"] = pd.to_numeric(df["engagement_score"], errors="coerce")
        if "total_gifts" in df: df["total_gifts"] = pd.to_numeric(df["total_gifts"], errors="coerce")
        if "event_participation" in df: df["event_participation"] = df["event_participation"].astype(str)

        df = df.dropna(subset=["donation_date","donor_id"])
        # Detect mode: tx rows vs per-donor summary (your CSV is per-donor summary)
        max_rows_per_donor = df.groupby("donor_id").size().max() if not df.empty else 0
        self._donations_mode = "tx" if max_rows_per_donor and max_rows_per_donor > 1 else "summary"
        return df

    # ---------- DB logic using semantics ----------
    def _answer_from_db(self, raw_text: str) -> Optional[str]:
        df = self._load_donations()
        if df is None:
            return None

        # Normalize the question with semantic synonyms
        norm_text = normalize_text_with_semantics(raw_text, self.semantic)

        if not self._is_donor_entity_query(norm_text):
            return None

        days = parse_time_window_days(norm_text, 90)
        cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=days))
        by_state = wants_group_by_state(norm_text)
        states_filter = extract_state_filter(raw_text)  # preserve uppercase state codes if present
        want_count, want_median = requested_metrics(norm_text)

        # Filter by time window
        recent = df[df["donation_date"] >= cutoff].copy()

        # Compute per-donor totals for tx mode; your dataset is summary, so this branch is short
        if self._donations_mode == "tx":
            per_donor_totals = recent.groupby("donor_id")["amount"].sum().rename("per_donor_total")
            recent = recent.merge(per_donor_totals, on="donor_id", how="left")
        else:
            # summary mode: amount already represents per-donor total
            recent["per_donor_total"] = recent["amount"]

        # Optional filter by states list (if provided)
        if states_filter and "state" in recent:
            recent = recent[recent["state"].isin(states_filter)]

        # Grouping
        if by_state and "state" in recent:
            grp = recent.groupby("state", dropna=False)
            parts: List[str] = []
            if want_count:
                donor_counts = grp["donor_id"].nunique().sort_values(ascending=False)
                # If no explicit states, show top 3
                donor_counts = donor_counts if states_filter else donor_counts.head(3)
                parts.append("Donor count by State: " + ", ".join(f"{k}={int(v)}" for k, v in donor_counts.items()))
            if want_median:
                medians = grp["per_donor_total"].median().sort_values(ascending=False)
                medians = medians if states_filter else medians.head(3)
                parts.append("Median per-donor total by State: " + ", ".join(f"{k}={float(v):.2f}" for k, v in medians.items()))
            if parts:
                return f"In the last {days} days:\n" + "\n".join(parts)
            # Fallback if metrics not detected
            donor_counts = grp["donor_id"].nunique().sort_values(ascending=False).head(3)
            medians = grp["per_donor_total"].median().sort_values(ascending=False).head(3)
            return f"In the last {days} days:\nDonor count by State: " + ", ".join(f"{k}={int(v)}" for k, v in donor_counts.items()) + \
                   "\nMedian per-donor total by State: " + ", ".join(f"{k}={float(v):.2f}" for k, v in medians.items())
        else:
            # No grouping
            parts: List[str] = []
            if want_count:
                donors = int(recent["donor_id"].nunique())
                parts.append(f"{donors} donors gave in the last {days} days.")
            if want_median:
                if recent.empty:
                    parts.append(f"No donors in the last {days} days.")
                else:
                    median_total = float(recent.groupby("donor_id")["per_donor_total"].max().median())
                    parts.append(f"Median per-donor total in the last {days} days is {median_total:.2f}.")
            if parts:
                return " ".join(parts)
            # Fallback default pair
            donors = int(recent["donor_id"].nunique())
            median_total = float(recent.groupby("donor_id")["per_donor_total"].max().median()) if not recent.empty else 0.0
            return f"In the last {days} days: donors={donors}; median per-donor total={median_total:.2f}."

    # ---------- Vector logic (prepend semantic context to the query) ----------
    def _answer_from_vectors(self, raw_text: str) -> Optional[str]:
        if self.vectordb is None:
            return None
        # Add short semantic context to steer retrieval
        semantic_context = "Dimensions: donor_id, donation_date, amount, total_gifts, state, engagement_score. " \
                           "Metrics: donor_count_last_n_days, median_total_last_n_days."
        query = f"{raw_text}\n\nSemantic context:\n{semantic_context}"
        try:
            docs = self.vectordb.similarity_search(query, k=3)
            if not docs:
                return None
            joined = "\n".join(getattr(d, "page_content", str(d)) for d in docs)
            return joined[:1500]
        except Exception:
            return None

    def _search_parallel(self, query: str) -> Tuple[str, str]:
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