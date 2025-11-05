# python
from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import re

import numpy as np
import pandas as pd

from .contracts import OrchestratorInput, OrchestratorOutput, QAResult
from .semantic import (
    load_semantics,
    normalize_text_with_semantics,
    parse_time_window_days,
    extract_state_filter,
    wants_group_by_state,
    requested_metrics,
    SemanticModel,
)

# ---------- Resilient imports for Chroma and HF Embeddings ----------
Chroma = None
HuggingFaceEmbeddings = None
try:
    from langchain_chroma import Chroma as _Chroma
    Chroma = _Chroma
except Exception:
    try:
        from langchain_community.vectorstores import Chroma as _Chroma  # fallback
        Chroma = _Chroma
    except Exception:
        Chroma = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings as _HFEmb
    HuggingFaceEmbeddings = _HFEmb
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings as _HFEmb  # fallback
        HuggingFaceEmbeddings = _HFEmb
    except Exception:
        HuggingFaceEmbeddings = None


# ---------- LLM Planner (agent) ----------
def _parse_json_object(txt: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from text."""
    txt = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", txt, flags=re.I | re.S)
    m = re.search(r"\{.*\}", txt, flags=re.S)
    raw = m.group(0) if m else txt
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


class LLMPlanner:
    """Provider-flexible planner to convert text into a structured analytics plan."""
    def __init__(self, provider: Optional[str], model: Optional[str], debug: bool = False) -> None:
        self.provider = (provider or os.getenv("PLANNER_PROVIDER") or "ollama").lower()
        # Model preference by provider
        self.model = model or {
            "ollama": os.getenv("OLLAMA_GEN_MODEL"),
            "openai": os.getenv("OPENAI_MODEL"),
            "deepseek": os.getenv("DEEPSEEK_MODEL"),
        }.get(self.provider) or os.getenv("OLLAMA_GEN_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("DEEPSEEK_MODEL") or "llama3.1"
        self.debug = bool(debug)

        # Provider env
        self.ollama_base = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.hf_endpoint = os.getenv("HF_TEXTGEN_ENDPOINT")

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(f"[Planner] {msg}")

    def _prompt(self, text: str) -> str:
        return (
            "You are a data analysis agent. Convert the user request into a strict JSON plan.\n"
            "Output ONLY JSON with keys: metric (donor_count|total_donation_amount|total_gifts|engagement_score), "
            "grain (monthly|weekly|yearly), window_days (int, optional), check_decline (bool), check_increase (bool), "
            "group_by_state (bool), states (array of 2-letter codes, optional).\n"
            f"Request: {text}\n"
            "JSON:"
        )

    def plan(self, text: str) -> Optional[Dict[str, Any]]:
        prompt = self._prompt(text)
        self._dbg(f"provider={self.provider} model={self.model}")
        self._dbg(f"prompt: {prompt}")

        # Ollama
        if self.provider in ("ollama", "auto"):
            try:
                from langchain_ollama import OllamaLLM
                llm = OllamaLLM(model=self.model, temperature=0, base_url=self.ollama_base)
                out = llm.invoke(prompt).strip()
                self._dbg(f"raw out: {out[:500]}")
                js = _parse_json_object(out)
                if js:
                    return js
            except Exception as e:
                self._dbg(f"ollama error: {e}")

        # OpenAI
        if self.provider in ("openai", "auto"):
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", self.model), temperature=0, api_key=self.openai_key)
                msg = llm.invoke(prompt)
                out = getattr(msg, "content", str(msg)).strip()
                self._dbg(f"raw out: {out[:500]}")
                js = _parse_json_object(out)
                if js:
                    return js
            except Exception as e:
                self._dbg(f"openai error: {e}")

        # DeepSeek (OpenAI-compatible)
        if self.provider in ("deepseek", "auto"):
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=os.getenv("DEEPSEEK_MODEL", self.model),
                    temperature=0,
                    api_key=self.deepseek_key,
                    base_url=self.deepseek_base,
                )
                msg = llm.invoke(prompt)
                out = getattr(msg, "content", str(msg)).strip()
                self._dbg(f"raw out: {out[:500]}")
                js = _parse_json_object(out)
                if js:
                    return js
            except Exception as e:
                self._dbg(f"deepseek error: {e}")

        # HF endpoint (optional)
        if self.provider in ("hf", "huggingface", "hugging_face", "auto"):
            try:
                from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
                repo = self.hf_endpoint or self.model or "mistralai/Mixtral-8x7B-Instruct-v0.1"
                endpoint = HuggingFaceEndpoint(repo_id=repo, temperature=0)
                llm = ChatHuggingFace(llm=endpoint)
                msg = llm.invoke(prompt)
                out = getattr(msg, "content", str(msg)).strip()
                self._dbg(f"raw out: {out[:500]}")
                js = _parse_json_object(out)
                if js:
                    return js
            except Exception as e:
                self._dbg(f"huggingface error: {e}")

        return None


class Orchestrator:
    """Semantic-aware Orchestrator with agent planning, DB/vector search, and verbose logging."""
    def __init__(
        self,
        vectordb_path: Optional[str] = "vectors",
        donations_csv: Optional[str] = "data/CRM_Donor_Simulation_Dataset.csv",
        semantic_path: Optional[str] = None,
        *,
        relative_to_data: bool = True,
        debug: bool = False,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        # Paths
        self.vectordb_path = str(Path(vectordb_path).resolve()) if vectordb_path else None
        self.donations_csv = str(Path(donations_csv).resolve()) if donations_csv else None

        # Config
        self.relative_to_data: bool = bool(relative_to_data)
        self.debug: bool = bool(debug)

        # Logger
        def _dbg(msg: str) -> None:
            if self.debug:
                print(f"[Orchestrator] {msg}")
        self._dbg = _dbg
        self._dbg(f"Init vectordb_path={self.vectordb_path}, donations_csv={self.donations_csv}")

        # Semantic model
        self.semantic: SemanticModel = load_semantics(semantic_path if semantic_path else None)
        self._dbg(f"Semantic loaded (source={'yaml' if semantic_path else 'semantic.py'})")

        # Planner (agent)
        self.planner = LLMPlanner(provider=llm_provider or os.getenv("PLANNER_PROVIDER"), model=llm_model, debug=self.debug)

        # Vector store
        self.vectordb = None
        if self.vectordb_path and Path(self.vectordb_path).exists() and Chroma and HuggingFaceEmbeddings:
            try:
                emb_model = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                emb = HuggingFaceEmbeddings(model_name=emb_model)
                self.vectordb = Chroma(persist_directory=self.vectordb_path, embedding_function=emb)
                self._dbg(f"Opened Chroma at {self.vectordb_path}")
            except Exception as e:
                self._dbg(f"Chroma unavailable: {e}")
                self.vectordb = None
        else:
            self._dbg("Vector DB not opened (missing path or embedding libs).")

        # Mode detection (set after CSV load)
        self._donations_mode: str = "summary"

    # ---------------- Utilities ----------------

    def _stringify_meta(self, meta: Dict[str, Any]) -> Dict[str, str]:
        """Ensure all meta values are strings to satisfy QAResult schema."""
        out: Dict[str, str] = {}
        for k, v in (meta or {}).items():
            if isinstance(v, str):
                out[k] = v
            elif isinstance(v, (dict, list)):
                out[k] = json.dumps(v)
            elif v is None:
                out[k] = ""
            else:
                out[k] = str(v)
        return out

    # ---------------- CSV loading and normalization ----------------

    @lru_cache(maxsize=1)
    def _load_donations(self) -> Optional[pd.DataFrame]:
        self._dbg(f"Loading CSV from: {self.donations_csv}")
        if not self.donations_csv or not Path(self.donations_csv).exists():
            self._dbg("CSV not found.")
            return None
        try:
            df = pd.read_csv(self.donations_csv)
        except Exception as e:
            self._dbg(f"Failed to read CSV: {e}")
            return None

        cols = set(df.columns)

        def first_present(names: List[str]) -> Optional[str]:
            for n in names:
                if n in cols:
                    return n
            return None

        dim = getattr(self.semantic, "dimensions", {}) or {}
        donor_hint = getattr(dim.get("donor_id"), "column", None) if "donor_id" in dim else None
        date_hint = getattr(dim.get("donation_date"), "column", None) if "donation_date" in dim else None
        amt_hint = getattr(dim.get("amount"), "column", None) if "amount" in dim else None
        gifts_hint = getattr(dim.get("total_gifts"), "column", None) if "total_gifts" in dim else None
        state_hint = getattr(dim.get("state"), "column", None) if "state" in dim else None
        engage_hint = getattr(dim.get("engagement_score"), "column", None) if "engagement_score" in dim else None
        event_hint = getattr(dim.get("event_participation"), "column", None) if "event_participation" in dim else None

        donor_col = donor_hint or first_present(["DonorID", "donor_id", "Donor Id", "ID"])
        date_col = date_hint or first_present(["LastDonationDate", "donation_date", "Date", "GiftDate"])
        amount_col = amt_hint or first_present(["TotalAmountDonated", "amount", "Amount", "DonationAmount", "Total"])
        gifts_col = gifts_hint or first_present(["TotalGifts", "total_gifts", "Gifts", "GiftCount"])
        state_col = state_hint or first_present(["State", "state"])
        engage_col = engage_hint or first_present(["EngagementScore", "engagement_score", "Engagement", "Score"])
        event_col = event_hint or first_present(["EventParticipation", "event_participation", "Participated", "Event"])

        if not donor_col or not date_col or not amount_col:
            self._dbg("CSV missing required columns (donor_id, donation_date, amount)")
            return None

        rename_map: Dict[str, str] = {donor_col: "donor_id", date_col: "donation_date", amount_col: "amount"}
        if gifts_col:
            rename_map[gifts_col] = "total_gifts"
        if state_col:
            rename_map[state_col] = "state"
        if engage_col:
            rename_map[engage_col] = "engagement_score"
        if event_col:
            rename_map[event_col] = "event_participation"
        self._dbg(f"Column mapping: {rename_map}")

        df = df.rename(columns=rename_map)

        df["donor_id"] = df["donor_id"].astype(str)
        df["donation_date"] = pd.to_datetime(df["donation_date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        if "total_gifts" in df.columns:
            df["total_gifts"] = pd.to_numeric(df["total_gifts"], errors="coerce")
        if "state" in df.columns:
            df["state"] = df["state"].astype(str).str.upper()
        if "engagement_score" in df.columns:
            df["engagement_score"] = pd.to_numeric(df["engagement_score"], errors="coerce")
        if "event_participation" in df.columns:
            df["event_participation"] = df["event_participation"].astype(str)

        df = df.dropna(subset=["donation_date", "donor_id"])

        max_rows_per_donor = df.groupby("donor_id").size().max() if not df.empty else 0
        self._donations_mode = "tx" if (max_rows_per_donor and max_rows_per_donor > 1) else "summary"
        present = [c for c in ["donor_id", "donation_date", "amount", "total_gifts", "state", "engagement_score"] if c in df.columns]
        self._dbg(f"CSV loaded: rows={len(df)}, mode={self._donations_mode}, cols={present}")
        return df

    # ---------------- Helpers for DB answering ----------------

    def _compute_cutoff(self, df: pd.DataFrame, days: int) -> Tuple[pd.Timestamp, Dict[str, str]]:
        if "donation_date" not in df.columns:
            ts = pd.Timestamp(datetime.utcnow())
            meta = {
                "relative_to_data": str(self.relative_to_data),
                "cutoff": ts.date().isoformat(),
                "days": str(days),
            }
            return ts, meta

        if self.relative_to_data and not df["donation_date"].isna().all():
            max_date = df["donation_date"].max()
            cutoff = max_date - pd.Timedelta(days=days)
        else:
            cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=days))

        meta = {
            "relative_to_data": str(self.relative_to_data),
            "cutoff": cutoff.date().isoformat(),
            "days": str(days),
        }
        self._dbg(f"Cutoff computed: {meta}")
        return cutoff, meta

    def _detect_instruction(self, text: str) -> Optional[Dict[str, Any]]:
        """Regex fallback if agent is unavailable."""
        t = text.lower()
        m = re.search(r"compute\s+([a-z_]+)\s*(?:\(([^)]+)\))?.*?(?:over time|trend)", t)
        if m:
            metric = m.group(1).strip()
            grain = (m.group(2) or "monthly").strip().lower()
            check_decline = "check decline" in t or "decline" in t
            check_increase = "check increase" in t or "increase" in t
            return {"metric": metric, "grain": grain, "check_decline": check_decline, "check_increase": check_increase}

        m = re.search(
            r"compute\s+([a-z_]+)\s+(monthly|weekly|yearly)\s+over\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)",
            t,
        )
        if m:
            metric = m.group(1).strip()
            grain = m.group(2).strip().lower()
            n = int(m.group(3))
            unit = m.group(4)
            days = n * 365 if "year" in unit else n * 30 if "month" in unit else n * 7 if "week" in unit else n
            return {"metric": metric, "grain": grain, "check_decline": ("decline" in t), "check_increase": ("increase" in t), "window_days": days}

        m = re.search(r"compute\s+([a-z_]+)\s+(monthly|weekly|yearly)\b", t)
        if m:
            metric = m.group(1).strip()
            grain = m.group(2).strip().lower()
            return {"metric": metric, "grain": grain, "check_decline": ("decline" in t), "check_increase": ("increase" in t)}

        return None

    def _series_for_metric(
        self, df: pd.DataFrame, metric: str, grain: str
    ) -> Tuple[Optional[pd.Series], bool, str]:
        if df.empty or "donation_date" not in df.columns:
            return None, False, "No dated records."

        df = df.copy()
        df["donation_date"] = pd.to_datetime(df["donation_date"], errors="coerce")
        df = df.dropna(subset=["donation_date"])
        df = df.set_index("donation_date").sort_index()

        rule = {"monthly": "MS", "weekly": "W-MON", "yearly": "YS"}.get(grain, "MS")

        if metric == "donor_count":
            s = df.groupby(pd.Grouper(freq=rule))["donor_id"].nunique()
            return s, False, ""

        if metric == "total_donation_amount":
            if self._donations_mode != "tx":
                return None, True, "Donor-level summary file; need gift-level (one row per donation) to compute period totals."
            s = df["amount"].resample(rule).sum()
            return s, False, ""

        if metric == "total_gifts":
            if self._donations_mode != "tx" or "total_gifts" not in df.columns:
                return None, True, "Gift counts per period unavailable in summary dataset."
            s = df["total_gifts"].resample(rule).sum()
            return s, False, ""

        if metric == "engagement_score":
            if "engagement_score" not in df.columns:
                return None, False, "Engagement score not present."
            s = df["engagement_score"].resample(rule).mean()
            return s, (self._donations_mode == "summary"), "Proxy: mean among donors active in period."

        return None, False, f"Unsupported metric: {metric}"

    def _trend_check(self, s: pd.Series) -> Dict[str, Any]:
        out: Dict[str, Any] = {"slope": 0.0, "pct_change": None, "decline": False, "increase": False}
        s = s.dropna()
        if len(s) < 4:
            return out
        y = s.values.astype(float)
        x = np.arange(len(y), dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        out["slope"] = slope
        last = float(np.mean(y[-3:])) if len(y) >= 3 else float(y[-1])
        prev = float(np.mean(y[-6:-3])) if len(y) >= 6 else float(np.mean(y[:-3])) if len(y) > 3 else last
        pct = None if prev == 0 else (last - prev) / prev
        out["pct_change"] = pct
        if pct is not None:
            out["decline"] = pct < -0.05 or (slope < 0 and last < prev)
            out["increase"] = pct > 0.05 or (slope > 0 and last > prev)
        else:
            out["decline"] = slope < 0
            out["increase"] = slope > 0
        return out

    def _humanize_trend(
        self, metric: str, grain: str, s: pd.Series, check_decline: bool, check_increase: bool, is_proxy: bool, note: str
    ) -> str:
        if s is None or s.dropna().empty:
            return "No data found."
        s = s.sort_index().dropna()
        if len(s) > 24:
            s = s.tail(24)
        tail = s.tail(6)
        last_val = float(tail.iloc[-1])
        start = tail.index[0].date().isoformat()
        end = tail.index[-1].date().isoformat()
        tc = self._trend_check(s)
        pct = tc["pct_change"]
        change_txt = "n/a" if pct is None else f"{pct*100:.1f}%"
        flag = ""
        if check_decline and tc["decline"]:
            flag = " A decline is detected."
        elif check_increase and tc["increase"]:
            flag = " An increase is detected."
        elif check_decline or check_increase:
            flag = " No significant change detected."
        proxy_note = f" Note: {note}" if is_proxy and note else (" Note: " + note if note else "")
        return f"{metric} ({grain}) from {start} to {end}: last={last_val:.2f}, change vs prior window={change_txt}.{flag}{proxy_note}"

    def _summarize_combined(self, parent_ctx: Dict[str, str]) -> str:
        qids = sorted(parent_ctx.keys())
        texts = [parent_ctx[qid] for qid in qids if parent_ctx[qid]]
        if not texts:
            return "No prior insights to summarize."
        summary_bits: List[str] = []
        declines = [("decline is detected" in t.lower()) for t in texts]
        if any(declines):
            if all(declines):
                summary_bits.append("Both metrics show a decline.")
            else:
                summary_bits.append("One metric declines while the other is flat or improving.")
        for t in texts[:2]:
            m = re.search(r"([a-z_]+) \((monthly|weekly|yearly)\) .*?change vs prior window=([-.\d%na/]+)\.", t, flags=re.I)
            if m:
                summary_bits.append(f"{m.group(1)} {m.group(2)} change {m.group(3)}.")
        return " ".join(summary_bits) if summary_bits else "Combined view: no strong signal across metrics."

    # ---------------- DB answering ----------------

    def _answer_from_db(self, raw_text: str, agent_plan: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Dict[str, str]]:
        self._dbg(f"DB path: raw='{raw_text}'")
        df = self._load_donations()
        if df is None or df.empty:
            self._dbg("DB path: DataFrame is None or empty.")
            return None, {}

        norm_text = normalize_text_with_semantics(raw_text, self.semantic)
        self._dbg(f"DB path: normalized='{norm_text}'")

        # Prefer agent plan
        instr = None
        if agent_plan:
            instr = {
                "metric": str(agent_plan.get("metric", "")).strip().lower(),
                "grain": str(agent_plan.get("grain", "monthly")).strip().lower(),
                "check_decline": bool(agent_plan.get("check_decline", False)),
                "check_increase": bool(agent_plan.get("check_increase", False)),
            }
            if "window_days" in agent_plan:
                instr["window_days"] = int(agent_plan["window_days"])
            self._dbg(f"DB plan (agent): {instr}")
        else:
            instr = self._detect_instruction(norm_text)
            self._dbg(f"DB plan (regex): {instr}")

        if instr is not None and instr.get("metric"):
            metric = instr["metric"]
            grain = instr["grain"]
            check_decl = instr.get("check_decline", False)
            check_incr = instr.get("check_increase", False)
            series, is_proxy, note = self._series_for_metric(df, metric, grain)
            latest_cut = df["donation_date"].max().date().isoformat() if "donation_date" in df.columns else ""
            meta = {
                "relative_to_data": str(self.relative_to_data),
                "cutoff": latest_cut,
                "days": str(int(instr.get("window_days", 0))),
            }
            if series is not None:
                sdrop = series.dropna()
                self._dbg(f"Series head:\n{sdrop.head().to_string() if not sdrop.empty else '<empty>'}")
                self._dbg(f"Series tail:\n{sdrop.tail().to_string() if not sdrop.empty else '<empty>'}")
            if series is None or series.dropna().empty:
                if is_proxy:
                    msg = (
                        f"Cannot compute {metric} ({grain}) trend from a donor-level summary file. "
                        "Please provide gift-level data (one row per donation) to compute period totals."
                    )
                    self._dbg(f"DB result: {msg}")
                    return msg, self._stringify_meta(meta)
                self._dbg("DB result: No data found.")
                return "No data found.", self._stringify_meta(meta)

            text = self._humanize_trend(metric, grain, series, check_decl, check_incr, is_proxy, note)
            self._dbg(f"DB result: {text}")
            return text, self._stringify_meta(meta)

        # Fallback: last-N-days donor stats with optional grouping
        days = parse_time_window_days(norm_text, 90)
        if agent_plan and "window_days" in agent_plan:
            days = int(agent_plan["window_days"])
        by_state = wants_group_by_state(norm_text) or bool(agent_plan and agent_plan.get("group_by_state"))
        states_filter = [s.upper() for s in (agent_plan.get("states", []) if agent_plan else (extract_state_filter(raw_text) or []))]
        want_count, want_median = requested_metrics(norm_text)
        if not (want_count or want_median):
            want_count, want_median = True, True

        self._dbg(f"Fallback plan: days={days}, by_state={by_state}, states={states_filter}, want_count={want_count}, want_median={want_median}")

        cutoff, meta = self._compute_cutoff(df, days)
        recent = df[df["donation_date"] >= cutoff].copy() if "donation_date" in df.columns else df.copy()
        self._dbg(f"Fallback filter: donation_date >= {meta.get('cutoff')} -> rows={len(recent)}")

        if "amount" in recent.columns:
            if self._donations_mode == "tx":
                per_donor_totals = recent.groupby("donor_id")["amount"].sum().rename("per_donor_total")
                recent = recent.merge(per_donor_totals, on="donor_id", how="left")
            else:
                recent["per_donor_total"] = recent["amount"]
        else:
            recent["per_donor_total"] = 0.0

        if states_filter and "state" in recent.columns:
            recent = recent[recent["state"].isin(states_filter)]
            self._dbg(f"State filter applied: {states_filter} -> rows={len(recent)}")

        if by_state and "state" in recent.columns and not recent.empty:
            grp = recent.groupby("state", dropna=False)
            parts: List[str] = []
            if want_count:
                donor_counts = grp["donor_id"].nunique().sort_values(ascending=False)
                donor_counts = donor_counts if states_filter else donor_counts.head(3)
                parts.append("Donor count by state: " + ", ".join(f"{k}={int(v)}" for k, v in donor_counts.items()))
            if want_median and "per_donor_total" in recent.columns:
                medians = grp["per_donor_total"].median().sort_values(ascending=False)
                medians = medians if states_filter else medians.head(3)
                parts.append("Median per-donor total by state: " + ", ".join(f"{k}={float(v):.2f}" for k, v in medians.items()))
            final = f"In the last {days} days: " + " ".join(parts)
            self._dbg(f"DB result: {final}")
            return final, self._stringify_meta(meta)

        if recent.empty:
            msg = f"No donors in the last {days} days."
            self._dbg(f"DB result: {msg}")
            return msg, self._stringify_meta(meta)

        parts: List[str] = []
        if want_count:
            donors = int(recent["donor_id"].nunique()) if "donor_id" in recent.columns else 0
            parts.append(f"{donors} donors gave in the last {days} days.")
        if want_median:
            if "donor_id" in recent.columns:
                median_total = float(recent.groupby("donor_id")["per_donor_total"].max().median())
            else:
                median_total = 0.0
            parts.append(f"Median per-donor total is {median_total:.2f}.")
        final = " ".join(parts) if parts else f"In the last {days} days: donors={int(recent['donor_id'].nunique()) if 'donor_id' in recent.columns else 0}; median per-donor total={float(recent.groupby('donor_id')['per_donor_total'].max().median()) if 'donor_id' in recent.columns else 0.0:.2f}."
        self._dbg(f"DB result: {final}")
        return final, self._stringify_meta(meta)

    # ---------------- Vector answering ----------------

    def _answer_from_vectors(self, raw_text: str, agent_plan: Optional[Dict[str, Any]]) -> Optional[str]:
        if self.vectordb is None:
            self._dbg("Vector path skipped: vectordb is None.")
            return None

        semantic_context = (
            "Dimensions: donor_id, donation_date, amount, total_gifts, state, engagement_score. "
            "Metrics: donor_count_last_n_days, median_total_last_n_days."
        )
        plan_part = f"Plan: {json.dumps(agent_plan)}" if agent_plan else ""
        query = f"{raw_text}\n\n{plan_part}\n\nSemantic context:\n{semantic_context}"
        self._dbg(f"Vector query(k=3): {query[:400].replace(chr(10),' ')}...")

        try:
            docs = self.vectordb.similarity_search(query, k=3)
            self._dbg(f"Vector hits={len(docs)}")
            for i, d in enumerate(docs or []):
                content = getattr(d, "page_content", str(d)) or ""
                meta = getattr(d, "metadata", {}) or {}
                self._dbg(f"Doc[{i}] meta={meta} content={content[:200].replace(chr(10),' ')}...")
        except Exception as e:
            self._dbg(f"Vector search failed: {e}")
            return None

        if not docs:
            return None

        joined = "\n".join(getattr(d, "page_content", str(d)) for d in docs)
        return joined[:1000] if joined else None

    # ---------------- Search orchestration ----------------

    def _search_parallel(self, query: str, agent_plan: Optional[Dict[str, Any]]) -> Tuple[str, str, Dict[str, str]]:
        self._dbg(f"Search start for: {query}")
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_db = ex.submit(self._answer_from_db, query, agent_plan)
            fut_vs = ex.submit(self._answer_from_vectors, query, agent_plan)

            db_text: Optional[str] = None
            db_meta: Dict[str, str] = {}
            vs_text: Optional[str] = None

            for fut in as_completed([fut_db, fut_vs]):
                try:
                    res = fut.result()
                except Exception as e:
                    self._dbg(f"Search task error: {e}")
                    res = None

                if fut is fut_db:
                    if isinstance(res, tuple):
                        db_text, db_meta = res
                else:
                    vs_text = res if isinstance(res, str) else None

        if db_text:
            if vs_text:
                self._dbg("Search result source=both (preferring DB)")
                return f"{db_text}\n\n(Vector context: {vs_text[:400]})", "both", db_meta
            self._dbg("Search result source=db")
            return db_text, "db", db_meta
        if vs_text:
            self._dbg("Search result source=chroma (no DB answer)")
            return vs_text, "chroma", db_meta
        self._dbg("Search result source=none")
        return "No data found.", "none", db_meta

    # ---------------- Public API ----------------

    def run(self, plan: OrchestratorInput) -> OrchestratorOutput:
        results: List[QAResult] = []
        errors: Dict[str, str] = {}
        parent_ctx: Dict[str, str] = {}

        for step in plan.ordered_steps:
            try:
                self._dbg(f"Step {step.id}: {step.text}")

                # Summarization step
                if "summarize combined insights" in step.text.lower():
                    text = self._summarize_combined(parent_ctx)
                    meta = {"source": "summary"}
                    results.append(QAResult(
                        question_id=step.id,
                        question_text=step.text,
                        answer=text,
                        meta=meta,
                    ))
                    parent_ctx[step.id] = text
                    self._dbg(f"Step {step.id} summary meta={meta}")
                    continue

                # Agent planning
                agent_plan = self.planner.plan(step.text)
                self._dbg(f"Agent plan: {agent_plan}")

                # Effective query with context (for vectors reference/logs)
                query = step.text
                if getattr(step, "requires", None):
                    ctx = {pid: parent_ctx.get(pid, "") for pid in step.requires}
                    if any(bool(v) for v in ctx.values()):
                        query = f"{step.text}\n\nContext:\n{json.dumps(ctx)[:2000]}"
                self._dbg(f"Effective query: {query}")

                text, source, db_meta = self._search_parallel(query, agent_plan)

                # Ensure meta types are strings
                meta: Dict[str, str] = {"source": str(source)}
                if source in {"db", "both"} and db_meta:
                    meta.update(self._stringify_meta(db_meta))

                results.append(
                    QAResult(
                        question_id=step.id,
                        question_text=step.text,
                        answer=text,
                        meta=meta,
                    )
                )
                parent_ctx[step.id] = text
                self._dbg(f"Step {step.id} done: source={source} meta={meta}")

            except Exception as e:
                errors[step.id] = str(e)
                self._dbg(f"Step {step.id} error: {e}")

        return OrchestratorOutput(results=results, errors=errors)