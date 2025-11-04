# python
# filepath: /Users/gauravsingh/study/AI/DeependraBhaiyaproject/questionier/questionnaires/src/qapipeline/semantic.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from pathlib import Path
import re

try:
    import yaml  # optional; if missing we’ll use built-in defaults
except Exception:
    yaml = None

class Dimension(BaseModel):
    name: str
    column: str
    type: str = "categorical"  # categorical | time | numeric | boolean
    synonyms: List[str] = []

class Metric(BaseModel):
    name: str
    description: str = ""
    # Human-readable; we’ll implement logic for a few common metrics in Orchestrator
    expression: str = ""
    synonyms: List[str] = []

class SemanticModel(BaseModel):
    dataset: str = "crm_donor"
    id_dimension: str = "donor_id"
    time_dimension: str = "donation_date"
    dimensions: Dict[str, Dimension] = {}
    metrics: Dict[str, Metric] = {}
    synonyms: Dict[str, str] = {}  # token → canonical term

    def all_synonyms(self) -> Dict[str, str]:
        mapping = dict(self.synonyms)
        for dim in self.dimensions.values():
            mapping[dim.name.lower()] = dim.name
            for s in dim.synonyms:
                mapping[s.lower()] = dim.name
        for met in self.metrics.values():
            mapping[met.name.lower()] = met.name
            for s in met.synonyms:
                mapping[s.lower()] = met.name
        return mapping

def default_crm_semantics() -> SemanticModel:
    # Canonical names we’ll use internally:
    # donor_id, donation_date, amount, total_gifts, state, engagement_score, event_participation
    return SemanticModel(
        dataset="crm_donor",
        id_dimension="donor_id",
        time_dimension="donation_date",
        dimensions={
            "donor_id": Dimension(name="donor_id", column="DonorID", type="categorical", synonyms=["donor", "donors", "id"]),
            "donation_date": Dimension(name="donation_date", column="LastDonationDate", type="time", synonyms=["last donation date", "last gift date"]),
            "amount": Dimension(name="amount", column="TotalAmountDonated", type="numeric", synonyms=["donation total", "total amount", "total"]),
            "total_gifts": Dimension(name="total_gifts", column="TotalGifts", type="numeric", synonyms=["gift count", "gifts"]),
            "state": Dimension(name="state", column="State", type="categorical", synonyms=["state", "states"]),
            "engagement_score": Dimension(name="engagement_score", column="EngagementScore", type="numeric", synonyms=["engagement", "score"]),
            "event_participation": Dimension(name="event_participation", column="EventParticipation", type="boolean", synonyms=["event participation", "event", "participation"]),
        },
        metrics={
            "donor_count_last_n_days": Metric(
                name="donor_count_last_n_days",
                description="Unique donors with donation_date within last N days.",
                synonyms=["how many donors", "donor count", "count donors"]
            ),
            "median_total_last_n_days": Metric(
                name="median_total_last_n_days",
                description="Median per-donor total amount within last N days.",
                synonyms=["median donation total", "median total", "median donated"]
            ),
        },
        synonyms={
            "donation": "amount",
            "donations": "amount",
            "totals": "amount",
        }
    )

def load_semantics(path: Optional[str]) -> SemanticModel:
    if not path:
        return default_crm_semantics()
    p = Path(path).expanduser()
    if p.exists() and yaml is not None:
        try:
            data = yaml.safe_load(p.read_text())
            dims = {d["name"]: Dimension(**d) for d in data.get("dimensions", [])}
            mets = {m["name"]: Metric(**m) for m in data.get("metrics", [])}
            return SemanticModel(
                dataset=data.get("dataset", "crm_donor"),
                id_dimension=data.get("id_dimension", "donor_id"),
                time_dimension=data.get("time_dimension", "donation_date"),
                dimensions=dims,
                metrics=mets,
                synonyms=data.get("synonyms", {}),
            )
        except Exception:
            return default_crm_semantics()
    return default_crm_semantics()

def normalize_text_with_semantics(text: str, sem: SemanticModel) -> str:
    norm = " " + text.lower() + " "
    for k, v in sem.all_synonyms().items():
        norm = re.sub(rf"(?<![a-z]){re.escape(k)}(?![a-z])", v.lower(), norm)
    return norm.strip()

def parse_time_window_days(text: str, default_days: int = 90) -> int:
    m = re.search(r"last\s+(\d+)\s*day", text.lower())
    return int(m.group(1)) if m else default_days

def extract_state_filter(text: str) -> List[str]:
    # Look for parenthetical state list (e.g., CA, NY, TX)
    m = re.search(r"\((?:e\.g\.,?\s*)?([A-Z]{2}(?:\s*,\s*[A-Z]{2})+)\)", text)
    if not m:
        return []
    raw = m.group(1)
    return [s.strip().upper() for s in re.split(r"\s*,\s*", raw) if s.strip()]

def wants_group_by_state(text: str) -> bool:
    t = text.lower()
    return "by state" in t or "across state" in t or "across states" in t or "by states" in t

def requested_metrics(text: str) -> Tuple[bool, bool]:
    t = text.lower()
    want_count = ("how many donors" in t) or ("donor count" in t) or ("count donors" in t)
    want_median = ("median donation total" in t) or ("median per-donor total" in t) or ("median total" in t)
    # Fallback heuristics
    if ("how many" in t and "donor" in t): want_count = True
    if "median" in t and ("total" in t or "donation" in t): want_median = True
    return want_count, want_median