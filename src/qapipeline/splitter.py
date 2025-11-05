# python
from __future__ import annotations
import re
from typing import Dict, List, Optional
import networkx as nx

from .contracts import SubQuestion, SplitPlan

# -------------------------
# Normalization helpers
# -------------------------
WS = re.compile(r"\s+")
def _norm(s: str) -> str:
    return WS.sub(" ", s).strip()

def _lower(s: str) -> str:
    return _norm(s).lower()

# -------------------------
# Domain vocabulary (extend as you like)
# -------------------------
METRIC_SYNONYMS: Dict[str, List[str]] = {
    "total_donation_amount": [
        "total donation amount", "total amount donated", "donation amount",
        "sum of donations", "total raised", "total amount", "donation totals",
        "donation total", "total donations", "total giving"
    ],
    "engagement_score": [
        "engagement score", "average engagement", "mean engagement", "avg engagement",
        "engagement", "engagement index"
    ],
    "donor_count": [
        "donor count", "count of donors", "how many donors", "number of donors", "unique donors"
    ],
    "total_gifts": [
        "total gifts", "gift count", "number of gifts", "gifts total"
    ],
}

OPERATIONS: Dict[str, List[str]] = {
    "trend": ["trend", "over time", "time series", "time-series"],
    "decline": ["decline", "decreasing", "drop", "downturn", "going down", "falling"],
    "increase": ["increase", "growing", "rising", "going up"],
    "compare": ["compare", "vs", "versus"],
    "group_by_state": ["by state", "across states", "by states"],
}

TIME_GRAINS: Dict[str, List[str]] = {
    "monthly": ["per month", "monthly", "by month"],
    "weekly": ["per week", "weekly", "by week"],
    "yearly": ["per year", "yearly", "annually", "annual", "by year"],
}

# -------------------------
# Heuristic splitting helpers
# -------------------------
SPLIT_PUNCT = re.compile(r"[;:?!]")
JOINERS = re.compile(
    r"\b(?:and then|then|next|after that|afterwards|subsequently|first|second|third|before|using|based on|given|with|by)\b",
    re.I,
)
INPUT_FINDER_PAT = re.compile(r"\b(find|fetch|lookup|extract|get|identify|search)\b.*\b(from|in|via|using)\b", re.I)

def _pre_normalize(text: str) -> str:
    t = text
    t = re.sub(r"\be\.g\.\b", "for example", t, flags=re.I)
    t = re.sub(r"\bi\.e\.\b", "that is", t, flags=re.I)
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    return t

def _maybe_split_compare(seg: str) -> List[str]:
    m = re.search(r"\b(vs\.?|versus)\b", seg, flags=re.I)
    if not m:
        return [seg]
    parts = re.split(r"\bvs\.?|\bversus\b", seg, flags=re.I)
    if len(parts) != 2:
        return [seg]
    left, right = _norm(parts[0]), _norm(parts[1])
    pref_match = re.match(r"^\s*(compare|show|find|report|list|give|display|analyze|compute|calculate)\b\s*", left, flags=re.I)
    prefix = pref_match.group(0) if pref_match else ""
    if prefix:
        left_obj = left[len(prefix):]
        return [_norm(prefix + left_obj), _norm(prefix + right)]
    return [left, right]

def _maybe_split_metrics_with_and(seg: str) -> List[str]:
    lower = seg.lower()
    splitter = " and " if " and " in lower else (" & " if " & " in lower else None)
    if not splitter:
        return [seg]
    left, right = seg.split(splitter, 1)
    flat_terms = sum(METRIC_SYNONYMS.values(), [])
    has_left = any(term in left.lower() for term in flat_terms)
    has_right = any(term in right.lower() for term in flat_terms)
    if has_left and has_right:
        pref = re.match(
            r"^\s*(show|find|report|list|give|display|compute|calculate|compare)\b.*?\b(of|for)\b\s*",
            seg,
            flags=re.I,
        )
        if pref:
            prefix = pref.group(0)
            l_obj = left[len(prefix):] if left.lower().startswith(prefix.lower()) else left
            return [_norm(prefix + l_obj), _norm(prefix + right)]
        return [_norm(left), _norm(right)]
    return [seg]

def _heuristic_split(question: str) -> List[str]:
    q = _pre_normalize(question)
    chunks: List[str] = []
    for sent in SPLIT_PUNCT.split(q):
        sent = _norm(sent)
        if not sent:
            continue
        for cmp_seg in _maybe_split_compare(sent):
            for ms in _maybe_split_metrics_with_and(cmp_seg):
                for p in re.split(JOINERS, ms):
                    p = _norm(p)
                    if p:
                        chunks.append(p)
    seen = set()
    out: List[str] = []
    for c in chunks:
        if c not in seen:
            out.append(c)
            seen.add(c)
    if len(out) == 1:
        extra = re.split(r"(?<=[a-z0-9])\.\s+(?=[A-Z])", out[0])
        extra = [_norm(x) for x in extra if _norm(x)]
        if len(extra) > 1:
            out = extra
    return out

def _parse_json_array(txt: str) -> Optional[List[str]]:
    txt = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", txt, flags=re.I | re.S)
    m = re.search(r'\[\s*(?:"[^"]*"\s*,\s*)*"[^"]*"\s*\]', txt, flags=re.S)
    raw = m.group(0) if m else txt
    try:
        import json
        data = json.loads(raw)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [_norm(x) for x in data if _norm(x)]
    except Exception:
        return None
    return None

def _has_strong_boundary(text: str) -> bool:
    t = text.lower()
    if re.search(SPLIT_PUNCT, text):
        return True
    if re.search(JOINERS, t):
        return True
    if re.search(r"\b(vs\.?|versus)\b", t):
        return True
    return False

def _is_degenerate_llm_split(original: str, parts: List[str]) -> bool:
    if not parts or len(parts) <= 1:
        return False
    word_counts = [len(p.split()) for p in parts]
    avg_words = sum(word_counts) / len(word_counts)
    many_short = len(parts) >= 3 and sum(1 for w in word_counts if w <= 2) >= max(2, len(parts) - 1)
    if many_short and not _has_strong_boundary(original):
        return True
    if "?" in original and len(parts) > 1 and avg_words < 5:
        return True
    return False

# --------- LLM providers for splitting ----------
def _llm_split_ollama(question: str, model: Optional[str]) -> Optional[List[str]]:
    try:
        from langchain_ollama import OllamaLLM
    except Exception:
        return None
    try:
        llm = OllamaLLM(model=model or "llama3.1", temperature=0)
        prompt = (
            "You split analytics requests into minimal sequential sub-questions.\n"
            "Rules:\n"
            "- Split only on strong punctuation (; : ? !), step joiners "
            "('and then','then','next','after that','before','using','based on','with','by'),\n"
            "- Or on 'vs/versus', or when two different metrics/entities are joined by 'and'.\n"
            "- Keep each sub-question a full phrase; never single words.\n"
            "- Return ONLY a JSON array of strings.\n"
            f"Request: {question}\n"
            "Output:"
        )
        txt = llm.invoke(prompt).strip()
        return _parse_json_array(txt)
    except Exception:
        return None

def _llm_split_openai(question: str, model: Optional[str]) -> Optional[List[str]]:
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None
    try:
        llm = ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
        prompt = (
            "Split the user request into minimal sequential sub-questions for analytics.\n"
            "Return ONLY a JSON array of strings.\n"
            f"Request: {question}\nOutput:"
        )
        msg = llm.invoke(prompt)
        txt = getattr(msg, "content", str(msg)).strip()
        return _parse_json_array(txt)
    except Exception:
        return None

def _llm_split_hf(question: str, model: Optional[str]) -> Optional[List[str]]:
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    except Exception:
        return None
    try:
        endpoint = HuggingFaceEndpoint(repo_id=model or "mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=0)
        llm = ChatHuggingFace(llm=endpoint)
        prompt = (
            "Split the user request into minimal sequential sub-questions for analytics.\n"
            "Return ONLY a JSON array of strings.\n"
            f"Request: {question}\nOutput:"
        )
        msg = llm.invoke(prompt)
        txt = getattr(msg, "content", str(msg)).strip()
        return _parse_json_array(txt)
    except Exception:
        return None

def _llm_split(question: str, provider: Optional[str], model: Optional[str]) -> Optional[List[str]]:
    prov = (provider or "ollama").lower()
    if prov == "openai":
        return _llm_split_openai(question, model) or _llm_split_ollama(question, model)
    if prov in ("hf", "huggingface", "hugging_face"):
        return _llm_split_hf(question, model) or _llm_split_ollama(question, model)
    # default ollama
    return _llm_split_ollama(question, model)

def detect_input_finder(clause: str) -> bool:
    return bool(INPUT_FINDER_PAT.search(clause))

def guess_dependencies(idx: int) -> List[str]:
    return [f"Q{idx-1}"] if idx > 1 else []

def build_graph(subqs: List[SubQuestion]) -> nx.DiGraph:
    g = nx.DiGraph()
    for sq in subqs:
        g.add_node(sq.id, obj=sq)
    for sq in subqs:
        for d in sq.requires:
            if d != sq.id and d in g:
                g.add_edge(d, sq.id)
    return g

def topo_sorted(g: nx.DiGraph) -> List[SubQuestion]:
    try:
        order = list(nx.topological_sort(g))
        return [g.nodes[n]["obj"] for n in order]
    except nx.NetworkXUnfeasible:
        return [g.nodes[n]["obj"] for n in g.nodes]

# -------------------------
# Domain-aware detection
# -------------------------
def detect_timegrain(q: str) -> str:
    L = _lower(q)
    for grain, phrases in TIME_GRAINS.items():
        if any(p in L for p in phrases):
            return grain
    return "monthly"

def detect_ops(q: str) -> List[str]:
    L = _lower(q)
    ops: List[str] = []
    for op, phrases in OPERATIONS.items():
        if any(p in L for p in phrases):
            ops.append(op)
    if "decline" in ops and "trend" not in ops:
        ops.append("trend")
    if not ops:
        ops = ["trend"]
    return ops

def extract_metrics(q: str) -> List[str]:
    L = _lower(q)
    found: List[str] = []

    def add(m: str):
        if m not in found:
            found.append(m)

    for canon in METRIC_SYNONYMS.keys():
        if canon.replace("_", " ") in L:
            add(canon)
    for canon, syns in METRIC_SYNONYMS.items():
        if any(s in L for s in syns):
            add(canon)
    if "how many" in L and "donor" in L and "donor_count" not in found:
        add("donor_count")
    return found

def _build_compute_clause(metric: str, grain: str, ops: List[str]) -> str:
    base = f"compute {metric} ({grain}) over time and trend"
    if "decline" in ops:
        base += " & check decline"
    elif "increase" in ops:
        base += " & check increase"
    return base

def _maybe_add_grouping(q: str, clause: str) -> str:
    L = _lower(q)
    if "by state" in L or "across states" in L or "by states" in L:
        return clause + " (group by state)"
    return clause

def _has_vs(q: str) -> bool:
    return bool(re.search(r"\b(vs\.?|versus)\b", _lower(q)))

def _build_summary_clause() -> str:
    return "summarize combined insights and compare/correlate metrics (highlight simultaneous declines if any)"

# -------------------------
# Splitter class
# -------------------------
class QuestionSplitter:
    """Rule-based splitter with provider-backed LLM splitting when enabled."""

    def __init__(self, try_llm: bool = False, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        self.try_llm = bool(try_llm)
        self.provider = (provider or "ollama").lower() if provider else "ollama"
        self.model = model

    def plan(self, question: str) -> SplitPlan:
        q = _norm(question)
        used_llm = False
        clauses: List[str] = []

        # If asked, try LLM-first splitting
        if self.try_llm:
            llm_subqs = _llm_split(q, self.provider, self.model)
            if llm_subqs and not _is_degenerate_llm_split(q, llm_subqs):
                clauses = llm_subqs
                used_llm = True

        # Fallback: domain-aware + heuristics
        if not clauses:
            grain = detect_timegrain(q)
            ops = detect_ops(q)
            metrics = extract_metrics(q)
            if metrics:
                for m in metrics:
                    clause = _build_compute_clause(m, grain, ops)
                    clause = _maybe_add_grouping(q, clause)
                    clauses.append(clause)
                if len(metrics) > 1 or "compare" in ops or _has_vs(q):
                    clauses.append(_build_summary_clause())
            else:
                clauses = _heuristic_split(q)

        # Last resort: LLM (if enabled) then heuristics
        if not clauses and self.try_llm:
            llm_subqs = _llm_split(q, self.provider, self.model)
            if llm_subqs and not _is_degenerate_llm_split(q, llm_subqs):
                clauses = llm_subqs
                used_llm = True
        if not clauses:
            clauses = [q]

        # Materialize sub-questions
        subqs: List[SubQuestion] = []
        for idx, clause in enumerate(clauses, 1):
            subqs.append(
                SubQuestion(
                    id=f"Q{idx}",
                    text=clause,
                    requires=guess_dependencies(idx),
                    is_input_finder=detect_input_finder(clause),
                )
            )

        # Build DAG and order
        g = build_graph(subqs)
        ordered = topo_sorted(g)
        start_nodes = [s.id for s in ordered if g.in_degree(s.id) == 0]

        return SplitPlan(
            original_question=q,
            start_nodes=start_nodes,
            ordered_steps=ordered,
            graph_edges=list(g.edges()),
            simplification=[s.text for s in ordered],
            used_llm=used_llm,
        )