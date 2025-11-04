# python
from __future__ import annotations
import re
from typing import List, Optional
import networkx as nx

from .contracts import SubQuestion, SplitPlan

# Strong boundaries (avoid '.' to keep e.g./initials/decimals intact)
SPLIT_PUNCT = re.compile(r"[;:?!]")
# Step joiners that indicate sequence; do not include bare "and"
JOINERS = re.compile(
    r"\b(?:and then|then|next|after that|afterwards|subsequently|first|second|third|before|using|based on|given|with|by)\b",
    re.I,
)
# Input-finder detector
INPUT_FINDER_PAT = re.compile(r"\b(find|fetch|lookup|extract|get|identify|search)\b.*\b(from|in|via|using)\b", re.I)

# Metric/field terms that commonly appear as parallel items joined by "and"/"&"
METRIC_TERMS = [
    "total donation amount", "donation total", "total amount donated", "total amount", "total donations",
    "engagement score", "engagement",
    "donor count", "count donors", "how many donors",
    "median per-donor total", "median donation total", "median total",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _pre_normalize(text: str) -> str:
    t = text
    t = re.sub(r"\be\.g\.\b", "for example", t, flags=re.I)
    t = re.sub(r"\bi\.e\.\b", "that is", t, flags=re.I)
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    return t


def _maybe_split_compare(seg: str) -> List[str]:
    """Split 'X vs Y' or 'X versus Y' into two symmetric clauses with a shared action if present."""
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
    """If 'X and Y' (or '&') where both sides contain metric terms, split into two clauses, keeping shared prefix."""
    lower = seg.lower()
    splitter = " and " if " and " in lower else (" & " if " & " in lower else None)
    if not splitter:
        return [seg]
    left, right = seg.split(splitter, 1)
    has_left = any(term in left.lower() for term in METRIC_TERMS)
    has_right = any(term in right.lower() for term in METRIC_TERMS)
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
    """Heuristic splitter: strong punctuation, compare, metric-aware 'and', step-joiners."""
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
    # Deduplicate preserving order
    seen = set()
    out: List[str] = []
    for c in chunks:
        if c not in seen:
            out.append(c)
            seen.add(c)
    # If only one long clause, try conservative dot-split: ". " followed by Capital
    if len(out) == 1:
        extra = re.split(r"(?<=[a-z0-9])\.\s+(?=[A-Z])", out[0])
        extra = [_norm(x) for x in extra if _norm(x)]
        if len(extra) > 1:
            out = extra
    return out


def _parse_json_array(txt: str) -> Optional[List[str]]:
    """Parse a JSON array of strings; strip code fences and extract the first array found."""
    # Strip code fences if present
    txt = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", txt, flags=re.I | re.S)
    # Find a JSON array of strings (allow newlines)
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
    """Detect over-fragmented LLM splits like ['How many','donors','gave','in the last','90 days']."""
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


def _llm_split_ollama(question: str, model: Optional[str]) -> Optional[List[str]]:
    """Use Ollama (via langchain_ollama) to split into sub-questions; return None on failure."""
    try:
        from langchain_ollama import OllamaLLM
    except Exception:
        return None
    try:
        llm = OllamaLLM(model=model or "llama3.1", temperature=0)
        prompt = (
            "You split user requests into minimal sequential sub-questions.\n"
            "Rules:\n"
            "- DO NOT split inside a single natural-language clause.\n"
            "- Split only on strong punctuation (; : ? !), step joiners "
            "('and then','then','next','after that','before','using','based on','with','by'),\n"
            "- Or on 'vs/versus', or when two different metrics/entities are joined by 'and' "
            "(e.g., 'total donation amount and engagement score').\n"
            "- Keep each sub-question as a full phrase; never return single words.\n"
            "- Return ONLY a JSON array of strings.\n"
            "Examples:\n"
            'Request: How many donors gave in the last 90 days?\n'
            'Output: ["How many donors gave in the last 90 days"]\n'
            'Request: show me the declining of total donation amount and engagement score\n'
            'Output: ["show me the declining of total donation amount", "show me the declining of engagement score"]\n'
            f"Request: {question}\n"
            "Output:"
        )
        txt = llm.invoke(prompt).strip()
        return _parse_json_array(txt)
    except Exception:
        return None


def detect_input_finder(clause: str) -> bool:
    return bool(INPUT_FINDER_PAT.search(clause))


def guess_dependencies(idx: int) -> List[str]:
    # Simple sequential dependency
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


class QuestionSplitter:
    """Heuristic splitter with optional Ollama assist. used_llm=True when Ollama is invoked (even on fallback)."""
    def __init__(self, try_llm: bool = False, provider: Optional[str] = None, model: Optional[str] = None):
        self.try_llm = bool(try_llm)
        # Force Ollama when try_llm is requested; provider is ignored otherwise
        self.provider = "ollama" if self.try_llm else None
        self.model = model

    def plan(self, question: str) -> SplitPlan:
        q = _norm(question)

        used_llm = False
        clauses: List[str] = []

        if self.try_llm:
            # Attempt LLM split first
            llm_subqs = _llm_split_ollama(q, self.model)
            used_llm = True
            if llm_subqs and not _is_degenerate_llm_split(q, llm_subqs):
                clauses = llm_subqs
            else:
                # Fallback to heuristic if LLM over-fragments or fails
                clauses = _heuristic_split(q)
        else:
            clauses = _heuristic_split(q)

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