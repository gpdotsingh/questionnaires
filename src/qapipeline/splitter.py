from __future__ import annotations
import re, json, os
from typing import List, Optional, Dict, Tuple
import networkx as nx
from pydantic import BaseModel

from .contracts import SubQuestion, SplitPlan

# Optional NLP (spaCy)
try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
except Exception:
    NLP = None

# Optional LLMs (LangChain)
try:
    from langchain_openai import ChatOpenAI  # also works with OpenAI-compatible endpoints if base_url is set
except Exception:
    ChatOpenAI = None
try:
    from langchain_ollama import OllamaLLM
except Exception:
    OllamaLLM = None

# ---------------- Heuristics for rule-based splitting ----------------
JOINERS = re.compile(
    r'\b(?:and then|then|next|after that|afterwards|subsequently|first|second|third|before|using|based on|given|with|by)\b',
    re.I
)
SPLIT_PUNCT = re.compile(r'[;:.?!]')
INPUT_FINDER_PAT = re.compile(r'\b(find|fetch|lookup|extract|get)\b.*\b(from|in|via|using)\b', re.I)

def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def naive_clause_split(q: str) -> List[str]:
    chunks = []
    for sent in SPLIT_PUNCT.split(q):
        sent = _norm(sent)
        if not sent:
            continue
        parts = re.split(JOINERS, sent)
        if len(parts) > 1:
            tokens = re.split(r'(\b(?:and then|then|next|after that|afterwards|subsequently|first|second|third|before|using|based on|given|with|by)\b)', sent)
            buf = ""
            for t in tokens:
                t = t.strip()
                if not t:
                    continue
                if buf and not JOINERS.fullmatch(t):
                    chunks.append(_norm(buf))
                    buf = t
                else:
                    if buf:
                        buf += " " + t
                    else:
                        buf = t
            if buf:
                chunks.append(_norm(buf))
        else:
            chunks.append(sent)
    seen = set(); out=[]
    for c in chunks:
        k = c.lower()
        if k not in seen:
            seen.add(k); out.append(c)
    return out

def spacy_split(q: str) -> List[str]:
    if not NLP:
        return naive_clause_split(q)
    doc = NLP(q)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    out = []
    for s in sents:
        out.extend(naive_clause_split(s))
    return out

def detect_input_finder(clause: str) -> bool:
    return bool(INPUT_FINDER_PAT.search(clause))

def guess_dependencies(clause: str, id_map: Dict[str, str]) -> List[str]:
    deps: List[str] = []
    text = clause.lower()
    cues = [
        (r'\busing (the )?(previous|above|earlier|that|those|it|result|output)\b', 'ANY'),
        (r'\b(based on|given|with)\b', 'ANY'),
        (r'\bthen\b', 'PREV'),
        (r'\bafter (that|this|finding|calculating)\b', 'PREV'),
        (r'\buse (that|those|the result)\b', 'PREV'),
        (r'\bfeed (it|that) into\b', 'PREV'),
    ]
    for pat, kind in cues:
        if re.search(pat, text):
            if kind == 'PREV' and id_map:
                deps.append(list(id_map.keys())[-1])
            elif kind == 'ANY' and id_map:
                candidates = [k for k, v in id_map.items() if any(t in v.lower() for t in ("find","fetch","extract","lookup","get"))]
                deps.append(candidates[-1] if candidates else list(id_map.keys())[-1])
    return deps

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
        return [g.nodes[i]["obj"] for i in order]
    except nx.NetworkXUnfeasible:
        # break a back-edge heuristically
        if g.edges:
            u, v = list(g.edges())[-1]
            g.remove_edge(u, v)
            return topo_sorted(g)
        raise


# ---------------- Optional LLM-assisted decomposition ----------------
SYS_DECOMP = (
    "Decompose a complex user question into atomic sub-questions with explicit dependencies.\n"
    "Rules:\n"
    "1) Return JSON: {\"subquestions\":[{id,text,requires,is_input_finder}], \"simplification\":[...]}.\n"
    "2) 'requires' lists ids required first.\n"
    "3) Mark input-gathering steps (find/fetch/extract) as is_input_finder=true.\n"
    "4) Keep sub-questions short and executable."
)
USR_DECOMP = "Question:\n{q}\n\nReturn ONLY JSON."

def _select_llm(provider: Optional[str] = None):
    p = (provider or os.getenv("PROVIDER", "auto")).lower()
    # OpenAI or OpenAI-compatible (if OPENAI_API_KEY or custom base_url available)
    if p in ("openai", "auto", "deepseek"):
        if ChatOpenAI:
            # deepseek through OpenAI-compatible base if provided
            base_url = os.getenv("DEEPSEEK_BASE_URL")
            api_key  = os.getenv("DEEPSEEK_API_KEY")
            if p == "deepseek" and base_url and api_key:
                return ChatOpenAI(model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                                  api_key=api_key, base_url=base_url, temperature=0)
            if os.getenv("OPENAI_API_KEY"):
                return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    # Ollama
    if p in ("ollama", "auto"):
        if OllamaLLM:
            return OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3.1"))
    return None

def _llm_decompose(question: str, provider: Optional[str]) -> Optional[SplitPlan]:
    llm = _select_llm(provider)
    if not llm:
        return None
    prompt = f"<SYS>\n{SYS_DECOMP}\n</SYS>\n" + USR_DECOMP.format(q=question)
    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or str(resp)
        m = re.search(r'\{.*\}\s*$', text)
        data = json.loads(m.group(0) if m else text)
        subqs = [SubQuestion(**sq) for sq in data.get("subquestions", [])]
        if not subqs:
            return None
        g = build_graph(subqs)
        ordered = topo_sorted(g)
        starts = [sq.id for sq in ordered if g.in_degree(sq.id) == 0]
        simpl = data.get("simplification", []) or [sq.text for sq in ordered]
        edges = list(g.edges())
        return SplitPlan(
            original_question=question,
            start_nodes=starts,
            ordered_steps=ordered,
            graph_edges=edges,
            simplification=simpl,
            used_llm=True
        )
    except Exception:
        return None

def _rule_decompose(question: str) -> SplitPlan:
    clauses = spacy_split(question) if NLP else naive_clause_split(question)
    subqs: List[SubQuestion] = []
    id_map: Dict[str, str] = {}
    for idx, c in enumerate(clauses, 1):
        cid = f"Q{idx}"
        deps = guess_dependencies(c, id_map)
        subqs.append(SubQuestion(
            id=cid, text=c, requires=deps, is_input_finder=detect_input_finder(c)
        ))
        id_map[cid] = c

    if not any(s.is_input_finder for s in subqs):
        for s in subqs:
            if "from " in s.text.lower() or "given " in s.text.lower():
                s.is_input_finder = True
                break

    g = build_graph(subqs)
    ordered = topo_sorted(g)
    starts = [sq.id for sq in ordered if g.in_degree(sq.id) == 0]
    simpl = []
    for s in ordered:
        t = _norm(re.sub(JOINERS, "", s.text))
        simpl.append(("Find " if s.is_input_finder else "Compute ") + t)

    return SplitPlan(
        original_question=question,
        start_nodes=starts,
        ordered_steps=ordered,
        graph_edges=list(g.edges()),
        simplification=simpl,
        used_llm=False
    )


# ---------------- Public API for Module 1 ----------------
class QuestionSplitter:
    """Module 1: splits a question into sub-questions + dependencies (DAG)."""

    def __init__(self, try_llm: bool = True, provider: Optional[str] = None):
        self.try_llm = try_llm
        self.provider = provider

    def plan(self, question: str) -> SplitPlan:
        question = _norm(question)
        if self.try_llm:
            p = _llm_decompose(question, self.provider)
            if p:
                return p
        return _rule_decompose(question)
