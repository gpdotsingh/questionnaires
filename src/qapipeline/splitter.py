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
# Regex patterns for splitting and detecting dependencies
JOINERS = re.compile(r'\b(?:and then|then|next|after that|afterwards|subsequently|first|second|third|before|using|based on|given|with|by)\b', re.I)
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
        chunks.extend(_norm(p) for p in parts if p)
    return list(dict.fromkeys(chunks))  # Remove duplicates while preserving order

def detect_input_finder(clause: str) -> bool:
    return bool(INPUT_FINDER_PAT.search(clause))

def guess_dependencies(clause: str, id_map: Dict[str, str]) -> List[str]:
    deps = []
    text = clause.lower()
    if "then" in text or "after" in text:
        if id_map:
            deps.append(list(id_map.keys())[-1])
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
        raise ValueError("Cyclic dependencies detected in the graph.")

class QuestionSplitter:
    """Splits a question into sub-questions + dependencies (DAG)."""
    def __init__(self, try_llm: bool = True, provider: Optional[str] = None):
        self.try_llm = try_llm
        self.provider = provider

    def plan(self, question: str) -> SplitPlan:
        question = _norm(question)
        clauses = naive_clause_split(question)
        subqs = []
        id_map = {}
        for idx, clause in enumerate(clauses, 1):
            cid = f"Q{idx}"
            deps = guess_dependencies(clause, id_map)
            subqs.append(SubQuestion(
                id=cid, text=clause, requires=deps, is_input_finder=detect_input_finder(clause)
            ))
            id_map[cid] = clause
        g = build_graph(subqs)
        ordered = topo_sorted(g)
        starts = [sq.id for sq in ordered if g.in_degree(sq.id) == 0]
        return SplitPlan(
            original_question=question,
            start_nodes=starts,
            ordered_steps=ordered,
            graph_edges=list(g.edges()),
            simplification=[sq.text for sq in ordered],
            used_llm=False
        )