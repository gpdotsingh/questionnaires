from __future__ import annotations
from typing import List, Optional, Tuple, Dict
from pydantic import BaseModel


# ---------- Common data contracts passed between modules ----------

class SubQuestion(BaseModel):
    id: str
    text: str
    requires: List[str] = []
    notes: Optional[str] = None
    is_input_finder: bool = False


class SplitPlan(BaseModel):
    original_question: str
    start_nodes: List[str]
    ordered_steps: List[SubQuestion]
    graph_edges: List[Tuple[str, str]]
    simplification: List[str]
    used_llm: bool = True


# Orchestrator input = SplitPlan
OrchestratorInput = SplitPlan

# Orchestrator output: map sub-question id -> answer payload (free-form)
class QAResult(BaseModel):
    question_id: str
    question_text: str
    answer: str
    meta: Dict[str, str] = {}

class OrchestratorOutput(BaseModel):
    results: List[QAResult]
    errors: Dict[str, str] = {}   # question_id -> error message


# Compiler input: OrchestratorOutput + original question
class CompilerInput(BaseModel):
    original_question: str
    answers: OrchestratorOutput

class CompilerOutput(BaseModel):
    final_answer: str
    reasoning: Optional[str] = None


# Validator input: original question + compiled answer
class ValidatorInput(BaseModel):
    original_question: str
    compiled_answer: str

class ValidatorOutput(BaseModel):
    is_valid: bool
    score: float
    feedback: Optional[str] = None
