from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path

from qapipeline import (
    QuestionSplitter, Orchestrator, LLMCompiler, Validator,
    CompilerInput, ValidatorInput
)

app = FastAPI(title="QA Pipeline Chat (Ollama Frontend)", version="1.1")

class ChatRequest(BaseModel):
    message: str
    try_llm: bool = True
    provider: Optional[str] = "ollama"  # ollama | openai | huggingface
    model: Optional[str] = None         # provider-specific model name

class ChatResponse(BaseModel):
    answer: str
    used_llm_in_splitter: bool
    plan_steps: List[str]
    validation_score: Optional[float] = None

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    splitter = QuestionSplitter(try_llm=req.try_llm, provider=req.provider, model=req.model)
    plan = splitter.plan(req.message)

    base_dir = Path(__file__).resolve().parent
    vectors_path = base_dir / "vectors"
    data_path = base_dir / "data" / "CRM_Donor_Simulation_Dataset.csv"

    # Use semantic.py defaults (no YAML)
    orch = Orchestrator(
        vectordb_path=str(vectors_path),
        donations_csv=str(data_path),
        semantic_path=None,
        relative_to_data=True,
        debug=True,
    )
    answers = orch.run(plan)

    compiler = LLMCompiler()  # uses Ollama if env COMPILER_USE_OLLAMA=true
    compiled = compiler.compile(CompilerInput(
        original_question=plan.original_question,
        answers=answers
    ))

    validator = Validator()
    verdict = validator.validate(ValidatorInput(
        original_question=plan.original_question,
        compiled_answer=compiled.final_answer
    ))

    return ChatResponse(
        answer=compiled.final_answer,
        used_llm_in_splitter=plan.used_llm,
        plan_steps=[f"{s.id}: {s.text}" for s in plan.ordered_steps],
        validation_score=verdict.score
    )

@app.get("/health")
def health():
    return {"status": "ok"}