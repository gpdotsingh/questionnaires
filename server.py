# python
# ...existing code...
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

from qapipeline import (
    QuestionSplitter, Orchestrator, LLMCompiler, Validator,
    CompilerInput, ValidatorInput
)

app = FastAPI(title="QA Pipeline Chat (Ollama Frontend)", version="1.0")

class ChatRequest(BaseModel):
    message: str
    try_llm: bool = True
    provider: Optional[str] = "ollama"

class ChatResponse(BaseModel):
    answer: str
    used_llm_in_splitter: bool
    plan_steps: List[str]
    validation_score: Optional[float] = None

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    splitter = QuestionSplitter(try_llm=req.try_llm, provider=req.provider)
    plan = splitter.plan(req.message)

    # Point to your CRM CSV
    orch = Orchestrator(vectordb_path="vectors", donations_csv="data/CRM_Donor_Simulation_Dataset.csv")
    answers = orch.run(plan)

    compiler = LLMCompiler()
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