# server.py
from fastapi import FastAPI
from pydantic import BaseModel
import os

from qapipeline import (
    QuestionSplitter, Orchestrator, LLMCompiler, Validator,
    CompilerInput, ValidatorInput
)

# Optional: use Ollama for local LLM responses inside compiler later if you like.
# For now, compiler is mock; splitter can use ollama by provider="ollama".
# Make sure Ollama is running: `ollama serve &` and you have pulled a model:
# `ollama pull llama3.1`

app = FastAPI(title="QA Pipeline Chat (Ollama Frontend)", version="1.0")

class ChatRequest(BaseModel):
    message: str
    # controls for splitter
    try_llm: bool = True
    provider: str | None = "ollama"   # force ollama for front-end

class ChatResponse(BaseModel):
    answer: str
    used_llm_in_splitter: bool
    plan_steps: list[str]
    validation_score: float

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1) split
    splitter = QuestionSplitter(try_llm=req.try_llm, provider=req.provider)
    plan = splitter.plan(req.message)

    # 2) orchestrate (stub)
    orch = Orchestrator()
    answers = orch.run(plan)

    # 3) compile (stub)
    compiler = LLMCompiler()
    compiled = compiler.compile(CompilerInput(
        original_question=plan.original_question, answers=answers
    ))

    # 4) validate
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
    return {"ok": True}
