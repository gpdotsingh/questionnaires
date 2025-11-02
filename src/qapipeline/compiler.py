# python
# filepath: /Users/gauravsingh/study/AI/DeependraBhaiyaproject/questionier/questionnaires/src/qapipeline/compiler.py
from __future__ import annotations
import os
from .contracts import CompilerInput, CompilerOutput

# Optional Ollama summarizer
try:
    from langchain_ollama import OllamaLLM
except Exception:
    OllamaLLM = None

class LLMCompiler:
    """Combines sub-answers into a human-readable final answer. Uses Ollama if available and enabled."""
    def __init__(self, use_ollama: bool | None = None, model: str | None = None):
        self.use_ollama = use_ollama if use_ollama is not None else (os.getenv("COMPILER_USE_OLLAMA", "false").lower() == "true")
        self.model = model or os.getenv("OLLAMA_GEN_MODEL", "llama3.1")

    def _summarize_with_ollama(self, question: str, bullets: str) -> str:
        if not OllamaLLM:
            return ""
        try:
            llm = OllamaLLM(model=self.model)
            prompt = f"You are a helpful assistant. Based on these step answers, write a concise final answer.\nQuestion:\n{question}\n\nSteps and answers:\n{bullets}\n\nFinal answer:"
            return llm.invoke(prompt).strip()
        except Exception:
            return ""

    def compile(self, inp: CompilerInput) -> CompilerOutput:
        bullets = "\n".join(f"{r.question_id}: {r.answer}" for r in inp.answers.results)
        final_answer = ""
        if self.use_ollama and OllamaLLM:
            final_answer = self._summarize_with_ollama(inp.original_question, bullets)
        if not final_answer:
            # Heuristic fallback
            final_answer = f"Summary for: {inp.original_question}\n\n{bullets}"
        reasoning = "Compiled from sub-answers using Ollama" if self.use_ollama and OllamaLLM else "Heuristic aggregation"
        return CompilerOutput(final_answer=final_answer, reasoning=reasoning)