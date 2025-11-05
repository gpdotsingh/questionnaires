# python
from __future__ import annotations
import re
import os
from typing import List
from .contracts import CompilerInput, CompilerOutput

# Optional Ollama summarizer
try:
    from langchain_ollama import OllamaLLM
except Exception:
    OllamaLLM = None

def _compact_text(t: str, max_len: int = 400) -> str:
    # Drop PII-ish lines and long row dumps, keep first few lines, cap length
    lines = [ln for ln in t.splitlines() if ln.strip()]
    keep: List[str] = []
    for ln in lines:
        if any(k in ln.lower() for k in ["@"," phone:", "email:", "zipcode:", "city:", "firstname:", "lastname:"]):
            continue
        if len(ln) > 300 and ln.count(":") >= 4:
            continue
        keep.append(ln.strip())
        if len(keep) >= 5:
            break
    s = " ".join(keep) if keep else t.strip()
    return s[:max_len]

class LLMCompiler:
    """Combines sub-answers into a concise, humanized final answer. Uses Ollama if enabled."""
    def __init__(self, use_ollama: bool | None = None, model: str | None = None):
        self.use_ollama = use_ollama if use_ollama is not None else (os.getenv("COMPILER_USE_OLLAMA", "false").lower() == "true")
        self.model = model or os.getenv("OLLAMA_GEN_MODEL", "llama3.1")

    def _summarize_with_ollama(self, question: str, bullets: str) -> str:
        if not OllamaLLM:
            return ""
        try:
            llm = OllamaLLM(model=self.model, temperature=0)
            prompt = (
                "You are a data analyst. Write a clear, concise answer (2-4 sentences), avoid PII and raw tables.\n"
                f"Question:\n{question}\n\nFindings:\n{bullets}\n\nFinal answer:"
            )
            return llm.invoke(prompt).strip()
        except Exception:
            return ""

    def compile(self, inp: CompilerInput) -> CompilerOutput:
        compact = [_compact_text(r.answer) for r in inp.answers.results if r.answer]
        bullets = "; ".join(f"{r.question_id}: {c}" for r, c in zip(inp.answers.results, compact))
        final_answer = ""
        if self.use_ollama and OllamaLLM:
            final_answer = self._summarize_with_ollama(inp.original_question, bullets)
        if not final_answer:
            final_answer = f"{inp.original_question.strip()}: " + (" ".join(compact) if compact else "No data found.")
        reasoning = "Compiled via Ollama" if self.use_ollama and OllamaLLM else "Heuristic summarization"
        return CompilerOutput(final_answer=final_answer, reasoning=reasoning)