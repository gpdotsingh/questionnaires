from __future__ import annotations
from .contracts import CompilerInput, CompilerOutput

class LLMCompiler:
    """Module 3: turns sub-answers into a human-readable final answer."""
    def __init__(self):
        pass

    def compile(self, inp: CompilerInput) -> CompilerOutput:
        # TODO: call LLM to weave answers + reasoning. For now, join.
        joined = "\n".join(f"{r.question_id}: {r.answer}" for r in inp.answers.results)
        return CompilerOutput(final_answer=f"Summary for: {inp.original_question}\n{joined}",
                              reasoning="(mock reasoning)")
