from __future__ import annotations
from .contracts import ValidatorInput, ValidatorOutput

class Validator:
    """Module 4: checks if compiled answer satisfies the original question."""
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def validate(self, inp: ValidatorInput) -> ValidatorOutput:
        # TODO: implement semantic / rule validation. For now, naive.
        ok = len(inp.compiled_answer.strip()) > 0
        score = 0.8 if ok else 0.0
        return ValidatorOutput(is_valid=ok, score=score,
                               feedback="(mock) Looks good." if ok else "Answer empty.")
