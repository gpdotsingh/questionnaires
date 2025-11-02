from __future__ import annotations
from .contracts import ValidatorInput, ValidatorOutput

class Validator:
    """Validates the compiled answer against the original question."""
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def validate(self, inp: ValidatorInput) -> ValidatorOutput:
        key_terms = set(inp.original_question.lower().split())
        answer_terms = set(inp.compiled_answer.lower().split())
        overlap = len(key_terms & answer_terms) / len(key_terms)
        is_valid = overlap >= self.threshold
        feedback = "Answer is valid." if is_valid else "Answer does not fully address the question."
        return ValidatorOutput(is_valid=is_valid, score=overlap, feedback=feedback)