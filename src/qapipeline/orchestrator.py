from __future__ import annotations
from typing import Dict, List
from .contracts import OrchestratorInput, OrchestratorOutput, QAResult

class Orchestrator:
    """Module 2: executes sub-questions (parallel where possible),
    fetches from web/docs/db/json, and returns answers."""
    def __init__(self):
        pass

    def run(self, plan: OrchestratorInput) -> OrchestratorOutput:
        # TODO: implement:
        # - build execution graph
        # - run ready nodes in parallel
        # - use toolset (web/file/db) per node
        # For now, return mocked answers:
        results: List[QAResult] = []
        for step in plan.ordered_steps:
            results.append(QAResult(
                question_id=step.id,
                question_text=step.text,
                answer=f"[MOCK ANSWER for {step.id}]",
                meta={"source": "mock"}
            ))
        return OrchestratorOutput(results=results, errors={})
