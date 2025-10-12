# main_demo.py
from qapipeline import (
    QuestionSplitter, Orchestrator, LLMCompiler, Validator,
    CompilerInput, ValidatorInput
)

def run_once(user_question: str, try_llm=True, provider="auto"):
    # 1) split
    splitter = QuestionSplitter(try_llm=try_llm, provider=provider)
    plan = splitter.plan(user_question)

    print("=== PLAN ===")
    print("Start nodes:", plan.start_nodes, "| used_llm:", plan.used_llm)
    for step in plan.ordered_steps:
        print(f"{step.id}: {step.text}  requires={step.requires}  input={step.is_input_finder}")

    # 2) orchestrate (stub returns mock answers for now)
    orch = Orchestrator()
    answers = orch.run(plan)

    # 3) compile
    compiler = LLMCompiler()
    compiled = compiler.compile(CompilerInput(
        original_question=plan.original_question, answers=answers
    ))

    print("\n=== COMPILED ===")
    print(compiled.final_answer)

    # 4) validate
    validator = Validator()
    verdict = validator.validate(ValidatorInput(
        original_question=plan.original_question,
        compiled_answer=compiled.final_answer
    ))

    print("\n=== VALIDATION ===")
    print("valid:", verdict.is_valid, "| score:", verdict.score, "| feedback:", verdict.feedback)

    return verdict.is_valid

if __name__ == "__main__":
    samples = [
        "Find the current EUR to USD rate and then price our €129 plan in USD, include 8% tax and summarize the method.",
        "From a CSV of events, compute active users per day; then 7-day retention; finally give a short summary.",
        "Fetch endpoints from an API docs URL and create Python and curl examples for each, then produce a quickstart."
    ]
    for q in samples:
        print("\n\n>>> QUESTION:", q)
        run_once(q, try_llm=True, provider="auto")
