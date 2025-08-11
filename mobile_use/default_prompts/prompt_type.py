from typing import Literal, Optional
from dataclasses import dataclass
from pathlib import Path

import yaml

def load_prompt(prompt_type = Literal["planner", "operator"], prompt_config: str=None) -> Optional["Prompt"]:
    match prompt_type:
        case "planner":
            return PlannerPrompt(config=prompt_config) if prompt_config else PlannerPrompt()
        case "operator":
            return OperatorPrompt(config=prompt_config) if prompt_config else OperatorPrompt()
        case _:
            raise KeyError(f"Unknown prompt type: {prompt_type}")

@dataclass
class Prompt:
    config: str
    name: str = ""

    def __post_init__(self):
        script_dir = Path(__file__).parent
        with open(script_dir / f"{self.config}", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for attr, value in data.items():
            setattr(self, attr, value)


@dataclass
class PlannerPrompt(Prompt):
    config: str = "planner.yaml"
    system_prompt: str = ""
    task_prompt: str = ""
    init_plan: str = ""
    continue_plan: str = ""


@dataclass
class OperatorPrompt(Prompt):
    config: str = "operator.yaml"
    system_prompt: str = ""
    init_tips: str = ""
    task_prompt: str = ""
    device_time_prompt: str = ""
    plan_prompt: str = ""
    subgoal_prompt: str = ""
    history_prompt: str = ""
    progress_prompt: str = ""
    memory_prompt: str = ""
    reflection_prompt: str = ""
    long_reflection_prompt: str = ""
    global_reflection_prompt: str = ""
    observation_prompt: str = ""
    response_prompt: str = ""


if __name__ == "__main__":
    # prompt = load_prompt("planner")
    # print(prompt)
    # print(prompt.system_prompt)
    # print(prompt.task_prompt)
    # print(prompt.init_plan)
    # print(prompt.continue_plan)
    prompt = load_prompt("operator")
    print(prompt)
