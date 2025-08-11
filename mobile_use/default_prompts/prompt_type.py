from typing import Literal, Optional
from dataclasses import dataclass
from pathlib import Path

import yaml

def load_prompt(prompt_type = Literal["planner"], prompt_config: str=None) -> Optional["Prompt"]:
    match prompt_type:
        case "planner":
            return PlannerPrompt(config=prompt_config) if prompt_config else PlannerPrompt()
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

if __name__ == "__main__":
    prompt = load_prompt("planner")
    print(prompt)
    print(prompt.system_prompt)
    print(prompt.task_prompt)
    print(prompt.init_plan)
    print(prompt.continue_plan)