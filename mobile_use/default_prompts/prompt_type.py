from typing import Literal, Optional
from dataclasses import dataclass
from pathlib import Path

import yaml

def load_prompt(prompt_type = Literal["planner", "operator", "reflector", "trajectory_reflector", "global_reflector", "progressor"], prompt_config: str=None) -> Optional["Prompt"]:
    match prompt_type:
        case "planner":
            return PlannerPrompt(config=prompt_config) if prompt_config else PlannerPrompt()
        case "operator":
            return OperatorPrompt(config=prompt_config) if prompt_config else OperatorPrompt()
        case "reflector":
            return ReflectorPrompt(config=prompt_config) if prompt_config else ReflectorPrompt()
        case "trajectory_reflector":
            return TrajectoryReflectorPrompt(config=prompt_config) if prompt_config else TrajectoryReflectorPrompt()
        case "global_reflector":
            return GlobalReflectorPrompt(config=prompt_config) if prompt_config else GlobalReflectorPrompt()
        case "progressor":
            return ProgressorPrompt(config=prompt_config) if prompt_config else ProgressorPrompt()
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


@dataclass
class ReflectorPrompt(Prompt):
    config: str = "reflector.yaml"
    system_prompt: str = ""
    task_prompt: str = ""
    subgoal_prompt: str = ""
    observation_prompt: str = ""
    diff_image_prompt: str = ""
    expection_prompt: str = ""
    response_prompt: str = ""


@dataclass
class TrajectoryReflectorPrompt(Prompt):
    config: str = "trajectory_reflector.yaml"
    system_prompt: str = ""
    task_prompt: str = ""
    plan_prompt: str = ""
    history_prompt: str = ""
    progress_prompt: str = ""
    observation_prompt: str = ""
    error_info_prompt: str = ""
    response_prompt: str = ""


@dataclass
class GlobalReflectorPrompt(Prompt):
    config: str = "global_reflector.yaml"
    system_prompt: str = ""
    task_prompt: str = ""
    plan_prompt: str = ""
    history_prompt: str = ""
    observation_prompt: str = ""
    response_prompt: str = ""


@dataclass
class ProgressorPrompt(Prompt):
    config: str = "progressor.yaml"
    system_prompt: str = ""
    init_progress: str = ""
    continue_progress_start: str = ""
    continue_progress_reflection: str = ""
    continue_progress_response: str = ""


if __name__ == "__main__":
    prompt = load_prompt("planner")
    # print(prompt)
    print(prompt.system_prompt)
    print(prompt.task_prompt)
    # print(prompt.init_plan)
    # print(prompt.continue_plan)
    # prompt = load_prompt("operator")
    # print(prompt)
    # prompt = load_prompt("reflector")
    # print(prompt)
    # prompt = load_prompt("trajectory_reflector")
    # print(prompt)
    # prompt = load_prompt("global_reflector")
    # print(prompt)
    # prompt = load_prompt("progressor")
    # print(prompt)
