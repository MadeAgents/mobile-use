from typing import Literal, Optional
from dataclasses import dataclass
from pathlib import Path

import yaml

def load_prompt(prompt_type: str, prompt_config: str=None) -> Optional["Prompt"]:
    match prompt_type:
        case "qwen_agent":
            return QwenAgentPrompt(config=prompt_config) if prompt_config else QwenAgentPrompt()
        case "planner":
            return PlannerPrompt(config=prompt_config) if prompt_config else PlannerPrompt()
        case "operator":
            return OperatorPrompt(config=prompt_config) if prompt_config else OperatorPrompt()
        case "answer_agent":
            return AnswerAgentPrompt(config=prompt_config) if prompt_config else AnswerAgentPrompt()
        case "reflector":
            return ReflectorPrompt(config=prompt_config) if prompt_config else ReflectorPrompt()
        case "trajectory_reflector":
            return TrajectoryReflectorPrompt(config=prompt_config) if prompt_config else TrajectoryReflectorPrompt()
        case "global_reflector":
            return GlobalReflectorPrompt(config=prompt_config) if prompt_config else GlobalReflectorPrompt()
        case "progressor":
            return ProgressorPrompt(config=prompt_config) if prompt_config else ProgressorPrompt()
        case "note_taker":
            return NoteTakerPrompt(config=prompt_config) if prompt_config else NoteTakerPrompt()
        case "task_classifier":
            return TaskClassifierPrompt(config=prompt_config) if prompt_config else TaskClassifierPrompt()
        case "task_orchestrator":
            return TaskOrchestratorPrompt(config=prompt_config) if prompt_config else TaskOrchestratorPrompt()
        case "task_extractor":
            return TaskExtractorPrompt(config=prompt_config) if prompt_config else TaskExtractorPrompt()
        case "task_rewriter":
            return TaskRewriterPrompt(config=prompt_config) if prompt_config else TaskRewriterPrompt()
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
class QwenAgentPrompt(Prompt):
    config: str = "qwen_agent.yaml"
    system_prompt: str = ""
    task_prompt: str = ""
    history_prompt: str = ""
    thinking_prompt: str = ""


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
    knowledge_prompt: str = ""
    memory_prompt: str = ""
    reflection_prompt: str = ""
    trajectory_reflection_prompt: str = ""
    global_reflection_prompt: str = ""
    observation_prompt: str = ""
    a11y_tree_prompt: str = ""
    response_prompt: str = ""


@dataclass
class AnswerAgentPrompt(OperatorPrompt):
    config: str = "answer_agent.yaml"


@dataclass
class ReflectorPrompt(Prompt):
    config: str = "reflector.yaml"
    system_prompt: str = ""
    task_prompt: str = ""
    subgoal_prompt: str = ""
    observation_prompt: str = ""
    same_image_prompt: str = ""
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
    progress_prompt: str = ""
    observation_prompt: str = ""
    response_prompt: str = ""


@dataclass
class ProgressorPrompt(Prompt):
    config: str = "progressor.yaml"
    system_prompt: str = ""
    task_prompt: str = ""
    init_progress: str = ""
    continue_progress_start: str = ""
    continue_progress_reflection: str = ""
    continue_progress_response: str = ""


@dataclass
class NoteTakerPrompt(Prompt):
    config: str = "note_taker.yaml"
    system_prompt: str = ""
    task_prompt: str = ""
    plan_prompt: str = ""
    subgoal_prompt: str = ""
    memory_prompt: str = ""
    observation_prompt: str = ""
    response_prompt: str = ""


@dataclass
class TaskClassifierPrompt(Prompt):
    config: str = "task_classifier.yaml"
    system_prompt: str = ""
    user_prompt: str = ""

@dataclass
class TaskOrchestratorPrompt(Prompt):
    config: str = "task_orchestrator.yaml"
    system_prompt: str = ""
    user_prompt: str = ""

@dataclass
class TaskExtractorPrompt(Prompt):
    config: str = "task_extractor.yaml"
    system_prompt: str = ""
    user_prompt: str = ""


@dataclass
class TaskRewriterPrompt(Prompt):
    config: str = "task_rewriter.yaml"
    system_prompt: str = ""
    user_prompt: str = ""


if __name__ == "__main__":
    prompt = load_prompt("qwen_agent")
    print(repr(prompt.system_prompt))
    print(repr(prompt.task_prompt))
    print(repr(prompt.history_prompt))
    print(repr(prompt.thinking_prompt))