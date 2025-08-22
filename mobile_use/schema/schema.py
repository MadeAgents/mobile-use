from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

from PIL import Image


@dataclass(frozen=True)
class EnvState:
    """State of the environment.

    Attributes:
        pixels: Screenshot of the current state.
        package: Current foreground app package name.
        device_time: Current device time in string format, e.g., "Tue Aug 12 03:04:58 GMT 2025".
    """

    pixels: Image.Image
    package: str
    device_time: Optional[str] = None


@dataclass()
class Action:
    """A structrued representation of an action.
    
    # Example
    result = {'name': 'click', 'parameters': {'x': %d, 'y': %d}}
    action = Action(**result)

    Attributes:
        name: The action type name.
        parameters: The parameters of the action.
    """

    name: str
    parameters: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        pass

    def __repr__(self) -> str:
        kv = []
        if self.parameters:
            for k, v in self.parameters.items():
                if v is not None:
                    kv.append(f"{k}={v}")
        params_str = ','.join(kv)
        return f"{self.name}({params_str})"
    
    def __str__(self) -> str:
        return self.__repr__()


class AgentState(Enum):
    READY = 'READY'
    RUNNING = 'RUNNING'
    CALLUSER = 'CALLUSER'

class AgentStatus(Enum):
    FINISHED = 'FINISHED'
    FAILED = 'FAILED'


@dataclass
class VLMCallingData:
    messages: List[Dict[str,Any]]
    response: str


@dataclass
class BaseStepData:
    step_idx: int
    curr_env_state: EnvState
    content: Optional[str] = None       # VLM response content
    action: Optional[Action] = None
    exec_env_state: Optional[EnvState] = None
    vlm_call_history: Optional[List[VLMCallingData]] = field(default_factory=list)


@dataclass
class SingleAgentStepData(BaseStepData):
    thought: Optional[str] = None
    action_s: Optional[str] = None
    action_desc: Optional[str] = None
    answer: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class MobileUseStepData(SingleAgentStepData):
    # Action related
    action_type_tokens: Optional[List[str]] = None
    action_type_logprobs: Optional[List[float]] = None
    # Plan related
    plan: Optional[str] = None
    sub_goal: Optional[str] = None
    # State related
    progress: Optional[str] = None
    memory: Optional[str] = None
    # Reflection related
    reflection_outcome: Optional[str] = None
    reflection_error: Optional[str] = None
    trajectory_reflection_outcome: Optional[str] = None
    trajectory_reflection_error: Optional[str] = None
    evaluation_result: Optional[str] = None
    evaluation_reason: Optional[str] = None
    # Time related
    step_duration: Optional[float] = None
    exec_duration: Optional[float] = None


@dataclass
class BaseEpisodeData:
    goal: str
    num_steps: int
    status: Optional[str] = None
    message: Optional[str] = None
    trajectory: Optional[List[BaseStepData]] = None


@dataclass
class MobileUseEpisodeData(BaseEpisodeData):
    trajectory: List[MobileUseStepData] = field(default_factory=list)
    finish_count: int = 0
    memory: Optional[str] = ""
