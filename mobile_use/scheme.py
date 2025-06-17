from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

from PIL import Image

from .action import ACTION_SPACE

############ Environment ############
@dataclass(frozen=True)
class EnvState:
    """State of the environment.

    Attributes:
        pixels: Screenshot of the current state.
        auxiliaries: Additional information about the state.
    """

    pixels: Image.Image
    package: str


############ Action ############
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
        # assert self.name in ACTION_SPACE, f"Invalid action name: {self.name}"
        # action_space = ACTION_SPACE[self.name]
        # if "parameters" in action_space:
        #     assert self.parameters is not None, f"Missing parameters: {action_space['parameters']}"
        #     for k, v in action_space['parameters'].items():
        #         assert k in self.parameters, f"Missing parameter: {k}"
        #     for k, v in self.parameters.items():
        #         assert k in action_space['parameters'], f"Invalid parameter: {k}"

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


############ Agent ############
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
class StepData:
    step_idx: int
    curr_env_state: EnvState
    content: Optional[str] = None       # VLM response content
    thought: Optional[str] = None
    action: Optional[Action] = None
    answer: Optional[str] = None        # The final answer for the task goal
    exec_env_state: Optional[EnvState] = None
    vlm_call_history: Optional[List[VLMCallingData]] = None
    plan: Optional[str] = None
    sub_goal: Optional[str] = None
    action_desc: Optional[str] = None
    action_s: Optional[str] = None
    summary: Optional[str] = None
    progress: Optional[str] = None
    memory: Optional[str] = None
    reflection_outcome: Optional[str] = None
    reflection_error: Optional[str] = None
    long_reflection_outcome: Optional[str] = None
    long_reflection_error: Optional[str] = None
    evaluation_result: Optional[str] = None
    evaluation_reason: Optional[str] = None
    action_type_tokens: Optional[List[str]] = None
    action_type_logprobs: Optional[List[float]] = None
    step_duration: Optional[float] = None
    exec_duration: Optional[float] = None

@dataclass
class EpisodeData:
    goal: str
    num_steps: int
    status: Optional[str] = None
    message: Optional[str] = None
    trajectory: Optional[List[StepData]] = None
    create_time: Optional[str] = field(default_factory=lambda: datetime.now().isoformat())
    input_tips: Optional[str] = None
    retrieved_tips: Optional[str] = None
    output_tips: Optional[str] = None
    finish_count: Optional[int] = 0
    memory: Optional[str] = ""
