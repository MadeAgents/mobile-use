import pydantic
import yaml
from typing import Optional, Union, Literal


class MobileEnvConfig(pydantic.BaseModel):
    serial_no: str = None
    host: str="127.0.0.1"
    port: int=5037
    wait_after_action_seconds: float = 2.0


class VLMConfig(pydantic.BaseModel):
    model_name: str
    api_key: str
    base_url: str
    max_retry: int = 3
    retry_waiting_seconds: int = 2
    max_tokens: int = 1024
    temperature: float = 0.0

    # This will allow arbitrary extra fields
    class Config:
        extra = 'allow'


class SubAgentConfig(pydantic.BaseModel):
    enabled: bool = False
    vlm: VLMConfig = None
    prompt_config: str = None

class PlannerConfig(SubAgentConfig):
    pass

class OperatorConfig(SubAgentConfig):
    num_histories: int = None
    include_device_time: bool = True
    include_tips: bool = True

class AnswerAgentConfig(SubAgentConfig):
    num_histories: int = None
    include_device_time: bool = True

class ReflectorConfig(SubAgentConfig):
    pass

class TrajectoryReflectorConfig(SubAgentConfig):
    evoke_every_steps: int = 5
    cold_steps: int = 3
    detect_error: bool = True
    num_histories: Union[Literal['auto'], int] = 'auto'
    num_latest_screenshots: int = 0
    max_repeat_action: int = 3
    max_repeat_action_series: int = 2
    max_repeat_screen: int = 3
    max_fail_count: int = 3

class GlobalReflectorConfig(SubAgentConfig):
    num_latest_screenshots: int = 3

class ProgressorConfig(SubAgentConfig):
    pass


class AgentConfig(pydantic.BaseModel):
    vlm: VLMConfig
    env: MobileEnvConfig = MobileEnvConfig()
    enable_log: bool = False
    log_dir: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_file: str):
        """Load configuration from a YAML file and create a instance"""
        with open(yaml_file, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)


class QwenAgentConfig(AgentConfig):
    max_action_retry: int = 3
    enable_think: bool = True
    prompt_config: str = None


class MultiAgentConfig(AgentConfig):
    planner: Optional[PlannerConfig] = None
    operator: Optional[OperatorConfig] = None
    answer_agent: Optional[AnswerAgentConfig] = None
    reflector: Optional[ReflectorConfig] = None
    trajectory_reflector: Optional[TrajectoryReflectorConfig] = None
    global_reflector: Optional[GlobalReflectorConfig] = None
    progressor: Optional[ProgressorConfig] = None
    max_action_retry: int = 3
    reflect_on_demand: bool = False
    logprob_threshold: float = -0.01
    enable_pre_reflection: bool = True
