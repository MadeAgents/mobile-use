import pydantic
import yaml
from typing import Optional, Union, Literal


class BaseConfig(pydantic.BaseModel):
    @classmethod
    def from_yaml(cls, yaml_file: str):
        """Load configuration from a YAML file and create a instance"""
        with open(yaml_file, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)


class MobileEnvConfig(BaseConfig):
    serial_no: Union[str, None] = None
    host: str = "127.0.0.1"
    port: int = 5037
    go_home: bool = False
    wait_after_action_seconds: float = 2.0


class VLMConfig(BaseConfig):
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


class SubAgentConfig(BaseConfig):
    enabled: bool = False
    vlm: VLMConfig = None
    prompt_config: str = None

class PlannerConfig(SubAgentConfig):
    pass

class KnowledgeConfig(BaseConfig):
    embedding_model_path: str = None
    knowledge_database_dir: str = None
    explored_knowledge_path: str = None

class OperatorConfig(SubAgentConfig):
    name: str = "Operator"
    num_histories: int = None
    include_device_time: bool = True
    include_tips: bool = True
    include_a11y_tree: bool = False
    max_pixels: int = None
    knowledge: Union[KnowledgeConfig, None] = None

class AnswerAgentConfig(SubAgentConfig):
    name: str = "AnswerAgent"
    num_histories: int = None
    include_device_time: bool = True
    include_tips: bool = False
    max_pixels: int = None
    knowledge: Union[KnowledgeConfig, None] = None

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

class NoteTakerConfig(SubAgentConfig):
    pass


class AgentConfig(BaseConfig):
    vlm: VLMConfig = None
    env: MobileEnvConfig = None
    max_steps: int = 10
    enable_log: bool = False
    log_dir: Optional[str] = None


class ReActAgentConfig(AgentConfig):
    num_latest_screenshots: int = 3
    max_action_retry: int = 3
    prompt_config: str = None


class QwenAgentConfig(AgentConfig):
    max_action_retry: int = 3
    enable_think: bool = True
    prompt_config: str = None
    min_pixels: int = 3136
    max_pixels: int = 10035200
    message_type: Literal['single', 'chat'] = 'single'
    num_image_limit: int = 2
    coordinate_type: Literal['absolute', 'relative'] = 'absolute'


class MultiAgentConfig(AgentConfig):
    planner: Optional[PlannerConfig] = None
    operator: Optional[OperatorConfig] = None
    answer_agent: Optional[AnswerAgentConfig] = None
    reflector: Optional[ReflectorConfig] = None
    trajectory_reflector: Optional[TrajectoryReflectorConfig] = None
    global_reflector: Optional[GlobalReflectorConfig] = None
    progressor: Optional[ProgressorConfig] = None
    note_taker: Optional[NoteTakerConfig] = None
    max_action_retry: int = 3
    reflect_on_demand: bool = False
    logprob_threshold: float = -0.01
    enable_pre_reflection: bool = True


class HierarchicalAgentConfig(MultiAgentConfig):
    task_classifier: Optional[SubAgentConfig] = None
    task_orchestrator: Optional[SubAgentConfig] = None
    task_extractor: Optional[SubAgentConfig] = None
    task_rewriter: Optional[SubAgentConfig] = None
    enable_hierarchical_planning: bool = True

